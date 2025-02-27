import argparse
import copy
import logging
import math
import os
import os.path as osp
import random
import time
import warnings
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np

import diffusers
import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPVisionModelWithProjection
from einops import repeat
from scipy.ndimage import zoom
from accelerate import Accelerator, DistributedType

from src.dataset.anime_video_dataset import AnimeVdoDataset

from src.models.mutual_mix_attention import ReferenceAttentionControl
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_2d_mix import (
    UNet2DConditionModel as UNet2DConditionModel_M,
)
from src.models.unet_3d_mix import UNet3DConditionModel

from src.pipelines.pipeline_stage2_vdo import Pose2VideoPipeline
from tools.scene_motion_tracking import camera_to_scene_motion
from skimage.transform import resize as ski_resize

from src.utils.util import (
    delete_additional_ckpt,
    import_filename,
    read_frames,
    save_videos_grid,
    seed_everything,
    get_fps,
)


warnings.filterwarnings("ignore")

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")


class Net(nn.Module):
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        reference_control_writer,
        reference_control_reader,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader

    def forward(
        self,
        noisy_latents,
        timesteps,
        ref_image_latents,
        clip_image_embeds,
        uncond_fwd: bool = False,
    ):
        if not uncond_fwd:
            ref_timesteps = torch.zeros_like(timesteps)
            self.reference_unet(
                ref_image_latents,
                ref_timesteps,
                encoder_hidden_states=clip_image_embeds,
                return_dict=False,
            )
            self.reference_control_reader.update(self.reference_control_writer)

        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=clip_image_embeds,
        ).sample

        return model_pred


def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[
        timesteps
    ].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
        device=timesteps.device
    )[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr


def log_validation(
    vae,
    image_enc,
    net,
    scheduler,
    accelerator,
    width,
    height,
    clip_length=24,
    generator=None,
):
    logger.info("Running validation... ")

    ori_net = accelerator.unwrap_model(net)

    denoising_unet = copy.deepcopy(ori_net.denoising_unet).to(dtype=torch.float16)
    reference_unet = copy.deepcopy(ori_net.reference_unet).to(dtype=torch.float16)

    vae = vae.to(dtype=torch.float16)
    image_enc = image_enc.to(dtype=torch.float16)

    if generator is None:
        generator = torch.manual_seed(42)

    pipe = Pose2VideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        scheduler=scheduler,
    )
    pipe = pipe.to(accelerator.device)

    ref_image_paths = [
        "./src/dataset/log_val/chars/img-kamisatoayakagenshinimpact.jpg",
    ]
    ref_skel_paths = [
        "./src/dataset/log_val/chars/skel-img-kamisatoayakagenshinimpact.jpg",
    ]
    ref_depth_paths = [
        "./src/dataset/log_val/chars/depm-img-kamisatoayakagenshinimpact.npy",
    ]
    pose_video_paths = [
        "./src/dataset/log_val/poses/pose-frames.mp4",
    ]
    face_video_paths = [
        "./src/dataset/log_val/poses/face-frames.mp4",
    ]
    hand_video_paths = [
        "./src/dataset/log_val/poses/hand-frames.mp4",
    ]
    w2c_paths = ["./src/dataset/log_val/poses/w2c-frames.npy"]
    c2w_paths = ["./src/dataset/log_val/poses/c2w-frames.npy"]

    results = []
    for j, ref_image_path in enumerate(ref_image_paths):
        for i, pose_video_path in enumerate(pose_video_paths):
            pose_tgt_pils = read_frames(pose_video_path)
            pose_tgt_pils = pose_tgt_pils[:clip_length]
            face_tgt_pils = read_frames(face_video_paths[i])
            face_tgt_pils = face_tgt_pils[:clip_length]
            hand_tgt_pils = read_frames(hand_video_paths[i])
            hand_tgt_pils = hand_tgt_pils[:clip_length]

            pose_name = os.path.splitext(os.path.basename(pose_video_path))[0]
            ref_name = os.path.splitext(os.path.basename(ref_image_path))[0]
            num_frames = len(pose_tgt_pils)
            print("Total frames: {}".format(num_frames))

            ref_skel_path = ref_skel_paths[j]
            ref_depth_path = ref_depth_paths[j]

            skel_ref_pil = Image.open(ref_skel_path).convert("RGB")
            depth_map = np.load(ref_depth_path)
            depth_map = ski_resize(depth_map, (1, width // 8, height // 8))
            ref_image_pil = Image.open(ref_image_path).convert("RGB")

            img_transform = transforms.Compose(
                [transforms.Resize((height, width)), transforms.ToTensor()]
            )
            skel_tensor_list = []
            for skel_image in pose_tgt_pils:
                skel_tensor_list.append(img_transform(skel_image))
            pose_tensor = torch.stack(skel_tensor_list, dim=0)  # (f, c, h, w)
            pose_tensor = pose_tensor.transpose(0, 1).unsqueeze(0)

            face_tensor_list = []
            for face_image in face_tgt_pils:
                face_tensor_list.append(img_transform(face_image))
            face_tensor = torch.stack(face_tensor_list, dim=0)  # (f, c, h, w)
            face_tensor = face_tensor.transpose(0, 1).unsqueeze(0)

            ref_image_tensor = img_transform(ref_image_pil)  # (c, h, w)
            ref_image_tensor = ref_image_tensor.unsqueeze(1).unsqueeze(
                0
            )  # (1, c, 1, h, w)
            ref_image_tensor = repeat(
                ref_image_tensor, "b c f h w -> b c (repeat f) h w", repeat=num_frames
            )

            w2c_npy = np.load(w2c_paths[i])
            c2w_npy = np.load(c2w_paths[i])

            w2c_npy_lst = [w2c_npy[k] for k in range(w2c_npy.shape[0])]
            c2w_npy_lst = [c2w_npy[k] for k in range(c2w_npy.shape[0])]

            K = [3.2, 3.2, 1.6, 1.6]
            scene_motion = camera_to_scene_motion(
                w2c_npy_lst, c2w_npy_lst, K, depth_map, width // 8, height // 8
            )
            scene_motion_npy = scene_motion[:clip_length]

            pipeline_output = pipe(
                ref_image_pil,
                skel_ref_pil,
                pose_tgt_pils,
                face_tgt_pils,
                hand_tgt_pils,
                scene_motion_npy,
                width,
                height,
                num_frames,
                20,
                3.5,
                generator=generator,
            )  # b c t h w

            video = pipeline_output.videos

            video = torch.cat(
                [ref_image_tensor, pose_tensor, face_tensor, video], dim=0
            )
            results.append({"name": f"{ref_name}_{pose_name}", "vid": video})

    del pipe
    torch.cuda.empty_cache()

    return results


def main(cfg):
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps,
        mixed_precision=cfg.solver.mixed_precision,
        log_with="mlflow",
        project_dir="./mlruns",
        kwargs_handlers=[kwargs],
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        seed_everything(cfg.seed)

    exp_name = cfg.exp_name
    save_dir = f"{cfg.output_dir}/{exp_name}"
    if accelerator.is_main_process:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    inference_config_path = "./configs/inference/mikudance_config.yaml"
    infer_config = OmegaConf.load(inference_config_path)

    if cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif cfg.weight_dtype == "fp32":
        weight_dtype = torch.float32
    else:
        raise ValueError(
            f"Do not support weight dtype: {cfg.weight_dtype} during training"
        )

    sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
    if cfg.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    val_noise_scheduler = DDIMScheduler(**sched_kwargs)
    sched_kwargs.update({"beta_schedule": "scaled_linear"})
    train_noise_scheduler = DDIMScheduler(**sched_kwargs)

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        cfg.image_encoder_path,
    ).to(dtype=weight_dtype, device="cuda")
    vae = AutoencoderKL.from_pretrained(cfg.vae_model_path).to(
        "cuda", dtype=weight_dtype
    )

    unet = UNet2DConditionModel.from_pretrained(
        cfg.base_model_path,
        subfolder="unet",
    )

    reference_unet = UNet2DConditionModel_M.from_unet(unet)
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        cfg.base_model_path,
        cfg.mm_path,
        subfolder="unet",
        unet_additional_kwargs=OmegaConf.to_container(
            infer_config.unet_additional_kwargs
        ),
    ).to(device="cuda")

    stage1_ckpt_dir = cfg.stage1_ckpt_dir
    stage1_ckpt_step = cfg.stage1_ckpt_step

    denoising_unet.load_state_dict(
        torch.load(
            os.path.join(stage1_ckpt_dir, f"denoising_unet-{stage1_ckpt_step}.pth"),
            map_location="cpu",
        ),
        strict=False,
    )
    reference_unet.load_state_dict(
        torch.load(
            os.path.join(stage1_ckpt_dir, f"reference_unet-{stage1_ckpt_step}.pth"),
            map_location="cpu",
        ),
        strict=False,
    )

    # Freeze
    vae.requires_grad_(False)
    image_enc.requires_grad_(False)
    reference_unet.requires_grad_(False)
    denoising_unet.requires_grad_(False)

    # Set motion module learnable
    for name, module in denoising_unet.named_modules():
        if "motion_modules" in name:
            for params in module.parameters():
                params.requires_grad = True

    for name, module in reference_unet.named_modules():
        if "man_blocks" in name:
            for params in module.parameters():
                params.requires_grad = True

    reference_control_writer = ReferenceAttentionControl(
        reference_unet,
        do_classifier_free_guidance=False,
        mode="write",
        fusion_blocks="full",
    )
    reference_control_reader = ReferenceAttentionControl(
        denoising_unet,
        do_classifier_free_guidance=False,
        mode="read",
        fusion_blocks="full",
    )

    net = Net(
        reference_unet,
        denoising_unet,
        reference_control_writer,
        reference_control_reader,
    )

    if cfg.solver.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            reference_unet.enable_xformers_memory_efficient_attention()
            denoising_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    if cfg.solver.gradient_checkpointing:
        reference_unet.enable_gradient_checkpointing()
        denoising_unet.enable_gradient_checkpointing()

    if cfg.solver.scale_lr:
        learning_rate = (
            cfg.solver.learning_rate
            * cfg.solver.gradient_accumulation_steps
            * cfg.data.train_bs
            * accelerator.num_processes
        )
    else:
        learning_rate = cfg.solver.learning_rate

    # Initialize the optimizer
    if cfg.solver.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    trainable_params = list(filter(lambda p: p.requires_grad, net.parameters()))
    logger.info(f"Total trainable params {len(trainable_params)}")
    optimizer = optimizer_cls(
        trainable_params,
        lr=learning_rate,
        betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2),
        weight_decay=cfg.solver.adam_weight_decay,
        eps=cfg.solver.adam_epsilon,
    )

    # Scheduler
    lr_scheduler = get_scheduler(
        cfg.solver.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.solver.lr_warmup_steps
        * cfg.solver.gradient_accumulation_steps,
        num_training_steps=cfg.solver.max_train_steps
        * cfg.solver.gradient_accumulation_steps,
    )

    train_dataset = AnimeVdoDataset(
        img_size=(cfg.data.train_width, cfg.data.train_height),
        n_sample_frames=cfg.data.n_sample_frames,
        sample_rate=cfg.data.sample_rate,
        img_scale=(1.0, 1.0),
        frame_ratio=cfg.data.frame_ratio,
        cam_ratio=cfg.data.cam_ratio,
        drop_ratio=cfg.data.drop_ratio,
        drop_vdo_ratio=cfg.data.drop_vdo_ratio,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.data.train_bs, shuffle=True, num_workers=4
    )

    # Prepare everything with our `accelerator`.
    (
        net,
        optimizer,
        train_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        net,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.solver.gradient_accumulation_steps
    )
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(
        cfg.solver.max_train_steps / num_update_steps_per_epoch
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run_time = datetime.now().strftime("%Y%m%d-%H%M")
        accelerator.init_trackers(
            exp_name,
            init_kwargs={"mlflow": {"run_name": run_time}},
        )
        # dump config file
        mlflow.log_dict(OmegaConf.to_container(cfg), "config.yaml")

    # Train!
    total_batch_size = (
        cfg.data.train_bs
        * accelerator.num_processes
        * cfg.solver.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.data.train_bs}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {cfg.solver.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {cfg.solver.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if cfg.resume_from_checkpoint:
        if cfg.resume_from_checkpoint != "latest":
            resume_dir = cfg.resume_from_checkpoint
        else:
            resume_dir = save_dir
        # Get the most recent checkpoint
        dirs = os.listdir(resume_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1]
        accelerator.load_state(os.path.join(resume_dir, path))
        accelerator.print(f"Resuming from checkpoint {path}")
        global_step = int(path.split("-")[1])

        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = global_step % num_update_steps_per_epoch

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, cfg.solver.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, num_train_epochs):
        train_loss = 0.0
        t_data_start = time.time()
        for step, batch in enumerate(train_dataloader):
            t_data = time.time() - t_data_start
            with accelerator.accumulate(net):
                # Convert videos to latent space
                target_vdo = batch["target_vdo_frames"].to(weight_dtype)
                with torch.no_grad():
                    video_length = target_vdo.shape[1]
                    target_vdo = rearrange(target_vdo, "b f c h w -> (b f) c h w")
                    latents = vae.encode(target_vdo).latent_dist.sample()
                    latents = rearrange(
                        latents, "(b f) c h w -> b c f h w", f=video_length
                    )
                    latents = latents * 0.18215

                noise = torch.randn_like(latents)
                if cfg.noise_offset > 0:
                    noise += cfg.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1, 1),
                        device=latents.device,
                    )
                bsz = latents.shape[0]
                # Sample a random timestep for each video
                timesteps = torch.randint(
                    0,
                    train_noise_scheduler.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()

                target_pose = batch["target_pose_frames"]  # (bs, f, c, H, W)
                bs, t, c, h, w = target_pose.shape
                target_pose = target_pose.reshape((bs * t, c, h, w)).to(weight_dtype)

                pose_image_latent_lst = []
                for pose_cond_i in range(bs * t):
                    pose_image_latent = vae.encode(
                        target_pose[[pose_cond_i], :]
                    ).latent_dist.mean  # (bst, d, 64, 64)
                    pose_image_latent = pose_image_latent * 0.18215
                    pose_image_latent_lst.append(pose_image_latent)
                pose_tgt_latents = torch.cat(pose_image_latent_lst, dim=0)

                target_face = batch["target_face_frames"]  # (bs, f, c, H, W)
                bs, t, c, h, w = target_face.shape
                target_face = target_face.reshape((bs * t, c, h, w)).to(weight_dtype)

                face_image_latent_lst = []
                for face_cond_i in range(bs * t):
                    face_image_latent = vae.encode(
                        target_face[[face_cond_i], :]
                    ).latent_dist.mean  # (bst, d, 64, 64)
                    face_image_latent = face_image_latent * 0.18215
                    face_image_latent_lst.append(face_image_latent)
                face_tgt_latents = torch.cat(face_image_latent_lst, dim=0)

                target_hand = batch["target_hand_frames"]  # (bs, f, c, H, W)
                bs, t, c, h, w = target_hand.shape
                target_hand = target_hand.reshape((bs * t, c, h, w)).to(weight_dtype)
                hand_image_latent_lst = []
                for hand_cond_i in range(bs * t):
                    hand_image_latent = vae.encode(
                        target_hand[[hand_cond_i], :]
                    ).latent_dist.mean  # (bst, d, 64, 64)
                    hand_image_latent = hand_image_latent * 0.18215
                    hand_image_latent_lst.append(hand_image_latent)
                hand_tgt_latents = torch.cat(hand_image_latent_lst, dim=0)

                tgt_scene_motion = batch[
                    "target_scene_motion_frames"
                ]  # (bs, f, c, H, W)
                bs, t, c, h, w = tgt_scene_motion.shape
                tgt_scene_motion = tgt_scene_motion.reshape((bs * t, c, h, w)).to(
                    weight_dtype
                )

                uncond_fwd = random.random() < cfg.uncond_ratio
                clip_image_list = []
                ref_image_list = []
                ref_pose_image_list = []
                for batch_idx, (
                    ref_img,
                    ref_pose_img,
                    clip_img,
                ) in enumerate(
                    zip(
                        batch["ref_img"],
                        batch["ref_skel_img"],
                        batch["clip_img"],
                    )
                ):
                    if uncond_fwd:
                        clip_image_list.append(torch.zeros_like(clip_img))
                    else:
                        clip_image_list.append(clip_img)
                    ref_image_list.append(ref_img)
                    ref_pose_image_list.append(ref_pose_img)

                with torch.no_grad():
                    ref_img = torch.stack(ref_image_list, dim=0).to(
                        dtype=vae.dtype, device=vae.device
                    )
                    ref_image_latents = vae.encode(
                        ref_img
                    ).latent_dist.sample()  # (bs, d, 64, 64)
                    ref_image_latents = ref_image_latents * 0.18215
                    ref_image_latents = rearrange(
                        ref_image_latents.unsqueeze(1).repeat(1, t, 1, 1, 1),
                        "b t d w h -> (b t) d w h",
                    )

                    ref_pose_img = torch.stack(ref_pose_image_list, dim=0).to(
                        dtype=vae.dtype, device=vae.device
                    )
                    pose_ref_latents = vae.encode(
                        ref_pose_img
                    ).latent_dist.sample()  # (bs, d, 64, 64)
                    pose_ref_latents = pose_ref_latents * 0.18215
                    pose_ref_latents = rearrange(
                        pose_ref_latents.unsqueeze(1).repeat(1, t, 1, 1, 1),
                        "b t d w h -> (b t) d w h",
                    )

                    ref_latents = torch.cat(
                        [
                            ref_image_latents,
                            pose_ref_latents,
                            pose_tgt_latents,
                            face_tgt_latents,
                            hand_tgt_latents,
                            tgt_scene_motion,
                        ],
                        dim=1,
                    )

                    clip_img = torch.stack(clip_image_list, dim=0).to(
                        dtype=image_enc.dtype, device=image_enc.device
                    )
                    clip_image_emb = image_enc(
                        clip_img.to("cuda", dtype=weight_dtype)
                    ).last_hidden_state

                    clip_image_emb_norm = image_enc.vision_model.post_layernorm(
                        clip_image_emb
                    )
                    image_prompt_embeds = image_enc.visual_projection(
                        clip_image_emb_norm
                    )
                    image_prompt_embeds = image_prompt_embeds.repeat((t, 1, 1))

                # add noise
                noisy_latents = train_noise_scheduler.add_noise(
                    latents, noise, timesteps
                )

                # Get the target for loss depending on the prediction type
                if train_noise_scheduler.prediction_type == "epsilon":
                    target = noise
                elif train_noise_scheduler.prediction_type == "v_prediction":
                    target = train_noise_scheduler.get_velocity(
                        latents, noise, timesteps
                    )
                else:
                    raise ValueError(
                        f"Unknown prediction type {train_noise_scheduler.prediction_type}"
                    )

                # ---- Forward!!! -----
                model_pred = net(
                    noisy_latents,
                    timesteps,
                    ref_latents,
                    image_prompt_embeds,
                    uncond_fwd=uncond_fwd,
                )

                if cfg.snr_gamma == 0:
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="mean"
                    )
                else:
                    snr = compute_snr(train_noise_scheduler, timesteps)
                    if train_noise_scheduler.config.prediction_type == "v_prediction":
                        # Velocity objective requires that we add one to SNR values before we divide by them.
                        snr = snr + 1
                    mse_loss_weights = (
                        torch.stack(
                            [snr, cfg.snr_gamma * torch.ones_like(timesteps)], dim=1
                        ).min(dim=1)[0]
                        / snr
                    )
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="none"
                    )
                    loss = (
                        loss.mean(dim=list(range(1, len(loss.shape))))
                        * mse_loss_weights
                    )
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(cfg.data.train_bs)).mean()
                train_loss += avg_loss.item() / cfg.solver.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        trainable_params,
                        cfg.solver.max_grad_norm,
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                reference_control_reader.clear()
                reference_control_writer.clear()
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
                    if global_step % cfg.checkpointing_steps == 0:
                        save_path = os.path.join(save_dir, f"checkpoint-{global_step}")
                        delete_additional_ckpt(save_dir, 3)
                        accelerator.save_state(save_path)

                if global_step % cfg.val.validation_steps == 0:
                    if accelerator.is_main_process:
                        generator = torch.Generator(device=accelerator.device)
                        generator.manual_seed(cfg.seed)

                        sample_dicts = log_validation(
                            vae=vae,
                            image_enc=image_enc,
                            net=net,
                            scheduler=val_noise_scheduler,
                            accelerator=accelerator,
                            width=cfg.data.train_width,
                            height=cfg.data.train_height,
                            clip_length=cfg.data.n_sample_frames,
                            generator=generator,
                        )

                        for sample_id, sample_dict in enumerate(sample_dicts):
                            sample_name = sample_dict["name"]
                            vid = sample_dict["vid"]
                            with TemporaryDirectory() as temp_dir:
                                out_file = Path(
                                    f"{temp_dir}/{global_step:06d}-{sample_name}.gif"
                                )
                                save_videos_grid(vid, out_file, n_rows=4)
                                mlflow.log_artifact(out_file)

                if (
                    global_step % cfg.save_model_step_interval == 0
                    and accelerator.is_main_process
                ):
                    unwrap_net = accelerator.unwrap_model(net)
                    save_checkpoint(
                        unwrap_net.denoising_unet,
                        save_dir,
                        "motion_module",
                        global_step,
                        total_limit=None,
                    )
                    save_checkpoint_full(
                        unwrap_net.denoising_unet,
                        save_dir,
                        "denoising_unet",
                        global_step,
                        total_limit=None,
                    )
                    save_checkpoint_full(
                        unwrap_net.reference_unet,
                        save_dir,
                        "reference_unet",
                        global_step,
                        total_limit=None,
                    )

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "td": f"{t_data:.2f}s",
            }
            t_data_start = time.time()
            progress_bar.set_postfix(**logs)

            if global_step >= cfg.solver.max_train_steps:
                break
        # save model after each epoch
        if (
            epoch + 1
        ) % cfg.save_model_epoch_interval == 0 and accelerator.is_main_process:

            save_path = os.path.join(save_dir, f"checkpoint-{global_step}")
            delete_additional_ckpt(save_dir, 1)
            accelerator.save_state(save_path)

            unwrap_net = accelerator.unwrap_model(net)
            save_checkpoint(
                unwrap_net.denoising_unet,
                save_dir,
                "motion_module",
                global_step,
                total_limit=None,
            )
            save_checkpoint_full(
                unwrap_net.denoising_unet,
                save_dir,
                "denoising_unet",
                global_step,
                total_limit=None,
            )
            save_checkpoint_full(
                unwrap_net.reference_unet,
                save_dir,
                "reference_unet",
                global_step,
                total_limit=None,
            )

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()


def save_checkpoint(model, save_dir, prefix, ckpt_num, total_limit=None):
    save_path = osp.join(save_dir, f"{prefix}-{ckpt_num}.pth")

    if total_limit is not None:
        checkpoints = os.listdir(save_dir)
        checkpoints = [d for d in checkpoints if d.startswith(prefix)]
        checkpoints = sorted(
            checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0])
        )

        if len(checkpoints) >= total_limit:
            num_to_remove = len(checkpoints) - total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]
            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(save_dir, removing_checkpoint)
                os.remove(removing_checkpoint)

    mm_state_dict = OrderedDict()
    state_dict = model.state_dict()
    for key in state_dict:
        if "motion_module" in key:
            mm_state_dict[key] = state_dict[key]

    torch.save(mm_state_dict, save_path)


def save_checkpoint_full(model, save_dir, prefix, ckpt_num, total_limit=None):
    save_path = osp.join(save_dir, f"{prefix}-{ckpt_num}.pth")

    if total_limit is not None:
        checkpoints = os.listdir(save_dir)
        checkpoints = [d for d in checkpoints if d.startswith(prefix)]
        checkpoints = sorted(
            checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0])
        )

        if len(checkpoints) >= total_limit:
            num_to_remove = len(checkpoints) - total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]
            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(save_dir, removing_checkpoint)
                os.remove(removing_checkpoint)

    state_dict = model.state_dict()
    torch.save(state_dict, save_path)


def decode_latents(vae, latents):
    video_length = latents.shape[2]
    latents = 1 / 0.18215 * latents
    latents = rearrange(latents, "b c f h w -> (b f) c h w")
    # video = self.vae.decode(latents).sample
    video = []
    for frame_idx in tqdm(range(latents.shape[0])):
        video.append(vae.decode(latents[frame_idx : frame_idx + 1]).sample)
    video = torch.cat(video)
    video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
    video = (video / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
    video = video.cpu().float().numpy()
    return video


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="./configs/training/train_stage2.yaml"
    )
    args = parser.parse_args()

    if args.config[-5:] == ".yaml":
        config = OmegaConf.load(args.config)
    elif args.config[-3:] == ".py":
        config = import_filename(args.config).cfg
    else:
        raise ValueError("Do not support this format config file")
    main(config)
