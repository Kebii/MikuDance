import argparse
import os
from datetime import datetime
from pathlib import Path
import random


import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler, AutoencoderKLTemporalDecoder
from einops import repeat
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection

from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_2d_mix import (
    UNet2DConditionModel as UNet2DConditionModel_MIX,
)
from src.models.unet_3d_mix import UNet3DConditionModel
from src.pipelines.pipeline_mikudance import MikuDanceVideoPipeline
from src.utils.util import get_fps, read_frames, save_videos_grid

from tools.scene_motion_tracking import camera_to_scene_motion
from skimage.transform import resize as ski_resize


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("-W", type=int, default=768)
    parser.add_argument("-H", type=int, default=768)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--cfg", type=float, default=3.5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--fps", type=int)
    parser.add_argument(
        "--video_decoder",
        action="store_true",
        help="The temporal decoder produces less noise in the results but leads to longer inference times.",
    )
    args = parser.parse_args()

    return args


def get_tensor(pils, height, width):
    img_transform = transforms.Compose(
        [transforms.Resize((height, width)), transforms.ToTensor()]
    )
    tensor_list = []
    for image_i in pils:
        tensor_list.append(img_transform(image_i))
    image_tensor = torch.stack(tensor_list, dim=0)  # (f, c, h, w)
    image_tensor = image_tensor.transpose(0, 1).unsqueeze(0)

    return image_tensor


def main():
    args = parse_args()

    config = OmegaConf.load(args.config)

    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32
    cuda_device = "cuda"

    if args.video_decoder:
        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            config.pretrained_temporal_vae_path
        ).to(cuda_device, dtype=weight_dtype)
    else:
        vae = AutoencoderKL.from_pretrained(
            config.pretrained_vae_path,
        ).to(cuda_device, dtype=weight_dtype)

    unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_base_model_path,
        subfolder="unet",
    ).to(device=cuda_device)
    reference_unet = UNet2DConditionModel_MIX.from_unet(unet)

    inference_config_path = config.inference_config
    infer_config = OmegaConf.load(inference_config_path)

    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        config.pretrained_base_model_path,
        config.motion_module_path,
        subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs,
    ).to(dtype=weight_dtype, device=cuda_device)

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        config.image_encoder_path
    ).to(dtype=weight_dtype, device=cuda_device)

    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)

    generator = torch.manual_seed(args.seed)

    width, height = args.W, args.H

    assert width % 8 == 0 and height % 8 == 0                 # width and height must be divisible by 8, since the vae is trained on 1/8 resolution

    # load pretrained weights
    denoising_unet.load_state_dict(
        torch.load(config.denoising_unet_path, map_location="cpu"),
        strict=False,
    )
    reference_unet.load_state_dict(
        torch.load(config.reference_unet_path, map_location="cpu"),
    )

    denoising_unet.eval()
    reference_unet.eval()

    pipe = MikuDanceVideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        scheduler=scheduler,
        video_decoder=args.video_decoder,
    )
    pipe = pipe.to("cuda", dtype=weight_dtype)

    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M%S")
    save_dir_name = f"{time_str}--seed_{args.seed}-{args.W}x{args.H}"

    save_dir = Path(f"output/{date_str}/{save_dir_name}")
    save_dir.mkdir(exist_ok=True, parents=True)

    random.seed(args.seed)
    ref_image_path = config.ref_image_path
    ref_depth_path = config.ref_depth_path
    ref_skel_path = config.ref_skel_path
    tgt_pose_path = config.tgt_pose_path
    tgt_face_path = config.tgt_face_path
    tgt_hand_path = config.tgt_hand_path
    tgt_w2c_path = config.tgt_w2c_path
    tgt_c2w_path = config.tgt_c2w_path

    if tgt_pose_path is None or tgt_pose_path == "None":
        raise ValueError("Target pose is required!")

    pose_pils = read_frames(tgt_pose_path)
    src_fps = get_fps(tgt_pose_path)
    num_frames = len(pose_pils)

    if tgt_face_path is None or tgt_face_path == "None":
        face_pils = [
            Image.new("RGB", pose_pils[0].size, (0, 0, 0)) for i in range(num_frames)
        ]
    else:
        face_pils = read_frames(tgt_face_path)

    if tgt_hand_path is None or tgt_hand_path == "None":
        hand_pils = [
            Image.new("RGB", pose_pils[0].size, (0, 0, 0)) for i in range(num_frames)
        ]
    else:
        hand_pils = read_frames(tgt_hand_path)

    if tgt_w2c_path is None or tgt_w2c_path == "None" or tgt_c2w_path is None or tgt_c2w_path == "None":
        w2c_npy = np.eye(4).reshape((1, 4, 4)).repeat(num_frames, axis=0)
        c2w_npy = np.eye(4).reshape((1, 4, 4)).repeat(num_frames, axis=0)
    else:
        w2c_npy = np.load(tgt_w2c_path)
        c2w_npy = np.load(tgt_c2w_path)

    if ref_depth_path is None or ref_depth_path == "None":
        depth_map = np.zeros((1, height, width))
    else:
        depth_map = np.load(ref_depth_path)

    w2c_npy_lst = [w2c_npy[k] for k in range(w2c_npy.shape[0])]
    c2w_npy_lst = [c2w_npy[k] for k in range(c2w_npy.shape[0])]

    K = [3.2, 3.2, 1.6, 1.6]
    depth_map = ski_resize(depth_map, (1, height // 8, width // 8))
    scene_motion_npy = camera_to_scene_motion(
        w2c_npy_lst, c2w_npy_lst, K, depth_map, width // 8, height // 8, False
    )

    print("Total frames: {}".format(num_frames))

    skel_name = os.path.splitext(os.path.basename(tgt_pose_path))[0]
    ref_name = os.path.splitext(os.path.basename(ref_image_path))[0]

    img_transform = transforms.Compose(
        [transforms.Resize((height, width)), transforms.ToTensor()]
    )

    pose_tensor = get_tensor(pose_pils, height, width)

    ref_image_pil = Image.open(ref_image_path).convert("RGB")
    ref_skel_pil = Image.open(ref_skel_path).convert("RGB")

    ref_image_tensor = img_transform(ref_image_pil)  # (c, h, w)
    ref_image_tensor = ref_image_tensor.unsqueeze(1).unsqueeze(0)  # (1, c, 1, h, w)
    ref_image_tensor = repeat(
        ref_image_tensor, "b c f h w -> b c (repeat f) h w", repeat=num_frames
    )

    pipeline_output = pipe(
        ref_image_pil,
        ref_skel_pil,
        pose_pils,
        face_pils,
        hand_pils,
        scene_motion_npy,
        width,
        height,
        num_frames,
        args.steps,
        args.cfg,
        generator=generator,
    )  # b c t h w

    video = pipeline_output.videos

    video = torch.cat([ref_image_tensor, pose_tensor, video], dim=0)
    save_videos_grid(
        video,
        f"{save_dir}/{skel_name}_{ref_name}_{args.H}x{args.W}_{int(args.cfg)}_{time_str}.mp4",
        n_rows=3,
        fps=src_fps if args.fps is None else args.fps,
    )


if __name__ == "__main__":
    main()
