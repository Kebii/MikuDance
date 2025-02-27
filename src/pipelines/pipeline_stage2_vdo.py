# Adapted from https://github.com/magic-research/magic-animate/blob/main/magicanimate/pipelines/pipeline_animation.py
import inspect
import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from diffusers import DiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import BaseOutput, deprecate, is_accelerate_available, logging
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from tqdm import tqdm
from transformers import CLIPImageProcessor

from src.models.mutual_mix_attention import ReferenceAttentionControl
from src.pipelines.context import get_context_scheduler
from src.pipelines.utils import get_tensor_interpolation_method
from torchvision.transforms.functional import pil_to_tensor


@dataclass
class Pose2VideoPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


class Pose2VideoPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae,
        image_encoder,
        reference_unet,
        denoising_unet,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        image_proj_model=None,
        tokenizer=None,
        text_encoder=None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            scheduler=scheduler,
            image_proj_model=image_proj_model,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.clip_image_processor = CLIPImageProcessor()
        self.ref_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True
        )
        self.cond_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_convert_rgb=True,
            do_normalize=False,
        )

        self.decode_chunk_size = 16

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def decode_latents(self, latents):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        # video = self.vae.decode(latents).sample
        video = []
        for frame_idx in tqdm(range(latents.shape[0])):
            frame = self.vae.decode(latents[frame_idx : frame_idx + 1]).sample
            frame = (frame / 2 + 0.5).clamp(0, 1)
            video.append(frame)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        # video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    def decode_temporal(self, latents):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")

        video = []
        for frame_idx in tqdm(range(0, latents.shape[0], self.decode_chunk_size)):
            in_frames = latents[frame_idx : frame_idx + self.decode_chunk_size]
            decode_kwargs = {}
            decode_kwargs["num_frames"] = in_frames.shape[0]
            frame = self.vae.decode(in_frames, **decode_kwargs).sample
            frame = (frame / 2 + 0.5).clamp(0, 1)
            frame = frame.cpu().float()
            video.append(frame)

        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = video.cpu().float().numpy()
        return video

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        width,
        height,
        video_length,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            video_length,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def _encode_prompt(
        self,
        prompt,
        device,
        num_videos_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
    ):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
            )

        if (
            hasattr(self.text_encoder.config, "use_attention_mask")
            and self.text_encoder.config.use_attention_mask
        ):
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(
            bs_embed * num_videos_per_prompt, seq_len, -1
        )

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if (
                hasattr(self.text_encoder.config, "use_attention_mask")
                and self.text_encoder.config.use_attention_mask
            ):
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(
                batch_size * num_videos_per_prompt, seq_len, -1
            )

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    def interpolate_latents(
        self, latents: torch.Tensor, interpolation_factor: int, device
    ):
        if interpolation_factor < 2:
            return latents

        new_latents = torch.zeros(
            (
                latents.shape[0],
                latents.shape[1],
                ((latents.shape[2] - 1) * interpolation_factor) + 1,
                latents.shape[3],
                latents.shape[4],
            ),
            device=latents.device,
            dtype=latents.dtype,
        )

        org_video_length = latents.shape[2]
        rate = [i / interpolation_factor for i in range(interpolation_factor)][1:]

        new_index = 0

        v0 = None
        v1 = None

        for i0, i1 in zip(range(org_video_length), range(org_video_length)[1:]):
            v0 = latents[:, :, i0, :, :]
            v1 = latents[:, :, i1, :, :]

            new_latents[:, :, new_index, :, :] = v0
            new_index += 1

            for f in rate:
                v = get_tensor_interpolation_method()(
                    v0.to(device=device), v1.to(device=device), f
                )
                new_latents[:, :, new_index, :, :] = v.to(latents.device)
                new_index += 1

        new_latents[:, :, new_index, :, :] = v1
        new_index += 1

        return new_latents

    @torch.no_grad()
    def __call__(
        self,
        ref_image,
        ref_skel_image,
        tgt_pose_images,
        tgt_face_images,
        tgt_hand_images,
        scene_motion_npy,
        width,
        height,
        video_length,
        num_inference_steps,
        guidance_scale,
        num_images_per_prompt=1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        context_schedule="uniform",
        context_frames=32,  # 30
        context_stride=1,
        context_overlap=8,  # 8
        context_batch_size=1,
        interpolation_factor=1,
        **kwargs,
    ):
        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        device = self._execution_device

        do_classifier_free_guidance = guidance_scale > 1.0

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        batch_size = 1

        # Prepare clip image embeds
        clip_image = self.clip_image_processor.preprocess(
            ref_image.resize((224, 224)), return_tensors="pt"
        ).pixel_values
        clip_image_emb = self.image_encoder(
            clip_image.to(device, dtype=self.image_encoder.dtype)
        ).last_hidden_state

        clip_image_emb_norm = self.image_encoder.vision_model.post_layernorm(
            clip_image_emb
        )
        image_prompt_embeds = self.image_encoder.visual_projection(clip_image_emb_norm)

        uncond_image_prompt_embeds = torch.zeros_like(image_prompt_embeds)

        if do_classifier_free_guidance:
            image_prompt_embeds = torch.cat(
                [uncond_image_prompt_embeds, image_prompt_embeds], dim=0
            )

        reference_control_writer = ReferenceAttentionControl(
            self.reference_unet,
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode="write",
            batch_size=batch_size,
            fusion_blocks="full",
        )
        reference_control_reader = ReferenceAttentionControl(
            self.denoising_unet,
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode="read",
            batch_size=batch_size,
            fusion_blocks="full",
        )

        num_channels_latents = self.denoising_unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            width,
            height,
            video_length,
            image_prompt_embeds.dtype,
            device,
            generator,
        )

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Prepare ref image latents
        ref_image_tensor = self.ref_image_processor.preprocess(
            ref_image, height=height, width=width
        )  # (bs, c, width, height)
        ref_image_tensor = ref_image_tensor.to(
            dtype=self.vae.dtype, device=self.vae.device
        )
        ref_image_latents = self.vae.encode(ref_image_tensor).latent_dist.mean
        ref_image_latents = ref_image_latents * 0.18215  # (b, 4, h, w)
        ref_image_latents = rearrange(
            ref_image_latents.unsqueeze(1).repeat(1, video_length, 1, 1, 1),
            "b t d w h -> (b t) d w h",
        )

        # Prepare a list of pose condition images
        pose_cond_tensor_list = []
        for pose_image in tgt_pose_images:
            pose_cond_tensor = self.cond_image_processor.preprocess(
                pose_image, height=height, width=width
            )
            pose_cond_tensor = pose_cond_tensor.unsqueeze(1)  # (bs, 1, c, h, w)
            pose_cond_tensor_list.append(pose_cond_tensor)
        pose_cond_tensor = torch.cat(pose_cond_tensor_list, dim=1)  # (bs, t, c, h, w)
        bs, f, c, h, w = pose_cond_tensor.shape
        pose_cond_tensor = pose_cond_tensor.reshape((bs * f, c, h, w)).to(
            dtype=self.vae.dtype, device=self.vae.device
        )

        pose_image_latent_lst = []
        for pose_cond_i in range(bs * f):
            pose_image_latent = self.vae.encode(
                pose_cond_tensor[[pose_cond_i], :]
            ).latent_dist.mean  # (bs, d, 64, 64)
            pose_image_latent = pose_image_latent * 0.18215
            pose_image_latent_lst.append(pose_image_latent)
        pose_tgt_latents = torch.cat(pose_image_latent_lst, dim=0)

        pose_ref_tensor = self.cond_image_processor.preprocess(
            ref_skel_image, height=height, width=width
        )
        pose_ref_tensor = pose_ref_tensor.to(
            dtype=self.vae.dtype, device=self.vae.device
        )
        pose_ref_latents = self.vae.encode(
            pose_ref_tensor
        ).latent_dist.mean  # (bs, d, 64, 64)
        pose_ref_latents = pose_ref_latents * 0.18215
        pose_ref_latents = rearrange(
            pose_ref_latents.unsqueeze(1).repeat(1, video_length, 1, 1, 1),
            "b f d w h -> (b f) d w h",
        )

        face_cond_tensor_list = []
        for face_image in tgt_face_images:
            face_cond_tensor = self.cond_image_processor.preprocess(
                face_image, height=height, width=width
            )
            face_cond_tensor = face_cond_tensor.unsqueeze(1)  # (bs, 1, c, h, w)
            face_cond_tensor_list.append(face_cond_tensor)
        face_cond_tensor = torch.cat(face_cond_tensor_list, dim=1)  # (bs, t, c, h, w)
        bs, f, c, h, w = face_cond_tensor.shape
        face_cond_tensor = face_cond_tensor.reshape((bs * f, c, h, w)).to(
            dtype=self.vae.dtype, device=self.vae.device
        )

        face_image_latent_lst = []
        for face_cond_i in range(bs * f):
            face_image_latent = self.vae.encode(
                face_cond_tensor[[face_cond_i], :]
            ).latent_dist.mean  # (bs, d, 64, 64)
            face_image_latent = face_image_latent * 0.18215
            face_image_latent_lst.append(face_image_latent)
        face_tgt_latents = torch.cat(face_image_latent_lst, dim=0)

        hand_cond_tensor_list = []
        for hand_image in tgt_hand_images:
            hand_cond_tensor = self.cond_image_processor.preprocess(
                hand_image, height=height, width=width
            )
            hand_cond_tensor = hand_cond_tensor.unsqueeze(1)  # (bs, 1, c, h, w)
            hand_cond_tensor_list.append(hand_cond_tensor)
        hand_cond_tensor = torch.cat(hand_cond_tensor_list, dim=1)  # (bs, t, c, h, w)
        bs, f, c, h, w = hand_cond_tensor.shape
        hand_cond_tensor = hand_cond_tensor.reshape((bs * f, c, h, w)).to(
            dtype=self.vae.dtype, device=self.vae.device
        )

        hand_image_latent_lst = []
        for hand_cond_i in range(bs * f):
            hand_image_latent = self.vae.encode(
                hand_cond_tensor[[hand_cond_i], :]
            ).latent_dist.mean  # (bs, d, 64, 64)
            hand_image_latent = hand_image_latent * 0.18215
            hand_image_latent_lst.append(hand_image_latent)
        hand_tgt_latents = torch.cat(hand_image_latent_lst, dim=0)

        scene_motion_ts = torch.from_numpy(scene_motion_npy).to(
            dtype=self.vae.dtype, device=self.vae.device
        )
        scene_motion_ts = scene_motion_ts[None,].repeat((bs, 1, 1, 1, 1))
        scene_motion_ts = rearrange(scene_motion_ts, "b f d w h -> (b f) d w h")

        ref_latents = torch.cat(
            [
                ref_image_latents,
                pose_ref_latents,
                pose_tgt_latents,
                face_tgt_latents,
                hand_tgt_latents,
                scene_motion_ts,
            ],
            dim=1,
        )

        ref_latents = rearrange(ref_latents, "(b f) d w h -> b f d w h", b=bs, f=f)

        context_scheduler = get_context_scheduler(context_schedule)

        # denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                noise_pred = torch.zeros(
                    (
                        latents.shape[0] * (2 if do_classifier_free_guidance else 1),
                        *latents.shape[1:],
                    ),
                    device=latents.device,
                    dtype=latents.dtype,
                )
                counter = torch.zeros(
                    (1, 1, latents.shape[2], 1, 1),
                    device=latents.device,
                    dtype=latents.dtype,
                )

                # 1. Forward reference image
                # if i == 0:
                #     self.reference_unet(
                #         ref_latents.repeat(
                #             (2 if do_classifier_free_guidance else 1), 1, 1, 1
                #         ),
                #         torch.zeros_like(t),
                #         # t,
                #         encoder_hidden_states=image_prompt_embeds,
                #         return_dict=False,
                #     )
                #     reference_control_reader.update(reference_control_writer)

                context_queue = list(
                    context_scheduler(
                        0,
                        num_inference_steps,
                        latents.shape[2],
                        context_frames,
                        context_stride,
                        0,
                    )
                )
                num_context_batches = math.ceil(len(context_queue) / context_batch_size)

                context_queue = list(
                    context_scheduler(
                        0,
                        num_inference_steps,
                        latents.shape[2],
                        context_frames,
                        context_stride,
                        context_overlap,
                    )
                )

                num_context_batches = math.ceil(len(context_queue) / context_batch_size)

                global_context = []
                for i in range(num_context_batches):
                    global_context.append(
                        context_queue[
                            i * context_batch_size : (i + 1) * context_batch_size
                        ]
                    )

                for context in global_context:
                    # 3.1 expand the latents if we are doing classifier free guidance
                    latent_model_input = (
                        torch.cat([latents[:, :, c] for c in context])
                        .to(device)
                        .repeat(2 if do_classifier_free_guidance else 1, 1, 1, 1, 1)
                    )
                    latent_model_input = self.scheduler.scale_model_input(
                        latent_model_input, t
                    )
                    b, c, f, h, w = latent_model_input.shape

                    ref_latent_input = (
                        torch.cat([ref_latents[:, c] for c in context])
                        .to(device)
                        .repeat(2 if do_classifier_free_guidance else 1, 1, 1, 1, 1)
                    )
                    ref_latent_input = rearrange(
                        ref_latent_input, "b f d w h -> (b f) d w h"
                    )

                    image_prompt_embeds_input = image_prompt_embeds.repeat((f, 1, 1))

                    self.reference_unet(
                        ref_latent_input,
                        torch.zeros_like(t),
                        encoder_hidden_states=image_prompt_embeds_input,
                        return_dict=False,
                    )
                    reference_control_reader.update(reference_control_writer)

                    pred = self.denoising_unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=image_prompt_embeds_input[:b],
                        return_dict=False,
                    )[0]

                    for j, c in enumerate(context):
                        noise_pred[:, :, c] = noise_pred[:, :, c] + pred
                        counter[:, :, c] = counter[:, :, c] + 1

                    reference_control_reader.clear()
                    reference_control_writer.clear()

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = (noise_pred / counter).chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                ).prev_sample

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        if interpolation_factor > 0:
            latents = self.interpolate_latents(latents, interpolation_factor, device)
        # Post-processing

        images = self.decode_latents(latents)  # (b, c, f, h, w)

        # Convert to tensor
        if output_type == "tensor":
            images = torch.from_numpy(images)

        if not return_dict:
            return images

        return Pose2VideoPipelineOutput(videos=images)
