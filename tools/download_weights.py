import os
from pathlib import Path, PurePosixPath
from huggingface_hub import hf_hub_download


def prepare_base_model():
    print(f"Preparing base stable-diffusion-v1-5 weights...")
    local_dir = "./pretrained_weights/stable-diffusion-v1-5"
    os.makedirs(local_dir, exist_ok=True)
    for hub_file in ["unet/config.json", "unet/diffusion_pytorch_model.bin"]:
        path = Path(hub_file)
        saved_path = local_dir / path
        if os.path.exists(saved_path):
            continue
        hf_hub_download(
            repo_id="runwayml/stable-diffusion-v1-5",
            subfolder=PurePosixPath(path.parent),
            filename=PurePosixPath(path.name),
            local_dir=local_dir,
        )


def prepare_image_encoder():
    print(f"Preparing image encoder weights...")
    local_dir = "./pretrained_weights"
    os.makedirs(local_dir, exist_ok=True)
    for hub_file in ["image_encoder/config.json", "image_encoder/pytorch_model.bin"]:
        path = Path(hub_file)
        saved_path = local_dir / path
        if os.path.exists(saved_path):
            continue
        hf_hub_download(
            repo_id="lambdalabs/sd-image-variations-diffusers",
            subfolder=PurePosixPath(path.parent),
            filename=PurePosixPath(path.name),
            local_dir=local_dir,
        )


def prepare_vae():
    print(f"Preparing vae weights...")
    local_dir = "./pretrained_weights/sd-vae-ft-mse"
    os.makedirs(local_dir, exist_ok=True)
    for hub_file in [
        "config.json",
        "diffusion_pytorch_model.bin",
    ]:
        path = Path(hub_file)
        saved_path = local_dir / path
        if os.path.exists(saved_path):
            continue

        hf_hub_download(
            repo_id="stabilityai/sd-vae-ft-mse",
            subfolder=PurePosixPath(path.parent),
            filename=PurePosixPath(path.name),
            local_dir=local_dir,
        )


def prepare_temporal_vae():
    print(f"Preparing temporal vae weights...")
    local_dir = "./pretrained_weights/vae_temporal_decoder"
    os.makedirs(local_dir, exist_ok=True)
    for hub_file in [
        "vae_temporal_decoder/config.json",
        "vae_temporal_decoder/diffusion_pytorch_model.safetensors",
    ]:
        path = Path(hub_file)
        saved_path = local_dir / path
        if os.path.exists(saved_path):
            continue

        hf_hub_download(
            repo_id="maxin-cn/Latte-1",
            subfolder=PurePosixPath(path.parent),
            filename=PurePosixPath(path.name),
            local_dir=local_dir,
        )

def prepare_mikudance():
    print(f"Preparing mikudance weights...")
    local_dir = "./pretrained_weights"
    os.makedirs(local_dir, exist_ok=True)
    for hub_file in [
        "denoising_unet-60000.pth",
        "reference_unet-60000.pth",
        "motion_module-60000.pth",
    ]:
        path = Path(hub_file)
        saved_path = local_dir / path
        if os.path.exists(saved_path):
            continue

        hf_hub_download(
            repo_id="JiaxuZ/MikuDance",
            subfolder=PurePosixPath(path.parent),
            filename=PurePosixPath(path.name),
            local_dir=local_dir,
        )


if __name__ == "__main__":
    prepare_base_model()
    prepare_image_encoder()
    prepare_vae()
    prepare_temporal_vae()
    prepare_mikudance()
