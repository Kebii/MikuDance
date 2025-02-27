import torch
import numpy as np
import cv2
import random
import os
import argparse

import diffusers
from diffusers import UniPCMultistepScheduler
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
from diffusers.utils import load_image
from PIL import Image
from tqdm import tqdm
from transformers import DPTFeatureExtractor, DPTForDepthEstimation

def get_depth_map(image, feature_extractor, depth_estimator, size, device):
    image = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad(), torch.autocast("cuda"):
        depth_map = depth_estimator(image).predicted_depth

    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=size,
        mode="bicubic",
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    depth_image = torch.cat([depth_map] * 3, dim=1)

    depth_image = depth_image.permute(0, 2, 3, 1).cpu().numpy()[0]
    depth_image = Image.fromarray((depth_image * 255.0).clip(0, 255).astype(np.uint8))
    return depth_map.squeeze(0), depth_image

def main(args):

    feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
    depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")

    frame = Image.open(args.image_path).convert("RGB")
    frame_name = os.path.basename(args.image_path)
    size = frame.size[::-1]
    depth, depth_image = get_depth_map(frame, feature_extractor, depth_estimator, size, "cuda")
    depth = depth.detach().cpu().numpy()

    save_path_depth = os.path.join(args.save_dir, "depm-"+frame_name.replace(".jpg", "").replace(".png", "")+".npy")
    save_image_depth = os.path.join(args.save_dir, "depi-"+frame_name)
    np.save(save_path_depth, depth)
    depth_image.save(save_image_depth)
            

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="./demo_samples/chars/img-kotobukitsumugik.jpg")
    parser.add_argument("--save_dir", default="./demo_samples/chars/")
    args = parser.parse_args()

    main(args)