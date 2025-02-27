import random
import os
from typing import List

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor
import numpy as np
import pickle
from skimage.transform import resize as ski_resize
from tools.scene_motion_tracking import camera_to_scene_motion


class AnimeVdoDataset(Dataset):
    def __init__(
        self,
        img_size,
        sample_rate,
        n_sample_frames,
        img_scale=(1.0, 1.0),
        img_ratio=(1.0, 1.1),
        vdo_length=24,
        drop_ratio=0.1,
        drop_vdo_ratio=0.05,
        frame_ratio=0.1,
        cam_ratio=0.02,
    ):
        super().__init__()

        self.img_size = img_size
        self.img_scale = img_scale
        self.img_ratio = img_ratio
        self.cam_ratio = cam_ratio
        self.vdo_length = vdo_length
        self.sample_rate = sample_rate
        self.n_sample_frames = n_sample_frames
        self.frame_ratio = frame_ratio
        self.drop_ratio = drop_ratio
        self.drop_vdo_ratio = drop_vdo_ratio

        self.video_path_lst = ["load your data"]
        self.video_frames_lst = ["load your data"]
        self.video_depth_lst = ["load your data"]
        self.video_skels_lst = ["load your data"]
        self.video_faces_lst = ["load your data"]
        self.video_hands_lst = ["load your data"]
        self.video_w2c_lst = ["load your data"]
        self.video_c2w_lst = ["load your data"]
        self.video_frames_name_lst = ["load your data"]

        self.cvideo_path_lst = ["load your data"]         # # You may collect a set of camera videos as introduced in our paper.
        self.cvideo_frames_lst = ["load your data"]
        self.cvideo_depth_lst = ["load your data"]
        self.cvideo_frames_name_lst = ["load your data"]
        self.cvideo_w2c_lst = ["load your data"]
        self.cvideo_c2w_lst = ["load your data"]

        self.camera_video_num = len(self.cvideo_path_lst)
        print("Camera videos: " + str(self.camera_video_num))
        print("Total videos: " + str(len(self.video_frames_lst)))

        self.clip_image_processor = CLIPImageProcessor()

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (self.img_size[1], self.img_size[0]),
                    scale=self.img_scale,
                    ratio=self.img_ratio,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.cond_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (self.img_size[1], self.img_size[0]),
                    scale=self.img_scale,
                    ratio=self.img_ratio,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
            ]
        )

    def augmentation(self, images, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        if isinstance(images, List):
            transformed_images = [transform(img) for img in images]
            ret_tensor = torch.stack(transformed_images, dim=0)  # (f, c, h, w)
        else:
            ret_tensor = transform(images)  # (c, h, w)
        return ret_tensor

    def __getitem__(self, index):

        # organize your own data here
        if random.random() < self.cam_ratio:
            video_path = ""
            ref_image_path = ""
            ref_depth_path = ""
            ref_skel = ""
            w2c_npy_lst = []
            c2w_npy_lst = []
            vdo_pil_image_list = []
            pose_pil_image_list = []
            face_pil_image_list = []
            hand_pil_image_list = []
        else:
            video_path = ""
            ref_image_path = ""
            ref_depth_path = ""
            ref_skel = ""
            w2c_npy_lst = []
            c2w_npy_lst = []
            vdo_pil_image_list = []
            pose_pil_image_list = []
            face_pil_image_list = []
            hand_pil_image_list = []
        
        if random.random() < self.drop_vdo_ratio:
            face_pil_image_list = []
        if random.random() < self.drop_vdo_ratio:
            hand_pil_image_list = []

        depth_map = np.load(ref_depth_path)
        depth_map = ski_resize(
            depth_map, (1, self.img_size[0] // 8, self.img_size[1] // 8)
        )

        K = [3.2, 3.2, 1.6, 1.6]
        scene_motion = camera_to_scene_motion(
            w2c_npy_lst,
            c2w_npy_lst,
            K,
            depth_map,
            self.img_size[0] // 8,
            self.img_size[1] // 8,
        )

        state = torch.get_rng_state()
        target_vdo = self.augmentation(vdo_pil_image_list, self.transform, state)
        target_pose = self.augmentation(pose_pil_image_list, self.cond_transform, state)
        target_face = self.augmentation(face_pil_image_list, self.cond_transform, state)
        target_hand = self.augmentation(hand_pil_image_list, self.cond_transform, state)
        ref_skel_img = self.augmentation(ref_skel, self.cond_transform, state)

        ref_pil = Image.open(ref_image_path).convert("RGB")
        clip_img = self.clip_image_processor(
            images=ref_pil, return_tensors="pt"
        ).pixel_values[0]

        ref_img = self.augmentation(ref_pil, self.transform, state)

        if random.random() < self.drop_vdo_ratio:
            scene_motion = np.zeros(scene_motion.shape)
        scene_motion_ts = torch.from_numpy(scene_motion)

        sample = dict(
            video_path=video_path,
            target_vdo_frames=target_vdo,
            target_pose_frames=target_pose,
            target_face_frames=target_face,
            target_hand_frames=target_hand,
            target_scene_motion_frames=scene_motion_ts,
            ref_img=ref_img,
            ref_skel_img=ref_skel_img,
            clip_img=clip_img,
        )

        return sample

    def __len__(self):
        return len(self.video_path_lst)


if __name__ == "__main__":
    pass
