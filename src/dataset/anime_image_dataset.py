import random
import os

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor
import pickle
from tqdm import tqdm


class AnimeImgDataset(Dataset):
    def __init__(
        self,
        img_size,
        img_scale=(1.0, 1.0),
        img_ratio=(1.0, 1.1),
        drop_ratio=0.1,
        style_ratio=0.01,
    ):
        super().__init__()

        self.img_size = img_size
        self.img_scale = img_scale
        self.img_ratio = img_ratio
        self.drop_ratio = drop_ratio
        self.style_ratio = style_ratio

        self.img_lst = ["load your image file path list"]
        self.img_name_lst = ["image name list"]
        self.skel_lst = ["image skeleton path list"]
        self.hand_lst = ["image hand path list"]
        self.face_lst = ["image face path list"]

        self.style_img_lst = ["load your image file path list"]   # You may pre-process the images for stylization as introdeced in our paper.
        self.style_img_name_lst = ["image name list"]
        self.style_skel_lst = ["image skeleton path list"]
        self.style_hand_lst = ["image hand path list"]
        self.style_face_lst = ["image face path list"]

        self.style_images_num = len(self.style_img_lst)
        self.images_num = len(self.img_lst)
        print(
            "-----------------------------------------------------------------------------------------"
        )
        print(
            "Total images: "
            + str(len(self.img_lst))
            + " Total style images:"
            + str(len(self.style_img_lst)),
        )
        print(
            "-----------------------------------------------------------------------------------------"
        )

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

    def augmentation(self, image, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        return transform(image)

    def select_random_element_with_index(self, lst):
        # Ensure the list is not empty
        if not lst:
            return None, None
        # Randomly select an index
        index = random.randrange(len(lst))
        # Retrieve the element at the randomly selected index
        element = lst[index]

        return element, index

    def __getitem__(self, index):

        if random.random() < self.style_ratio:
            # organize your own data here
            ref_img_path = ""
            ref_skel_path = ""
            tgt_img_path = ""
            tgt_pose_path = ""
            tgt_face_path = ""
            tgt_hand_path = ""
            ref_img_name = ""
            tgt_img_name = ""
        else:
            # organize your own data here
            ref_img_path = ""
            ref_skel_path = ""
            tgt_img_path = ""
            tgt_pose_path = ""
            tgt_face_path = ""
            tgt_hand_path = ""
            ref_img_name = ""
            tgt_img_name = ""

        ref_pil = Image.open(ref_img_path)
        ref_skel_pil = Image.open(ref_skel_path)
        tgt_pil = Image.open(tgt_img_path)
        tgt_pose_pil = Image.open(tgt_pose_path)
        tgt_face_pil = Image.open(tgt_face_path)
        tgt_hand_pil = Image.open(tgt_hand_path)

        if random.random() < self.drop_ratio:
            tgt_face_pil = Image.new("RGB", tgt_face_pil.size, (0, 0, 0))
        if random.random() < self.drop_ratio:
            tgt_hand_pil = Image.new("RGB", tgt_hand_pil.size, (0, 0, 0))

        state = torch.get_rng_state()
        ref_img = self.augmentation(ref_pil, self.transform, state)
        tgt_img = self.augmentation(tgt_pil, self.transform, state)
        tgt_pose_img = self.augmentation(tgt_pose_pil, self.cond_transform, state)
        ref_skel_img = self.augmentation(ref_skel_pil, self.cond_transform, state)
        tgt_face_img = self.augmentation(tgt_face_pil, self.cond_transform, state)
        tgt_hand_img = self.augmentation(tgt_hand_pil, self.cond_transform, state)

        clip_img = self.clip_image_processor(
            images=ref_pil, return_tensors="pt"
        ).pixel_values[0]

        sample = dict(
            ref_img_name=ref_img_name,
            tgt_img_name=tgt_img_name,
            ref_img=ref_img,
            tgt_img=tgt_img,
            ref_skel_img=ref_skel_img,
            tgt_pose_img=tgt_pose_img,
            tgt_face_img=tgt_face_img,
            tgt_hand_img=tgt_hand_img,
            clip_img=clip_img,
        )

        return sample

    def __len__(self):
        return len(self.img_lst)


if __name__ == "__main__":
    pass
