import argparse
import os
import sys
import numpy as np
import torch
from PIL import Image
import clip
import transforms as T
from models import build_model
from predefined_keypoints import *
from util import box_ops
from util.config import Config
from util.utils import clean_state_dict
import matplotlib.pyplot as plt
from torchvision.ops import nms
from io import BytesIO
import cv2
from tqdm import tqdm
from torch.utils.data import Dataset


def text_encoding(instance_names, keypoints_names, model, device):

    ins_text_embeddings = []
    for cat in instance_names:
        instance_description = (
            f"a photo of {cat.lower().replace('_', ' ').replace('-', ' ')}"
        )
        text = clip.tokenize(instance_description).to(device)
        text_features = model.encode_text(text)  # 1*512
        ins_text_embeddings.append(text_features)
    ins_text_embeddings = torch.cat(ins_text_embeddings, dim=0)

    kpt_text_embeddings = []

    for kpt in keypoints_names:
        kpt_description = f"a photo of {kpt.lower().replace('_', ' ')}"
        text = clip.tokenize(kpt_description).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text)  # 1*512
        kpt_text_embeddings.append(text_features)

    kpt_text_embeddings = torch.cat(kpt_text_embeddings, dim=0)

    return ins_text_embeddings, kpt_text_embeddings


def get_pose_image(image_pil, tgt, keypoint_skeleton, keypoint_text_prompt):
    num_kpts = len(keypoint_text_prompt)
    W, H = tgt["size"]
    fig = plt.figure(frameon=False)
    dpi = plt.gcf().dpi
    fig.set_size_inches(W / dpi, H / dpi)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax = plt.gca()
    ax.imshow(image_pil, aspect="equal")
    ax = plt.gca()
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")
    kpt_color = [
        [255, 255, 100],
        [255, 255, 100],
        [255, 255, 100],
        [255, 255, 100],
        [255, 255, 100],
        [255, 0, 0],
        [160, 32, 240],
        [255, 0, 0],
        [160, 32, 240],
        [255, 0, 0],
        [160, 32, 240],
        [0, 255, 0],
        [51, 153, 255],
        [0, 255, 0],
        [51, 153, 255],
        [0, 255, 0],
        [51, 153, 255],
    ]
    link_color = [
        [0, 255, 0],
        [0, 255, 0],
        [51, 153, 255],
        [51, 153, 255],
        [255, 128, 0],
        [255, 128, 0],
        [255, 128, 0],
        [255, 128, 0],
        [255, 0, 0],
        [160, 32, 240],
        [255, 0, 0],
        [160, 32, 240],
        [255, 255, 100],
        [255, 255, 100],
        [255, 255, 100],
        [255, 255, 100],
        [255, 255, 100],
        [255, 255, 100],
        [255, 255, 100],
    ]

    linewidth = (13 / 1344) * min(W, H)

    if "keypoints" in tgt:

        sks = np.array(keypoint_skeleton)
        if sks != []:
            if sks.min() == 1:
                sks = sks - 1

        for idx, ann in enumerate(tgt["keypoints"]):
            kp = np.array(ann.cpu())
            Z = kp[: num_kpts * 2] * np.array([W, H] * num_kpts)
            x = Z[0::2]
            y = Z[1::2]
            for i, sk in enumerate(sks):
                c = link_color[i]
                c_ = [ci / 255 for ci in c]
                plt.plot(
                    x[sk], y[sk], linewidth=linewidth, color=c_, solid_capstyle="round"
                )

            for i in range(num_kpts):
                c_kpt = kpt_color[i]
                c_kpt_ = [ci / 255 for ci in c_kpt]
                plt.plot(x[i], y[i], "o", markersize=0, markerfacecolor=c_kpt_)
    ax.set_axis_off()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=dpi)
    plt.close()
    buf.seek(0)
    # Load the image from the buffer into PIL
    image_pil = Image.open(buf).convert("RGB")
    image_pil = image_pil.crop((0, 0, W, H - 1))
    image_pil = image_pil.resize((W, H))

    return image_pil


def get_face_image(image_pil, tgt, keypoint_skeleton, keypoint_text_prompt):
    num_kpts = len(keypoint_text_prompt)
    W, H = tgt["size"]
    fig = plt.figure(frameon=False)
    dpi = plt.gcf().dpi
    dpi = 108
    fig.set_size_inches(W / dpi, H / dpi)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax = plt.gca()
    ax.imshow(image_pil, aspect="equal")
    ax = plt.gca()
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")
    ax.grid(False)

    kp_map = [
        "right cheekbone 1",
        "right cheekbone 2",
        "right cheek 1",
        "right cheek 2",
        "right cheek 3",
        "right cheek 4",
        "right cheek 5",
        "right chin",
        "chin center",
        "left chin",
        "left cheek 5",
        "left cheek 4",
        "left cheek 3",
        "left cheek 2",
        "left cheek 1",
        "left cheekbone 2",
        "left cheekbone 1",
        "right eyebrow 1",
        "right eyebrow 2",
        "right eyebrow 3",
        "right eyebrow 4",
        "right eyebrow 5",
        "left eyebrow 1",
        "left eyebrow 2",
        "left eyebrow 3",
        "left eyebrow 4",
        "left eyebrow 5",
        "nasal bridge 1",
        "nasal bridge 2",
        "nasal bridge 3",
        "nasal bridge 4",
        "right nasal wing 1",
        "right nasal wing 2",
        "nasal wing center",
        "left nasal wing 1",
        "left nasal wing 2",
        "right eye eye corner 1",
        "right eye upper eyelid 1",
        "right eye upper eyelid 2",
        "right eye eye corner 2",
        "right eye lower eyelid 2",
        "right eye lower eyelid 1",
        "left eye eye corner 1",
        "left eye upper eyelid 1",
        "left eye upper eyelid 2",
        "left eye eye corner 2",
        "left eye lower eyelid 2",
        "left eye lower eyelid 1",
        "right mouth corner",
        "upper lip outer edge 1",
        "upper lip outer edge 2",
        "upper lip outer edge 3",
        "upper lip outer edge 4",
        "upper lip outer edge 5",
        "left mouth corner",
        "lower lip outer edge 5",
        "lower lip outer edge 4",
        "lower lip outer edge 3",
        "lower lip outer edge 2",
        "lower lip outer edge 1",
        "upper lip inter edge 1",
        "upper lip inter edge 2",
        "upper lip inter edge 3",
        "upper lip inter edge 4",
        "upper lip inter edge 5",
        "lower lip inter edge 3",
        "lower lip inter edge 2",
        "lower lip inter edge 1",
    ]
    color_kpt = []
    for kp in kp_map:
        if "cheekbone" in kp:
            color_kpt.append([1.00, 1.00, 1.00])
        elif "cheek" in kp:
            color_kpt.append([0.00, 1.00, 1.00])
        elif "chin" in kp:
            color_kpt.append([1.00, 0.00, 1.00])
        elif "eyebrow" in kp:
            color_kpt.append([1.00, 1.00, 0.00])
        elif "nasal" in kp:
            color_kpt.append([1.00, 0.00, 0.00])
        elif "eye" in kp:
            color_kpt.append([0.00, 1.00, 0.00])
        elif "lip" in kp:
            color_kpt.append([0.00, 0.00, 1.00])
        else:
            color_kpt.append([1.00, 1.00, 1.00])

    if "keypoints" in tgt:
        if len(tgt["keypoints"]) > 1:
            image_np = np.array(image_pil)
            # Convert RGB (Matplotlib & PIL) to BGR (OpenCV)
            image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            return image_cv2

        sks = np.array(keypoint_skeleton)
        # import pdb;pdb.set_trace()
        if sks != []:
            if sks.min() == 1:
                sks = sks - 1

        for idx, ann in enumerate(tgt["keypoints"]):
            kp = np.array(ann.cpu())
            Z = kp[: num_kpts * 2] * np.array([W, H] * num_kpts)
            x = Z[0::2]
            y = Z[1::2]
            for sk in sks:
                plt.plot(x[sk], y[sk], linewidth=2, color="white")

            for i in range(num_kpts):
                c_kpt = color_kpt[i]
                plt.plot(
                    x[i],
                    y[i],
                    "o",
                    markersize=4,
                    markerfacecolor=c_kpt,
                    markeredgecolor="k",
                    markeredgewidth=0.0,
                )
    ax.set_axis_off()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=dpi)
    plt.close()
    buf.seek(0)
    # Load the image from the buffer into PIL
    image_pil = Image.open(buf).convert("RGB")
    image_pil = image_pil.crop((0, 0, W, H - 1))
    image_pil = image_pil.resize((W, H))

    return image_pil


def get_hand_image(image_pil, tgt, keypoint_skeleton, keypoint_text_prompt):
    num_kpts = len(keypoint_text_prompt)
    W, H = tgt["size"]
    fig = plt.figure(frameon=False)
    dpi = plt.gcf().dpi
    dpi = 108
    fig.set_size_inches(W / dpi, H / dpi)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax = plt.gca()
    ax.imshow(image_pil, aspect="equal")
    ax = plt.gca()
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")
    ax.grid(False)

    kp_map = [
        "wrist",
        "thumb root",
        "thumb's third knuckle",
        "thumb's second knuckle",
        "thumb’s first knuckle",
        "forefinger's root",
        "forefinger's third knuckle",
        "forefinger's second knuckle",
        "forefinger's first knuckle",
        "middle finger's root",
        "middle finger's third knuckle",
        "middle finger's second knuckle",
        "middle finger's first knuckle",
        "ring finger's root",
        "ring finger's third knuckle",
        "ring finger's second knuckle",
        "ring finger's first knuckle",
        "pinky finger's root",
        "pinky finger's third knuckle",
        "pinky finger's second knuckle",
        "pinky finger's first knuckle",
    ]
    color_kpt = []
    for kp in kp_map:
        if "thumb" in kp:
            color_kpt.append([0.00, 0.00, 1.00])
        elif "forefinger" in kp:
            color_kpt.append([0.00, 1.00, 0.00])
        elif "middle" in kp:
            color_kpt.append([1.00, 0.00, 0.00])
        elif "ring" in kp:
            color_kpt.append([1.00, 1.00, 0.00])
        elif "pinky" in kp:
            color_kpt.append([1.00, 0.00, 1.00])
        elif "wrist" in kp:
            color_kpt.append([0.00, 1.00, 1.00])
        else:
            color_kpt.append([1.00, 1.00, 1.00])

    if "keypoints" in tgt:
        sks = np.array(keypoint_skeleton)
        # import pdb;pdb.set_trace()
        if sks != []:
            if sks.min() == 1:
                sks = sks - 1

        for idx, ann in enumerate(tgt["keypoints"]):
            kp = np.array(ann.cpu())
            Z = kp[: num_kpts * 2] * np.array([W, H] * num_kpts)
            x = Z[0::2]
            y = Z[1::2]
            for sk in sks:
                plt.plot(x[sk], y[sk], linewidth=2, color="white")

            for i in range(num_kpts):
                c_kpt = color_kpt[i]
                plt.plot(
                    x[i],
                    y[i],
                    "o",
                    markersize=4,
                    markerfacecolor=c_kpt,
                    markeredgecolor="k",
                    markeredgewidth=0.0,
                )
    ax.set_axis_off()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=dpi)
    plt.close()
    buf.seek(0)
    # Load the image from the buffer into PIL
    image_pil = Image.open(buf).convert("RGB")
    image_pil = image_pil.crop((0, 0, W, H - 1))
    image_pil = image_pil.resize((W, H))

    return image_pil


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = Config.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(
        clean_state_dict(checkpoint["model"]), strict=False
    )
    print(load_res)
    _ = model.eval()
    return model


def get_unipose_output(
    model,
    image,
    instance_text_prompt,
    keypoint_text_prompt,
    box_threshold,
    iou_threshold,
    device="cuda",
):
    # instance_text_prompt: A, B, C, ...
    # keypoint_text_prompt: skeleton
    instance_list = instance_text_prompt.split(",")

    # clip_model, _ = clip.load("ViT-B/32", device=device)
    ins_text_embeddings, kpt_text_embeddings = text_encoding(
        instance_list, keypoint_text_prompt, model.clip_model, device
    )
    target = {}
    target["instance_text_prompt"] = instance_list
    target["keypoint_text_prompt"] = keypoint_text_prompt
    target["object_embeddings_text"] = ins_text_embeddings.float()
    kpt_text_embeddings = kpt_text_embeddings.float()
    kpts_embeddings_text_pad = torch.zeros(
        100 - kpt_text_embeddings.shape[0], 512, device=device
    )
    target["kpts_embeddings_text"] = torch.cat(
        (kpt_text_embeddings, kpts_embeddings_text_pad), dim=0
    )
    kpt_vis_text = torch.ones(kpt_text_embeddings.shape[0], device=device)
    kpt_vis_text_pad = torch.zeros(kpts_embeddings_text_pad.shape[0], device=device)
    target["kpt_vis_text"] = torch.cat((kpt_vis_text, kpt_vis_text_pad), dim=0)
    # import pdb;pdb.set_trace()
    model = model.to(device)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image[None], [target])

    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)
    keypoints = outputs["pred_keypoints"][0][
        :, : 2 * len(keypoint_text_prompt)
    ]  # (nq, n_kpts * 2)
    # filter output
    logits_filt = logits.cpu().clone()
    boxes_filt = boxes.cpu().clone()
    keypoints_filt = keypoints.cpu().clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    keypoints_filt = keypoints_filt[filt_mask]  # num_filt, 4

    keep_indices = nms(
        box_ops.box_cxcywh_to_xyxy(boxes_filt),
        logits_filt.max(dim=1)[0],
        iou_threshold=iou_threshold,
    )

    # Use keep_indices to filter boxes and keypoints
    filtered_boxes = boxes_filt[keep_indices]
    filtered_keypoints = keypoints_filt[keep_indices]

    return filtered_boxes, filtered_keypoints


def get_unipose_output_batch(
    model,
    images,
    instance_text_prompt,
    keypoint_text_prompt,
    box_threshold,
    iou_threshold,
    device="cuda",
):
    # instance_text_prompt: A, B, C, ...
    # keypoint_text_prompt: skeleton
    instance_list = instance_text_prompt.split(",")
    device = device
    bs = images.shape[0]

    ins_text_embeddings, kpt_text_embeddings = text_encoding(
        instance_list, keypoint_text_prompt, model.clip_model, device
    )
    target = {}
    target["instance_text_prompt"] = instance_list
    target["keypoint_text_prompt"] = keypoint_text_prompt
    target["object_embeddings_text"] = ins_text_embeddings.float()
    kpt_text_embeddings = kpt_text_embeddings.float()
    kpts_embeddings_text_pad = torch.zeros(
        100 - kpt_text_embeddings.shape[0], 512, device=device
    )
    target["kpts_embeddings_text"] = torch.cat(
        (kpt_text_embeddings, kpts_embeddings_text_pad), dim=0
    )
    kpt_vis_text = torch.ones(kpt_text_embeddings.shape[0], device=device)
    kpt_vis_text_pad = torch.zeros(kpts_embeddings_text_pad.shape[0], device=device)
    target["kpt_vis_text"] = torch.cat((kpt_vis_text, kpt_vis_text_pad), dim=0)

    model = model.to(device)
    images = images.to(device)

    with torch.no_grad():
        outputs = model(images, [target] * bs)

    logits = outputs["pred_logits"].sigmoid()  # (bs, nq, 256)
    boxes = outputs["pred_boxes"]  # (bs, nq, 4)
    keypoints = outputs["pred_keypoints"]

    filtered_boxes_lst = []
    filtered_keypoints_lst = []

    for i in range(bs):
        logit_i = logits[i]
        boxes_i = boxes[i]
        keypoints_i = keypoints[i][:, : 2 * len(keypoint_text_prompt)]

        # filter output
        logits_filt = logit_i.cpu().clone()
        boxes_filt = boxes_i.cpu().clone()
        keypoints_filt = keypoints_i.cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        keypoints_filt = keypoints_filt[filt_mask]  # num_filt, 4`
        keep_indices = nms(
            box_ops.box_cxcywh_to_xyxy(boxes_filt),
            logits_filt.max(dim=1)[0],
            iou_threshold=iou_threshold,
        )

        # Use keep_indices to filter boxes and keypoints
        filtered_boxes = boxes_filt[keep_indices]
        filtered_keypoints = keypoints_filt[keep_indices]

        filtered_boxes_lst.append(filtered_boxes)
        filtered_keypoints_lst.append(filtered_keypoints)

    return filtered_boxes_lst, filtered_keypoints_lst


if __name__ == "__main__":

    parser = argparse.ArgumentParser("UniPose Inference", add_help=True)
    parser.add_argument(
        "--config_file", "-c", type=str, required=True, help="path to config file"
    )
    parser.add_argument(
        "--checkpoint_path",
        "-p",
        type=str,
        required=True,
        help="path to checkpoint file",
    )
    parser.add_argument(
        "--image_path", "-i", type=str, required=True, help="path to image file"
    )
    parser.add_argument(
        "--instance_text_prompt",
        "-t",
        type=str,
        required=True,
        help="instance text prompt",
    )
    parser.add_argument(
        "--keypoint_text_example",
        "-k",
        type=str,
        default=None,
        help="keypoint text prompt",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="outputs",
        required=True,
        help="output directory",
    )
    parser.add_argument(
        "--box_threshold", type=float, default=0.1, help="box threshold"
    )
    parser.add_argument(
        "--iou_threshold", type=float, default=0.4, help="iou threshold"
    )
    parser.add_argument(
        "--cpu_only", action="store_true", help="running on cpu only!, default=False"
    )
    args = parser.parse_args()

    # cfg
    config_file = args.config_file  # change the path of the model config file
    checkpoint_path = args.checkpoint_path  # change the path of the model
    image_path = args.image_path
    instance_text_prompt = args.instance_text_prompt
    keypoint_text_example = args.keypoint_text_example

    if keypoint_text_example in globals():
        keypoint_dict = globals()[keypoint_text_example]
        keypoint_text_prompt = keypoint_dict.get("keypoints")
        keypoint_skeleton = keypoint_dict.get("skeleton")
    elif instance_text_prompt in globals():
        keypoint_dict = globals()[instance_text_prompt]
        keypoint_text_prompt = keypoint_dict.get("keypoints")
        keypoint_skeleton = keypoint_dict.get("skeleton")
    else:
        keypoint_dict = globals()["animal"]
        keypoint_text_prompt = keypoint_dict.get("keypoints")
        keypoint_skeleton = keypoint_dict.get("skeleton")

    device = "cuda" if torch.cuda.is_available() and not args.cpu_only else "cpu"

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    box_threshold = args.box_threshold

    iou_threshold = args.iou_threshold

    # load model
    model = load_model(config_file, checkpoint_path, cpu_only=args.cpu_only)
    image_name = os.path.basename(image_path)
    output_filename = instance_text_prompt + "-" + image_name
    save_path = os.path.join(output_dir, output_filename)

    # load video
    pil_image = Image.open(image_path).convert("RGB")
    image_size = pil_image.size
    transform = T.Compose(
        [
            T.RandomResize([800]),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(pil_image, None)

    boxes_filt, keypoints_filt = get_unipose_output(
        model,
        image,
        instance_text_prompt,
        keypoint_text_prompt,
        box_threshold,
        iou_threshold,
        device=device,
    )

    pred_dict = {
        "boxes": boxes_filt,
        "keypoints": keypoints_filt,
        "size": image_size,
    }
    image_raw = Image.new("RGB", image_size, (0, 0, 0))

    if instance_text_prompt == "person":
        image_result = get_pose_image(
            image_raw, pred_dict, keypoint_skeleton, keypoint_text_prompt
        )

    elif instance_text_prompt == "face":
        image_result = get_face_image(
            image_raw, pred_dict, keypoint_skeleton, keypoint_text_prompt
        )

    elif instance_text_prompt == "hand":
        image_result = get_hand_image(
            image_raw, pred_dict, keypoint_skeleton, keypoint_text_prompt
        )

    image_result.save(save_path)
