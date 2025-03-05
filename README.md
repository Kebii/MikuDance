<div align="center">
  <img src="https://kebii.github.io/MikuDance/static/images/logo.png" alt="MikuDance Logo" style="width:100px;"><br>
</div>

<div align="center">
<h2><font> MikuDance </font></center> <br> <center>Animating Character Art with Mixed Motion Dynamics</h2>

[Jiaxu Zhang](https://kebii.github.io/), [Xianfang Zeng](https://scholar.google.com/citations?user=tgDc0fsAAAAJ&hl=en), [Xin Chen](https://chenxin.tech/), [Wei Zuo](https://liris.cnrs.fr/en/member-page/wei-zuo), [Gang Yu](https://www.skicyyu.org/), [Zhigang Tu](http://tuzhigang.cn/)

<a href='https://arxiv.org/abs/2411.08656'><img src='https://img.shields.io/badge/ArXiv-2411.08656-red'></a> <a href='https://kebii.github.io/MikuDance/'><img src='https://img.shields.io/badge/Project-Page-purple'></a>
 <a href='https://m.lipuhome.com/'><img src='https://img.shields.io/badge/Lipu-狸谱-yellow'></a>

<img src="https://github.com/Kebii/MikuDance/blob/main/assets/mdimage.gif" alt="MD image" style="width:100%;"><br>

</div>

# 📣 Updates
- **[2025.2.27]** 🔥 The code is released! If you have any questions, please feel free to open an issue.

- **[2025.1.10]** 🕹️ Our MikuDance has recently been launched on the [Lipu](https://m.lipuhome.com/), an AI creation community designed for animation enthusiasts. We invite everyone to download and try it out. 

- **[2024.11.15]** ✨️ Paper and project page are released! Please see our demo videos on the project page. Considering the company's policy, the code release will be delayed. 
We will do our best to make it open source as soon as possible.

# ⚒️ Getting Started

## Build Environtment

We Recommend a python version `>=3.10` and cuda version `=11.7`. Then build environment as follows:

```shell
# [Optional] Create a virtual env
conda create -n MikuDance python=3.10
conda activate MikuDance
# Install with pip:
pip install -r requirements.txt  
```

## Download Weights

**Automatically downloading**: You can run the following command to download weights automatically:

```shell
python tools/download_weights.py
```

Weights will be placed under the `./pretrained_weights` direcotry. The whole downloading process may take a long time.

**Manually downloading**: You can also download weights manually, which has some steps:

1. Download MikuDance [weights](https://huggingface.co/JiaxuZ/MikuDance/tree/main), which include three parts: `denoising_unet.pth`, `reference_unet.pth` and `motion_module.pth`.

2. Download pretrained weight of based models and other components: 
    - [StableDiffusion V1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
    - [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)
    - [image_encoder](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/tree/main/image_encoder)
    - [vae_temporal_decoder](https://huggingface.co/maxin-cn/Latte-1/tree/main/vae_temporal_decoder)

Finally, these weights should be orgnized as follows:

```text
./pretrained_weights/
|-- image_encoder
|   |-- config.json
|   `-- pytorch_model.bin
|-- sd-vae-ft-mse
|   |-- config.json
|   |-- diffusion_pytorch_model.bin
|   `-- diffusion_pytorch_model.safetensors
|-- stable-diffusion-v1-5
|   |-- feature_extractor
|   |   `-- preprocessor_config.json
|   |-- model_index.json
|   |-- unet
|   |   |-- config.json
|   |   `-- diffusion_pytorch_model.bin
|   `-- v1-inference.yaml
|-- vae_temporal_decoder
|   |-- config.json
|   `-- diffusion_pytorch_model.safetensors
|-- denoising_unet.pth
|-- motion_module.pth
|-- reference_unet.pth
```

**Note:** If you have installed some of the pretrained models, such as `StableDiffusion V1.5`, you can specify their paths in the config file.

# 🚀 Training and Inference 

## Inference of MikuDance

Running inference scripts:

```shell
python -m scripts.inference_video \
--config ./configs/inference/inference_video.yaml \
-W 768 -H 768 --fps 30 --steps 20
```

You can refer the format of `inference_video.yaml` to animate your own reference images and pose videos.

**Note:** The target face, hand, w2c, c2w, and the reference depth are optional. If you don't have them, you can set them to `null` in the config file.

**Note:** The `-W` and `-H` are the width and height of the output video, respectively. The width and height must be an integer multiple of 8. The `--fps` is the frame rate of the output video. The `--steps` is the denoising steps.

## Training of MikuDance

### Training Data Preparation

You can refer the `src/dataset/anime_image_dataset.py` and `src/dataset/anime_video_dataset.py` to prepare your own dataset for the two training stages respectively.

Our dataset was organized as follows:

```text
./data/
|-- video_1/
|   |-- frame_0001.jpg
|   |-- pose_0001.jpg
|   |-- face_0001.jpg
|   |-- hand_0001.jpg
|   |-- depth_0001.npy
|   |-- w2c_0001.npy
|   |-- c2w_0001.npy
|   |-- frame_0002.jpg
|   |-- ...
|-- video_2/
|   |-- ...
```
**Note:** `w2c` and `c2w` are the camera parameters (world2camera and camera2world matrix) of the frame, `depth` is the depth map of the frame. You can organize your own dataset format according to your needs.

### Stage1

```shell
accelerate launch scripts/train_stage1.py --config configs/train/train_stage1.yaml
```

### Stage2

Put the pretrained motion module weights `mm_sd_v15_v2.ckpt` ([download link](https://huggingface.co/guoyww/animatediff/blob/main/mm_sd_v15_v2.ckpt)) under `./pretrained_weights`. 

```shell
accelerate launch scripts/train_stage2.py --config configs/train/train_stage2.yaml
```

# 🧩 Data Preparation

## Pose Estimation
We utilize [Xpose](https://github.com/IDEA-Research/X-Pose) to estimate the pose of the character. You can download the pretrained weights of Xpose from [here](https://drive.google.com/file/d/13gANvGWyWApMFTAtC3ntrMgx0fOocjIa/view) and put it under `./src/XPose/weights`.

Pose estimation for driving videos:
```shell
cd ./src/XPose
python inference_on_video.py \
-c config_model/UniPose_SwinT.py \
-p weights/unipose_swint.pth \
-i /input_video_path \
-o /output_video_path \
-t "person" -k "person" \ # change to "face" or "hand" for face and hands keypoints
# -- real_human # If the driving video is a real human video, we recommend to add this flag to adjust the head-body scale of the keypoints.
```

Pose estimation for reference images:
```shell
cd ./src/XPose
python inference_on_image.py \
-c config_model/UniPose_SwinT.py \
-p weights/unipose_swint.pth \
-i /input_image_path \
-o /output_image_path \
-t "person" -k "person" 
```

**Note:** We predefined the color map for the character keypoints. It is necessary to use the same color map and visualization settings as ours during inference.

**Note:** If the driving video features a real human and there is a significant difference in face scale compared to anime characters, we recommend setting the `tgt_face_path` to `null` in the config file.

## Camera Parameters Estimation
We utilize [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM) to estimate the camera parameters of the driving video. You can follow the instructions in the DROID-SLAM repository to install it in the `./src/DROID-SLAM` directory. Then you can run the following command to estimate the camera parameters:

```shell
cd ./src/DROID-SLAM
python get_camera_from_video.py -i /input_video_path -o /output_path
```

**Note:** The environment of DROID-SLAM is different from MikuDance, you may need to install it by following the instructions in the DROID-SLAM repository.

**Note:** The camera parameters are optional for the inference of MikuDance. If you don't have them, you can set them to `null` in the config file.

**Note:** In inference, the camera parameters are saved at the video level. But in our training dataset, the camera parameters are saved at the frame level.

## Depth Estimation
We utilize [Intel/dpt-hybrid-midas](https://huggingface.co/Intel/dpt-hybrid-midas) for depth estimation.

```shell
python tools/depth_from_image.py --image_path /input_image_path --save_dir /output_path
```

# 📄 Citation
If MikuDance is useful for your research, welcome to 🌟 this repo and cite our work using the following BibTeX:

```bibtex
@misc{zhang2024mikudance,
      title={MikuDance: Animating Character Art with Mixed Motion Dynamics}, 
      author={Jiaxu Zhang and Xianfang Zeng and Xin Chen and Wei Zuo and Gang Yu and Zhigang Tu},
      year={2024},
      eprint={2411.08656},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
