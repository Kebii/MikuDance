import sys

sys.path.append("droid_slam")

from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob
import time
import argparse

from torch.multiprocessing import Process
from droid import Droid
import quaternion

import torch.nn.functional as F
from packaging import version as pver
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R


def cam_pose_vis(
    output_path,
    camera_poses,
    cds="cv",
    pose_type="w2c",
    rgbs=None,
    use_tex=True,
    camera_ids=None,
):
    """Visualize camera poses
    output_path: Output path in obj format
    camera_poses: List or array of camera poses with size [Nx4x4]
    cds: Camera cordinate system, whether the camera cordinate system is
         'gl' (xyz->right,up,backward) or 'cv' (xyz->right,down,forward)
    pose_type: Camera pose type, whether the camera pose is
               'w2c' world to camera or 'c2w' camera to world
    rgbs: Desired rgb value for each camera with size [Nx3], None if no desired value
    use_tex: If True, outputs file with camera id texture
    camera_ids: Camera ids, None with default ranking id, or [N] array with specific id
    """
    # path
    if output_path[-4:] != ".obj":
        if "." not in output_path:
            output_path += ".obj"
        else:
            output_path = os.path.splitext(output_path)[0] + ".obj"
    # convert to c2w
    if pose_type == "w2c":
        c2ws = np.linalg.inv(np.array(camera_poses))
    else:
        c2ws = np.array(camera_poses)
    # scaling the camera pose
    tex_cir_rad = 40
    tex_text_size = 1.4
    transl = c2ws[:, :3, 3]
    min_, max_ = np.min(transl, axis=0), np.max(transl, axis=0)
    scale = np.mean(max_ - min_) * 0.1
    # scale = 2
    camera_num = len(camera_poses)
    # defining camera vertices, faces and tex
    cam_verts = (
        np.array(
            [
                [0, 0, 0],
                [0.5, 0.5, -1],
                [-0.5, 0.5, -1],
                [-0.5, -0.5, -1],
                [0.5, -0.5, -1],
                [0.5, 0.6, -1],
                [-0.5, 0.6, -1],
                [0, 0.8, -1],
            ]
        )
        * scale
    )  # camera vertex coordinate
    # convert cv camera coordinate to gl (default camera system in meshlab is gl)
    if cds == "cv":
        cam_verts = cam_verts * np.array([1, -1, -1])
    face_map = np.array(
        [[1, 2, 3], [1, 3, 4], [1, 4, 5], [1, 5, 2], [4, 3, 2], [2, 5, 4], [6, 7, 8]]
    )  # faces by vertex index
    tex_map = np.array(
        [[1, 0], [0.5, 0.5], [0, 1], [0.5, 1.5], [1, 2], [1.5, 1.5], [2, 1], [1.5, 0.5]]
    )  # vertex texture coordinate
    tex_face_map = np.array(
        [[1, 8, 2], [3, 2, 4], [5, 4, 6], [7, 6, 8], [6, 8, 2], [2, 4, 6], [1, 8, 2]]
    )  # faces by texture index
    with open(os.path.join(output_path), "w") as f:
        # if use texture, prepare material file and texture image
        if use_tex:
            mtl_file = output_path[:-4] + ".mtl"
            mtl_base = os.path.basename(mtl_file)
            tex_file = output_path[:-4] + ".png"
            tex_base = os.path.basename(tex_file)
            f.write(f"mtllib {mtl_base}\n")
            n_row = int(np.ceil(np.sqrt(camera_num)))
            im_size = n_row * tex_cir_rad * 2
            tex_im = np.zeros([im_size, im_size, 3], dtype=np.uint8)
        # write vertices
        for i in range(camera_num):
            verts = np.concatenate([cam_verts, np.ones((len(cam_verts), 1))], axis=1)
            for j in range(verts.shape[0]):
                p = np.dot(c2ws[i], np.transpose(verts[j]))[:3]
                rgb = (
                    list(rgbs[i]) if rgbs is not None else [0, 0, (i + 1) / camera_num]
                )
                if not use_tex:
                    f.write(
                        "v %f %f %f %f %f %f\n" % tuple(list(p) + rgb)
                    )  # vertex coloring
                else:
                    x, y = i % n_row, i // n_row
                    cam_text = "%02d" % i if camera_ids is None else str(camera_ids[i])
                    cx, cy = int((x * 2 + 1) * tex_cir_rad), int(
                        (y * 2 + 1) * tex_cir_rad
                    )
                    tex_im = cv2.circle(
                        tex_im,
                        (cx, cy),
                        tex_cir_rad,
                        [int(c * 255) for c in rgb],
                        cv2.FILLED,
                    )
                    tex_im = cv2.putText(
                        tex_im,
                        cam_text,
                        (
                            int((x * 2 + 0.64) * tex_cir_rad),
                            int((y * 2 + 1.2) * tex_cir_rad),
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        tex_text_size / len(cam_text),
                        [255, 255, 255],
                        thickness=2,
                    )
                    f.write("v %f %f %f\n" % tuple(list(p)))
        # write texture
        if use_tex:
            for view_id in range(camera_num):
                x, y = view_id % n_row, view_id // n_row
                for tex in tex_map:
                    tex = ((np.array([x, y]) * 2 + tex) * tex_cir_rad) / im_size
                    tex[1] = 1 - tex[1]
                    f.write("vt %f %f\n" % tuple(list(tex)))
            f.write("usemtl mymtl\n")
            cv2.imwrite(tex_file, tex_im)
            with open(mtl_file, "w") as f_mtl:
                f_mtl.write("newmtl mymtl\n")
                f_mtl.write("map_Kd {}\n".format(tex_base))
        # write faces
        for i in range(camera_num):
            face_step = i * cam_verts.shape[0]
            tex_step = i * tex_map.shape[0]
            for j in range(face_map.shape[0]):
                face = face_map[j] + face_step
                if not use_tex:
                    f.write("f %d %d %d\n" % tuple(list(face)))
                else:
                    tex_face = tex_face_map[j] + tex_step
                    face = np.stack([face, tex_face], axis=0).T.reshape(-1)
                    f.write("f %d/%d %d/%d %d/%d\n" % tuple(list(face)))


def pil_image_stream(frames, calib):
    for t, imfile in enumerate(frames):
        width, height = imfile.size
        image = np.array(imfile)  # Convert PIL image to NumPy array
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

        fx, fy, cx, cy = calib[:4]

        K = np.eye(3)
        K[0, 0] = fx
        K[0, 2] = cx
        K[1, 1] = fy
        K[1, 2] = cy

        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((width * height) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((width * height) / (h0 * w0)))

        image = cv2.resize(image, (w1, h1))
        image = image[: h1 - h1 % 8, : w1 - w1 % 8]
        image = torch.as_tensor(image).permute(2, 0, 1)

        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= w1 / w0
        intrinsics[1::2] *= h1 / h0

        yield t, image[None], intrinsics


def cv2_image_stream(frames, calib):
    for t, imfile in enumerate(frames):
        image = cv2.resize(imfile, (512, 512))
        height, width, channel = image.shape

        fx, fy, cx, cy = calib[:4]

        K = np.eye(3)
        K[0, 0] = fx
        K[0, 2] = cx
        K[1, 1] = fy
        K[1, 2] = cy

        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((width * height) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((width * height) / (h0 * w0)))

        image = cv2.resize(image, (w1, h1))
        image = image[: h1 - h1 % 8, : w1 - w1 % 8]
        image = torch.as_tensor(image).permute(2, 0, 1)

        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= w1 / w0
        intrinsics[1::2] *= h1 / h0

        yield t, image[None], intrinsics


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse("1.10"):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing="ij")


def slerp(t, q0, q1):
    """
    Spherical Linear Interpolation (SLERP) between two quaternions.

    Parameters:
    t (float): Interpolation factor between 0 and 1.
    q0 (np.array): Starting quaternion.
    q1 (np.array): Ending quaternion.

    Returns:
    np.array: Interpolated quaternion.
    """
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)

    dot_product = np.dot(q0, q1)

    if dot_product < 0.0:
        q1 = -q1
        dot_product = -dot_product

    if dot_product > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)

    theta_0 = np.arccos(dot_product)
    theta = theta_0 * t

    q2 = q1 - q0 * dot_product
    q2 = q2 / np.linalg.norm(q2)

    return q0 * np.cos(theta) + q2 * np.sin(theta)


def interpolate_camera_poses(steps, camera_poses, total_length):
    """
    Interpolates camera poses for a continuous period.

    Parameters:
    steps (np.array): Array of time steps.
    camera_poses (np.array): Array of camera poses corresponding to the time steps.
    total_length (int): The total length of the continuous period.

    Returns:
    np.array: Interpolated camera poses for the continuous period.
    """
    # Separate positions and quaternions
    positions = camera_poses[:, :3]
    quaternions = camera_poses[:, 3:]

    # Create an interpolation function for the positions
    position_interp_funcs = [
        interp1d(
            steps,
            positions[:, i],
            kind="linear",
            fill_value=positions[-1, i],
            bounds_error=False,
        )
        for i in range(positions.shape[1])
    ]

    # Generate new time steps for the continuous period
    new_steps = np.arange(total_length)

    # Interpolate the positions for the new time steps
    interpolated_positions = np.array(
        [interp_func(new_steps) for interp_func in position_interp_funcs]
    ).T

    # Interpolate the quaternions for the new time steps using SLERP
    interpolated_quaternions = []
    for t in new_steps:
        # Find the two nearest time steps
        idx = np.searchsorted(steps, t, side="right")
        if idx == 0:
            interpolated_quaternions.append(quaternions[0])
        elif idx == len(steps):
            interpolated_quaternions.append(quaternions[-1])
        else:
            t0, t1 = steps[idx - 1], steps[idx]
            q0, q1 = quaternions[idx - 1], quaternions[idx]
            t_interp = (t - t0) / (t1 - t0)
            interpolated_quaternions.append(slerp(t_interp, q0, q1))

    interpolated_quaternions = np.array(interpolated_quaternions)

    # Combine interpolated positions and quaternions
    interpolated_camera_poses = np.hstack(
        (interpolated_positions, interpolated_quaternions)
    )

    return interpolated_camera_poses


def split_video_to_frames(video_path):
    # Step 1: Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return []

    frames = []

    # Step 2: Read frames from the video
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Step 3: Store frames in a list
        frames.append(frame)

    # Step 4: Release the video capture object
    cap.release()

    return frames


def main(calib, video_path, save_dir):
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--t0", default=0, type=int, help="starting frame")
    parser.add_argument("--stride", default=1, type=int, help="frame stride")
    parser.add_argument("--buffer", type=int, default=1024)
    parser.add_argument("--image_size", default=[512, 512])
    parser.add_argument("--disable_vis", default=True, action="store_true")

    parser.add_argument(
        "--beta",
        type=float,
        default=0.3,
        help="weight for translation / rotation components of flow",
    )
    parser.add_argument(
        "--filter_thresh",
        type=float,
        default=2.4,
        help="how much motion before considering new keyframe",
    )
    parser.add_argument("--warmup", type=int, default=8, help="number of warmup frames")
    parser.add_argument(
        "--keyframe_thresh",
        type=float,
        default=4.0,
        help="threshold to create a new keyframe",
    )
    parser.add_argument(
        "--frontend_thresh",
        type=float,
        default=16.0,
        help="add edges between frames whithin this distance",
    )
    parser.add_argument(
        "--frontend_window", type=int, default=25, help="frontend optimization window"
    )
    parser.add_argument(
        "--frontend_radius",
        type=int,
        default=2,
        help="force edges between frames within radius",
    )
    parser.add_argument(
        "--frontend_nms", type=int, default=1, help="non-maximal supression of edges"
    )

    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--upsample", action="store_true")
    args = parser.parse_args()

    args.stereo = False
    torch.multiprocessing.set_start_method("spawn")

    full_video = split_video_to_frames(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    video_length = len(full_video)

    clip_frames = full_video
    clip_frames_length = len(clip_frames)
    frame_stream = cv2_image_stream(clip_frames, calib)
    droid = None

    for t, image, intrinsics in tqdm(frame_stream):
        if droid is None:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(args)
        droid.track(t, image, intrinsics=intrinsics)

    sample_t = droid.video.counter.value
    tstamps = droid.video.tstamp[:sample_t].cpu().numpy()
    poses = droid.video.poses[:sample_t].cpu().numpy()
    intrinsics = droid.video.intrinsics[:sample_t].cpu().numpy()

    interped_poses = interpolate_camera_poses(tstamps, poses, clip_frames_length)

    w2cs = []
    c2ws = []
    for j in range(interped_poses.shape[0]):
        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = quaternion.as_rotation_matrix(
            quaternion.from_float_array(interped_poses[j, 3:])
        )
        pose_matrix[:3, 3] = interped_poses[j, :3]
        w2cs.append(pose_matrix)
        c2ws.append(np.linalg.inv(pose_matrix))

    if not os.path.exists(os.path.join(save_dir, video_name)):
        os.makedirs(os.path.join(save_dir, video_name))

    save_cam_path = os.path.join(save_dir, video_name, "cam-" + video_name + ".obj")
    cam_pose_vis(save_cam_path, w2cs)

    w2c_vdo = np.stack(w2cs, axis=0)
    c2w_vdo = np.stack(c2ws, axis=0)
    save_path_w2c = os.path.join(save_dir, video_name, "w2c-" + video_name + ".npy")
    save_path_c2w = os.path.join(save_dir, video_name, "c2w-" + video_name + ".npy")
    save_path_pose = os.path.join(save_dir, video_name, "cps-" + video_name + ".npy")
    np.save(save_path_w2c, w2c_vdo)
    np.save(save_path_c2w, c2w_vdo)
    np.save(save_path_pose, interped_poses)


if __name__ == "__main__":
    # Before running this script, you should install DROID-SLAM (https://github.com/princeton-vl/DROID-SLAM)
    def parse_args():
        parser = argparse.ArgumentParser("DROID SLAM Inference", add_help=True)
        parser.add_argument(
            "--video_path",
            "-i",
            type=str,
            required=True,
            help="path of the input video file",
        )
        parser.add_argument(
            "--save_path",
            "-o",
            type=str,
            required=True,
            help="path for the output",
        )
        args = parser.parse_args()

        return args

    calib = [512.0, 512.0, 256.0, 256.0]
    args = parse_args()
    main(calib, args.video_path, args.save_path)
