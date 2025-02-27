import numpy as np


def get_K_matrix(K, T):
    K_matrix = np.zeros((T, 3, 4))
    K_matrix[:, 0, 0] = K[0]  # fx
    K_matrix[:, 1, 1] = K[1]  # fy
    K_matrix[:, 0, 2] = K[2]  # cx
    K_matrix[:, 1, 2] = K[3]  # cy
    K_matrix[:, 2, 2] = 1
    return K_matrix


def camera_to_scene_motion(w2cs, c2ws, K, depth_map, width, height, istrain=True):
    T = len(w2cs)
    K_matrix = get_K_matrix(K, T)

    x = np.arange(-width // 2, width // 2, 1)
    y = np.arange(-height // 2, height // 2, 1)
    xx, yy = np.meshgrid(x, y)

    xx = xx.reshape((1, width * height)).repeat(T, axis=0)
    yy = yy.reshape((1, width * height)).repeat(T, axis=0)
    zz = 100 - depth_map.reshape((1, width * height)).repeat(T, axis=0) * 50

    init_points = np.stack((xx, yy, zz), axis=-1)

    points_homogeneous = np.concatenate(
        (init_points, np.ones((T, width * height, 1))), axis=-1
    )

    init_img_points = np.einsum("tij,taj->tai", K_matrix, points_homogeneous)
    init_img_points /= init_img_points[..., 2:3]

    flow = np.zeros((T, 2, height, width))

    w2c_array = np.stack(w2cs, axis=0)
    c2w_array = np.stack(c2ws, axis=0)
    world_points = np.einsum("tij,taj->tai", c2w_array, points_homogeneous)
    camera_points = np.einsum(
        "tij,taj->tai", w2c_array[1:, ...], world_points[:-1, ...]
    )

    camera_img_points = np.einsum("tij,taj->tai", K_matrix[1:, ...], camera_points)
    camera_img_points /= camera_img_points[..., 2:3]

    camera_flow = camera_img_points[..., :2] - init_img_points[:-1, :, :2]
    camera_flow = camera_flow.transpose(0, 2, 1).reshape((T - 1, 2, height, width))

    flow_mean = np.mean(camera_flow)
    flow_std = np.std(camera_flow)

    if istrain and np.isfinite(camera_flow).all() and np.abs(flow_std) < 10.0:
        camera_flow = np.clip(
            camera_flow, flow_mean - (3 * flow_std), flow_mean + (3 * flow_std)
        )
        flow[1:, 0, ...] = camera_flow[:, 0, ...]
        flow[1:, 1, ...] = camera_flow[:, 1, ...]

    elif np.isfinite(camera_flow).all():
        camera_flow = np.clip(
            camera_flow, flow_mean - (3 * flow_std), flow_mean + (3 * flow_std)
        )
        flow[1:, 0, ...] = camera_flow[:, 0, ...]
        flow[1:, 1, ...] = camera_flow[:, 1, ...]

    return flow


if __name__ == "__main__":
    pass
