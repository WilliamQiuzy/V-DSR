"""
VDPM 点云轨迹可视化模块

将动态点云的运动轨迹渲染到图片上，用彩虹色表示时间顺序：
- 红色 → 橙色 → 黄色 → 绿色 → 青色 → 蓝色 → 紫色
- 颜色从红到紫表示从第一帧到最后一帧的时间演进

输入: VDPM .npz 文件
输出: 带轨迹的可视化图片
"""

import numpy as np
import cv2
from typing import Optional, Tuple, List
import matplotlib.pyplot as plt
from matplotlib import cm
import os


def load_vdpm_npz(npz_path: str) -> dict:
    """加载 VDPM 输出"""
    data = np.load(npz_path)
    return {
        'points': data['points'],  # (T, N, 3)
        'conf': data['conf'],      # (T, N)
    }


def project_points_to_image(points_3d: np.ndarray,
                            intrinsic: np.ndarray,
                            image_size: Tuple[int, int]) -> np.ndarray:
    """将 3D 点投影到 2D 图像平面

    Args:
        points_3d: (N, 3) 相机坐标系下的 3D 点
        intrinsic: (3, 3) 相机内参
        image_size: (H, W) 图像尺寸

    Returns:
        points_2d: (N, 2) 图像坐标
    """
    # 投影: p_2d = K @ p_3d / z
    z = points_3d[:, 2:3]  # (N, 1)
    z = np.clip(z, 1e-6, None)  # 避免除零

    points_homo = points_3d / z  # (N, 3), normalized
    points_2d = (intrinsic @ points_homo.T).T  # (N, 3)

    return points_2d[:, :2]  # (N, 2)


def get_rainbow_colors(n_frames: int) -> List[Tuple[int, int, int]]:
    """生成彩虹色序列 (BGR 格式用于 OpenCV)

    红 → 橙 → 黄 → 绿 → 青 → 蓝 → 紫
    """
    colors = []
    cmap = cm.get_cmap('rainbow')

    for i in range(n_frames):
        # 从红(0)到紫(1)
        t = i / max(n_frames - 1, 1)
        rgba = cmap(t)
        # 转为 BGR (OpenCV 格式)
        bgr = (int(rgba[2] * 255), int(rgba[1] * 255), int(rgba[0] * 255))
        colors.append(bgr)

    return colors


def visualize_trajectories_on_image(
    background_image: np.ndarray,
    points_sequence: np.ndarray,
    conf_sequence: np.ndarray,
    intrinsic: Optional[np.ndarray] = None,
    conf_threshold: float = 5.0,
    subsample: int = 100,
    line_thickness: int = 1,
    point_radius: int = 2,
) -> np.ndarray:
    """在背景图上绘制点云轨迹

    Args:
        background_image: (H, W, 3) 背景图像
        points_sequence: (T, N, 3) 点云序列
        conf_sequence: (T, N) 置信度序列
        intrinsic: (3, 3) 相机内参，如果 None 则假设点已经是 2D
        conf_threshold: 置信度阈值
        subsample: 每隔多少个点采样一次（减少绘制量）
        line_thickness: 轨迹线宽度
        point_radius: 点的半径

    Returns:
        output_image: 带轨迹的图像
    """
    H, W = background_image.shape[:2]
    output = background_image.copy()

    T, N, _ = points_sequence.shape

    # 按置信度过滤
    mean_conf = conf_sequence.mean(axis=0)  # (N,)
    valid_mask = mean_conf >= conf_threshold

    # 子采样
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) > subsample:
        selected_indices = np.random.choice(valid_indices, subsample, replace=False)
    else:
        selected_indices = valid_indices

    # 获取彩虹色
    colors = get_rainbow_colors(T)

    # 投影所有帧的点到 2D
    points_2d_sequence = []
    for t in range(T):
        points_3d = points_sequence[t, selected_indices]  # (M, 3)

        if intrinsic is not None:
            points_2d = project_points_to_image(points_3d, intrinsic, (H, W))
        else:
            # 假设 points_3d 的 x, y 已经是归一化的图像坐标
            points_2d = points_3d[:, :2] * np.array([W, H])

        points_2d_sequence.append(points_2d)

    points_2d_sequence = np.array(points_2d_sequence)  # (T, M, 2)

    # 绘制每个点的轨迹
    M = len(selected_indices)
    for i in range(M):
        trajectory = points_2d_sequence[:, i, :]  # (T, 2)

        # 绘制轨迹线段
        for t in range(T - 1):
            pt1 = tuple(trajectory[t].astype(int))
            pt2 = tuple(trajectory[t + 1].astype(int))

            # 检查点是否在图像内
            if (0 <= pt1[0] < W and 0 <= pt1[1] < H and
                0 <= pt2[0] < W and 0 <= pt2[1] < H):
                color = colors[t]
                cv2.line(output, pt1, pt2, color, line_thickness, cv2.LINE_AA)

    return output


def visualize_trajectories_3d_projection(
    points_sequence: np.ndarray,
    conf_sequence: np.ndarray,
    image_size: Tuple[int, int] = (512, 512),
    conf_threshold: float = 5.0,
    subsample: int = 500,
    line_thickness: int = 1,
    background_color: Tuple[int, int, int] = (255, 255, 255),
    view_angle: str = "xy",
) -> np.ndarray:
    """将 3D 轨迹投影到 2D 平面可视化（无需相机内参）

    Args:
        points_sequence: (T, N, 3) 点云序列
        conf_sequence: (T, N) 置信度
        image_size: 输出图像尺寸 (H, W)
        conf_threshold: 置信度阈值
        subsample: 采样点数
        line_thickness: 线宽
        background_color: 背景颜色 (BGR)
        view_angle: 投影视角 "xy", "xz", "yz"

    Returns:
        output_image: 可视化图像
    """
    H, W = image_size
    output = np.full((H, W, 3), background_color, dtype=np.uint8)

    T, N, _ = points_sequence.shape

    # 按置信度过滤
    mean_conf = conf_sequence.mean(axis=0)
    valid_mask = mean_conf >= conf_threshold
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) > subsample:
        selected_indices = np.random.choice(valid_indices, subsample, replace=False)
    else:
        selected_indices = valid_indices

    # 选择投影轴
    axis_map = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    ax1, ax2 = axis_map.get(view_angle, (0, 1))

    # 获取所有选中点的坐标范围
    selected_points = points_sequence[:, selected_indices, :]  # (T, M, 3)
    coords_1 = selected_points[:, :, ax1].flatten()
    coords_2 = selected_points[:, :, ax2].flatten()

    min_1, max_1 = coords_1.min(), coords_1.max()
    min_2, max_2 = coords_2.min(), coords_2.max()

    # 添加边距
    margin = 0.1
    range_1 = max_1 - min_1
    range_2 = max_2 - min_2
    min_1 -= range_1 * margin
    max_1 += range_1 * margin
    min_2 -= range_2 * margin
    max_2 += range_2 * margin

    # 归一化到图像坐标
    def normalize_to_image(val, min_val, max_val, size):
        return int((val - min_val) / (max_val - min_val + 1e-6) * (size - 1))

    # 获取彩虹色
    colors = get_rainbow_colors(T)

    # 绘制轨迹
    M = len(selected_indices)
    for i in range(M):
        trajectory = selected_points[:, i, :]  # (T, 3)

        for t in range(T - 1):
            x1 = normalize_to_image(trajectory[t, ax1], min_1, max_1, W)
            y1 = normalize_to_image(trajectory[t, ax2], min_2, max_2, H)
            x2 = normalize_to_image(trajectory[t + 1, ax1], min_1, max_1, W)
            y2 = normalize_to_image(trajectory[t + 1, ax2], min_2, max_2, H)

            # 翻转 Y 轴（图像坐标系 Y 向下）
            y1 = H - 1 - y1
            y2 = H - 1 - y2

            cv2.line(output, (x1, y1), (x2, y2), colors[t], line_thickness, cv2.LINE_AA)

    return output


def create_trajectory_overlay(
    npz_path: str,
    background_image_path: Optional[str] = None,
    output_path: str = "trajectory.png",
    conf_threshold: float = 3.0,
    subsample: int = 300,
) -> str:
    """创建轨迹叠加可视化

    Args:
        npz_path: VDPM .npz 文件路径
        background_image_path: 背景图片路径（可选，如无则用白色背景）
        output_path: 输出图片路径
        conf_threshold: 置信度阈值
        subsample: 采样点数

    Returns:
        输出文件路径
    """
    data = load_vdpm_npz(npz_path)
    points = data['points']  # (T, N, 3)
    conf = data['conf']      # (T, N)

    if points.ndim == 2:
        print("单帧数据，无法生成轨迹")
        return None

    if background_image_path and os.path.exists(background_image_path):
        bg_image = cv2.imread(background_image_path)
        # 需要相机内参来投影，这里暂时用 3D 投影
        output_image = visualize_trajectories_3d_projection(
            points, conf,
            image_size=(bg_image.shape[0], bg_image.shape[1]),
            conf_threshold=conf_threshold,
            subsample=subsample,
            background_color=(255, 255, 255),
        )
        # 混合背景
        alpha = 0.7
        output_image = cv2.addWeighted(bg_image, alpha, output_image, 1 - alpha, 0)
    else:
        output_image = visualize_trajectories_3d_projection(
            points, conf,
            image_size=(512, 512),
            conf_threshold=conf_threshold,
            subsample=subsample,
        )

    cv2.imwrite(output_path, output_image)
    print(f"轨迹图已保存到: {output_path}")
    return output_path


def infer_pointmap_resolution(n_points: int) -> Tuple[int, int]:
    """根据点数推断 VDPM 使用的分辨率

    VDPM 输出的点云是 H*W 个点，按行优先排列
    """
    # 常见的 VDPM 分辨率
    common_resolutions = [
        (518, 294),  # camel 视频
        (512, 288),
        (640, 360),
        (384, 216),
        (320, 180),
    ]

    for w, h in common_resolutions:
        if w * h == n_points:
            return h, w  # 返回 (H, W)

    # 尝试分解
    import math
    for h in range(200, 600):
        if n_points % h == 0:
            w = n_points // h
            if 1.5 < w / h < 2.0:  # 合理的宽高比
                return h, w

    raise ValueError(f"无法推断分辨率，点数: {n_points}")


def draw_trajectory_on_video_frame(
    video_path: str,
    npz_path: str,
    output_path: str = "trajectory_on_frame.png",
    target_frame: int = -1,  # -1 表示最后一帧
    conf_threshold: float = 3.0,
    subsample: int = 500,
    line_thickness: int = 2,
) -> str:
    """在视频帧上绘制 2D 轨迹

    利用 VDPM 点云的索引对应像素位置这一特性

    Args:
        video_path: 视频路径
        npz_path: VDPM .npz 文件
        output_path: 输出路径
        target_frame: 目标帧（背景）
        conf_threshold: 置信度阈值
        subsample: 采样轨迹数
        line_thickness: 线宽

    Returns:
        输出文件路径
    """
    # 加载点云
    data = load_vdpm_npz(npz_path)
    points = data['points']  # (T, N, 3)
    conf = data['conf']

    if points.ndim == 2:
        print("单帧数据，无法生成轨迹")
        return None

    T, N, _ = points.shape

    # 推断 VDPM 内部分辨率
    pm_H, pm_W = infer_pointmap_resolution(N)
    print(f"推断的点云分辨率: {pm_W}x{pm_H}")

    # 读取视频帧
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_W = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_H = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if target_frame == -1:
        target_frame = total_frames - 1

    video.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = video.read()
    video.release()

    if not ret:
        print(f"无法读取帧 {target_frame}")
        return None

    output = frame.copy()

    # 计算缩放比例
    scale_x = vid_W / pm_W
    scale_y = vid_H / pm_H

    # 按置信度过滤
    mean_conf = conf.mean(axis=0)
    valid_mask = mean_conf >= conf_threshold

    # 计算运动量（在 3D 空间）
    displacement = np.linalg.norm(points[-1] - points[0], axis=1)
    motion_mask = displacement > 0.005  # 过滤静态点

    combined_mask = valid_mask & motion_mask
    valid_indices = np.where(combined_mask)[0]

    if len(valid_indices) == 0:
        print("没有有效的动态点")
        return None

    print(f"动态点数: {len(valid_indices)}")

    # 按运动量排序，选择运动最大的
    motion_scores = displacement[valid_indices]
    sorted_order = np.argsort(-motion_scores)
    selected_indices = valid_indices[sorted_order[:subsample]]

    # 点索引 -> 像素坐标 (VDPM 按行优先存储)
    def idx_to_pixel(idx):
        py_pm = idx // pm_W  # 点云空间的 y
        px_pm = idx % pm_W   # 点云空间的 x
        # 缩放到视频分辨率
        px = int(px_pm * scale_x)
        py = int(py_pm * scale_y)
        return px, py

    colors = get_rainbow_colors(T)

    # 绘制轨迹
    # 注意：由于 3D 点会移动，我们需要追踪它们在每帧的 2D 位置
    # 但 VDPM 的点索引是固定的，表示"同一个物理点"
    # 所以我们直接用点的 3D 位置投影到图像上

    # 方法 1：使用索引位置（起始帧的 2D 位置）+ 3D 位移映射
    # 方法 2：直接用 3D 坐标的 x, y 分量（相机坐标系）

    # 使用方法 2：3D 点的 x, y 就是相机坐标系下的位置
    # 投影公式：u = fx * X/Z + cx, v = fy * Y/Z + cy
    # 简化：假设 Z 变化不大，直接用 X, Y 的相对变化

    # 获取第一帧的 2D 位置作为参考
    for idx in selected_indices:
        # 起始 2D 位置
        base_px, base_py = idx_to_pixel(idx)

        trajectory_3d = points[:, idx, :]  # (T, 3)

        # 计算相对于第一帧的 3D 位移
        delta_xyz = trajectory_3d - trajectory_3d[0]  # (T, 3)

        # 将 3D 位移映射到 2D 位移（简化：忽略 Z 变化的影响）
        # 假设相机焦距约等于图像宽度的 1.5 倍（典型值）
        focal = pm_W * 1.5
        z_ref = max(trajectory_3d[:, 2].mean(), 0.1)  # 使用平均深度作为参考

        for t in range(T - 1):
            # 当前帧的 2D 位置
            dx1 = delta_xyz[t, 0] / z_ref * focal * scale_x
            dy1 = delta_xyz[t, 1] / z_ref * focal * scale_y
            dx2 = delta_xyz[t + 1, 0] / z_ref * focal * scale_x
            dy2 = delta_xyz[t + 1, 1] / z_ref * focal * scale_y

            px1 = int(base_px + dx1)
            py1 = int(base_py + dy1)
            px2 = int(base_px + dx2)
            py2 = int(base_py + dy2)

            # 边界检查
            if (0 <= px1 < vid_W and 0 <= py1 < vid_H and
                0 <= px2 < vid_W and 0 <= py2 < vid_H):
                cv2.line(output, (px1, py1), (px2, py2), colors[t], line_thickness, cv2.LINE_AA)

    cv2.imwrite(output_path, output)
    print(f"轨迹图已保存到: {output_path}")
    return output_path


def visualize_with_pointcloud_and_trajectory(
    npz_path: str,
    first_frame_image_path: str,
    output_path: str = "trajectory_overlay.png",
    conf_threshold: float = 3.0,
    trajectory_subsample: int = 500,
    point_subsample: int = 5000,
    line_thickness: int = 2,
) -> str:
    """在点云渲染上叠加轨迹（类似你发的骆驼图）

    这个函数会:
    1. 渲染最后一帧的彩色点云
    2. 在上面叠加彩虹色轨迹线

    Args:
        npz_path: VDPM .npz 文件
        first_frame_image_path: 第一帧原图（用于获取点云颜色）
        output_path: 输出路径
        conf_threshold: 置信度阈值
        trajectory_subsample: 轨迹采样点数
        point_subsample: 点云渲染采样点数
        line_thickness: 轨迹线宽

    Returns:
        输出文件路径
    """
    data = load_vdpm_npz(npz_path)
    points = data['points']  # (T, N, 3)
    conf = data['conf']      # (T, N)

    if points.ndim == 2:
        print("单帧数据，无法生成轨迹")
        return None

    T, N, _ = points.shape

    # 加载原图获取颜色
    original_image = cv2.imread(first_frame_image_path)
    if original_image is None:
        print(f"无法加载图片: {first_frame_image_path}")
        return None

    H_img, W_img = original_image.shape[:2]

    # 计算投影边界
    all_points = points.reshape(-1, 3)
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()

    # 添加边距
    margin = 0.1
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= x_range * margin
    x_max += x_range * margin
    y_min -= y_range * margin
    y_max += y_range * margin

    # 创建输出图像
    output_size = 800
    output = np.ones((output_size, output_size, 3), dtype=np.uint8) * 255

    def project_to_image(x, y):
        """将 3D 点的 x, y 坐标映射到图像坐标"""
        px = int((x - x_min) / (x_max - x_min + 1e-6) * (output_size - 1))
        py = int((y - y_min) / (y_max - y_min + 1e-6) * (output_size - 1))
        # 翻转 Y
        py = output_size - 1 - py
        return px, py

    # 按置信度过滤
    mean_conf = conf.mean(axis=0)
    valid_mask = mean_conf >= conf_threshold
    valid_indices = np.where(valid_mask)[0]

    # --- 1. 渲染最后一帧的点云 ---
    last_frame_points = points[-1]  # (N, 3)

    # 采样点云
    if len(valid_indices) > point_subsample:
        point_indices = np.random.choice(valid_indices, point_subsample, replace=False)
    else:
        point_indices = valid_indices

    # 按深度排序（远的先画）
    depths = last_frame_points[point_indices, 2]
    sorted_order = np.argsort(-depths)  # 从远到近
    point_indices = point_indices[sorted_order]

    for idx in point_indices:
        x, y, z = last_frame_points[idx]
        px, py = project_to_image(x, y)

        if 0 <= px < output_size and 0 <= py < output_size:
            # 从原图获取颜色（使用第一帧的点位置）
            first_x, first_y = points[0, idx, 0], points[0, idx, 1]

            # 将 3D 坐标映射到原图像素（简化：使用归一化）
            img_x = int((first_x - x_min) / (x_max - x_min + 1e-6) * (W_img - 1))
            img_y = int((first_y - y_min) / (y_max - y_min + 1e-6) * (H_img - 1))
            img_y = H_img - 1 - img_y

            img_x = np.clip(img_x, 0, W_img - 1)
            img_y = np.clip(img_y, 0, H_img - 1)

            color = tuple(map(int, original_image[img_y, img_x]))

            # 根据深度调整点大小
            point_size = max(1, int(3 - z * 2))
            cv2.circle(output, (px, py), point_size, color, -1)

    # --- 2. 绘制轨迹 ---
    if len(valid_indices) > trajectory_subsample:
        traj_indices = np.random.choice(valid_indices, trajectory_subsample, replace=False)
    else:
        traj_indices = valid_indices

    colors = get_rainbow_colors(T)

    for idx in traj_indices:
        trajectory = points[:, idx, :]  # (T, 3)

        for t in range(T - 1):
            x1, y1, _ = trajectory[t]
            x2, y2, _ = trajectory[t + 1]

            px1, py1 = project_to_image(x1, y1)
            px2, py2 = project_to_image(x2, y2)

            if (0 <= px1 < output_size and 0 <= py1 < output_size and
                0 <= px2 < output_size and 0 <= py2 < output_size):
                cv2.line(output, (px1, py1), (px2, py2), colors[t], line_thickness, cv2.LINE_AA)

    cv2.imwrite(output_path, output)
    print(f"轨迹图已保存到: {output_path}")
    return output_path


# ============ CLI ============
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VDPM 轨迹可视化")
    parser.add_argument("--npz", type=str, required=True, help="VDPM .npz 文件")
    parser.add_argument("--background", type=str, default=None, help="背景图片（可选）")
    parser.add_argument("--output", type=str, default="trajectory.png", help="输出路径")
    parser.add_argument("--conf", type=float, default=3.0, help="置信度阈值")
    parser.add_argument("--subsample", type=int, default=300, help="采样点数")

    args = parser.parse_args()

    create_trajectory_overlay(
        args.npz,
        args.background,
        args.output,
        args.conf,
        args.subsample,
    )
