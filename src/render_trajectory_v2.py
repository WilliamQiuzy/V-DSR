"""
VDPM 轨迹渲染脚本

基于 VDPM gradio_demo.py 的 Plotly 渲染逻辑，
直接从 .npz 文件生成带轨迹的点云图片。

依赖: pip install plotly kaleido
"""

import numpy as np
import matplotlib
import matplotlib.colors
import plotly.graph_objects as go
from typing import Tuple, Optional
import cv2


# 参数（来自 gradio_demo.py）
MAX_POINTS_PER_FRAME = 50_000
TRAIL_LENGTH = 16
MAX_TRACKS = 200
STATIC_THRESHOLD = 0.025


def load_vdpm_data(npz_path: str, video_path: str) -> dict:
    """加载 VDPM 数据和视频帧颜色"""
    data = np.load(npz_path)
    points = data['points']  # (T, N, 3)
    conf = data['conf']      # (T, N)

    T, N, _ = points.shape

    # 推断分辨率
    # 常见: 518x294, 512x288, 518x518, 588x296 等
    H, W = None, None
    for h in range(200, 600):
        if N % h == 0:
            w = N // h
            # 支持宽高比 1.0 到 2.0
            if 0.9 < w / h < 2.1:
                H, W = h, w
                break

    if H is None:
        raise ValueError(f"无法推断分辨率，点数: {N}")

    print(f"点云分辨率: {W}x{H}")

    # 读取视频帧获取颜色
    video = cv2.VideoCapture(video_path)
    images = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        # BGR -> RGB, 归一化
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
        # 缩放到点云分辨率
        frame_resized = cv2.resize(frame_rgb, (W, H))
        images.append(frame_resized)
    video.release()

    # 均匀采样到 T 帧
    indices = np.linspace(0, len(images) - 1, T, dtype=int)
    images = np.array([images[i] for i in indices])  # (T, H, W, 3)

    # 重塑点云为 (T, H, W, 3)
    world_points = points.reshape(T, H, W, 3)

    return {
        'world_points': world_points,  # (T, H, W, 3)
        'conf': conf.reshape(T, H, W) if conf is not None else None,
        'images': images,  # (T, H, W, 3)
    }


def compute_scene_bounds(world_points: np.ndarray):
    """计算场景边界（来自 gradio_demo.py）"""
    all_pts = world_points.reshape(-1, 3)
    raw_min = all_pts.min(axis=0)
    raw_max = all_pts.max(axis=0)

    center = 0.5 * (raw_min + raw_max)
    half_extent = 0.5 * (raw_max - raw_min) * 1.05

    if np.all(half_extent < 1e-6):
        half_extent[:] = 1.0
    else:
        half_extent[half_extent < 1e-6] = half_extent.max()

    global_min = center - half_extent
    global_max = center + half_extent

    max_half = half_extent.max()
    aspectratio = {
        "x": float(half_extent[0] / max_half),
        "y": float(half_extent[1] / max_half),
        "z": float(half_extent[2] / max_half),
    }
    return global_min, global_max, aspectratio


def prepare_tracks(
    world_points: np.ndarray,
    images: np.ndarray,
    conf: Optional[np.ndarray],
    conf_thres: float = 1.5,
    color_mode: str = "rainbow",  # "rainbow" 或 "depth" (由浅到深)
) -> Tuple[Optional[np.ndarray], Optional[list], Optional[np.ndarray]]:
    """准备轨迹数据（来自 gradio_demo.py）"""
    S, H, W, _ = world_points.shape
    N = H * W
    if S < 2 or N == 0:
        return None, None, None

    tracks_xyz = world_points.reshape(S, N, 3)

    # 计算位移，筛选动态点
    disp = np.linalg.norm(tracks_xyz - tracks_xyz[0:1], axis=-1)
    dynamic_mask = disp.max(axis=0) > STATIC_THRESHOLD

    # 置信度过滤
    if conf is not None:
        conf_flat = conf.reshape(S, N)
        conf_score = conf_flat.mean(axis=0)
        dynamic_mask &= (conf_score >= conf_thres)

    idx_tracks = np.nonzero(dynamic_mask)[0]
    if idx_tracks.size == 0:
        return None, None, None

    # 限制轨迹数量
    if idx_tracks.size > MAX_TRACKS:
        step = int(np.ceil(idx_tracks.size / MAX_TRACKS))
        idx_tracks = idx_tracks[::step][:MAX_TRACKS]

    tracks_xyz = tracks_xyz[:, idx_tracks, :]

    # 按 y 排序（用于颜色分配）
    order = np.argsort(tracks_xyz[0, :, 1])
    tracks_xyz = tracks_xyz[:, order, :]

    num_tracks = tracks_xyz.shape[1]
    num_frames = tracks_xyz.shape[0]

    if color_mode == "depth":
        # 由浅到深：从浅青色到深红色，更容易区分方向
        # colorscale 按时间（帧）来定义
        colorscale = []
        for t in range(num_frames):
            ratio = t / max(num_frames - 1, 1)
            # 浅青色 (150, 230, 255) -> 深红色 (180, 0, 30)
            r = int(150 + (180 - 150) * ratio)
            g = int(230 - 230 * ratio)
            b = int(255 - 225 * ratio)
            pos = ratio
            colorscale.append([pos, f"rgb({r},{g},{b})"])
        # track_ids 现在表示时间索引
        track_ids = np.arange(num_frames, dtype=float)
    else:
        # 原来的 HSV 彩虹色（按轨迹分配）
        cmap = matplotlib.cm.get_cmap("hsv")
        norm = matplotlib.colors.Normalize(vmin=0, vmax=max(num_tracks - 1, 1))

        colorscale = []
        for t in range(num_tracks):
            r, g, b, _ = cmap(norm(t))
            r, g, b = int(r * 255), int(g * 255), int(b * 255)
            pos = t / max(num_tracks - 1, 1)
            colorscale.append([pos, f"rgb({r},{g},{b})"])
        track_ids = np.arange(num_tracks, dtype=float)

    return tracks_xyz, colorscale, track_ids, color_mode, num_frames


def track_segments_for_frame(
    tracks_xyz: Optional[np.ndarray],
    track_ids: Optional[np.ndarray],
    f: int,
    trail_length: int = TRAIL_LENGTH,
    color_mode: str = "rainbow",
    num_frames: int = 1,
):
    """获取某帧的轨迹线段（来自 gradio_demo.py）"""
    if tracks_xyz is None or track_ids is None or f <= 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    start_t = max(0, f - trail_length)
    num_tracks = tracks_xyz.shape[1]

    xs, ys, zs, cs = [], [], [], []
    for j in range(num_tracks):
        seg = tracks_xyz[start_t: f + 1, j, :]
        if seg.shape[0] < 2:
            continue

        xs.extend([seg[:, 0], np.array([np.nan])])
        ys.extend([seg[:, 1], np.array([np.nan])])
        zs.extend([seg[:, 2], np.array([np.nan])])

        if color_mode == "depth":
            # 按时间分配颜色：每个点的颜色是它的时间索引
            time_indices = np.arange(start_t, f + 1, dtype=float)
            cs.append(np.concatenate([time_indices, np.array([np.nan])]))
        else:
            # 按轨迹分配颜色
            cs.append(np.full(seg.shape[0] + 1, track_ids[j], dtype=float))

    x = np.concatenate(xs) if xs else np.array([])
    y = np.concatenate(ys) if ys else np.array([])
    z = np.concatenate(zs) if zs else np.array([])
    c = np.concatenate(cs) if cs else np.array([])

    return x, y, z, c


def sample_frame_points(
    world_points: np.ndarray,
    images: np.ndarray,
    conf: Optional[np.ndarray],
    frame_idx: int,
    conf_thres: float = 1.5,
    max_points: int = MAX_POINTS_PER_FRAME,
):
    """采样某帧的点和颜色（来自 gradio_demo.py）"""
    S, H, W, _ = world_points.shape
    pts = world_points[frame_idx].reshape(-1, 3)  # (N, 3)
    cols = (images[frame_idx].reshape(-1, 3) * 255).astype(np.uint8)  # (N, 3)

    # 置信度过滤
    mask = np.ones(pts.shape[0], dtype=bool)
    if conf is not None:
        conf_flat = conf[frame_idx].reshape(-1)
        mask &= (conf_flat >= conf_thres)

    pts = pts[mask]
    cols = cols[mask]

    # 下采样
    n = pts.shape[0]
    if n > max_points:
        step = int(np.ceil(n / max_points))
        pts = pts[::step]
        cols = cols[::step]

    colors_str = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in cols]
    return pts, colors_str


def render_frame_with_tracks(
    data: dict,
    frame_idx: int,
    output_path: str,
    conf_thres: float = 1.5,
    width: int = 1200,
    height: int = 900,
    show_tracks: bool = True,
    camera_eye: tuple = (0.0, 0.0, -2.0),
    camera_center: tuple = (0.0, 0.0, 0.5),
    color_mode: str = "rainbow",  # "rainbow" 或 "depth"
) -> str:
    """渲染某帧的点云和轨迹，保存为图片

    Args:
        data: load_vdpm_data 返回的数据
        frame_idx: 要渲染的帧索引
        output_path: 输出图片路径
        conf_thres: 置信度阈值
        width: 图片宽度
        height: 图片高度
        show_tracks: 是否显示轨迹

    Returns:
        输出文件路径
    """
    world_points = data['world_points']
    images = data['images']
    conf = data.get('conf')

    S = world_points.shape[0]
    frame_idx = min(frame_idx, S - 1)

    # 计算场景边界
    global_min, global_max, aspectratio = compute_scene_bounds(world_points)

    # 准备轨迹
    if show_tracks:
        result = prepare_tracks(
            world_points, images, conf, conf_thres, color_mode
        )
        tracks_xyz, colorscale, track_ids, actual_color_mode, num_frames = result
        if actual_color_mode == "depth":
            track_cmax = max(num_frames - 1, 1)
        else:
            track_cmax = max(len(track_ids) - 1, 1) if track_ids is not None else 1
    else:
        tracks_xyz, colorscale, track_ids = None, None, None
        track_cmax = 1
        actual_color_mode = color_mode
        num_frames = S

    # 获取点云数据
    pts, cols = sample_frame_points(
        world_points, images, conf, frame_idx, conf_thres
    )

    # 获取轨迹数据
    x, y, z, c = track_segments_for_frame(
        tracks_xyz, track_ids, frame_idx,
        color_mode=actual_color_mode, num_frames=num_frames
    )

    # 创建 Plotly 图形
    traces = [
        go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode="markers",
            marker=dict(size=2, color=cols),
            showlegend=False,
            name="points",
        ),
    ]

    if show_tracks and len(x) > 0:
        traces.append(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                line=dict(
                    width=3,
                    color=c if c is not None and c.size else None,
                    colorscale=colorscale if colorscale else None,
                    cmin=0,
                    cmax=track_cmax,
                ),
                hoverinfo="skip",
                showlegend=False,
                name="tracks",
            )
        )

    fig = go.Figure(data=traces)

    # 场景配置
    scene_cfg = dict(
        xaxis=dict(visible=False, showbackground=False, range=[float(global_min[0]), float(global_max[0])]),
        yaxis=dict(visible=False, showbackground=False, range=[float(global_min[1]), float(global_max[1])]),
        zaxis=dict(visible=False, showbackground=False, range=[float(global_min[2]), float(global_max[2])]),
        aspectmode="manual",
        aspectratio=aspectratio,
        camera=dict(
            eye=dict(x=0.0, y=0.0, z=-1.0),
            center=dict(x=0.0, y=0.0, z=0.0),
            up=dict(x=0.0, y=-1.0, z=0.0),
        ),
        bgcolor='white',
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        scene=scene_cfg,
        showlegend=False,
        width=width,
        height=height,
        paper_bgcolor='white',
    )

    # 保存图片
    fig.write_image(output_path, scale=2)
    print(f"已保存: {output_path}")
    return output_path


def render_trajectory_image(
    npz_path: str,
    video_path: str,
    output_path: str,
    frame_idx: int = -1,
    conf_thres: float = 1.5,
    width: int = 1200,
    height: int = 900,
    camera_eye: tuple = (0.0, 0.0, -2.0),
    camera_center: tuple = (0.0, 0.0, 0.5),
    color_mode: str = "rainbow",
) -> str:
    """主函数：从 npz 和视频生成轨迹图片

    Args:
        npz_path: VDPM .npz 文件路径
        video_path: 原视频路径
        output_path: 输出图片路径
        frame_idx: 帧索引，-1 表示最后一帧
        conf_thres: 置信度阈值
        width: 图片宽度
        height: 图片高度
        color_mode: "rainbow" (彩虹色按轨迹) 或 "depth" (由浅到深按时间)

    Returns:
        输出文件路径
    """
    print(f"加载数据: {npz_path}")
    data = load_vdpm_data(npz_path, video_path)

    T = data['world_points'].shape[0]
    if frame_idx == -1:
        frame_idx = T - 1

    print(f"渲染帧 {frame_idx}/{T-1}, 颜色模式: {color_mode}")
    return render_frame_with_tracks(
        data, frame_idx, output_path,
        conf_thres=conf_thres,
        width=width,
        height=height,
        camera_eye=camera_eye,
        camera_center=camera_center,
        color_mode=color_mode,
    )


# ============ CLI ============
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VDPM 轨迹渲染")
    parser.add_argument("--npz", type=str, required=True, help="VDPM .npz 文件")
    parser.add_argument("--video", type=str, required=True, help="原视频路径")
    parser.add_argument("--output", type=str, default="trajectory.png", help="输出路径")
    parser.add_argument("--frame", type=int, default=-1, help="帧索引，-1 为最后一帧")
    parser.add_argument("--conf", type=float, default=1.5, help="置信度阈值")
    parser.add_argument("--color", type=str, default="rainbow",
                        choices=["rainbow", "depth"],
                        help="颜色模式: rainbow (彩虹色) 或 depth (由浅到深)")

    args = parser.parse_args()

    render_trajectory_image(
        args.npz,
        args.video,
        args.output,
        args.frame,
        args.conf,
        color_mode=args.color,
    )
