"""
SAM2 单帧图像分割
输入: 一帧图片 + 点击坐标
输出: mask 后的物体，其他地方留黑

使用方法:
    python sam2_segment.py --image input.jpg --point 500,300 --output output.png

安装依赖:
    pip install torch torchvision
    pip install segment-anything-2
"""

import argparse
import numpy as np
from PIL import Image
import torch
import os
import urllib.request


def download_checkpoint(model_size="large"):
    """下载 SAM2 模型权重"""
    checkpoints = {
        "tiny": ("sam2_hiera_tiny.pt", "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt"),
        "small": ("sam2_hiera_small.pt", "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt"),
        "base_plus": ("sam2_hiera_base_plus.pt", "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt"),
        "large": ("sam2_hiera_large.pt", "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"),
    }

    filename, url = checkpoints[model_size]
    cache_dir = os.path.expanduser("~/.cache/sam2")
    os.makedirs(cache_dir, exist_ok=True)
    checkpoint_path = os.path.join(cache_dir, filename)

    if not os.path.exists(checkpoint_path):
        print(f"下载模型权重: {filename}...")
        urllib.request.urlretrieve(url, checkpoint_path)
        print("下载完成")

    return checkpoint_path


def segment_with_sam2(image_path, point, output_path, background="black", device="cuda", model_size="large"):
    """使用 SAM2 官方包分割"""
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    # 加载图片
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    print(f"图片尺寸: {image_np.shape}")
    print(f"点击坐标: {point}")

    # 下载模型
    checkpoint_path = download_checkpoint(model_size)

    # 模型配置
    configs = {
        "tiny": "sam2_hiera_t.yaml",
        "small": "sam2_hiera_s.yaml",
        "base_plus": "sam2_hiera_b+.yaml",
        "large": "sam2_hiera_l.yaml",
    }

    print("加载 SAM2 模型...")
    sam2 = build_sam2(
        config_file=configs[model_size],
        ckpt_path=checkpoint_path,
        device=device,
    )
    predictor = SAM2ImagePredictor(sam2)

    # 设置图片
    predictor.set_image(image_np)

    # 预测
    print("分割中...")
    point_coords = np.array([[point[0], point[1]]])
    point_labels = np.array([1])  # 1 表示前景点

    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )

    # 选择得分最高的 mask
    best_idx = np.argmax(scores)
    mask = masks[best_idx]

    print(f"得分: {scores[best_idx]:.3f}")

    # 生成输出
    return save_masked_image(image_np, mask, output_path, background)


def segment_with_sam1(image_path, point, output_path, background="black", device="cuda"):
    """使用 SAM1 (更稳定，兼容性更好)

    安装: pip install segment-anything
    """
    from segment_anything import sam_model_registry, SamPredictor

    # 加载图片
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    print(f"图片尺寸: {image_np.shape}")
    print(f"点击坐标: {point}")

    # 下载 SAM1 模型
    cache_dir = os.path.expanduser("~/.cache/sam")
    os.makedirs(cache_dir, exist_ok=True)
    checkpoint_path = os.path.join(cache_dir, "sam_vit_h_4b8939.pth")

    if not os.path.exists(checkpoint_path):
        print("下载 SAM1 模型权重...")
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        urllib.request.urlretrieve(url, checkpoint_path)
        print("下载完成")

    print("加载 SAM1 模型...")
    sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
    sam.to(device)
    predictor = SamPredictor(sam)

    # 设置图片
    predictor.set_image(image_np)

    # 预测
    print("分割中...")
    point_coords = np.array([[point[0], point[1]]])
    point_labels = np.array([1])

    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )

    # 选择得分最高的 mask
    best_idx = np.argmax(scores)
    mask = masks[best_idx]

    print(f"得分: {scores[best_idx]:.3f}")

    return save_masked_image(image_np, mask, output_path, background)


def segment_simple(image_path, point, output_path, background="black", device="cuda"):
    """最简单的方式: 使用 ultralytics 的 SAM

    安装: pip install ultralytics
    """
    from ultralytics import SAM

    # 加载图片
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    print(f"图片尺寸: {image_np.shape}")
    print(f"点击坐标: {point}")

    # 加载模型 (会自动下载)
    print("加载 SAM 模型...")
    model = SAM("sam2.1_l.pt")  # 或 "sam_l.pt" for SAM1

    # 预测
    print("分割中...")
    results = model(image_path, points=[point], labels=[1])

    # 获取 mask
    if results[0].masks is not None:
        mask = results[0].masks.data[0].cpu().numpy() > 0.5
    else:
        print("未检测到物体")
        return None

    return save_masked_image(image_np, mask, output_path, background)


def save_masked_image(image_np, mask, output_path, background="black"):
    """保存 mask 后的图片"""
    # 生成输出图片
    if background == "black":
        output = np.zeros_like(image_np)
    else:
        output = np.ones_like(image_np) * 255

    # 将 mask 区域填充为原图内容
    output[mask] = image_np[mask]

    # 保存
    output_image = Image.fromarray(output)
    output_image.save(output_path)
    print(f"已保存到: {output_path}")

    # 同时保存 mask
    mask_path = output_path.rsplit(".", 1)[0] + "_mask.png"
    mask_image = Image.fromarray((mask * 255).astype(np.uint8))
    mask_image.save(mask_path)
    print(f"Mask 保存到: {mask_path}")

    return mask


def main():
    parser = argparse.ArgumentParser(description="SAM 单帧图像分割")
    parser.add_argument("--image", type=str, required=True, help="输入图片路径")
    parser.add_argument("--point", type=str, required=True, help="点击坐标，格式: x,y")
    parser.add_argument("--output", type=str, default="output.png", help="输出图片路径")
    parser.add_argument("--background", type=str, default="black", choices=["black", "white"], help="背景颜色")
    parser.add_argument("--device", type=str, default="cuda", help="设备: cuda 或 cpu")
    parser.add_argument("--method", type=str, default="ultralytics",
                        choices=["sam2", "sam1", "ultralytics"],
                        help="使用的方法: sam2 (官方), sam1 (旧版), ultralytics (最简单)")

    args = parser.parse_args()

    # 解析点击坐标
    x, y = map(int, args.point.split(","))
    point = (x, y)

    # 检查设备
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，使用 CPU")
        args.device = "cpu"

    # 分割
    if args.method == "sam2":
        segment_with_sam2(args.image, point, args.output, args.background, args.device)
    elif args.method == "sam1":
        segment_with_sam1(args.image, point, args.output, args.background, args.device)
    else:
        segment_simple(args.image, point, args.output, args.background, args.device)


if __name__ == "__main__":
    main()
