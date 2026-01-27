import numpy as np
from grounding_dino.groundingdino.util.inference import load_image, predict
import random
import torch
from torchvision.ops import box_convert
import cv2
from PIL import Image
from inference import get_3angle
from paths import *
from vision_tower import DINOv2_MLP
from transformers import AutoImageProcessor
import torch.nn.functional as F
from utils import *
from inference import *
from grounding_dino.groundingdino.util.vl_utils import create_positive_map_from_span
from grounding_dino.groundingdino.util.utils import get_phrases_from_posmap
from pathlib import Path
from typing import List, Tuple, Optional
import os
def sample_uniform_frames(
    video_path: str,
    num_frames: int = 32,
    save_dir: Optional[str] = None,
    round_decimals: int = 1,
) -> List[Tuple[np.ndarray, float]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    # 帧数和 fps
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        # 有些视频元数据缺失时，opencv 报 fps = 0
        fps = None

    # 视频时长（秒）；若 frame_count 和 fps 可用则计算
    duration = None
    if fps:
        duration = frame_count / fps

    # 如果 frame_count 不可用或 < num_frames，改用时间线均匀采样（0..duration）
    if frame_count <= 0 or fps is None:
        # 尝试用 CAP_PROP_POS_MSEC 探测（不可靠），这里直接均匀用时间位置
        raise RuntimeError("invalid frame_count or fps")

    # 生成要读取的帧索引（整数），在 [0, frame_count-1] 均匀分布
    indices = np.linspace(0, frame_count - 1, num_frames)
    # 为了更稳定地 seek，取每个索引的四舍五入整数
    indices_int = np.round(indices).astype(int)

    results = []
    save_path = Path(save_dir) if save_dir else None
    if save_path:
        save_path.mkdir(parents=True, exist_ok=True)

    last_idx = -1
    for i, idx in enumerate(indices_int):
        # 避免重复 seek 太多次（如果重复索引就复用上一帧）
        if idx == last_idx and results:
            raise RuntimeError(f"repeat frame: {idx}")
        else:
            # 用 CAP_PROP_POS_FRAMES 定位并读取
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame_img = cap.read()
            if not ret:
                raise RuntimeError(f"read fail: {idx}")
            # 计算时间戳（秒）
            timestamp = idx / fps if fps else 0.0
            # 四舍五入到指定小数位
            timestamp = float(round(timestamp, round_decimals))
        results.append(timestamp)
        last_idx = idx

        # 可选保存
        if save_path:
            fname = f"{i:03d}.jpg"
            out_file = os.path.join(save_path,fname)
            cv2.imwrite(str(out_file), frame_img)

    cap.release()
    return results

def nms(boxes, scores, threshold=0.6):
    
    # If there are no boxes, return an empty list
    if boxes.size(0) == 0:
        return torch.empty(0, dtype=torch.long)
    
    # Convert from [cx, cy, w, h] to [x1, y1, x2, y2]
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    boxes = torch.stack([x1, y1, x2, y2], dim=1)
    
    # Compute area of each box
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    # Sort the boxes by their scores (in descending order)
    _, indices = scores.sort(descending=True)
    
    keep = []
    
    while indices.size(0) > 0:
        # Take the box with the highest score
        current = indices[0]
        keep.append(current)
        
        # Compute IoU between the highest score box and the rest
        xx1 = torch.maximum(boxes[current, 0], boxes[indices[1:], 0])
        yy1 = torch.maximum(boxes[current, 1], boxes[indices[1:], 1])
        xx2 = torch.minimum(boxes[current, 2], boxes[indices[1:], 2])
        yy2 = torch.minimum(boxes[current, 3], boxes[indices[1:], 3])
        
        # Compute intersection area
        inter_width = torch.maximum(torch.tensor(0.0), xx2 - xx1 + 1)
        inter_height = torch.maximum(torch.tensor(0.0), yy2 - yy1 + 1)
        inter_area = inter_width * inter_height
        
        # Compute IoU
        iou = inter_area / (areas[current] + areas[indices[1:]] - inter_area)
        
        # Keep boxes with IoU lower than the threshold
        remaining_indices = torch.nonzero(iou <= threshold).squeeze(1)
        indices = indices[remaining_indices + 1]
    
    return torch.tensor(keep)


def get_grounding_output(model, image, caption, box_threshold, device, text_threshold=None, with_logits=False, cpu_only=False, token_spans=None):
    assert text_threshold is not None or token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)
    # filter output
    if token_spans is None:
        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)
    else:
        # given-phrase mode
        positive_maps = create_positive_map_from_span(
            model.tokenizer(caption),
            token_span=token_spans
        ).to(image.device) # n_phrase, 256
        phrase = [' '.join([caption[_s:_e] for (_s, _e) in token_span]) for token_span in token_spans]
        logits_for_phrases = positive_maps @ logits.T # n_phrase, nq
        all_logits = []
        all_phrases = []
        all_boxes = []
        logits_for_phrases_score, corr_phrases = torch.max(logits_for_phrases,0)
        valid_box = boxes[logits_for_phrases_score>box_threshold]
        pred_phrases = [phrase[item] for item in corr_phrases[logits_for_phrases_score>box_threshold].tolist()]

    return valid_box.cpu(), pred_phrases, logits_for_phrases_score[logits_for_phrases_score>box_threshold].cpu()

def cal_dis(vector):
    return np.sqrt(np.sum(vector**2))

def camera_mutual_coordinate_transform(coords, mat_1, mat_2):
    rot_1 = mat_1[:3,:3]
    trans_1 = mat_1[:3,3:]
    rot_2 = mat_2[:3,:3]
    trans_2 = mat_2[:3,3:]
    coords_world = np.dot(rot_1, coords.T) + trans_1
    
    # 将P_world从世界坐标系转换到目标坐标系
    coords_new = np.dot(np.linalg.inv(rot_2), coords_world - trans_2).T
    
    return coords_new

def depth2pc(depth_image, K):
    f_x = K[0, 0]
    f_y = K[1, 1]
    c_x = K[0, 2]
    c_y = K[1, 2]
    
    # 获取深度图的高度和宽度
    height, width = depth_image.shape
    
    # 初始化空的点云数组
    point_cloud = []
    
    # 遍历深度图的每个像素
    for v in range(height):
        for u in range(width):
            # 获取深度值
            d = depth_image[v, u]
            
            # 计算3D点坐标
            if d > 0:  # 只处理有效深度值
                X = (u - c_x) * d / f_x
                Y = (v - c_y) * d / f_y
                Z = d
                point_cloud.append([X, Y, Z])
    
    # 转换为 numpy 数组
    point_cloud = np.array(point_cloud)
    
    return point_cloud

def camera_self_coord_trans(coords):
    new_x = coords[:,2:3]
    new_y = -coords[:,0:1]
    new_z = -coords[:,1:2]
    return np.hstack([new_x,new_y,new_z])

def rotation_matrix_x(angle):
    rad = np.radians(angle)
    return np.array([
        [1, 0, 0],
        [0, np.cos(rad), np.sin(rad)],
        [0, -np.sin(rad), np.cos(rad)]
    ])

def rotation_matrix_y(angle):
    rad = np.radians(angle)
    return np.array([
        [np.cos(rad), 0, -np.sin(rad)],
        [0, 1, 0],
        [np.sin(rad), 0, np.cos(rad)]
    ])

def rotation_matrix_z(angle):
    rad = np.radians(angle)
    return np.array([
        [np.cos(rad), np.sin(rad), 0],
        [-np.sin(rad), np.cos(rad), 0],
        [0, 0, 1]
    ])

def rotate_point_cloud(points, angle_x, angle_y, angle_z):
    Rx = rotation_matrix_x(angle_x)
    Ry = rotation_matrix_y(angle_y)
    Rz = rotation_matrix_z(angle_z)
    Rp1 = rotation_matrix_x(90)
    Rp2 = rotation_matrix_z(270)
    R = Rp2 @ Rp1 @ Ry @ Rx @ Rz

    rotated_points = np.dot(R, points.T)
    
    return rotated_points.T

def camera_agent_coord_trans(coords, polar, azimuth, rotation, trans):
    coords -= trans
    new_coords = rotate_point_cloud(coords,polar,azimuth,rotation)
    return new_coords

def agent_camera_coord_trans(coord, polar, azimuth, rotation, trans):
    rev_r_1 = rotation_matrix_z(270).T
    rev_r_2 = rotation_matrix_x(90).T
    rev_r_3 = rotation_matrix_y(azimuth).T
    rev_r_4 = rotation_matrix_x(polar).T
    rev_r_5 = rotation_matrix_z(rotation).T
    R = rev_r_5 @ rev_r_4 @ rev_r_3 @ rev_r_2 @ rev_r_1
    coord_reverse = np.dot(R, coord.T).T
    coord_reverse += trans
    return coord_reverse

def extract_obj_center(coords, mask, conf_mask):
    mask_and = np.logical_and(mask,conf_mask)
    obj = coords[mask_and.flatten()]
    return np.mean(obj,0)

def extract_token(obj_str):
    objs = obj_str.split(".")[:-1]
    token = []
    pos=0
    for obj in objs:
        obj = obj.strip()
        token_tmp = [[pos, pos+len(obj)]]
        pos = pos+len(obj)+2
        #parts = obj.split(" ")
        #for part in parts:
        #    pos_end = pos+len(part)
        #    token_tmp.append([pos,pos_end])
        #    pos = pos_end+1
        token.append(token_tmp)
    return token
        
def extract_obj_image_mask(grounding_model, segment_model, image, obj, device, box_threshold=0.35, text_threshold=0.25):
    image_source, image = load_image(image)
    token = extract_token(obj)
    boxes, labels, scores = get_grounding_output(
            model=grounding_model,
            image=image,
            caption=obj,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            token_spans=token,
            device=device
        )
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    keep_index = nms(boxes, scores)
    boxes = boxes[keep_index]
    labels = [labels[index] for index in keep_index.tolist()]
    idx = random.sample(range(len(boxes)), 1)
    boxes = boxes[idx]
    label_output = labels[idx[0]]
    segment_model.set_image(image_source)

    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    masks, scores, logits = segment_model.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    return label_output, input_boxes[0].astype(np.uint32), masks[0].astype(np.bool_)

def extract_obj_video_mask(grounding_model, segment_model, video, obj, num, device, box_threshold=0.35, text_threshold=0.25):
    image_source, image = load_image(video[0])
    token = extract_token(obj)
    boxes, labels, scores = get_grounding_output(
            model=grounding_model,
            image=image,
            caption=obj,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            token_spans=token,
            device=device
        )
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    keep_index = nms(boxes,scores)
    boxes = boxes[keep_index]
    scores = scores[keep_index]
    labels = [labels[index] for index in keep_index.tolist()]
    idx = random.sample(range(len(boxes)), num)
    boxes = boxes[idx]
    label_output = [labels[i] for i in idx]
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    inference_state = segment_model.init_state(video_path=video)
    for object_id, box in enumerate(input_boxes, start=1):
        _, out_obj_ids, out_mask_logits = segment_model.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=object_id,
            box=box,
        )
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in segment_model.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    masks = []
    for object_id, _ in enumerate(input_boxes, start=1):
        mask_tmp = [video_segments[key][object_id].squeeze() for key in video_segments]
        mask_tmp = np.stack(mask_tmp)
        masks.append(mask_tmp)
    masks = np.stack(masks)
    return label_output, input_boxes.astype(np.uint32), masks

def extract_all_video_mask(grounding_model, segment_model, video, obj, device, box_threshold=0.35, text_threshold=0.25):
    image_source, image = load_image(video[0])
    token = extract_token(obj)
    boxes, labels, scores = get_grounding_output(
            model=grounding_model,
            image=image,
            caption=obj,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            token_spans=token,
            device=device
        )
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    keep_index = nms(boxes,scores)
    boxes = boxes[keep_index]
    scores = scores[keep_index]
    labels = [labels[index] for index in keep_index.tolist()]
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    inference_state = segment_model.init_state(video_path=video)
    for object_id, box in enumerate(input_boxes, start=1):
        _, out_obj_ids, out_mask_logits = segment_model.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=object_id,
            box=box,
        )
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in segment_model.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    masks = []
    for object_id, _ in enumerate(input_boxes, start=1):
        mask_tmp = [video_segments[key][object_id].squeeze() for key in video_segments]
        mask_tmp = np.stack(mask_tmp)
        masks.append(mask_tmp)
    masks = np.stack(masks)
    return labels, input_boxes.astype(np.uint32), masks

def mask2bbox(mask):
    nonzero_indices = np.nonzero(mask)
    min_y, min_x = np.min(nonzero_indices, axis=1)
    max_y, max_x = np.max(nonzero_indices, axis=1)
    bbox = [min_x, min_y, max_x, max_y]
    return bbox

def extract_orient(dino, val_preprocess, image, mask, device):
    origin_image = cv2.imread(image)
    origin_image = origin_image[:,:,::-1]
    h,w,_ = origin_image.shape
    alpha_channel = np.ones((origin_image.shape[0], origin_image.shape[1]), dtype=origin_image.dtype) * 255
    origin_image = cv2.merge([origin_image[:, :, 0], origin_image[:, :, 1], origin_image[:, :, 2], alpha_channel])
    rm_bkg_img = origin_image
    rm_bkg_img[mask==0][:,-1]=0
    bbox = mask2bbox(mask)
    min_y = min(min(10,bbox[1]),min(10,h-bbox[3]))
    min_x = min(min(10,bbox[0]),min(10,w-bbox[2]))
    crop_image = rm_bkg_img[(bbox[1]-min_y):(bbox[3]+min_y),(bbox[0]-min_x):(bbox[2]+min_x)]
    crop_image = Image.fromarray(crop_image)
    angles = get_3angle(crop_image, dino, val_preprocess, device)
    azimuth     = float(angles[0])
    polar       = float(angles[1])
    rotation    = float(angles[2])
    confidence = float(angles[3])
    return azimuth, polar, rotation, confidence