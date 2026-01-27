from template import *
import os
import random
import numpy as np
from qa_utils import *
import cv2
import json
import cv2
import torch
import supervision as sv
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor 
from grounding_dino.groundingdino.util.inference import load_model
from transformers import AutoImageProcessor
import pandas as pd
import warnings
import multiprocessing
import torch.multiprocessing as mp
import requests
from pi3.utils.basic import load_images_as_tensor, write_ply
from pi3.utils.geometry import depth_edge
from pi3.models.pi3 import Pi3
import glob
import math

import argparse

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--part_len', type=int, default=2)
    parser.add_argument('--process_num', type=int, default=10)
    parser.add_argument('--video_root', type=str, default='./video_root')
    parser.add_argument('--qa_num', type=int, default=2)
    args = parser.parse_args()
    
    return args

def run_pi3(model, proc_id, device):
    imgs = load_images_as_tensor(f"./{proc_id}_frames_nontemp/", interval=1).to(device) # (N, 3, H, W)
    h,w,c = cv2.imread(f"./{proc_id}_frames_nontemp/000.jpg").shape
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            res = model(imgs[None]) # Add batch dimension

    masks = torch.sigmoid(res['conf'][..., 0]) > 0.1
    non_edge = ~depth_edge(res['local_points'][..., 2], rtol=0.03)
    masks = torch.logical_and(masks, non_edge)[0]
    masks = masks.float()
    local_points = res['local_points'][0].permute(0,3,1,2)
    local_points = F.interpolate(local_points,(h,w))
    local_points = local_points.cpu().numpy()
    camera_pose = res['camera_poses'][0].cpu().numpy()
    masks = F.interpolate(masks.unsqueeze(0),(h,w))[0]
    masks = masks.round().to(torch.bool)
    masks = masks.cpu().numpy()

    return local_points, masks, camera_pose

def generate_one_video(proc_id, timestamps, agents, objs, points, masks, poses, grounding_model, video_predictor, sam2_predictor, val_preprocess, dino, device):
    timestamps = sorted(timestamps)
    timestamps = timestamps[:-2]
    agents_with_cam = ["camera"]
    agents_with_cam.extend(agents)
    objs.extend(agents)
    pcs = points
    pcs = np.transpose(pcs,(0,2,3,1))
    pcs = np.reshape(pcs,(pcs.shape[0],-1,pcs.shape[-1]))
    trial = 0
    while True:
       trial+=1
       if trial>50:
        return "fail", None, None, None, None
       try:
        success = True
        qa_type = "abs"
        if random.uniform(0,1)<0.5:
            qa_type = "rel"
        if "abs" in qa_type:
            camera_poses = poses
            agent = random.sample(agents_with_cam,1)[0]
            time_1 = random.sample(timestamps,1)[0]
            idx_1 = timestamps.index(time_1)
            time_2, time_3 = random.sample(timestamps,2)
            time_2, time_3 = sorted([time_2, time_3])
            idx_2 = timestamps.index(time_2)
            idx_3 = timestamps.index(time_3)
            while (idx_3-idx_2)<3:
                time_2, time_3 = random.sample(timestamps,2)
                time_2, time_3 = sorted([time_2, time_3])
                idx_2 = timestamps.index(time_2)
                idx_3 = timestamps.index(time_3)
            if agent != "camera":
                agent_label, agent_box, agent_mask = extract_obj_image_mask(grounding_model, sam2_predictor, f"./{proc_id}_frames_nontemp/{idx_1:03d}.jpg", ". ".join(agents)+".", device)
                if np.sum(np.logical_and(agent_mask,masks[idx_1]))==0:
                    continue
                agent_ori = extract_orient(dino, val_preprocess, f"./{proc_id}_frames_nontemp/{idx_1:03d}.jpg", agent_mask, device)
                azimuth, polar, rotation = agent_ori[0], agent_ori[1], agent_ori[2]
                if agent_ori[3]<0.6:
                    continue
                trans = extract_obj_center(pcs[idx_1],agent_mask,masks[idx_1])
            video_frames = [f"./{proc_id}_frames_nontemp/{idx:03d}.jpg" for idx in range(idx_2, idx_3+1)]
            obj_labels, obj_boxes, obj_masks = extract_all_video_mask(grounding_model, video_predictor, video_frames, ". ".join(objs)+".", device)
            obj_coords = {}
            for idx in range(len(obj_labels)):
                obj_coords[f"{obj_labels[idx]} with initial bounding box coordinates [{obj_boxes[idx][0]},{obj_boxes[idx][1]},{obj_boxes[idx][2]},{obj_boxes[idx][3]}]"] = []
            for idx in range(idx_2,idx_3+1):
                for obj_idx in range(obj_boxes.shape[0]):
                    obj_mask = obj_masks[obj_idx][idx-idx_2]
                    if np.sum(np.logical_and(obj_mask,masks[idx]))==0:
                        success = False
                        break
                    obj_coord = np.array([extract_obj_center(pcs[idx], obj_mask, masks[idx])])
                    mat_1 = camera_poses[idx]
                    mat_2 = camera_poses[idx_1]
                    coords_trans = camera_mutual_coordinate_transform(obj_coord, mat_1, mat_2)
                    if agent != "camera":
                        coords_trans = camera_agent_coord_trans(coords_trans,polar, azimuth, rotation,trans)
                    else:
                        coords_trans = camera_self_coord_trans(coords_trans)
                    obj_coords[f"{obj_labels[obj_idx]} with initial bounding box coordinates [{obj_boxes[obj_idx][0]},{obj_boxes[obj_idx][1]},{obj_boxes[obj_idx][2]},{obj_boxes[obj_idx][3]}]"].append(coords_trans.squeeze().tolist())
                if success == False:
                    for obj_before_idx in range(obj_idx):
                        obj_coords[f"{obj_labels[obj_before_idx]} with initial bounding box coordinates [{obj_boxes[obj_before_idx][0]},{obj_boxes[obj_before_idx][1]},{obj_boxes[obj_before_idx][2]},{obj_boxes[obj_before_idx][3]}]"] = obj_coords[f"{obj_labels[obj_before_idx]} with initial bounding box coordinates [{obj_boxes[obj_before_idx][0]},{obj_boxes[obj_before_idx][1]},{obj_boxes[obj_before_idx][2]},{obj_boxes[obj_before_idx][3]}]"][:-1]
                    break
            if success == False:
                idx = idx-1
            time_s = time_2
            time_e = timestamps[idx]
            if (timestamps.index(time_e)-idx_2)<3:
                continue
            if agent != "camera":
                agent_ret = f"{agent_label} with bounding box coordinates [{agent_box[0]},{agent_box[1]},{agent_box[2]},{agent_box[3]}] at {time_1:.1f}s"
            else:
                agent_ret = f"camera at {time_1:.1f}s"
            return qa_type, agent_ret, time_s, time_e, obj_coords
        elif "rel" in qa_type:
            time_1, time_2 = random.sample(timestamps,2)
            time_1, time_2 = sorted([time_1, time_2])
            idx_1 = timestamps.index(time_1)
            idx_2 = timestamps.index(time_2)
            while (idx_2-idx_1)<3:
                time_1, time_2 = random.sample(timestamps,2)
                time_1, time_2 = sorted([time_1, time_2])
                idx_1 = timestamps.index(time_1)
                idx_2 = timestamps.index(time_2)
            video_frames = [f"./{proc_id}_frames_nontemp/{idx:03d}.jpg" for idx in range(idx_1, idx_2+1)]
            obj_labels, obj_boxes, obj_masks = extract_all_video_mask(grounding_model, video_predictor, video_frames, ". ".join(objs)+".", device)
            if random.uniform(0,1)<0.5:
                agent = "camera"
            else:
                agent = "rand"
                agent_idx = random.sample(range(len(obj_labels)),1)[0]
                agent_label = obj_labels[agent_idx]
                if agent_label not in agents_with_cam:
                    continue
                agent_box = obj_boxes[agent_idx]
                agent_mask = obj_masks[agent_idx]
                obj_labels.pop(agent_idx)
                obj_boxes = np.delete(obj_boxes, agent_idx, axis=0)
                obj_masks = np.delete(obj_masks, agent_idx, axis=0)
            obj_coords = {}
            for idx in range(len(obj_labels)):
                obj_coords[f"{obj_labels[idx]} with initial bounding box coordinates [{obj_boxes[idx][0]},{obj_boxes[idx][1]},{obj_boxes[idx][2]},{obj_boxes[idx][3]}]"] = []
            success = True
            for idx in range(idx_1,idx_2+1):
                if agent != "camera":
                    if np.sum(np.logical_and(agent_mask[idx-idx_1],masks[idx]))==0:
                        success = False
                        break
                    agent_ori = extract_orient(dino, val_preprocess, f"./{proc_id}_frames_nontemp/{idx:03d}.jpg", agent_mask[idx-idx_1], device)
                    azimuth, polar, rotation = agent_ori[0], agent_ori[1], agent_ori[2]
                    if agent_ori[3]<0.6:
                        success = False
                        break
                    trans = extract_obj_center(pcs[idx],agent_mask[idx-idx_1],masks[idx])
                for obj_idx in range(obj_boxes.shape[0]):
                    obj_mask = obj_masks[obj_idx][idx-idx_1]
                    if np.sum(np.logical_and(obj_mask,masks[idx]))==0:
                        success = False
                        break
                    obj_coord = np.array([extract_obj_center(pcs[idx], obj_mask, masks[idx])])
                    if agent != "camera":
                        coords_trans = camera_agent_coord_trans(obj_coord,polar, azimuth, rotation,trans)
                    else:
                        coords_trans = camera_self_coord_trans(obj_coord)
                    obj_coords[f"{obj_labels[obj_idx]} with initial bounding box coordinates [{obj_boxes[obj_idx][0]},{obj_boxes[obj_idx][1]},{obj_boxes[obj_idx][2]},{obj_boxes[obj_idx][3]}]"].append(coords_trans.squeeze().tolist())
                if success == False:
                    for obj_before_idx in range(obj_idx):
                        obj_coords[f"{obj_labels[obj_before_idx]} with initial bounding box coordinates [{obj_boxes[obj_before_idx][0]},{obj_boxes[obj_before_idx][1]},{obj_boxes[obj_before_idx][2]},{obj_boxes[obj_before_idx][3]}]"] = obj_coords[f"{obj_labels[obj_before_idx]} with initial bounding box coordinates [{obj_boxes[obj_before_idx][0]},{obj_boxes[obj_before_idx][1]},{obj_boxes[obj_before_idx][2]},{obj_boxes[obj_before_idx][3]}]"][:-1]
                    break
            if success == False:
                idx = idx-1
            time_s = time_1
            time_e = timestamps[idx]
            if (timestamps.index(time_e)-idx_1)<3:
                continue
            if agent != "camera":
                agent_ret = f"{agent_label} with initial bounding box coordinates [{agent_box[0]},{agent_box[1]},{agent_box[2]},{agent_box[3]}]"
            else:
                agent_ret = f"camera"
            return qa_type, agent_ret, time_s, time_e, obj_coords
       except:
        pass

def generate_multi_videos(physical_gpu_id, proc_id, part_info, qa_num, step_size, video_dynamic):
    device = torch.device(f"cuda:{physical_gpu_id}" if torch.cuda.is_available() else "cpu")

    GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_CHECKPOINT = "./models/groundingdino_swint_ogc.pth"
    BOX_THRESHOLD = 0.35
    TEXT_THRESHOLD = 0.25
    PROMPT_TYPE_FOR_VIDEO = "box" # choose from ["point", "box", "mask"]
    grounding_model = load_model(
        model_config_path=GROUNDING_DINO_CONFIG, 
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
        device=device
    )


    # init sam image predictor and video predictor model
    sam2_checkpoint = "./models/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    dino = DINOv2_MLP(
                    dino_mode   = 'large',
                    in_dim      = 1024,
                    out_dim     = 360+180+180+2,
                    evaluate    = True,
                    mask_dino   = False,
                    frozen_back = False
                )
    dino.eval()
    print('model create')
    dino.load_state_dict(torch.load("./models/croplargeEX2/dino_weight.pt", map_location='cpu'))
    dino = dino.to(device)
    print('weight loaded')
    pi3 = Pi3.from_pretrained("./models/Pi3").to(device).eval()
    val_preprocess   = AutoImageProcessor.from_pretrained(DINO_LARGE, cache_dir='./')
    save_files = glob.glob(f"qa_pairs_nontemp_{proc_id}_*.json")
    if os.path.exists(f"qa_pairs_nontemp_{proc_id}.json"):
        return
    if save_files==[]:
        qa_total = {}
        part_num = 0
    else:
        qa_total = json.load(open(save_files[0],"r"))
        part_num = int(save_files[0].split(".")[0].split("_")[-1])
    for idx,(_,row) in enumerate(part_info.iloc[part_num*step_size:].iterrows()):
        if (part_num*step_size+idx)%step_size==0:
            with open(f"qa_pairs_nontemp_{proc_id}_{(part_num*step_size+idx)//step_size}.json", "w", encoding="utf-8") as f:
                json.dump(qa_total, f, ensure_ascii=False, indent=4)
            if os.path.exists(f"qa_pairs_nontemp_{proc_id}_{(part_num*step_size+idx)//step_size-1}.json"):
                os.remove(f"qa_pairs_nontemp_{proc_id}_{(part_num*step_size+idx)//step_size-1}.json")
        video_name = row['videoID']
        if video_name not in video_dynamic:
            continue
        timestamps = sample_uniform_frames(video_path=f"{video_root}/{video_name}.mp4",save_dir=f"./{proc_id}_frames_nontemp")
        qa_total[video_name] = []
        agents = row['agent']
        obj = row['obj']
        if pd.isna(agents):
            continue
        else:
            agents = agents.lower()
            agents = agents.split('.')
        if pd.isna(obj):
            obj=[]
        else:
            obj = obj.lower()
            obj = obj.split('.')
        points, masks, poses = run_pi3(pi3, proc_id, device)
        trial_num = 0
        while trial_num<qa_num:
            qa_type, agent, time_s, time_e, obj_coords = generate_one_video(proc_id, timestamps, agents, obj, points, masks, poses, grounding_model, video_predictor, sam2_predictor, val_preprocess, dino, device)
            trial_num += 1
            if qa_type=="fail":
                continue
            s_idx = timestamps.index(time_s)
            e_idx = timestamps.index(time_e)
            timestamps_tmp = timestamps[s_idx:(e_idx+1)]
            token = "YOUR_TOKEN"
            url = "YOUR_URL"

            payload = {
                'model': 'deepseek-r1',
                'messages': [
                    {
                        'role': 'system',
                        'content': 'You are a helpful assistant.'
                    },
                    {
                        'role': 'user',
                        'content': 'From the perspective of {} in a video, {} are the 3D coordinates (x,y,z) of some objects between {:.1f}s and {:.1f}s. The positive direction of x, y, z axis are forward, left, up respectively. The coordinates are collected at {}. According to the change of 3D relationship between objects, please generate one question, which cannot be answered only with 2D knowledge and without 3D knowledge, and 4 answers where one of them is the correct answer. The question should not include description of perspective. The question should not be quantitative and related to accurate 3D coordinates. All objects must be refered with their bounding box coordinates given at the beginning. In a new line, only output your question. Then in 4 new lines, only output your 4 answers. Then in a new line, only output the symbol of correct answer.'.format(agent, str(obj_coords), time_s, time_e, ", ".join(str(x) + 's' for x in timestamps_tmp))
                    }
                ]
            }
            
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {token}'
            }

            response = requests.post(url, headers=headers, data=json.dumps(payload))
            if response.status_code != 200:
                print(f"Deepseek error:{proc_id}")
                continue
            output = response.json()['choices'][0]['message']['content']
            output = output.split("\n")
            output = [item for item in output if item !=""]
            while len(output)!=6:
                response = requests.post(url, headers=headers, data=json.dumps(payload))

                if response.status_code != 200:
                    print(f"Deepseek error:{proc_id}")
                    continue
                output = response.json()['choices'][0]['message']['content']
                output = output.split("\n")
                output = [item for item in output if item !=""]
            if qa_type=="rel":
                qa_total[video_name].append({"Question":f"Following the perspective of {agent}, between {time_s:.1f}s and {time_e:.1f}s, {output[0].lower()}", "A":output[1], "B":output[2], "C":output[3], "D":output[4], "Correct":output[5]})
            else:
                qa_total[video_name].append({"Question":f"From the perspective of {agent}, between {time_s:.1f}s and {time_e:.1f}s, {output[0].lower()}", "A":output[1], "B":output[2], "C":output[3], "D":output[4], "Correct":output[5]})
    with open(f'qa_pairs_nontemp_{proc_id}.json', 'w') as json_file:
        json.dump(qa_total, json_file, ensure_ascii=False, indent=4)
    save_files = glob.glob(f"qa_pairs_nontemp_{proc_id}_*.json")
    if save_files != []:
        for file in save_files:
            os.remove(file)

def split_data(data, num_processes):
    avg_len = len(data)//num_processes
    return [data.iloc[i:i + avg_len] for i in range(0, len(data), avg_len)]

args = parse_arg()
video_root = args.video_root
warnings.filterwarnings("error", category=RuntimeWarning)
video_dynamic = json.load(open("./dynamic_videos.json","r"))
video_dynamic = [item.split(".")[0] for item in video_dynamic]
num_processes = args.process_num
meta_info = pd.read_csv("./agent_object.csv")
data_parts = split_data(meta_info,num_processes)
mp.set_start_method("spawn", force=True)
procs = []
if __name__ == '__main__':
    for proc_idx in range(num_processes):
        physical_gpu_id = proc_idx
        p = mp.Process(
        target=generate_multi_videos,
        args=(physical_gpu_id, proc_idx, data_parts[proc_idx], args.qa_num, args.part_len, video_dynamic),
        daemon=False
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    save_files = glob.glob("qa_pairs_nontemp_*.json")
    merged = {}

    for f in save_files:
        data = json.load(open(f))
        merged.update(data)   # 直接合并字典

    with open("qa_pairs_nontemp.json", 'w') as json_file:
        json.dump(merged, json_file, ensure_ascii=False, indent=4)