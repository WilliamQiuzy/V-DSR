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
from pi3.utils.basic import load_images_as_tensor, write_ply
from pi3.utils.geometry import depth_edge
from pi3.models.pi3 import Pi3
import glob
import torch.nn.functional as F
import math
import argparse

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--part_len', type=int, default=2)
    parser.add_argument('--process_num', type=int, default=10)
    parser.add_argument('--video_root', type=str, default='./video_root')
    parser.add_argument('--qa_num', type=int, default=10)
    args = parser.parse_args()
    
    return args

def split_data(data, num_processes):
    avg_len = len(data)//num_processes
    return [data.iloc[i:i + avg_len] for i in range(0, len(data), avg_len)]

def prob_cal(static):
    total_num = sum(static.values())+len(static)
    prob = {}
    for item in static:
        prob[item] = 1/((static[item]+1)/total_num)
    return prob

def run_pi3(model, proc_id, device):
    imgs = load_images_as_tensor(f"./{proc_id}_frames/", interval=1).to(device) # (N, 3, H, W)
    h,w,c = cv2.imread(f"./{proc_id}_frames/000.jpg").shape
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

def generate_one_video(proc_id, timestamps, agents, objs, qa_num, points, masks, poses, grounding_model, video_predictor, sam2_predictor, val_preprocess, dino, device):
    timestamps = sorted(timestamps)
    timestamps = timestamps[:-2]
    agents_with_cam = ["camera"]
    agents_with_cam.extend(agents)
    objs.extend(agents)
    pcs = points
    pcs = np.transpose(pcs,(0,2,3,1))
    pcs = np.reshape(pcs,(pcs.shape[0],-1,pcs.shape[-1]))
    qa_pairs = []
    qa_count = {}
    for q_type in question_type:
        qa_count[q_type]=0
    for qa_idx in range(qa_num):
       try:
        success = True
        qa_prob = prob_cal(qa_count)
        qa_prob = [qa_prob[item] for item in question_type]
        qa_type = random.choices(list(range(len(qa_count))),weights=qa_prob,k=1)[0]
        qa_type = question_type[qa_type]
        if "abs" in qa_type and "pred" not in qa_type:
            camera_poses = poses
            agent = random.sample(agents_with_cam,1)[0]
            time_1 = random.sample(timestamps,1)[0]
            idx_1 = timestamps.index(time_1)
            time_2, time_3 = random.sample(timestamps,2)
            time_2, time_3 = sorted([time_2, time_3])
            idx_2 = timestamps.index(time_2)
            idx_3 = timestamps.index(time_3)
            while (idx_3-idx_2)<4:
                time_2, time_3 = random.sample(timestamps,2)
                time_2, time_3 = sorted([time_2, time_3])
                idx_2 = timestamps.index(time_2)
                idx_3 = timestamps.index(time_3)
            if agent != "camera":
                agent_label, agent_box, agent_mask = extract_obj_image_mask(grounding_model, sam2_predictor, f"./{proc_id}_frames/{idx_1:03d}.jpg", ". ".join(agents)+".", device)
                if np.sum(np.logical_and(agent_mask,masks[idx_1]))==0:
                    continue
                agent_ori = extract_orient(dino, val_preprocess, f"./{proc_id}_frames/{idx_1:03d}.jpg", agent_mask, device)
                if agent_ori[3]<0.6:
                    continue
            video_frames = [f"./{proc_id}_frames/{idx:03d}.jpg" for idx in range(idx_2, idx_3+1)]
            if "dir" in qa_type:
                coords_record = []
                objects = ['rand','rand']
                for idx in range(2):
                    rand_num = random.uniform(0,1)
                    if rand_num < 0.5:
                        objects[idx] = "camera"
                        break
                obj_labels, obj_boxes, obj_masks = extract_obj_video_mask(grounding_model, video_predictor, video_frames, ". ".join(objs)+".", 2, device)
                if agent == "camera":
                    for idx in range(idx_2,idx_3+1):
                        if objects[0] != "camera":
                            obj1_mask = obj_masks[0][idx-idx_2]
                            if np.sum(np.logical_and(obj1_mask,masks[idx]))==0:
                                success = False
                                break
                            obj1_coord = extract_obj_center(pcs[idx], obj1_mask, masks[idx])
                        else:
                            obj1_coord = np.array([[0,0,0]]).astype(np.float32)
                        if objects[1] != "camera":
                            obj2_mask = obj_masks[1][idx-idx_2]
                            if np.sum(np.logical_and(obj2_mask,masks[idx]))==0:
                                success = False
                                break
                            obj2_coord = extract_obj_center(pcs[idx], obj2_mask, masks[idx])
                        else:
                            obj2_coord = np.array([[0,0,0]]).astype(np.float32)
                        mat_1 = camera_poses[idx]
                        mat_2 = camera_poses[idx_1]
                        coords_trans = camera_mutual_coordinate_transform(np.vstack((obj1_coord, obj2_coord)), mat_1, mat_2)
                        coords_trans = camera_self_coord_trans(coords_trans)
                        coords_record.append(coords_trans[0]-coords_trans[1])
                else:
                    for idx in range(idx_2,idx_3+1):
                        if objects[0] != "camera":
                            obj1_mask = obj_masks[0][idx-idx_2]
                            if np.sum(np.logical_and(obj1_mask,masks[idx]))==0:
                                success = False
                                break
                            obj1_coord = extract_obj_center(pcs[idx], obj1_mask, masks[idx])
                        else:
                            obj1_coord = np.array([[0,0,0]]).astype(np.float32)
                        if objects[1] != "camera":
                            obj2_mask = obj_masks[1][idx-idx_2]
                            if np.sum(np.logical_and(obj2_mask,masks[idx]))==0:
                                success = False
                                break
                            obj2_coord = extract_obj_center(pcs[idx], obj2_mask, masks[idx])
                        else:
                            obj2_coord = np.array([[0,0,0]]).astype(np.float32)
                        mat_1 = camera_poses[idx]
                        mat_2 = camera_poses[idx_1]
                        coords_trans = camera_mutual_coordinate_transform(np.vstack((obj1_coord, obj2_coord)), mat_1, mat_2)
                        azimuth, polar, rotation = agent_ori[0], agent_ori[1], agent_ori[2]
                        if np.sum(np.logical_and(agent_mask,masks[idx_1]))==0:
                            success = False
                            break
                        trans = extract_obj_center(pcs[idx_1],agent_mask, masks[idx_1])
                        coords_trans = camera_agent_coord_trans(coords_trans,polar,azimuth,rotation,trans)
                        coords_record.append(coords_trans[0]-coords_trans[1])
                answer = describe_dir(coords_record)
                len_answer = len(answer.split("."))-1
                agent = "camera" if agent=="camera" else f"{agent_label} with bounding box coordinates [{agent_box[0]},{agent_box[1]},{agent_box[2]},{agent_box[3]}]"
                object_1 = "camera" if objects[0]=="camera" else f"{obj_labels[0]} with initial bounding box coordinates [{obj_boxes[0][0]},{obj_boxes[0][1]},{obj_boxes[0][2]},{obj_boxes[0][3]}]"
                object_2 = "camera" if objects[1]=="camera" else f"{obj_labels[1]} with initial bounding box coordinates [{obj_boxes[1][0]},{obj_boxes[1][1]},{obj_boxes[1][2]},{obj_boxes[1][3]}]"
                idx = idx-1 if success == False else idx
                if len_answer>15 or (idx-idx_2+1)<4:
                    continue
                time_end = timestamps[idx]
                question = template[qa_type].format(time_2,time_end,agent,time_1,object_1,object_2)
                wrong = []
                for i in range(3):
                    rand_num = np.random.uniform(-1,1,size=(idx-idx_2+1,3))
                    wrong_tmp = describe_dir(rand_num).split(".")
                    len_tmp = random.sample(range(1 if (len(wrong_tmp)-1)<=(len_answer-3) else max(1,len_answer-3),min(len_answer+3,len(wrong_tmp)-1)+1),1)[0]
                    wrong_tmp = wrong_tmp[:len_tmp]
                    while (".".join(wrong_tmp)+".") == answer or (".".join(wrong_tmp)+".") in wrong:
                        rand_num = np.random.uniform(-1,1,size=(idx-idx_2+1,3))
                        wrong_tmp = describe_dir(rand_num).split(".")
                        len_tmp = random.sample(range(1 if (len(wrong_tmp)-1)<=(len_answer-3) else max(1,len_answer-3),min(len_answer+3,len(wrong_tmp)-1)+1),1)[0]
                        wrong_tmp = wrong_tmp[:len_tmp]
                    wrong.append(".".join(wrong_tmp)+".")
                options = ["A","B","C","D"]
                rand_num = np.random.uniform(0,1)
                if rand_num<0.25:
                    c_option = "A"
                elif rand_num<0.5 and rand_num>=0.25:
                    c_option = "B"
                elif rand_num<0.75 and rand_num>=0.5:
                    c_option = "C"
                else:
                    c_option = "D"
                options.remove(c_option)
                qa_pairs.append({"Type":qa_type, "Question":question, options[0]:wrong[0], options[1]:wrong[1], options[2]:wrong[2], c_option:answer, "Correct":c_option})
            elif "dis" in qa_type:
                coords_record = []
                objects = ['rand','rand']
                for idx in range(2):
                    rand_num = random.uniform(0,1)
                    if rand_num < 0.5:
                        objects[idx] = "camera"
                        break
                obj_labels, obj_boxes, obj_masks = extract_obj_video_mask(grounding_model, video_predictor, video_frames, ". ".join(objs)+".", 2, device)
                if objects[1] == "camera":
                    for idx in range(idx_2,idx_3+1):
                        obj_mask = obj_masks[0][idx-idx_2]
                        pc_tmp = camera_self_coord_trans(pcs[idx])
                        if np.sum(np.logical_and(obj_mask,masks[idx]))==0:
                            success = False
                            break
                        obj_coord = extract_obj_center(pc_tmp, obj_mask, masks[idx])
                        coords_record.append(cal_dis(obj_coord))
                elif objects[0] == "camera":
                    for idx in range(idx_2,idx_3+1):
                        obj_mask = obj_masks[1][idx-idx_2]
                        pc_tmp = camera_self_coord_trans(pcs[idx])
                        if np.sum(np.logical_and(obj_mask,masks[idx]))==0:
                            success = False
                            break
                        obj_coord = extract_obj_center(pc_tmp, obj_mask, masks[idx])
                        coords_record.append(cal_dis(obj_coord))
                else:
                    for idx in range(idx_2,idx_3+1):
                        obj1_mask = obj_masks[0][idx-idx_2]
                        obj2_mask = obj_masks[1][idx-idx_2]
                        pc_tmp = camera_self_coord_trans(pcs[idx])
                        if np.sum(np.logical_and(obj1_mask,masks[idx]))==0 or np.sum(np.logical_and(obj2_mask,masks[idx]))==0:
                            success = False
                            break
                        obj1_coord = extract_obj_center(pc_tmp, obj1_mask, masks[idx])
                        obj2_coord = extract_obj_center(pc_tmp, obj2_mask, masks[idx])
                        coords_record.append(cal_dis(obj1_coord-obj2_coord))
                coords_record = np.array(coords_record)
                answer = describe_dist_runs(coords_record)
                len_answer = len(answer.split("."))-1
                agent = "camera" if agent=="camera" else f"{agent_label} with bounding box coordinates [{agent_box[0]},{agent_box[1]},{agent_box[2]},{agent_box[3]}]"
                object_1 = "camera" if objects[0]=="camera" else f"{obj_labels[0]} with initial bounding box coordinates [{obj_boxes[0][0]},{obj_boxes[0][1]},{obj_boxes[0][2]},{obj_boxes[0][3]}]"
                object_2 = "camera" if objects[1]=="camera" else f"{obj_labels[1]} with initial bounding box coordinates [{obj_boxes[1][0]},{obj_boxes[1][1]},{obj_boxes[1][2]},{obj_boxes[1][3]}]"
                idx = idx-1 if success == False else idx
                if len_answer>15 or (idx-idx_2+1)<4:
                    continue
                time_end = timestamps[idx]
                question = template[qa_type].format(time_2,time_end,agent,time_1,object_1,object_2)
                wrong = []
                for i in range(3):
                    rand_num = np.random.uniform(0,2,size=(idx-idx_2+1))
                    wrong_tmp = describe_dist_runs(rand_num).split(".")
                    len_tmp = random.sample(range(1 if (len(wrong_tmp)-1)<=(len_answer-3) else max(1,len_answer-3),min(len_answer+3,len(wrong_tmp)-1)+1),1)[0]
                    wrong_tmp = wrong_tmp[:len_tmp]
                    while (".".join(wrong_tmp)+".") == answer or (".".join(wrong_tmp)+".") in wrong:
                        rand_num = np.random.uniform(0,2,size=(idx-idx_2+1))
                        wrong_tmp = describe_dist_runs(rand_num).split(".")
                        len_tmp = random.sample(range(1 if (len(wrong_tmp)-1)<=(len_answer-3) else max(1,len_answer-3),min(len_answer+3,len(wrong_tmp)-1)+1),1)[0]
                        wrong_tmp = wrong_tmp[:len_tmp]
                    wrong.append(".".join(wrong_tmp)+".")
                options = ["A","B","C","D"]
                rand_num = np.random.uniform(0,1)
                if rand_num<0.25:
                    c_option = "A"
                elif rand_num<0.5 and rand_num>=0.25:
                    c_option = "B"
                elif rand_num<0.75 and rand_num>=0.5:
                    c_option = "C"
                else:
                    c_option = "D"
                options.remove(c_option)
                qa_pairs.append({"Type":qa_type, "Question":question, options[0]:wrong[0], options[1]:wrong[1], options[2]:wrong[2], c_option:answer, "Correct":c_option})
            elif "ori" in qa_type:
                coords_record = []
                objects = 'rand'
                rand_num = random.uniform(0,1)
                if rand_num < 0.5:
                    objects = "camera"
                obj_labels, obj_boxes, obj_masks = extract_obj_video_mask(grounding_model, video_predictor, video_frames, ". ".join(agents)+".", 1, device)
                if agent == "camera":
                    for idx in range(idx_2,idx_3+1):
                        if objects != "camera":
                            if np.sum(np.logical_and(obj_masks[0][idx-idx_2],masks[idx]))==0:
                                success = False
                                break
                            pose_obj = extract_orient(dino, val_preprocess, video_frames[idx-idx_2], obj_masks[0][idx-idx_2], device)
                            if pose_obj[3]<0.6:
                                success = False
                                break
                            obj_coord = extract_obj_center(pcs[idx], obj_masks[0][idx-idx_2], masks[idx])
                            azimuth, polar, rotation = pose_obj[0], pose_obj[1], pose_obj[2]
                            vector = np.array([[0,0,0],[1,0,0]]).astype(np.float32)
                            vector_ori = agent_camera_coord_trans(vector,polar,azimuth,rotation,obj_coord)
                            vector_ori = camera_mutual_coordinate_transform(vector_ori, camera_poses[idx], camera_poses[idx_1])
                            vector_ori = camera_self_coord_trans(vector_ori)
                            coords_record.append(vector_ori[1]-vector_ori[0])
                        else:
                            vector_ori = np.array([[0,0,0],[0,0,1]]).astype(np.float32)
                            vector_ori = camera_mutual_coordinate_transform(vector_ori, camera_poses[idx], camera_poses[idx_1])
                            vector_ori = camera_self_coord_trans(vector_ori)
                            coords_record.append(vector_ori[1]-vector_ori[0])
                else:
                    for idx in range(idx_2,idx_3+1):
                        if objects != "camera":
                            if np.sum(np.logical_and(obj_masks[0][idx-idx_2],masks[idx]))==0:
                                success = False
                                break
                            pose_obj = extract_orient(dino, val_preprocess, video_frames[idx-idx_2], obj_masks[0][idx-idx_2], device)
                            if pose_obj[3]<0.6:
                                success = False
                                break
                            obj_coord = extract_obj_center(pcs[idx], obj_masks[0][idx-idx_2], masks[idx])
                            azimuth, polar, rotation = pose_obj[0], pose_obj[1], pose_obj[2]
                            vector = np.array([[0,0,0],[1,0,0]]).astype(np.float32)
                            vector_ori = agent_camera_coord_trans(vector,polar,azimuth,rotation,obj_coord)
                            vector_ori = camera_mutual_coordinate_transform(vector_ori, camera_poses[idx], camera_poses[idx_1])
                            if np.sum(np.logical_and(agent_mask,masks[idx_1]))==0:
                                success = False
                                break
                            trans = extract_obj_center(pcs[idx_1],agent_mask, masks[idx_1])
                            azimuth, polar, rotation = agent_ori[0], agent_ori[1], agent_ori[2]
                            vector_ori = camera_agent_coord_trans(vector_ori, polar, azimuth, rotation, trans)
                            coords_record.append(vector_ori[1]-vector_ori[0])
                        else:
                            vector_ori = np.array([[0,0,0],[0,0,1]]).astype(np.float32)
                            vector_ori = camera_mutual_coordinate_transform(vector_ori, camera_poses[idx], camera_poses[idx_1])
                            if np.sum(np.logical_and(agent_mask,masks[idx_1]))==0:
                                success = False
                                break
                            trans = extract_obj_center(pcs[idx_1],agent_mask, masks[idx_1])
                            azimuth, polar, rotation = agent_ori[0], agent_ori[1], agent_ori[2]
                            vector_ori = camera_agent_coord_trans(vector_ori, polar, azimuth, rotation, trans)
                            coords_record.append(vector_ori[1]-vector_ori[0])
                answer = describe_dir(coords_record)
                len_answer = len(answer.split("."))-1
                agent = "camera" if agent=="camera" else f"{agent_label} with bounding box coordinates [{agent_box[0]},{agent_box[1]},{agent_box[2]},{agent_box[3]}]"
                objects = "camera" if objects=="camera" else f"{obj_labels[0]} with initial bounding box coordinates [{obj_boxes[0][0]},{obj_boxes[0][1]},{obj_boxes[0][2]},{obj_boxes[0][3]}]"
                idx = idx-1 if success==False else idx
                if len_answer>15 or (idx+1-idx_2)<4:
                    continue
                time_end = timestamps[idx]
                question = template[qa_type].format(time_2,time_end,agent,time_1,objects)
                wrong = []
                for i in range(3):
                    rand_num = np.random.uniform(-1,1,size=(idx+1-idx_2,3))
                    wrong_tmp = describe_dir(rand_num).split(".")
                    len_tmp = random.sample(range(1 if (len(wrong_tmp)-1)<=(len_answer-3) else max(1,len_answer-3),min(len_answer+3,len(wrong_tmp)-1)+1),1)[0]
                    wrong_tmp = wrong_tmp[:len_tmp]
                    while (".".join(wrong_tmp)+".") == answer or (".".join(wrong_tmp)+".") in wrong:
                        rand_num = np.random.uniform(-1,1,size=(idx+1-idx_2,3))
                        wrong_tmp = describe_dir(rand_num).split(".")
                        len_tmp = random.sample(range(1 if (len(wrong_tmp)-1)<=(len_answer-3) else max(1,len_answer-3),min(len_answer+3,len(wrong_tmp)-1)+1),1)[0]
                        wrong_tmp = wrong_tmp[:len_tmp]
                    wrong.append(".".join(wrong_tmp)+".")
                options = ["A","B","C","D"]
                rand_num = np.random.uniform(0,1)
                if rand_num<0.25:
                    c_option = "A"
                elif rand_num<0.5 and rand_num>=0.25:
                    c_option = "B"
                elif rand_num<0.75 and rand_num>=0.5:
                    c_option = "C"
                else:
                    c_option = "D"
                options.remove(c_option)
                qa_pairs.append({"Type":qa_type, "Question":question, options[0]:wrong[0], options[1]:wrong[1], options[2]:wrong[2], c_option:answer, "Correct":c_option})
            elif "spd" in qa_type and "comp" in qa_type:
                coords_record_1 = []
                coords_record_2 = []
                objects = ['rand','rand']
                for idx in range(2):
                    rand_num = random.uniform(0,1)
                    if rand_num < 0.5:
                        objects[idx] = "camera"
                        break
                obj_labels, obj_boxes, obj_masks = extract_obj_video_mask(grounding_model, video_predictor, video_frames, ". ".join(objs)+".", 2, device)
                if agent == "camera":
                    for idx in range(idx_2,idx_3+1):
                        if objects[0] == "camera":
                            obj1_coord = np.array([[0,0,0]]).astype(np.float32)
                        else:
                            if np.sum(np.logical_and(obj_masks[0][idx-idx_2],masks[idx]))==0:
                                success = False
                                break
                            obj1_coord = np.array([extract_obj_center(pcs[idx], obj_masks[0][idx-idx_2], masks[idx])])
                        if objects[1] == "camera":
                            obj2_coord = np.array([[0,0,0]]).astype(np.float32)
                        else:
                            if np.sum(np.logical_and(obj_masks[1][idx-idx_2],masks[idx]))==0:
                                success = False
                                break
                            obj2_coord = np.array([extract_obj_center(pcs[idx], obj_masks[1][idx-idx_2], masks[idx])])
                        coord_trans = camera_mutual_coordinate_transform(np.vstack((obj1_coord,obj2_coord)), camera_poses[idx], camera_poses[idx_1])
                        coord_trans = camera_self_coord_trans(coord_trans)
                        coords_record_1.append(coord_trans[0])
                        coords_record_2.append(coord_trans[1])
                else:
                    for idx in range(idx_2,idx_3+1):
                        if objects[0] == "camera":
                            obj1_coord = np.array([[0,0,0]]).astype(np.float32)
                        else:
                            if np.sum(np.logical_and(obj_masks[0][idx-idx_2],masks[idx]))==0:
                                success = False
                                break
                            obj1_coord = np.array([extract_obj_center(pcs[idx], obj_masks[0][idx-idx_2], masks[idx])])
                        if objects[1] == "camera":
                            obj2_coord = np.array([[0,0,0]]).astype(np.float32)
                        else:
                            if np.sum(np.logical_and(obj_masks[1][idx-idx_2],masks[idx]))==0:
                                success = False
                                break
                            obj2_coord = np.array([extract_obj_center(pcs[idx], obj_masks[1][idx-idx_2], masks[idx])])
                        coord_trans = camera_mutual_coordinate_transform(np.vstack((obj1_coord,obj2_coord)), camera_poses[idx], camera_poses[idx_1])
                        azimuth, polar, rotation = agent_ori[0], agent_ori[1], agent_ori[2]
                        if np.sum(np.logical_and(agent_mask,masks[idx_1]))==0:
                            success = False
                            break
                        trans = extract_obj_center(pcs[idx_1],agent_mask,masks[idx_1])
                        coord_trans = camera_agent_coord_trans(coord_trans, polar, azimuth, rotation, trans)
                        coords_record_1.append(coord_trans[0])
                        coords_record_2.append(coord_trans[1])
                coords_record_1 = np.array(coords_record_1)
                coords_record_2 = np.array(coords_record_2)
                coords_dis_1 = coords_record_1[1:]-coords_record_1[:-1]
                coords_dis_2 = coords_record_2[1:]-coords_record_2[:-1]
                coords_dis_1 = np.array([cal_dis(item) for item in coords_dis_1])
                coords_dis_2 = np.array([cal_dis(item) for item in coords_dis_2])
                answer = describe_spd_comp(coords_dis_1,coords_dis_2)
                len_answer = len(answer.split("."))-1
                agent = "camera" if agent=="camera" else f"{agent_label} with bounding box coordinates [{agent_box[0]},{agent_box[1]},{agent_box[2]},{agent_box[3]}]"
                object_1 = "camera" if objects[0]=="camera" else f"{obj_labels[0]} with initial bounding box coordinates [{obj_boxes[0][0]},{obj_boxes[0][1]},{obj_boxes[0][2]},{obj_boxes[0][3]}]"
                object_2 = "camera" if objects[1]=="camera" else f"{obj_labels[1]} with initial bounding box coordinates [{obj_boxes[1][0]},{obj_boxes[1][1]},{obj_boxes[1][2]},{obj_boxes[1][3]}]"
                idx = idx-1 if success==False else idx
                if len_answer>15 or (idx-idx_2)<4:
                    continue
                time_end = timestamps[idx]
                question = template[qa_type].format(time_2,time_end,agent,time_1,object_1,object_2)
                wrong = []
                for i in range(3):
                    rand_num_1 = np.random.uniform(0,2,size=(idx-idx_2))
                    rand_num_2 = np.random.uniform(0,2,size=(idx-idx_2))
                    wrong_tmp = describe_spd_comp(rand_num_1,rand_num_2).split(".")
                    len_tmp = random.sample(range(1 if (len(wrong_tmp)-1)<=(len_answer-3) else max(1,len_answer-3),min(len_answer+3,len(wrong_tmp)-1)+1),1)[0]
                    wrong_tmp = wrong_tmp[:len_tmp]
                    while (".".join(wrong_tmp)+".") == answer or (".".join(wrong_tmp)+".") in wrong:
                        rand_num_1 = np.random.uniform(0,2,size=(idx-idx_2))
                        rand_num_2 = np.random.uniform(0,2,size=(idx-idx_2))
                        wrong_tmp = describe_spd_comp(rand_num_1,rand_num_2).split(".")
                        len_tmp = random.sample(range(1 if (len(wrong_tmp)-1)<=(len_answer-3) else max(1,len_answer-3),min(len_answer+3,len(wrong_tmp)-1)+1),1)[0]
                        wrong_tmp = wrong_tmp[:len_tmp]
                    wrong.append(".".join(wrong_tmp)+".")
                options = ["A","B","C","D"]
                rand_num = np.random.uniform(0,1)
                if rand_num<0.25:
                    c_option = "A"
                elif rand_num<0.5 and rand_num>=0.25:
                    c_option = "B"
                elif rand_num<0.75 and rand_num>=0.5:
                    c_option = "C"
                else:
                    c_option = "D"
                options.remove(c_option)
                qa_pairs.append({"Type":qa_type, "Question":question, options[0]:wrong[0], options[1]:wrong[1], options[2]:wrong[2], c_option:answer, "Correct":c_option})
            elif "spd" in qa_type:
                coords_record_1 = []
                objects = 'rand'
                rand_num = random.uniform(0,1)
                if rand_num < 0.5:
                    objects = "camera"
                obj_labels, obj_boxes, obj_masks = extract_obj_video_mask(grounding_model, video_predictor, video_frames, ". ".join(agents)+".", 1, device)
                if agent == "camera":
                    for idx in range(idx_2,idx_3+1):
                        if objects == "camera":
                            obj1_coord = np.array([[0,0,0]]).astype(np.float32)
                        else:
                            if np.sum(np.logical_and(obj_masks[0][idx-idx_2],masks[idx]))==0:
                                success = False
                                break
                            obj1_coord = np.array([extract_obj_center(pcs[idx], obj_masks[0][idx-idx_2], masks[idx])])
                        obj1_coord = camera_mutual_coordinate_transform(obj1_coord, camera_poses[idx], camera_poses[idx_1])
                        obj1_coord = camera_self_coord_trans(obj1_coord)
                        coords_record_1.append(obj1_coord[0])
                else:
                    for idx in range(idx_2,idx_3+1):
                        if objects == "camera":
                            obj1_coord = np.array([[0,0,0]]).astype(np.float32)
                        else:
                            if np.sum(np.logical_and(obj_masks[0][idx-idx_2],masks[idx]))==0:
                                success = False
                                break
                            obj1_coord = np.array([extract_obj_center(pcs[idx], obj_masks[0][idx-idx_2], masks[idx])])
                        obj1_coord = camera_mutual_coordinate_transform(obj1_coord, camera_poses[idx], camera_poses[idx_1])
                        azimuth, polar, rotation = agent_ori[0], agent_ori[1], agent_ori[2]
                        if np.sum(np.logical_and(agent_mask,masks[idx_1]))==0:
                            success = False
                            break
                        trans = extract_obj_center(pcs[idx_1], agent_mask, masks[idx_1])
                        obj1_coord = camera_agent_coord_trans(obj1_coord, polar, azimuth, rotation, trans)
                        coords_record_1.append(obj1_coord[0])
                coords_record_1 = np.array(coords_record_1)
                coords_dis_1 = coords_record_1[1:]-coords_record_1[:-1]
                coords_dis_1 = [cal_dis(item) for item in coords_dis_1]
                coords_dis_1 = np.array(coords_dis_1)
                answer = describe_dist_runs(coords_dis_1)
                len_answer = len(answer.split("."))-1
                agent = "camera" if agent=="camera" else f"{agent_label} with bounding box coordinates [{agent_box[0]},{agent_box[1]},{agent_box[2]},{agent_box[3]}]"
                objects = "camera" if objects=="camera" else f"{obj_labels[0]} with initial bounding box coordinates [{obj_boxes[0][0]},{obj_boxes[0][1]},{obj_boxes[0][2]},{obj_boxes[0][3]}]"
                idx = idx-1 if success == False else idx
                if len_answer>15 or (idx-idx_2)<4:
                    continue
                time_end = timestamps[idx]
                question = template[qa_type].format(time_2,time_end,agent,time_1,objects)
                wrong = []
                for i in range(3):
                    rand_num = np.random.uniform(0,2,size=(idx-idx_2))
                    wrong_tmp = describe_dist_runs(rand_num).split(".")
                    len_tmp = random.sample(range(1 if (len(wrong_tmp)-1)<=(len_answer-3) else max(1,len_answer-3),min(len_answer+3,len(wrong_tmp)-1)+1),1)[0]
                    wrong_tmp = wrong_tmp[:len_tmp]
                    while (".".join(wrong_tmp)+".") == answer or (".".join(wrong_tmp)+".") in wrong:
                        rand_num = np.random.uniform(0,2,size=(idx-idx_2))
                        wrong_tmp = describe_dist_runs(rand_num).split(".")
                        len_tmp = random.sample(range(1 if (len(wrong_tmp)-1)<=(len_answer-3) else max(1,len_answer-3),min(len_answer+3,len(wrong_tmp)-1)+1),1)[0]
                        wrong_tmp = wrong_tmp[:len_tmp]
                    wrong.append(".".join(wrong_tmp)+".")
                options = ["A","B","C","D"]
                rand_num = np.random.uniform(0,1)
                if rand_num<0.25:
                    c_option = "A"
                elif rand_num<0.5 and rand_num>=0.25:
                    c_option = "B"
                elif rand_num<0.75 and rand_num>=0.5:
                    c_option = "C"
                else:
                    c_option = "D"
                options.remove(c_option)
                qa_pairs.append({"Type":qa_type, "Question":question, options[0]:wrong[0], options[1]:wrong[1], options[2]:wrong[2], c_option:answer, "Correct":c_option})
        elif "rel" in qa_type and "pred" not in qa_type:
            time_1, time_2 = random.sample(timestamps,2)
            time_1, time_2 = sorted([time_1, time_2])
            idx_1 = timestamps.index(time_1)
            idx_2 = timestamps.index(time_2)
            while (idx_2-idx_1)<4:
                time_1, time_2 = random.sample(timestamps,2)
                time_1, time_2 = sorted([time_1, time_2])
                idx_1 = timestamps.index(time_1)
                idx_2 = timestamps.index(time_2)
            video_frames = [f"./{proc_id}_frames/{idx:03d}.jpg" for idx in range(idx_1, idx_2+1)]
            if "dir" in qa_type:
                coords_record = []
                agent = random.sample(agents_with_cam,1)[0]
                if agent != "camera":
                    agent_label, agent_box, agent_mask = extract_obj_video_mask(grounding_model, video_predictor, video_frames, ". ".join(agents)+".", 1, device)
                objects = ['rand','rand']
                for idx in range(2):
                    rand_num = random.uniform(0,1)
                    if rand_num < 0.5:
                        objects[idx] = "camera"
                        break
                obj_labels, obj_boxes, obj_masks = extract_obj_video_mask(grounding_model, video_predictor, video_frames, ". ".join(objs)+".", 2, device)
                if agent == "camera":
                    for idx in range(idx_1,idx_2+1):
                        pc_tmp = camera_self_coord_trans(pcs[idx])
                        if objects[0] != "camera":
                            if np.sum(np.logical_and(obj_masks[0][idx-idx_1],masks[idx]))==0:
                                success = False
                                break
                            obj1_coord = extract_obj_center(pc_tmp, obj_masks[0][idx-idx_1], masks[idx])
                        else:
                            obj1_coord = np.array([0,0,0])
                        if objects[1] != "camera":
                            if np.sum(np.logical_and(obj_masks[1][idx-idx_1],masks[idx]))==0:
                                success = False
                                break
                            obj2_coord = extract_obj_center(pc_tmp, obj_masks[1][idx-idx_1], masks[idx])
                        else:
                            obj2_coord = np.array([0,0,0])
                        coords_record.append(obj1_coord-obj2_coord)
                else:
                    for idx in range(idx_1,idx_2+1):
                        if np.sum(np.logical_and(agent_mask[0][idx-idx_1],masks[idx]))==0:
                            success = False
                            break
                        pose_agent = extract_orient(dino, val_preprocess, f"./{proc_id}_frames/{idx:03d}.jpg", agent_mask[0][idx-idx_1], device)
                        if pose_agent[3]<0.6:
                            success = False
                            break
                        azimuth, polar, rotation = pose_agent[0], pose_agent[1], pose_agent[2]
                        trans = extract_obj_center(pcs[idx], agent_mask[0][idx-idx_1], masks[idx])
                        if objects[0] != "camera":
                            if np.sum(np.logical_and(obj_masks[0][idx-idx_1],masks[idx]))==0:
                                success = False
                                break
                            obj1_coord = extract_obj_center(pcs[idx], obj_masks[0][idx-idx_1], masks[idx])
                        else:
                            obj1_coord = np.array([[0,0,0]]).astype(np.float32)
                        if objects[1] != "camera":
                            if np.sum(np.logical_and(obj_masks[1][idx-idx_1],masks[idx]))==0:
                                success = False
                                break
                            obj2_coord = extract_obj_center(pcs[idx], obj_masks[1][idx-idx_1], masks[idx])
                        else:
                            obj2_coord = np.array([[0,0,0]]).astype(np.float32)
                        coord_trans = camera_agent_coord_trans(np.vstack((obj1_coord,obj2_coord)), polar, azimuth, rotation, trans)
                        coords_record.append(coord_trans[0]-coord_trans[1])
                answer = describe_dir(coords_record)
                len_answer = len(answer.split("."))-1
                agent = "camera" if agent=="camera" else f"{agent_label[0]} with initial bounding box coordinates [{agent_box[0][0]},{agent_box[0][1]},{agent_box[0][2]},{agent_box[0][3]}]"
                object_1 = "camera" if objects[0]=="camera" else f"{obj_labels[0]} with initial bounding box coordinates [{obj_boxes[0][0]},{obj_boxes[0][1]},{obj_boxes[0][2]},{obj_boxes[0][3]}]"
                object_2 = "camera" if objects[1]=="camera" else f"{obj_labels[1]} with initial bounding box coordinates [{obj_boxes[1][0]},{obj_boxes[1][1]},{obj_boxes[1][2]},{obj_boxes[1][3]}]"
                idx = idx-1 if success==False else idx
                if len_answer>15 or (idx-idx_1+1)<4:
                    continue
                time_end = timestamps[idx]
                question = template[qa_type].format(time_1,time_end,agent,object_1,object_2)
                wrong = []
                for i in range(3):
                    rand_num = np.random.uniform(-1,1,size=(idx-idx_1+1,3))
                    wrong_tmp = describe_dir(rand_num).split(".")
                    len_tmp = random.sample(range(1 if (len(wrong_tmp)-1)<=(len_answer-3) else max(1,len_answer-3),min(len_answer+3,len(wrong_tmp)-1)+1),1)[0]
                    wrong_tmp = wrong_tmp[:len_tmp]
                    while (".".join(wrong_tmp)+".") == answer or (".".join(wrong_tmp)+".") in wrong:
                        rand_num = np.random.uniform(-1,1,size=(idx-idx_1+1,3))
                        wrong_tmp = describe_dir(rand_num).split(".")
                        len_tmp = random.sample(range(1 if (len(wrong_tmp)-1)<=(len_answer-3) else max(1,len_answer-3),min(len_answer+3,len(wrong_tmp)-1)+1),1)[0]
                        wrong_tmp = wrong_tmp[:len_tmp]
                    wrong.append(".".join(wrong_tmp)+".")
                options = ["A","B","C","D"]
                rand_num = np.random.uniform(0,1)
                if rand_num<0.25:
                    c_option = "A"
                elif rand_num<0.5 and rand_num>=0.25:
                    c_option = "B"
                elif rand_num<0.75 and rand_num>=0.5:
                    c_option = "C"
                else:
                    c_option = "D"
                options.remove(c_option)
                qa_pairs.append({"Type":qa_type, "Question":question, options[0]:wrong[0], options[1]:wrong[1], options[2]:wrong[2], c_option:answer, "Correct":c_option})
            elif "dis" in qa_type:
                coords_record = []
                agent = random.sample(agents_with_cam,1)[0]
                if agent != "camera":
                    agent_label, agent_box, agent_mask = extract_obj_video_mask(grounding_model, video_predictor, video_frames, ". ".join(agents)+".", 1, device)
                objects = ['rand','rand']
                for idx in range(2):
                    rand_num = random.uniform(0,1)
                    if rand_num < 0.5:
                        objects[idx] = "camera"
                        break
                obj_labels, obj_boxes, obj_masks = extract_obj_video_mask(grounding_model, video_predictor, video_frames, ". ".join(objs)+".", 2, device)
                if objects[1] == "camera":
                    for idx in range(idx_1,idx_2+1):
                        pc_tmp = camera_self_coord_trans(pcs[idx])
                        if np.sum(np.logical_and(obj_masks[0][idx-idx_1],masks[idx]))==0:
                            success = False
                            break
                        obj_coord = extract_obj_center(pc_tmp, obj_masks[0][idx-idx_1], masks[idx])
                        coords_record.append(cal_dis(obj_coord))
                elif objects[0] == "camera":
                    for idx in range(idx_1,idx_2+1):
                        pc_tmp = camera_self_coord_trans(pcs[idx])
                        if np.sum(np.logical_and(obj_masks[1][idx-idx_1],masks[idx]))==0:
                            success = False
                            break
                        obj_coord = extract_obj_center(pc_tmp, obj_masks[1][idx-idx_1], masks[idx])
                        coords_record.append(cal_dis(obj_coord))
                else:
                    for idx in range(idx_1,idx_2+1):
                        pc_tmp = camera_self_coord_trans(pcs[idx])
                        if np.sum(np.logical_and(obj_masks[0][idx-idx_1],masks[idx]))==0 or np.sum(np.logical_and(obj_masks[1][idx-idx_1],masks[idx]))==0:
                            success = False
                            break
                        obj1_coord = extract_obj_center(pc_tmp, obj_masks[0][idx-idx_1], masks[idx])
                        obj2_coord = extract_obj_center(pc_tmp, obj_masks[1][idx-idx_1], masks[idx])
                        coords_record.append(cal_dis(obj1_coord-obj2_coord))
                answer = describe_dist_runs(coords_record)
                len_answer = len(answer.split("."))-1
                agent = "camera" if agent=="camera" else f"{agent_label[0]} with initial bounding box coordinates [{agent_box[0][0]},{agent_box[0][1]},{agent_box[0][2]},{agent_box[0][3]}]"
                object_1 = "camera" if objects[0]=="camera" else f"{obj_labels[0]} with initial bounding box coordinates [{obj_boxes[0][0]},{obj_boxes[0][1]},{obj_boxes[0][2]},{obj_boxes[0][3]}]"
                object_2 = "camera" if objects[1]=="camera" else f"{obj_labels[1]} with initial bounding box coordinates [{obj_boxes[1][0]},{obj_boxes[1][1]},{obj_boxes[1][2]},{obj_boxes[1][3]}]"
                idx = idx-1 if success==False else idx
                if len_answer>15 or (idx-idx_1+1)<4:
                    continue
                time_end = timestamps[idx]
                question = template[qa_type].format(time_1,time_end,agent,object_1,object_2)
                wrong = []
                for i in range(3):
                    rand_num = np.random.uniform(0,2,size=(idx-idx_1+1))
                    wrong_tmp = describe_dist_runs(rand_num).split(".")
                    len_tmp = random.sample(range(1 if (len(wrong_tmp)-1)<=(len_answer-3) else max(1,len_answer-3),min(len_answer+3,len(wrong_tmp)-1)+1),1)[0]
                    wrong_tmp = wrong_tmp[:len_tmp]
                    while (".".join(wrong_tmp)+".") == answer or (".".join(wrong_tmp)+".") in wrong:
                        rand_num = np.random.uniform(0,2,size=(idx-idx_1+1))
                        wrong_tmp = describe_dist_runs(rand_num).split(".")
                        len_tmp = random.sample(range(1 if (len(wrong_tmp)-1)<=(len_answer-3) else max(1,len_answer-3),min(len_answer+3,len(wrong_tmp)-1)+1),1)[0]
                        wrong_tmp = wrong_tmp[:len_tmp]
                    wrong.append(".".join(wrong_tmp)+".")
                options = ["A","B","C","D"]
                rand_num = np.random.uniform(0,1)
                if rand_num<0.25:
                    c_option = "A"
                elif rand_num<0.5 and rand_num>=0.25:
                    c_option = "B"
                elif rand_num<0.75 and rand_num>=0.5:
                    c_option = "C"
                else:
                    c_option = "D"
                options.remove(c_option)
                qa_pairs.append({"Type":qa_type, "Question":question, options[0]:wrong[0], options[1]:wrong[1], options[2]:wrong[2], c_option:answer, "Correct":c_option})
            elif "ori" in qa_type:
                coords_record = []
                objects = ['rand','rand']
                for idx in range(2):
                    rand_num = random.uniform(0,1)
                    if rand_num < 0.5:
                        objects[idx] = "camera"
                        break
                obj_labels, obj_boxes, obj_masks = extract_obj_video_mask(grounding_model, video_predictor, video_frames, ". ".join(agents)+".", 2, device)
                agent = objects[0]
                if agent == "camera":
                    for idx in range(idx_1,idx_2+1):
                        if np.sum(np.logical_and(obj_masks[1][idx-idx_1],masks[idx]))==0:
                            success = False
                            break
                        pose_obj = extract_orient(dino, val_preprocess, f"./{proc_id}_frames/{idx:03d}.jpg", obj_masks[1][idx-idx_1], device)
                        if pose_obj[3]<0.6:
                            success = False
                            break
                        trans = extract_obj_center(pcs[idx],obj_masks[1][idx-idx_1],masks[idx])
                        azimuth, polar, rotation = pose_obj[0], pose_obj[1], pose_obj[2]
                        vector = np.array([[0,0,0],[1,0,0]]).astype(np.float32)
                        vector_ori = agent_camera_coord_trans(vector,polar,azimuth,rotation,trans)
                        vector_ori = camera_self_coord_trans(vector_ori)
                        coords_record.append(vector_ori[1]-vector_ori[0])
                else:
                    for idx in range(idx_1,idx_2+1):
                        if objects[1] != "camera":
                            if np.sum(np.logical_and(obj_masks[1][idx-idx_1],masks[idx]))==0:
                                success = False
                                break
                            pose_obj = extract_orient(dino, val_preprocess, f"./{proc_id}_frames/{idx:03d}.jpg", obj_masks[1][idx-idx_1], device)
                            if pose_obj[3]<0.6:
                                success = False
                                break
                            trans = extract_obj_center(pcs[idx],obj_masks[1][idx-idx_1],masks[idx])
                            azimuth, polar, rotation = pose_obj[0], pose_obj[1], pose_obj[2]
                            vector = np.array([[0,0,0],[1,0,0]]).astype(np.float32)
                            vector_ori = agent_camera_coord_trans(vector,polar,azimuth,rotation,trans)
                        else:
                            vector_ori = np.array([[0,0,0],[0,0,1]]).astype(np.float32)
                        if np.sum(np.logical_and(obj_masks[0][idx-idx_1],masks[idx]))==0:
                            success = False
                            break
                        pose_agent = extract_orient(dino, val_preprocess, f"./{proc_id}_frames/{idx:03d}.jpg", obj_masks[0][idx-idx_1], device)
                        if pose_agent[3]<0.6:
                            success = False
                            break
                        trans = extract_obj_center(pcs[idx],obj_masks[0][idx-idx_1],masks[idx])
                        azimuth, polar, rotation = pose_agent[0], pose_agent[1], pose_agent[2]
                        vector_ori = camera_agent_coord_trans(vector_ori, polar, azimuth, rotation, trans)
                        coords_record.append(vector_ori[1]-vector_ori[0])
                answer = describe_dir(coords_record)
                len_answer = len(answer.split("."))-1
                agent = "camera" if objects[0]=="camera" else f"{obj_labels[0]} with initial bounding box coordinates [{obj_boxes[0][0]},{obj_boxes[0][1]},{obj_boxes[0][2]},{obj_boxes[0][3]}]"
                object_1 = "camera" if objects[1]=="camera" else f"{obj_labels[1]} with initial bounding box coordinates [{obj_boxes[1][0]},{obj_boxes[1][1]},{obj_boxes[1][2]},{obj_boxes[1][3]}]"
                idx = idx-1 if success==False else idx
                if len_answer>15 or (idx+1-idx_1)<4:
                    continue
                time_end = timestamps[idx]
                question = template[qa_type].format(time_1,time_end,agent,object_1)
                wrong = []
                for i in range(3):
                    rand_num = np.random.uniform(-1,1,size=(idx+1-idx_1,3))
                    wrong_tmp = describe_dir(rand_num).split(".")
                    len_tmp = random.sample(range(1 if (len(wrong_tmp)-1)<=(len_answer-3) else max(1,len_answer-3),min(len_answer+3,len(wrong_tmp)-1)+1),1)[0]
                    wrong_tmp = wrong_tmp[:len_tmp]
                    while (".".join(wrong_tmp)+".") == answer or (".".join(wrong_tmp)+".") in wrong:
                        rand_num = np.random.uniform(-1,1,size=(idx+1-idx_1,3))
                        wrong_tmp = describe_dir(rand_num).split(".")
                        len_tmp = random.sample(range(1 if (len(wrong_tmp)-1)<=(len_answer-3) else max(1,len_answer-3),min(len_answer+3,len(wrong_tmp)-1)+1),1)[0]
                        wrong_tmp = wrong_tmp[:len_tmp]
                    wrong.append(".".join(wrong_tmp)+".")
                options = ["A","B","C","D"]
                rand_num = np.random.uniform(0,1)
                if rand_num<0.25:
                    c_option = "A"
                elif rand_num<0.5 and rand_num>=0.25:
                    c_option = "B"
                elif rand_num<0.75 and rand_num>=0.5:
                    c_option = "C"
                else:
                    c_option = "D"
                options.remove(c_option)
                qa_pairs.append({"Type":qa_type, "Question":question, options[0]:wrong[0], options[1]:wrong[1], options[2]:wrong[2], c_option:answer, "Correct":c_option})
            elif "spd" in qa_type and "comp" in qa_type:
                coords_record_1 = []
                coords_record_2 = []
                objects = ['rand','rand','rand']
                for idx in range(3):
                    rand_num = random.uniform(0,1)
                    if rand_num < 0.5:
                        objects[idx] = "camera"
                        break
                obj_labels, obj_boxes, obj_masks = extract_obj_video_mask(grounding_model, video_predictor, video_frames, ". ".join(objs)+".", 3, device)
                if obj_labels[0] not in agents_with_cam:
                    continue
                agent = objects[0]
                if agent == "camera":
                    for idx in range(idx_1,idx_2+1):
                        pc_tmp = camera_self_coord_trans(pcs[idx])
                        if np.sum(np.logical_and(obj_masks[1][idx-idx_1],masks[idx]))==0 or np.sum(np.logical_and(obj_masks[2][idx-idx_1],masks[idx]))==0:
                            success = False
                            break
                        obj1_coord = extract_obj_center(pc_tmp, obj_masks[1][idx-idx_1], masks[idx])
                        obj2_coord = extract_obj_center(pc_tmp, obj_masks[2][idx-idx_1], masks[idx])
                        coords_record_1.append(obj1_coord)
                        coords_record_2.append(obj2_coord)
                else:
                    for idx in range(idx_1,idx_2+1):
                        if objects[1] != "camera":
                            if np.sum(np.logical_and(obj_masks[1][idx-idx_1],masks[idx]))==0:
                                success = False
                                break
                            obj1_coord = extract_obj_center(pcs[idx], obj_masks[1][idx-idx_1], masks[idx])
                        else:
                            obj1_coord = np.array([[0,0,0]]).astype(np.float32)
                        if objects[2] != "camera":
                            if np.sum(np.logical_and(obj_masks[2][idx-idx_1],masks[idx]))==0:
                                success = False
                                break
                            obj2_coord = extract_obj_center(pcs[idx], obj_masks[2][idx-idx_1], masks[idx])
                        else:
                            obj2_coord = np.array([[0,0,0]]).astype(np.float32)
                        if np.sum(np.logical_and(obj_masks[0][idx-idx_1],masks[idx]))==0:
                            success = False
                            break
                        pose_agent = extract_orient(dino, val_preprocess, f"./{proc_id}_frames/{idx:03d}.jpg", obj_masks[0][idx-idx_1], device)
                        if pose_agent[3]<0.6:
                            success = False
                            break
                        azimuth, polar, rotation = pose_agent[0], pose_agent[1], pose_agent[2]
                        trans = extract_obj_center(pcs[idx], obj_masks[0][idx-idx_1], masks[idx])
                        coord_trans = camera_agent_coord_trans(np.vstack((obj1_coord,obj2_coord)), polar, azimuth, rotation, trans)
                        coords_record_1.append(coord_trans[0])
                        coords_record_2.append(coord_trans[1])
                coords_record_1 = np.array(coords_record_1)
                coords_record_2 = np.array(coords_record_2)
                coords_dis_1 = coords_record_1[1:]-coords_record_1[:-1]
                coords_dis_2 = coords_record_2[1:]-coords_record_2[:-1]
                coords_dis_1 = np.array([cal_dis(item) for item in coords_dis_1])
                coords_dis_2 = np.array([cal_dis(item) for item in coords_dis_2])
                answer = describe_spd_comp(coords_dis_1,coords_dis_2)
                len_answer = len(answer.split("."))-1
                agent = "camera" if agent=="camera" else f"{obj_labels[0]} with initial bounding box coordinates [{obj_boxes[0][0]},{obj_boxes[0][1]},{obj_boxes[0][2]},{obj_boxes[0][3]}]"
                object_1 = "camera" if objects[1]=="camera" else f"{obj_labels[1]} with initial bounding box coordinates [{obj_boxes[1][0]},{obj_boxes[1][1]},{obj_boxes[1][2]},{obj_boxes[1][3]}]"
                object_2 = "camera" if objects[2]=="camera" else f"{obj_labels[2]} with initial bounding box coordinates [{obj_boxes[2][0]},{obj_boxes[2][1]},{obj_boxes[2][2]},{obj_boxes[2][3]}]"
                idx = idx-1 if success==False else idx
                time_end = timestamps[idx]
                question = template[qa_type].format(time_1,time_end,agent,object_1,object_2)
                wrong = []
                if len_answer>15 or (idx-idx_1)<4:
                    continue
                for i in range(3):
                    rand_num_1 = np.random.uniform(0,2,size=(idx-idx_1))
                    rand_num_2 = np.random.uniform(0,2,size=(idx-idx_1))
                    wrong_tmp = describe_spd_comp(rand_num_1,rand_num_2).split(".")
                    len_tmp = random.sample(range(1 if (len(wrong_tmp)-1)<=(len_answer-3) else max(1,len_answer-3),min(len_answer+3,len(wrong_tmp)-1)+1),1)[0]
                    wrong_tmp = wrong_tmp[:len_tmp]
                    while (".".join(wrong_tmp)+".") == answer or (".".join(wrong_tmp)+".") in wrong:
                        rand_num_1 = np.random.uniform(0,2,size=(idx-idx_1))
                        rand_num_2 = np.random_uniform(0,2,size=(idx-idx_1)) 
                        wrong_tmp = describe_spd_comp(rand_num_1,rand_num_2).split(".")
                        len_tmp = random.sample(range(1 if (len(wrong_tmp)-1)<=(len_answer-3) else max(1,len_answer-3),min(len_answer+3,len(wrong_tmp)-1)+1),1)[0]
                        wrong_tmp = wrong_tmp[:len_tmp]
                    wrong.append(".".join(wrong_tmp)+".")
                options = ["A","B","C","D"]
                rand_num = np.random.uniform(0,1)
                if rand_num<0.25:
                    c_option = "A"
                elif rand_num<0.5 and rand_num>=0.25:
                    c_option = "B"
                elif rand_num<0.75 and rand_num>=0.5:
                    c_option = "C"
                else:
                    c_option = "D"
                options.remove(c_option)
                qa_pairs.append({"Type":qa_type, "Question":question, options[0]:wrong[0], options[1]:wrong[1], options[2]:wrong[2], c_option:answer, "Correct":c_option})
            elif "spd" in qa_type:
                coords_record_1 = []
                objects = ['rand','rand']
                for idx in range(2):
                    rand_num = random.uniform(0,1)
                    if rand_num < 0.5:
                        objects[idx] = "camera"
                        break
                obj_labels, obj_boxes, obj_masks = extract_obj_video_mask(grounding_model, video_predictor, video_frames, ". ".join(objs)+".", 2, device)
                if obj_labels[0] not in agents_with_cam:
                    continue
                agent = objects[0]
                if agent == "camera":
                    for idx in range(idx_1,idx_2+1):
                        pc_tmp = camera_self_coord_trans(pcs[idx])
                        if np.sum(np.logical_and(obj_masks[1][idx-idx_1],masks[idx]))==0:
                            success = False
                            break
                        obj1_coord = extract_obj_center(pc_tmp, obj_masks[1][idx-idx_1], masks[idx])
                        coords_record_1.append(obj1_coord)
                else:
                    for idx in range(idx_1,idx_2+1):
                        if objects[1] != "camera":
                            if np.sum(np.logical_and(obj_masks[1][idx-idx_1],masks[idx]))==0:
                                success = False
                                break
                            obj1_coord = np.array([extract_obj_center(pcs[idx], obj_masks[1][idx-idx_1], masks[idx])])
                        else:
                            obj1_coord = np.array([[0,0,0]]).astype(np.float32)
                        if np.sum(np.logical_and(obj_masks[0][idx-idx_1],masks[idx]))==0:
                            success = False
                            break
                        pose_agent = extract_orient(dino, val_preprocess, f"./{proc_id}_frames/{idx:03d}.jpg", obj_masks[0][idx-idx_1], device)
                        if pose_agent[3]<0.6:
                            success = False
                            break
                        azimuth, polar, rotation = pose_agent[0], pose_agent[1], pose_agent[2]
                        trans = extract_obj_center(pcs[idx],obj_masks[0][idx-idx_1],masks[idx])
                        obj1_coord = camera_agent_coord_trans(obj1_coord, polar, azimuth, rotation, trans)
                        coords_record_1.append(obj1_coord[0])
                coords_record_1 = np.array(coords_record_1)
                coords_dis_1 = coords_record_1[1:]-coords_record_1[:-1]
                coords_dis_1 = [cal_dis(item) for item in coords_dis_1]
                coords_dis_1 = np.array(coords_dis_1)
                answer = describe_dist_runs(coords_dis_1)
                len_answer = len(answer.split("."))-1
                agent = "camera" if objects[0]=="camera" else f"{obj_labels[0]} with initial bounding box coordinates [{obj_boxes[0][0]},{obj_boxes[0][1]},{obj_boxes[0][2]},{obj_boxes[0][3]}]"
                object_1 = "camera" if objects[1]=="camera" else f"{obj_labels[1]} with initial bounding box coordinates [{obj_boxes[1][0]},{obj_boxes[1][1]},{obj_boxes[1][2]},{obj_boxes[1][3]}]"
                idx = idx-1 if success==False else idx
                if len_answer>15 or (idx-idx_1)<4:
                    continue
                time_end = timestamps[idx]
                question = template[qa_type].format(time_1,time_end,agent,object_1)
                wrong = []
                for i in range(3):
                    rand_num = np.random.uniform(-1,1,size=(idx-idx_1))
                    wrong_tmp = describe_dist_runs(rand_num).split(".")
                    len_tmp = random.sample(range(1 if (len(wrong_tmp)-1)<=(len_answer-3) else max(1,len_answer-3),min(len_answer+3,len(wrong_tmp)-1)+1),1)[0]
                    wrong_tmp = wrong_tmp[:len_tmp]
                    while (".".join(wrong_tmp)+".") == answer or (".".join(wrong_tmp)+".") in wrong:
                        rand_num = np.random.uniform(-1,1,size=(idx-idx_1))
                        wrong_tmp = describe_dist_runs(rand_num).split(".")
                        len_tmp = random.sample(range(1 if (len(wrong_tmp)-1)<=(len_answer-3) else max(1,len_answer-3),min(len_answer+3,len(wrong_tmp)-1)+1),1)[0]
                        wrong_tmp = wrong_tmp[:len_tmp]
                    wrong.append(".".join(wrong_tmp)+".")
                options = ["A","B","C","D"]
                rand_num = np.random.uniform(0,1)
                if rand_num<0.25:
                    c_option = "A"
                elif rand_num<0.5 and rand_num>=0.25:
                    c_option = "B"
                elif rand_num<0.75 and rand_num>=0.5:
                    c_option = "C"
                else:
                    c_option = "D"
                options.remove(c_option)
                qa_pairs.append({"Type":qa_type, "Question":question, options[0]:wrong[0], options[1]:wrong[1], options[2]:wrong[2], c_option:answer, "Correct":c_option})
        elif qa_type=="abs_dir_pred":
            camera_poses = poses
            agent = random.sample(agents_with_cam,1)[0]
            time_1 = random.sample(timestamps,1)[0]
            idx_1 = timestamps.index(time_1)
            obj = "rand"
            if random.uniform(0,1)<0.5:
                obj = "camera"
            video_frames = [f"./{proc_id}_frames/029.jpg", f"./{proc_id}_frames/031.jpg"]
            obj_labels, obj_boxes, obj_masks = extract_obj_video_mask(grounding_model, video_predictor, video_frames, ". ".join(objs)+".", 1, device)
            if agent != "camera":
                agent_label, agent_box, agent_mask = extract_obj_image_mask(grounding_model, sam2_predictor, f"./{proc_id}_frames/{idx_1:03d}.jpg", ". ".join(agents)+".", device)
            if agent == "camera":
                if obj == "camera":
                    pos_1 = np.array([[0,0,0]]).astype(np.float32)
                    pos_1 = camera_mutual_coordinate_transform(pos_1, camera_poses[-3], camera_poses[idx_1])
                    pos_2 = np.array([[0,0,0]]).astype(np.float32)
                    pos_2 = camera_mutual_coordinate_transform(pos_2, camera_poses[-1], camera_poses[idx_1])
                    pos_1 = camera_self_coord_trans(pos_1)
                    pos_2 = camera_self_coord_trans(pos_2)
                else:
                    if np.sum(np.logical_and(obj_masks[0][0],masks[-3]))==0 or np.sum(np.logical_and(obj_masks[0][1],masks[-1]))==0:
                        success = False
                        continue
                    obj_coord1 = np.array([extract_obj_center(pcs[-3],obj_masks[0][0],masks[-3])])
                    pos_1 = camera_mutual_coordinate_transform(obj_coord1, camera_poses[-3], camera_poses[idx_1])
                    pos_1 = camera_self_coord_trans(pos_1)
                    obj_coord2 = np.array([extract_obj_center(pcs[-1],obj_masks[0][1],masks[-1])])
                    pos_2 = camera_mutual_coordinate_transform(obj_coord2, camera_poses[-1], camera_poses[idx_1])
                    pos_2 = camera_self_coord_trans(pos_2)
            else:
                if obj == "camera":
                    pos_1 = np.array([[0,0,0]]).astype(np.float32)
                    pos_1 = camera_mutual_coordinate_transform(pos_1, camera_poses[-3], camera_poses[idx_1])
                    pos_2 = np.array([[0,0,0]]).astype(np.float32)
                    pos_2 = camera_mutual_coordinate_transform(pos_2, camera_poses[-1], camera_poses[idx_1])
                    if np.sum(np.logical_and(agent_mask,masks[idx_1]))==0:
                        success = False
                        continue
                    trans = extract_obj_center(pcs[idx_1],agent_mask,masks[idx_1])
                    pose_agent = extract_orient(dino, val_preprocess, f"./{proc_id}_frames/{idx_1:03d}.jpg", agent_mask, device)
                    if pose_agent[3]<0.6:
                        success = False
                        continue
                    azimuth, polar, rotation = pose_agent[0], pose_agent[1], pose_agent[2]
                    pos_1 = camera_agent_coord_trans(pos_1, polar, azimuth, rotation, trans)
                    pos_2 = camera_agent_coord_trans(pos_2, polar, azimuth, rotation, trans)
                else:
                    if np.sum(np.logical_and(obj_masks[0][0],masks[-3]))==0 or np.sum(np.logical_and(obj_masks[0][1],masks[-1]))==0 or np.sum(np.logical_and(agent_mask,masks[idx_1]))==0:
                        success = False
                        continue
                    obj_coord1 = np.array([extract_obj_center(pcs[-3],obj_masks[0][0],masks[-3])])
                    pos_1 = camera_mutual_coordinate_transform(obj_coord1, camera_poses[-3], camera_poses[idx_1])
                    obj_coord2 = np.array([extract_obj_center(pcs[-1],obj_masks[0][1],masks[-1])])
                    pos_2 = camera_mutual_coordinate_transform(obj_coord2, camera_poses[-1], camera_poses[idx_1])
                    trans = extract_obj_center(pcs[idx_1],agent_mask,masks[idx_1])
                    pose_agent = extract_orient(dino, val_preprocess, f"./{proc_id}_frames/{idx_1:03d}.jpg", agent_mask, device)
                    if pose_agent[3]<0.6:
                        success = False
                        continue
                    azimuth, polar, rotation = pose_agent[0], pose_agent[1], pose_agent[2]
                    pos_1 = camera_agent_coord_trans(pos_1, polar, azimuth, rotation, trans)
                    pos_2 = camera_agent_coord_trans(pos_2, polar, azimuth, rotation, trans)
            coords_dis = pos_2-pos_1
            answer = describe_dir(coords_dis)
            len_answer = len(answer.split("."))-1
            agent = "camera" if agent=="camera" else f"{agent_label} with bounding box coordinates [{agent_box[0]},{agent_box[1]},{agent_box[2]},{agent_box[3]}]"
            object_1 = "camera" if obj=="camera" else f"{obj_labels[0]} with final bounding box coordinates [{obj_boxes[0][0]},{obj_boxes[0][1]},{obj_boxes[0][2]},{obj_boxes[0][3]}]"
            question = template[qa_type].format(agent,time_1, object_1)
            wrong = []
            for _ in range(3):
                rand_num = np.random.uniform(-1,1,size=((1,3)))
                wrong_tmp = describe_dir(rand_num).split(".")
                while (".".join(wrong_tmp)) == answer or (".".join(wrong_tmp)) in wrong:
                    rand_num = np.random.uniform(-1,1,size=((1,3)))
                    wrong_tmp = describe_dir(rand_num).split(".")
                wrong.append(".".join(wrong_tmp))
            options = ["A","B","C","D"]
            rand_num = np.random.uniform(0,1)
            if rand_num<0.25:
                c_option = "A"
            elif rand_num<0.5 and rand_num>=0.25:
                c_option = "B"
            elif rand_num<0.75 and rand_num>=0.5:
                c_option = "C"
            else:
                c_option = "D"
            options.remove(c_option)
            qa_pairs.append({"Type":qa_type, "Question":question, options[0]:wrong[0], options[1]:wrong[1], options[2]:wrong[2], c_option:answer, "Correct":c_option})
        elif qa_type=="rel_dir_pred":
            camera_poses = poses
            agent = random.sample(agents_with_cam,1)[0]
            time_1 = timestamps[-1]
            idx_1 = timestamps.index(time_1)
            obj = "rand"
            if random.uniform(0,1)<0.5:
                obj = "camera"
            video_frames = [f"./{proc_id}_frames/029.jpg", f"./{proc_id}_frames/031.jpg"]
            obj_labels, obj_boxes, obj_masks = extract_obj_video_mask(grounding_model, video_predictor, video_frames, ". ".join(objs)+".", 1, device)
            if agent != "camera":
                agent_label, agent_box, agent_mask = extract_obj_video_mask(grounding_model, video_predictor, video_frames, ". ".join(agents)+".", 1, device)
            if agent == "camera":
                if obj == "camera":
                    pos_1 = np.array([[0,0,0]]).astype(np.float32)
                    pos_1 = camera_mutual_coordinate_transform(pos_1, camera_poses[-3], camera_poses[idx_1])
                    pos_2 = np.array([[0,0,0]]).astype(np.float32)
                    pos_2 = camera_mutual_coordinate_transform(pos_2, camera_poses[-1], camera_poses[idx_1])
                    pos_1 = camera_self_coord_trans(pos_1)
                    pos_2 = camera_self_coord_trans(pos_2)
                else:
                    if np.sum(np.logical_and(obj_masks[0][0],masks[-3]))==0 or np.sum(np.logical_and(obj_masks[0][1],masks[-1]))==0:
                        success = False
                        continue
                    obj_coord1 = np.array([extract_obj_center(pcs[-3],obj_masks[0][0],masks[-3])])
                    pos_1 = camera_self_coord_trans(obj_coord1)
                    obj_coord2 = np.array([extract_obj_center(pcs[-1],obj_masks[0][1],masks[-1])])
                    pos_2 = camera_self_coord_trans(obj_coord2)
            else:
                if obj == "camera":
                    pos_1 = np.array([[0,0,0]]).astype(np.float32)
                    if np.sum(np.logical_and(agent_mask[0][0],masks[-3]))==0:
                        success = False
                        continue
                    trans = extract_obj_center(pcs[-3],agent_mask[0][0],masks[-3])
                    pose_agent = extract_orient(dino, val_preprocess, f"./{proc_id}_frames/029.jpg", agent_mask[0][0], device)
                    if pose_agent[3]<0.6:
                        success = False
                        continue
                    azimuth, polar, rotation = pose_agent[0], pose_agent[1], pose_agent[2]
                    pos_1 = camera_agent_coord_trans(pos_1, polar, azimuth, rotation, trans)
                    pos_2 = np.array([[0,0,0]]).astype(np.float32)
                    if np.sum(np.logical_and(agent_mask[0][1],masks[-1]))==0:
                        success = False
                        continue
                    trans = extract_obj_center(pcs[-1],agent_mask[0][1],masks[-1])
                    pose_agent = extract_orient(dino, val_preprocess, f"./{proc_id}_frames/031.jpg", agent_mask[0][1], device)
                    if pose_agent[3]<0.6:
                        success = False
                        continue
                    azimuth, polar, rotation = pose_agent[0], pose_agent[1], pose_agent[2]
                    pos_2 = camera_agent_coord_trans(pos_2, polar, azimuth, rotation, trans)
                else:
                    if np.sum(np.logical_and(obj_masks[0][0],masks[-3]))==0 or np.sum(np.logical_and(obj_masks[0][1],masks[-1]))==0 or np.sum(np.logical_and(agent_mask[0][0],masks[-3]))==0 or np.sum(np.logical_and(agent_mask[0][1],masks[-1]))==0:
                        success = False
                        continue
                    obj_coord1 = np.array([extract_obj_center(pcs[-3],obj_masks[0][0],masks[-3])])
                    trans = extract_obj_center(pcs[-3],agent_mask[0][0],masks[-3])
                    pose_agent = extract_orient(dino, val_preprocess, f"./{proc_id}_frames/029.jpg", agent_mask[0][0], device)
                    if pose_agent[3]<0.6:
                        success = False
                        continue
                    intersect = np.sum(np.logical_and(obj_masks[0][0],agent_mask[0][0]))
                    if (intersect/(np.sum(obj_masks[0][0])+np.sum(agent_mask[0][0])-intersect))>0.5:
                        continue
                    azimuth, polar, rotation = pose_agent[0], pose_agent[1], pose_agent[2]
                    pos_1 = camera_agent_coord_trans(obj_coord1, polar, azimuth, rotation, trans)

                    obj_coord2 = np.array([extract_obj_center(pcs[-1],obj_masks[0][1],masks[-1])])
                    trans = extract_obj_center(pcs[-1],agent_mask[0][1],masks[-1])
                    pose_agent = extract_orient(dino, val_preprocess, f"./{proc_id}_frames/031.jpg", agent_mask[0][1], device)
                    if pose_agent[3]<0.6:
                        success = False
                        continue
                    azimuth, polar, rotation = pose_agent[0], pose_agent[1], pose_agent[2]
                    pos_2 = camera_agent_coord_trans(obj_coord2, polar, azimuth, rotation, trans)
            coords_dis = pos_2-pos_1
            answer = describe_dir(coords_dis)
            len_answer = len(answer.split("."))-1
            agent = "camera" if agent=="camera" else f"{agent_label[0]} with final bounding box coordinates [{agent_box[0][0]},{agent_box[0][1]},{agent_box[0][2]},{agent_box[0][3]}]"
            object_1 = "camera" if obj=="camera" else f"{obj_labels[0]} with final bounding box coordinates [{obj_boxes[0][0]},{obj_boxes[0][1]},{obj_boxes[0][2]},{obj_boxes[0][3]}]"
            if agent == object_1:
                continue
            question = template[qa_type].format(agent,object_1)
            wrong = []
            for _ in range(3):
                rand_num = np.random.uniform(-1,1,size=((1,3)))
                wrong_tmp = describe_dir(rand_num).split(".")
                while (".".join(wrong_tmp)) == answer or (".".join(wrong_tmp)) in wrong:
                    rand_num = np.random.uniform(-1,1,size=((1,3)))
                    wrong_tmp = describe_dir(rand_num).split(".")
                wrong.append(".".join(wrong_tmp))
            options = ["A","B","C","D"]
            rand_num = np.random.uniform(0,1)
            if rand_num<0.25:
                c_option = "A"
            elif rand_num<0.5 and rand_num>=0.25:
                c_option = "B"
            elif rand_num<0.75 and rand_num>=0.5:
                c_option = "C"
            else:
                c_option = "D"
            options.remove(c_option)
            qa_pairs.append({"Type":qa_type, "Question":question, options[0]:wrong[0], options[1]:wrong[1], options[2]:wrong[2], c_option:answer, "Correct":c_option})
        qa_count[qa_type] += 1
       except RuntimeWarning as rw:
        print(f"runtimewarning: {rw}")
       except Exception as e:
        pass
    return qa_pairs

def generate_multi_videos(physical_gpu_id, part_info, video_dynamic, qa_num, step_size, save_idx):
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
    val_preprocess = AutoImageProcessor.from_pretrained(DINO_LARGE, cache_dir='./')
    pi3 = Pi3.from_pretrained("./models/Pi3").to(device).eval()
    save_files = glob.glob(f"qa_pairs_{save_idx}_*.json")
    if os.path.exists(f"qa_pairs_{save_idx}.json"):
        return
    if save_files==[]:
        qa_total = {}
        part_num = 0
    else:
        qa_total = json.load(open(save_files[0],"r"))
        part_num = int(save_files[0].split(".")[0].split("_")[-1])
    for idx,(_,row) in enumerate(part_info.iloc[part_num*step_size:].iterrows()):
        if (part_num*step_size+idx)%step_size==0:
            with open(f"qa_pairs_{save_idx}_{(part_num*step_size+idx)//step_size}.json", "w", encoding="utf-8") as f:
                json.dump(qa_total, f, ensure_ascii=False, indent=4)
            if os.path.exists(f"qa_pairs_{save_idx}_{(part_num*step_size+idx)//step_size-1}.json"):
                os.remove(f"qa_pairs_{save_idx}_{(part_num*step_size+idx)//step_size-1}.json")
        video_name = row['videoID']
        if video_name not in video_dynamic:
            continue
        timestamps = sample_uniform_frames(video_path=f"{video_root}/{video_name}.mp4",save_dir=f"./{save_idx}_frames")
        agent = row['agent']
        obj = row['obj']
        if pd.isna(agent):
            continue
        else:
            agent = agent.lower()
            agent = agent.split('.')
        if pd.isna(obj):
            obj=[]
        else:
            obj = obj.lower()
            obj = obj.split('.')
        points, masks, poses = run_pi3(pi3, save_idx, device)
        qa_pairs = generate_one_video(save_idx, timestamps, agent, obj, qa_num, points, masks, poses, grounding_model, video_predictor, sam2_predictor, val_preprocess, dino, device)
        qa_total[video_name]=qa_pairs
    with open(f'qa_pairs_{save_idx}.json', 'w') as json_file:
        json.dump(qa_total, json_file, ensure_ascii=False, indent=4)
    save_files = glob.glob(f"qa_pairs_{save_idx}_*.json")
    if save_files != []:
        for file in save_files:
            os.remove(file)

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
        args=(physical_gpu_id, data_parts[proc_idx], video_dynamic, args.qa_num, args.part_len, proc_idx),
        daemon=False
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    save_files = glob.glob("qa_pairs_*.json")
    merged = {}

    for f in save_files:
        data = json.load(open(f))
        merged.update(data)   # 直接合并字典

    with open("qa_pairs.json", 'w') as json_file:
        json.dump(merged, json_file, ensure_ascii=False, indent=4)