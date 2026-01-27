import json

import requests
import random
import os
import pandas as pd
from tqdm import tqdm
from functools import partial
import multiprocessing
import glob
import argparse

os.makedirs("./dynamic_videos",exist_ok=True)
def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--part_len', type=int, default=2)
    parser.add_argument('--process_num', type=int, default=10)
    parser.add_argument('--video_root', type=str, default="./video_root")
    args = parser.parse_args()
    
    return args

def video_collect(video_part,proc_id,step_size):
    video_classes = ["Sports & Recreation", "Transportation & Vehicle Operation", "Art Performance", "Manual Labor & Craftsmanship", "Daily Activities & Hobbies", "Nature & Wildlife"]
    save_files = glob.glob(f"dynamic_videos/dynamic_videos_{proc_id}_*.json")
    if os.path.exists(f"dynamic_videos/dynamic_videos_{proc_id}.json"):
        return
    if save_files==[]:
        video_filter = {}
        video_num = {}
        for v_class in video_classes:
            video_filter[v_class] = []
            video_num[v_class] = 0
        part_num = 0
    else:
        video_filter = json.load(open(save_files[0],"r"))
        video_num = {}
        for v_class in video_classes:
            video_num[v_class] = len(video_filter[v_class])
        part_num = int(save_files[0].split(".")[-2].split("_")[-1])
    for idx, video in tqdm(enumerate(video_part[part_num*step_size:])):
        if (part_num*step_size+idx)%step_size==0:
            with open(f"dynamic_videos/dynamic_videos_{proc_id}_{(part_num*step_size+idx)//step_size}.json", "w", encoding="utf-8") as f:
                json.dump(video_filter, f, ensure_ascii=False, indent=4)
            if os.path.exists(f"dynamic_videos/dynamic_videos_{proc_id}_{(part_num*step_size+idx)//step_size-1}.json"):
                os.remove(f"dynamic_videos/dynamic_videos_{proc_id}_{(part_num*step_size+idx)//step_size-1}.json")
        try:
        # 文件名
            filename = f'{video}'
            # 对话模型
            agent = agent_obj_info.loc[agent_obj_info['videoID']==video[:-4],'agent'].tolist()[0]
            obj = agent_obj_info.loc[agent_obj_info['videoID']==video[:-4],'obj'].tolist()[0]
            if pd.isna(agent):
                continue
            else:
                agent = agent.lower()
                agent = agent.replace(".",", ")
            if pd.isna(obj):
                obj=[]
            else:
                obj = obj.lower()
                obj = obj.replace(".",", ")
            model = 'gemini-2.5-pro'
            # 请求生成临时密钥
            prompt = f'The video belongs to one of following categories: {",".join(video_classes)}. The agent in the video is {agent}. The object in the video is {obj}. Think about whether the video satisfies the following conditions: (1) The camera is not mainly filming the inside of one moving object; (2) Regardless of camera movement, articulated motion and shape variation, the world coordinates of at least 3 agent (exclude hands) or object change substantially; (3) Every agent or object cannot occupy less than 1% of the screen; (4) Every agent or object cannot occupy more than 50% of the screen. Only consider the objects that can be regarded as a whole and not part of or containing other objects. Only consider objects that are discrete, countable items (e.g., a bottle, a rock). Only consider objects that are not in contact with each other. Only consider agents and objects belonging to classes given at the beginning. \n If all conditions are satisfied, output "yes" in a new line. If any condition is not satisfied, output "no" in a new line. Then in a new line, output the category of the video.'
            response = "YOUR_REQUEST_FUNCTION"
            response = response.split("\n")
            while len(response)!=2:
                response = "YOUR_REQUEST_FUNCTION"
            if "no" in response[0]:
                continue
            elif "yes" in response[0]:
                video_filter[response[1]].append(filename)
                video_num[response[1]]+=1
        except:
            continue
    with open(f"dynamic_videos/dynamic_videos_{proc_id}.json", "w", encoding="utf-8") as f:
        json.dump(video_filter, f, ensure_ascii=False, indent=4)
    save_files = glob.glob(f"dynamic_videos/dynamic_videos_{proc_id}_*.json")
    if save_files != []:
        for file in save_files:
            os.remove(file)

args = parse_arg()
video_root = args.video_root
videos = os.listdir(video_root)
total_process = args.process_num
process_func = partial(video_collect)
total_num = len(videos)
part_num = total_num//total_process
agent_obj_info = pd.read_csv("./agent_object.csv")
process_func(videos[0:part_num],0,args.part_len)
with multiprocessing.Pool(processes=total_process) as pool:
    pool.starmap(process_func,[(videos[i*part_num:(i+1)*part_num],i,args.part_len) for i in range(total_process)])
save_files = glob.glob("dynamic_videos/dynamic_videos_*.json")
merged = []
video_classes = ["Sports & Recreation", "Transportation & Vehicle Operation", "Art Performance", "Manual Labor & Craftsmanship", "Daily Activities & Hobbies", "Nature & Wildlife"]
for v_class in video_classes:
    merged[v_class] = []
for f in save_files:
    data = json.load(open(f))
    for v_class in video_classes:
        merged[v_class].extend(data[v_class])
with open("dynamic_videos.json", "w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False, indent=4)