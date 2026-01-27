import os
import json
import requests
import pandas as pd
import multiprocessing
from functools import partial
from tqdm import tqdm
import glob
import argparse
token = "YOUR_TOKEN"
url = "YOUR_URL"

os.makedirs("./dynamic_videos",exist_ok=True)
def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--part_len', type=int, default=2)
    parser.add_argument('--process_num', type=int, default=10)
    parser.add_argument('--koala_csv_path', type=str, default="./koala_videos.csv")
    args = parser.parse_args()
    
    return args

def process_data(s, e, part_len):
    if os.path.exists(f"dynamic_videos/dynamic_videos_complete_{s}_{e}.json"):
        return
    process_part = captions.iloc[s:e]
    save_files = glob.glob(f"dynamic_videos/dynamic_videos_{s}_{e}_*.json")
    process_num = 0
    filtered_rows = []
    if save_files != []:
        process_num = int(save_files[0].split(".")[-2].split('_')[-1])
        filtered_rows = json.load(open(save_files[0]))
        process_part = process_part.iloc[process_num*part_len:]
    
    for index, (_, row) in tqdm(enumerate(process_part.iterrows())):
        caption = row["caption"]
        payload = {
        'model': 'deepseek-r1',
        'messages': [
            {
                'role': 'system',
                'content': 'You are a helpful assistant.'
            },
            {
                'role': 'user',
                'content': f'"{caption}" is the caption of a video. Think about whether the video satisfies the following condition: Regardless of camera movement and articulated motion, the world coordinates of at least 3 main subjects in the video change substantially. Only consider the objects that can be regarded as a whole and not part of other objects. \n If the condition is satisfied, output "yes" in a new line. If the condition is not satisfied, output "no" in a new line.'
            }
            ]
        }

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {token}'
        }

        response = requests.post(url, headers=headers, data=json.dumps(payload))

        # 判断是否异常
        while response.status_code != 200:
            print(response.json())
            response = requests.post(url, headers=headers, data=json.dumps(payload))
        if "no" not in response.json()['choices'][0]['message']['content'].split('\n')[-1] and "yes" in response.json()['choices'][0]['message']['content'].split('\n')[-1]:
            filtered_rows.append(row['videoID'])
        if (index+1+process_num*part_len)%part_len==0:
            with open(f"dynamic_videos/dynamic_videos_{s}_{e}_{(index+1+process_num*part_len)//part_len}.json", "w", encoding="utf-8") as f:
                json.dump(filtered_rows, f, ensure_ascii=False, indent=4)
            if os.path.exists(f'dynamic_videos/dynamic_videos_{s}_{e}_{(index+1+process_num*part_len)//part_len-1}.json'):
                os.remove(f'dynamic_videos/dynamic_videos_{s}_{e}_{(index+1+process_num*part_len)//part_len-1}.json')
    with open(f"dynamic_videos/dynamic_videos_complete_{s}_{e}.json", "w", encoding="utf-8") as f:
        json.dump(filtered_rows, f, ensure_ascii=False, indent=4)
    save_files = glob.glob(f"dynamic_videos/dynamic_videos_{s}_{e}_*.json")
    if save_files != []:
        for file in save_files:
            os.remove(file)

args = parse_arg()
captions = pd.read_csv(args.koala_csv_path)
total_num = len(captions)
total_process = args.process_num
process_func = partial(process_data)
part_num = total_num//total_process
with multiprocessing.Pool(processes=total_process) as pool:
    pool.starmap(process_func,[(i*part_num,(i+1)*part_num,args.part_len) for i in range(total_process)])
save_files = glob.glob("dynamic_videos/dynamic_videos_*.json")
merged = []
for f in save_files:
    data = json.load(open(f))
    merged.extend(data)
with open("dynamic_videos.json", "w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False, indent=4)