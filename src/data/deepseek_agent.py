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

os.makedirs("./agent_object",exist_ok=True)
def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--part_len', type=int, default=2)
    parser.add_argument('--process_num', type=int, default=10)
    parser.add_argument('--koala_csv_path', type=str, default="./koala_videos.csv")
    args = parser.parse_args()
    
    return args

def split_output(answer):
    parts = answer.split('\n')
    parts_left = []
    for item in parts:
        if item=='':
            continue
        else:
            parts_left.append(item.strip())
    return parts_left

def process_data(s, e, part_len):
    if os.path.exists(f"agent_object/agent_object_{s}_{e}.csv"):
        return
    process_part = captions.iloc[s:e]
    save_files = glob.glob(f"agent_object/agent_object_{s}_{e}_*.csv")
    process_num = 0
    filtered_rows = pd.DataFrame()
    if save_files != []:
        process_num = int(save_files[0].split(".")[-2].split('_')[-1])
        try:
            filtered_rows = pd.read_csv(save_files[0])
        except:
            filtered_rows = pd.DataFrame()
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
                'content': f'"{caption}" is the caption of a video. In one line, list all classes of agents in the video and use dot to separate them. If there is no agent, output "None" in this line. Then, in a new line, list the classes of remaining foreground objects in the video and use dot to separate them. An object should be a discrete, countable item (e.g., a bottle, a rock). The object must not be a part of other objects. If there is no object, output "None" in this line.'
            }
            ]
        }

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {token}'
        }

        response = requests.post(url, headers=headers, data=json.dumps(payload))
        while response.status_code != 200 or len(split_output(response.json()['choices'][0]['message']['content']))!=2:
            if response.status_code != 200:
            # 异常处理
                print(response.json())
            response = requests.post(url, headers=headers, data=json.dumps(payload))
        agent_objs = split_output(response.json()['choices'][0]['message']['content'])
        new_data = {"videoID":row["videoID"], "agent":agent_objs[0], "obj":agent_objs[1]}
        filtered_rows = pd.concat([filtered_rows, pd.DataFrame(new_data,index=[index])], ignore_index=True)
        if (index+1+process_num*part_len)%part_len==0:
            filtered_rows.to_csv(f'agent_object/agent_object_{s}_{e}_{(index+1+process_num*part_len)//part_len}.csv', index=False)
            if os.path.exists(f'agent_object/agent_object_{s}_{e}_{(index+1+process_num*part_len)//part_len-1}.csv'):
                os.remove(f'agent_object/agent_object_{s}_{e}_{(index+1+process_num*part_len)//part_len-1}.csv')
    filtered_rows.to_csv(f'agent_object/agent_object_{s}_{e}.csv', index=False)
    save_files = glob.glob(f"agent_object/agent_object_{s}_{e}_*.csv")
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
save_files = glob.glob("agent_object/agent_object_*.csv")
dfs = [pd.read_csv(f) for f in save_files]
merged_df = pd.concat(dfs, ignore_index=True)
merged_df.to_csv("agent_object.csv", index=False)