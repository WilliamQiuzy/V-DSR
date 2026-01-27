import json
import argparse

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qa_path', type=str)
    args = parser.parse_args()
    
    return args

args = parse_arg()
qa_file = args.qa_path
qa_pairs = json.load(open(qa_file,"r"))
qa_save = []
for video in qa_pairs:
    for qa in qa_pairs[video]:
        video_path = f"{video}.mp4"
        question = "<video>\n"+qa['Question']+'\nOptions:\nA '+qa['A']+'\nB '+qa['B']+'\nC '+qa['C']+'\nD '+qa['D']
        answer = f"{qa['Correct']}"
        sample = {
            'video':video_path,
            'conversations':[
                {'from':'human','value':question},
                {'from':'gpt','value':answer}
            ]   
        }
        qa_save.append(sample)
with open("train_qas.json", "w", encoding="utf-8") as f:
    json.dump(qa_save, f, ensure_ascii=False, indent=4)