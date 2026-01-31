import json, os

video_root = "/home/datatset/data/h30081741/VLM4D-video/videos_synthetic"
data = json.load(open("synthetic_mc.json"))

for d in data:
    fname = os.path.basename(d["video"])  # synth_001.mp4
    d["video"] = os.path.join(video_root, fname)

json.dump(data, open("synthetic_mc_local.json", "w"), ensure_ascii=False, indent=2)
