from typing import List, Tuple
import numpy as np
question_type = ["rel_dis","rel_dir","rel_ori","abs_dis","abs_dir","abs_ori","rel_spd","abs_spd","rel_spd_comp","abs_spd_comp","rel_dir_pred","abs_dir_pred"]
template={"rel_dis":"Between {:.1f}s and {:.1f}s, following the perspective of {}, how does the distance between {} and {} change?",
          "rel_dir":"Between {:.1f}s and {:.1f}s, following the perspective of {}, how does the direction of {} to {} change?",
          "rel_ori":"Between {:.1f}s and {:.1f}s, following the perspective of {}, how does the orientation of {} change?",
          "abs_dis":"Between {:.1f}s and {:.1f}s, from the perspective of {} at {:.1f}s, how does the distance between {} and {} change?",
          "abs_dir":"Between {:.1f}s and {:.1f}s, from the perspective of {} at {:.1f}s, how does the direction of {} to {} change?",
          "abs_ori":"Between {:.1f}s and {:.1f}s, from the perspective of {} at {:.1f}s, how does the orientation of {} change?",
          "rel_spd":"Between {:.1f}s and {:.1f}s, following the perspective of {}, how does the speed of {} change?",
          "abs_spd":"Between {:.1f}s and {:.1f}s, from the perspective of {} at {:.1f}s, how does the speed of {} change?",
          "rel_spd_comp":"Between {:.1f}s and {:.1f}s, following the perspective of {}, compare the speed between {} and {}.",
          "abs_spd_comp":"Between {:.1f}s and {:.1f}s, from the perspective of {} at {:.1f}s, compare the speed between {} and {}.",
          "rel_dir_pred":"Following the perspective of {}, predict the moving direction of {}",
          "abs_dir_pred":"From the perspective of {} at {:.1f}s, predict the moving direction of {}"
}

def _sign(x, eps=0.2):
    x_1,x_2 = x
    if x_1<0.1 and x_2<0.1:
        return 0
    if x_1/x_2 > (1+eps):
        return 1
    if x_1/x_2 < 1/(1+eps):
        return -1
    return 0

def _direction(x, eps=0.2):
    dirs = []
    x = x/np.sqrt(np.sum(x**2))
    for item in x:
        if item > eps:
            dirs.append("2")
        elif item < -eps:
            dirs.append("0")
        else:
            dirs.append("1")
    return "".join(dirs)

def _direction_trans(x):
    dirs = []
    if x[0] == "2":
        dirs.append("Front")
    elif x[0] == "0":
        dirs.append("Behind")
    if x[1] == "2":
        dirs.append("Left")
    elif x[1] == "0":
        dirs.append("Right")
    if x[2] == "2":
        dirs.append("Above")
    elif x[2] == "0":
        dirs.append("Below")
    return "|".join(dirs)

def describe_dist_runs(
    distances,
    epsilon=0.2,
    min_run_length=2
):
    descriptions = []  # 用于存储每段时间的描述
    start_idx = 0  # 每段时间的起始索引
    count = 1
    while start_idx < len(distances) - 1:
        end_idx = start_idx + 1
        # 判断一段时间内的变化
        if (distances[end_idx] / distances[start_idx] >= (1-epsilon) and distances[end_idx] / distances[start_idx] <= (1+epsilon)) or (distances[end_idx]<0.1 and distances[start_idx]<0.1):
            while end_idx < len(distances) and ((distances[end_idx] / distances[start_idx] >= (1-epsilon) and distances[end_idx] / distances[start_idx] <= (1+epsilon)) or (distances[end_idx]<0.1 and distances[start_idx]<0.1)):
                end_idx += 1
            if end_idx == len(distances) and start_idx==0:
                descriptions.append(f"({count}) Keep nearly constant.")
            elif end_idx == len(distances):
                descriptions.append(f"({count}) Keep nearly constant.")
            elif distances[end_idx] / distances[start_idx] < (1-epsilon):
                descriptions.append(f"({count}) Keep nearly constant and then become smaller.")
            elif distances[end_idx] / distances[start_idx] > (1+epsilon):
                descriptions.append(f"({count}) Keep nearly constant and then become larger.")
        elif (distances[end_idx] / distances[end_idx-1] > (1+epsilon)):
            while end_idx < len(distances) and (distances[end_idx] / distances[end_idx-1] > (1+epsilon)):
                end_idx += 1
            if end_idx == len(distances) and start_idx==0:
                descriptions.append(f"({count}) Become larger.")
            elif end_idx == len(distances):
                descriptions.append(f"({count}) Become larger.")
            else:
                descriptions.append(f"({count}) Become larger.")
                end_idx -= 1
        else:
            while end_idx < len(distances) and (distances[end_idx] / distances[end_idx-1] < (1-epsilon)) and distances[end_idx]>=0.1:
                end_idx += 1
            if end_idx == len(distances) and start_idx==0:
                descriptions.append(f"({count}) Become smaller.")
            elif end_idx == len(distances):
                descriptions.append(f"({count}) Become smaller.")
            elif distances[end_idx]<0.1:
                descriptions.append(f"({count}) Become smaller.")
            else:
                descriptions.append(f"({count}) Become smaller.")
                end_idx -= 1
        start_idx = end_idx  # 更新起始索引
        count += 1
    descriptions = " ".join(descriptions)
    return descriptions

def describe_spd_comp(
    speed_1,
    speed_2,
    epsilon=0.2,
    min_run_length=2
):
    n = len(speed_1)
    signs = [_sign(x, epsilon) for x in zip(speed_1,speed_2)]

    out = []
    i = 0
    count = 1
    while i < n:
        s = signs[i]
        j = i + 1
        # 扩展到同号段
        while j < n and signs[j] == s:
            j += 1
        # 统计该段的总变化量（正段直接相加，负段取绝对值便于表述，零段忽略数值）
        if i==0 and j==n and s>0:
            desc = f"({count}) The former is always faster."
        elif i==0 and j==n and s==0:
            desc = f"({count}) Their speed is nearly the same."
        elif i==0 and j==n and s<0:
            desc = "The latter is always faster."
        elif i==0 and s > 0:
            desc = f"({count}) The former is faster."
        elif s>0:
            desc = f"({count}) The former is faster."
        elif i==0 and s==0:
            desc = f"({count}) Nearly the same."
        elif s==0:
            desc = f"({count}) Nearly the same."
        elif i==0 and s < 0:
            desc = f"({count}) The latter is faster."
        else:
            desc = f"({count}) The latter is faster."
        out.append(desc)
        count += 1
        i = j
    final_desc = " ".join(out)
    return final_desc

def describe_dir(
    changes,
    epsilon=0.3,
    min_run_length=2
):
    n = len(changes)
    dirs = [_direction(x, epsilon) for x in changes]
    out = []
    i = 0
    count = 1
    while i < n:
        s = dirs[i]
        j = i + 1
        # 扩展到同号段
        dir_trans = _direction_trans(s)
        while j < n and dirs[j] == s:
            j += 1
        # 统计该段的总变化量（正段直接相加，负段取绝对值便于表述，零段忽略数值）
        if i==0 and j==n:
            desc = f"({count}) {dir_trans}."
        else:
            desc = f"({count}) {dir_trans}."
        out.append(desc)
        i = j
        count += 1
    final_desc = " ".join(out)
    return final_desc
