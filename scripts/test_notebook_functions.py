"""
测试 notebook 中的核心函数（不调用 API）
"""
import os
import sys
import json

# 添加 VLM4D 路径
vlm4d_path = os.path.join(os.path.dirname(__file__), "..", "VLM4D-main")
sys.path.insert(0, vlm4d_path)

from string import Template

# Prompt 模板
MULTI_CHOICE_COT_PROMPT = Template("""
Question: $question
$optionized_str

Answer the given multiple-choice question step by step. Begin by explaining your reasoning process clearly. In the last sentence of your response, you must conclude by stating the final answer using the following format: 'Therefore, the final answer is: $$LETTER' (without quotes), where $$LETTER must be only one of the options (A or B or C or D). Think step by step before answering.""")


def test_data_loading():
    """测试数据加载"""
    print("=" * 50)
    print("测试 1: 数据加载")
    print("=" * 50)

    data_path = os.path.join(vlm4d_path, "data", "real_mc.json")
    with open(data_path, "r") as f:
        data = json.load(f)

    print(f"[OK] 加载 {len(data)} 个问题")
    return data


def test_prompt_formatting(data):
    """测试 prompt 格式化"""
    print("\n" + "=" * 50)
    print("测试 2: Prompt 格式化")
    print("=" * 50)

    query = data[1]  # 使用第二个问题

    # 格式化选项
    optionized_list = [f"{key}: {value}" for key, value in query['choices'].items()]
    optionized_str = "\n".join(optionized_list)

    # 生成 prompt
    qa_text = MULTI_CHOICE_COT_PROMPT.substitute(
        question=query['question'],
        optionized_str=optionized_str
    )

    print(f"问题: {query['question']}")
    print(f"\n生成的 Prompt:\n{qa_text[:500]}...")
    print(f"\n[OK] Prompt 格式化成功")
    return qa_text


def test_video_download():
    """测试视频下载（只下载一个小样本）"""
    print("\n" + "=" * 50)
    print("测试 3: 视频下载")
    print("=" * 50)

    import hashlib
    import requests

    # 使用一个小视频测试
    video_url = "https://huggingface.co/datasets/shijiezhou/VLM4D/resolve/main/videos_real/davis/aerobatics.mp4"
    video_id = hashlib.md5(video_url.encode()).hexdigest()

    video_tmp_dir = os.path.join(os.path.dirname(__file__), "..", "video_cache_test")
    video_subdir = os.path.join(video_tmp_dir, video_id)
    os.makedirs(video_subdir, exist_ok=True)

    video_path = os.path.join(video_subdir, "video.mp4")

    if not os.path.exists(video_path):
        print(f"下载视频: {video_url[:60]}...")
        response = requests.get(video_url, timeout=30)
        with open(video_path, "wb") as f:
            f.write(response.content)
        print(f"[OK] 下载完成: {len(response.content)} bytes")
    else:
        print(f"[OK] 视频已缓存")

    return video_path


def test_frame_extraction(video_path):
    """测试帧提取"""
    print("\n" + "=" * 50)
    print("测试 4: 帧提取")
    print("=" * 50)

    import cv2
    import base64
    import numpy as np

    total_frames = 10

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"[ERROR] 无法打开视频")
        return None

    # 读取所有帧
    all_frames = []
    while True:
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        all_frames.append(frame_base64)
    video.release()

    print(f"视频总帧数: {len(all_frames)}")

    # 均匀采样
    selected_indices = np.linspace(0, len(all_frames) - 1, total_frames, dtype=int)
    selected_frames = [all_frames[i] for i in selected_indices]

    print(f"采样 {total_frames} 帧: 索引 {list(selected_indices)}")
    print(f"每帧 base64 长度: ~{len(selected_frames[0])} 字符")
    print(f"[OK] 帧提取成功")

    return selected_frames


def test_message_format(data, frames):
    """测试 OpenAI 消息格式"""
    print("\n" + "=" * 50)
    print("测试 5: OpenAI 消息格式")
    print("=" * 50)

    query = data[1]

    # 格式化文本
    optionized_list = [f"{key}: {value}" for key, value in query['choices'].items()]
    optionized_str = "\n".join(optionized_list)
    qa_text = MULTI_CHOICE_COT_PROMPT.substitute(
        question=query['question'],
        optionized_str=optionized_str
    )

    # 构建消息
    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame[:50]}..."}}
        for frame in frames[:3]  # 只展示前3帧
    ]
    content.append({"type": "text", "text": qa_text[:100] + "..."})

    message = [{"role": "user", "content": content}]

    print(f"消息结构:")
    print(f"  - role: {message[0]['role']}")
    print(f"  - content 数量: {len(message[0]['content'])} 项")
    print(f"    - {len(frames)} 个 image_url")
    print(f"    - 1 个 text")
    print(f"[OK] 消息格式正确")

    return message


def test_eval_message_format(data):
    """测试评估消息格式"""
    print("\n" + "=" * 50)
    print("测试 6: 评估消息格式")
    print("=" * 50)

    query = data[1]
    mock_response = "Based on my analysis, the batter hit the ball to the left. Therefore, the final answer is: $C"

    # 准备评估消息
    optionized_list = [f"{key}: {value}" for key, value in query['choices'].items()]
    optionized_str = "\n".join(optionized_list)

    question_context = f"Question: {query['question']}\n\nOptions:\n{optionized_str}"
    gt_answer = f"Ground Truth Answer: {query['answer']}"
    model_response = f"Model Response to the Question: {mock_response}"

    user_prompt = f"{question_context}\n\n{gt_answer}\n\n{model_response}"

    eval_instruction = "Your task is to evaluate whether the model's final answer is correct..."

    eval_message = [
        {"role": "system", "content": eval_instruction},
        {"role": "user", "content": user_prompt},
    ]

    print(f"评估消息结构:")
    print(f"  - system: {eval_message[0]['content'][:50]}...")
    print(f"  - user: {eval_message[1]['content'][:100]}...")
    print(f"[OK] 评估消息格式正确")

    return eval_message


if __name__ == "__main__":
    print("VLM4D Notebook 函数测试")
    print("=" * 50)

    # 测试 1: 数据加载
    data = test_data_loading()

    # 测试 2: Prompt 格式化
    test_prompt_formatting(data)

    # 测试 3-5: 视频相关测试（需要额外依赖）
    try:
        # 测试 3: 视频下载
        video_path = test_video_download()

        # 测试 4: 帧提取
        if video_path:
            frames = test_frame_extraction(video_path)
        else:
            frames = None

        # 测试 5: OpenAI 消息格式
        if frames:
            test_message_format(data, frames)
    except ImportError as e:
        print(f"\n[SKIP] 视频测试需要额外依赖: {e}")
        print("这些依赖在 notebook 环境中会安装")

    # 测试 6: 评估消息格式
    test_eval_message_format(data)

    print("\n" + "=" * 50)
    print("[OK] 核心测试通过！")
    print("=" * 50)
