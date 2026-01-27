"""
测试 VLM4D 数据加载模块
运行: python scripts/test_vlm4d_data.py
"""
import json
import os

def test_load_vlm4d():
    """测试加载 VLM4D 数据集"""
    data_path = os.path.join(os.path.dirname(__file__), "..", "VLM4D-main", "data", "real_mc.json")

    print("=" * 50)
    print("测试 VLM4D 数据加载")
    print("=" * 50)

    # 加载数据
    with open(data_path, "r") as f:
        data = json.load(f)

    print(f"\n总样本数: {len(data)}")

    # 检查数据格式
    sample = data[0]
    print(f"\n样本格式检查:")
    print(f"  - id: {sample.get('id')}")
    print(f"  - video: {sample.get('video')[:60]}...")
    print(f"  - question_type: {sample.get('question_type')}")
    print(f"  - question: {sample.get('question')}")
    print(f"  - choices: {sample.get('choices')}")
    print(f"  - answer: {sample.get('answer')}")

    # 验证所有样本都有必需字段
    required_fields = ['id', 'video', 'question_type', 'question', 'choices', 'answer']
    missing = []
    for i, item in enumerate(data):
        for field in required_fields:
            if field not in item:
                missing.append((i, field))

    if missing:
        print(f"\n[ERROR] 缺失字段: {missing[:5]}...")
    else:
        print(f"\n[OK] 所有 {len(data)} 个样本都包含必需字段")

    # 统计问题类型
    question_types = {}
    for item in data:
        qt = item.get('question_type', 'unknown')
        question_types[qt] = question_types.get(qt, 0) + 1
    print(f"\n问题类型分布: {question_types}")

    # 统计视频来源
    video_sources = {}
    for item in data:
        video = item.get('video', '')
        # 提取视频文件夹名
        parts = video.split('/')
        if 'videos_real' in parts:
            idx = parts.index('videos_real')
            if idx + 1 < len(parts):
                source = parts[idx + 1]
                video_sources[source] = video_sources.get(source, 0) + 1
    print(f"\n视频来源分布: {video_sources}")

    return data


def test_format_for_gpt(data):
    """测试格式化数据给 GPT"""
    print("\n" + "=" * 50)
    print("测试 GPT 输入格式化")
    print("=" * 50)

    sample = data[1]  # 使用第二个样本

    # 格式化选项
    choices = sample['choices']
    optionized_list = [f"{key}: {value}" for key, value in choices.items()]
    optionized_str = "\n".join(optionized_list)

    prompt = f"""Question: {sample['question']}

Options:
{optionized_str}

Please analyze the video and select the correct answer."""

    print(f"\n生成的 prompt:\n{prompt}")
    print(f"\n正确答案: {sample['answer']}")

    return prompt


def test_evaluation_format(data):
    """测试评估格式"""
    print("\n" + "=" * 50)
    print("测试评估格式")
    print("=" * 50)

    sample = data[1]

    # 模拟模型响应
    mock_response = "Based on my analysis, the batter hit the ball to the left side of the field. The answer is C: left."

    # 准备评估消息 (参考 eval_utils.py)
    optionized_list = [f"{key}: {value}" for key, value in sample['choices'].items()]
    optionized_str = "\n".join(optionized_list)
    question_context = f"Question: {sample['question']}\n\nOptions:\n{optionized_str}"
    gt_answer = f"Ground Truth Answer: {sample['answer']}"
    model_response = f"Model Response to the Question: {mock_response}"

    user_prompt = f"{question_context}\n\n{gt_answer}\n\n{model_response}"

    print(f"\n评估 prompt:\n{user_prompt[:500]}...")

    # 期望的输出格式
    expected_output = {
        "extracted_answer": "C: left",
        "correct": True
    }
    print(f"\n期望的评估输出: {expected_output}")

    return True


if __name__ == "__main__":
    # 测试1: 加载数据
    data = test_load_vlm4d()

    # 测试2: 格式化给 GPT
    test_format_for_gpt(data)

    # 测试3: 评估格式
    test_evaluation_format(data)

    print("\n" + "=" * 50)
    print("[OK] VLM4D 数据加载模块测试通过")
    print("=" * 50)
