# DSR Model Architecture

> **Dynamic Spatial Reasoning (DSR)** 通过 Geometry Selection Module (GSM) 增强 VLM 的 3D 空间理解能力

---

## Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           整体架构                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   pixel_values_videos                                                    │
│   (原始视频像素)                                                          │
│         │                                                                │
│         ▼                                                                │
│   ┌───────────────────────────────┐                                      │
│   │  Qwen Vision 预处理            │                                      │
│   │  (flatten + 归一化)            │                                      │
│   └───────────────┬───────────────┘                                      │
│                   │                                                      │
│                   ▼                                                      │
│         hidden_states                                                    │
│         (flattened patch tokens)                                         │
│         shape: [N_vis, C]                                                │
│                   │                                                      │
│     ┌─────────────┴─────────────┐                                        │
│     │                           │                                        │
│     ▼                           ▼                                        │
│ ┌─────────────────┐     ┌─────────────────┐                              │
│ │  Vision 分支     │     │  Geometry 分支   │                              │
│ │  (Qwen ViT)     │     │  (DINOv2 + π³)  │                              │
│ └────────┬────────┘     └────────┬────────┘                              │
│          │                       │                                       │
│          ▼                       ▼                                       │
│   vision_tokens           geometry_tokens                                │
│   [N_vis, 3584]           [32, 3584] (per sample)                        │
│          │                       │                                       │
│          └───────────┬───────────┘                                       │
│                      │                                                   │
│                      ▼                                                   │
│            torch.cat([vision, geometry])                                 │
│            [N_vis + 32, 3584]                                            │
│                      │                                                   │
│                      ▼                                                   │
│              ┌──────────────┐                                            │
│              │   Qwen LLM   │                                            │
│              └──────────────┘                                            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Key Components

| 组件 | 输入 | 输出 | 作用 |
|------|------|------|------|
| **Qwen Vision 预处理** | pixel_values | hidden_states | flatten + 归一化 |
| **Qwen ViT** | hidden_states | vision_tokens [N_vis, 3584] | 视觉语义特征 |
| **DINOv2 (π³)** | 重构的像素 | spatial_states [N_patches, 1024] | 3D 几何特征 |
| **spatial_merger** | [N, 1024×8] | [N/4, 3584] | 融合 2×2 空间 + 2 时间 |
| **Q-Former 1** | queries + text | [32, 3584] | 问题语义编码 |
| **Q-Former 2** | Q1_out + spatial | [32, 3584] | 几何特征选择 |

---

## Detailed Data Flow

### Vision 分支 (Qwen ViT)

```
hidden_states [N_vis, C]
       │
       ▼
┌──────────────────────────────┐
│  patch_embed                 │
│  像素 patch → token 表示      │
└──────────────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│  rotary_pos_emb              │
│  添加旋转位置编码             │
└──────────────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│  window_index 重排           │
│  窗口注意力准备               │
└──────────────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│  Qwen ViT Blocks × N         │
│  Transformer 编码            │
└──────────────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│  merger                      │
│  patch tokens 合并           │
└──────────────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│  reverse window order        │
│  恢复原始顺序                 │
└──────────────────────────────┘
       │
       ▼
vision_tokens [N_vis, 3584]
```

### Geometry 分支 (DINOv2 + Q-Former)

```
hidden_states [N_vis, C]
       │
       ▼
┌──────────────────────────────┐
│  reshape + permute           │
│  重构为 [N, 3, H, W] 像素布局 │
└──────────────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│  反归一化                     │
│  * before_std + before_mean  │
│  Qwen 格式 → 原始像素范围     │
└──────────────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│  重新归一化                   │
│  (x - mean) / std            │
│  原始像素 → DINOv2 格式       │
└──────────────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│  DINOv2 Encoder (π³ weights) │
│  spatial_encoder             │
│  输出: x_norm_patchtokens    │
└──────────────────────────────┘
       │
       ▼
spatial_states [N_patches, 1024]
       │
       ▼
┌──────────────────────────────┐
│  reshape + permute           │
│  重排 spatial tokens         │
└──────────────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│  spatial_merger              │
│  融合 2×2 空间 + 2 时间       │
│  1024 × 8 → 3584             │
└──────────────────────────────┘
       │
       ▼
spatial_states [N_patches/4, 3584]
       │
       │
       │    ┌─────────────────────────────────────┐
       │    │          Q-Former 1                 │
       │    │  ┌─────────────────────────────┐    │
       │    │  │ queries [32, 3584]          │    │
       │    │  │     │                       │    │
       │    │  │     ▼                       │    │
       │    │  │ Cross-Attention             │    │
       │    │  │     ▲                       │    │
       │    │  │     │                       │    │
       │    │  │ text_embeds (问题文本)       │    │
       │    │  │     │                       │    │
       │    │  │     ▼                       │    │
       │    │  │ q_former_res_1 [32, 3584]   │    │
       │    │  └─────────────────────────────┘    │
       │    └─────────────────────────────────────┘
       │                    │
       ▼                    ▼
┌─────────────────────────────────────────────────┐
│                Q-Former 2                        │
│  ┌───────────────────────────────────────────┐  │
│  │ q_former_res_1 [32, 3584]                 │  │
│  │     │                                     │  │
│  │     ▼                                     │  │
│  │ Cross-Attention ◄── spatial_states       │  │
│  │     │               (3D 几何特征)          │  │
│  │     ▼                                     │  │
│  │ q_former_res_2 [32, 3584]                 │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│  q_former_norm (RMSNorm)     │
└──────────────────────────────┘
       │
       ▼
geometry_tokens [32, 3584] (per sample)
```

### Fusion

```python
# 拼接 vision 和 geometry tokens
output = torch.cat([vision_tokens, geometry_tokens])
# shape: [N_vis + 32, 3584]
# 返回给 Qwen LLM 继续处理
```

---

## Q-Former Details

### Q-Former 1: Semantic Condenser

| 属性 | 值 |
|------|-----|
| **Query** | 32 个可学习的 embeddings (`self.q_former_queries`) |
| **Key/Value** | 问题文本的 embeddings (`text_embeds`) |
| **输出** | 问题语义增强的 queries [32×B, 3584] |

### Q-Former 2: Geometry Selector

| 属性 | 值 |
|------|-----|
| **Query** | Q-Former 1 的输出 (`q_former_res_1`) |
| **Key/Value** | 几何特征 (`spatial_states`) |
| **输出** | 与问题相关的几何 tokens [32×B, 3584] |

> **Note**: Q-Former 的形状是 batch 合并后的 `[32 * B, hidden]`，上图 `[32, 3584]` 表示单样本。

---

## Code Reference

源文件: `src/model/qwen_vl_finetune/train/qwen_vl_spatial.py`

```python
class Qwen2_5_VisionTransformerPretrainedModel_Spatial:
    def __init__(self, config):
        # DINOv2 encoder (加载 π³ 权重)
        self.spatial_encoder = dinov2_vitl14_reg(pretrained=False)

        # 维度转换: 融合 2×2 空间 + 2 时间
        self.spatial_merger = Qwen2_5_VLPatchMerger(
            dim=config.out_hidden_size,        # 3584
            context_dim=spatial_dim * 8,       # 1024 * 8 = 8192
            spatial_merge_size=1,
        )

        # 两个 Q-Former
        self.q_former_1 = Qwen2_5_QFormer(config)  # 语义压缩
        self.q_former_2 = Qwen2_5_QFormer(config)  # 几何选择

        # 32 个可学习 queries
        self.q_former_queries = nn.Parameter(torch.randn(32, 3584))
```

---

## Notes

1. **输入处理**
   - `pixel_values_videos` 是原始视频像素
   - 经 Qwen Vision 预处理后变成 `hidden_states` (flattened patch tokens)
   - 两个分支共享同一个 `hidden_states` 作为输入

2. **归一化转换** (Geometry 分支)
   - Qwen 格式 → 原始像素 (反归一化: `* std + mean`)
   - 原始像素 → DINOv2 格式 (重新归一化: `(x - mean) / std`)

3. **spatial_merger**
   - 融合 2×2 空间 + 2 时间 = 8 个 patch
   - 维度: 1024 × 8 = 8192 → 投影到 3584

4. **权重来源**
   - Qwen ViT: Qwen2.5-VL 预训练权重
   - DINOv2: π³ 预训练权重 (3D 任务训练)

5. **Token 数量**
   - Vision tokens: 取决于图像分辨率 (N_vis)
   - Geometry tokens: 固定 32 个 (per sample)
