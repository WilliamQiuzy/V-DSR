from transformers.utils import add_start_docstrings_to_model_forward, replace_return_docstrings
from transformers.activations import ACT2FN
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (Qwen2_5_VLForConditionalGeneration,Qwen2_5_VLConfig, Qwen2_5_VLPatchMerger,
                                                                Qwen2_5_VLCausalLMOutputWithPast,Qwen2_5_VLRotaryEmbedding, Qwen2_5_VisionTransformerPretrainedModel)
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch
import torch.nn.functional as F
from qwenvl.train.dinov2.hub.backbones import dinov2_vitl14_reg
import math
from typing import Any, Dict, List, Optional, Tuple, Union

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def obtain_rotary_pos_id_vision(grid_thw, second_per_grid_ts, spatial_merge_size):
    if isinstance(second_per_grid_ts, torch.Tensor) is False:
        second_per_grid_ts = torch.tensor(second_per_grid_ts, device=grid_thw.device)
    elif second_per_grid_ts.device != grid_thw.device:
        second_per_grid_ts = second_per_grid_ts.to(grid_thw.device)
    position_ids = []
    for i in range(grid_thw.shape[0]):
        t, h, w = grid_thw[i]
        llm_grid_t, llm_grid_h, llm_grid_w = (
            t.item(),
            h.item() // spatial_merge_size,
            w.item() // spatial_merge_size,
        )
        range_tensor = torch.arange(llm_grid_t).view(-1, 1)
        expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)
        expanded_range = expanded_range.to(grid_thw.device, second_per_grid_ts.dtype)
        second_per_grid_t = second_per_grid_ts[i]
        time_tensor = expanded_range * second_per_grid_t * 2

        time_tensor_long = time_tensor.long()
        t_index = time_tensor_long.flatten()

        h_index = (
            torch.arange(llm_grid_h)
            .view(1, -1, 1)
            .expand(llm_grid_t, -1, llm_grid_w)
            .flatten()
        ).to(t_index.device, t_index.dtype)
        w_index = (
            torch.arange(llm_grid_w)
            .view(1, 1, -1)
            .expand(llm_grid_t, llm_grid_h, -1)
            .flatten()
        ).to(t_index.device, t_index.dtype)
        position_ids.append(
                    torch.stack([t_index, h_index, w_index])
        )
    position_ids = torch.cat(position_ids,dim=1)
    return position_ids

def obtain_rotary_pos_id_text(text_lens):
    position_ids = []
    for item in text_lens:
        position_ids.append(
            torch.arange(item).view(1, -1).expand(3, -1)
        )
    position_ids = torch.cat(position_ids,dim=1)
    return position_ids

def obtain_rotary_pos_id_query(num):
    position_ids = []
    for item in range(num):
        position_ids.append(
            torch.arange(32).view(1, -1).expand(3, -1)
        )
    position_ids = torch.cat(position_ids,dim=1)
    return position_ids

def apply_multimodal_rotary_pos_emb(q, cos, sin, mrope_section, unsqueeze_dim=0):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors (https://qwenlm.github.io/blog/qwen2-vl/).

    Explanation:
        Multimodal 3D rotary position embedding is an extension to 1D rotary position embedding. The input embedding
        sequence contains vision (images / videos) embedding and text embedding or just contains text embedding. For
        vision embedding part, we apply rotary position embedding on temporal, height and width dimension separately.
        Here we split the channel dimension to 3 chunks for the temporal, height and width rotary position embedding.
        For text embedding part, we just apply 1D rotary position embedding. The three rotary position index (temporal,
        height and width) of text embedding is always the same, so the text embedding rotary position embedding has no
        difference with modern LLMs.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        mrope_section(`List(int)`):
            Multimodal rope section is for channel dimension of temporal, height and width in rope calculation.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    q_embed = (q * cos) + (rotate_half(q) * sin)
    return q_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, None, :, :].expand(num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(num_key_value_heads * n_rep, slen, head_dim)

class Qwen2_5_VLRotaryEmbedding_Mine(Qwen2_5_VLRotaryEmbedding):

    def forward(self, x, position_ids):
        # In contrast to other models, Qwen2_5_VL has different position ids for the grids
        # So we expand the inv_freq to shape (3, ...)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(3, -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()  # shape (3, 1, positions)
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class Qwen2_5_VLCrossAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: Qwen2_5_VLConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        self.rope_scaling = config.rope_scaling

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        target: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        query_position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        key_position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        q_len, _ = query.size()
        k_len, _ = target.size()
        query_states = self.q_proj(query)
        key_states = self.k_proj(target)
        value_states = self.v_proj(target)

        query_states = query_states.view(q_len, -1, self.head_dim).transpose(0, 1)
        key_states = key_states.view(k_len, -1, self.head_dim).transpose(0, 1)
        value_states = value_states.view(k_len, -1, self.head_dim).transpose(0, 1)

        cos_q, sin_q = query_position_embeddings
        query_states = apply_multimodal_rotary_pos_emb(
            query_states, cos_q, sin_q, self.rope_scaling["mrope_section"]
        )
        cos_k, sin_k = key_position_embeddings
        key_states = apply_multimodal_rotary_pos_emb(
            key_states, cos_k, sin_k, self.rope_scaling["mrope_section"]
        )
        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(1, 2)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask.unsqueeze(0).repeat(self.num_heads,1,1)
            attn_weights = attn_weights + causal_mask

        # Fix precision issues in Qwen2-VL float16 inference
        # Replace inf values with zeros in attention weights to prevent NaN propagation
        if query_states.dtype == torch.float16:
            attn_weights = torch.where(torch.isinf(attn_weights), torch.zeros_like(attn_weights), attn_weights)

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(0, 1).contiguous()
        attn_output = attn_output.reshape(q_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output

class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

class Qwen2_5_VLMLP(nn.Module):
    def __init__(self, config, bias: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))

class Qwen2_5_QFormer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.norm1 = Qwen2RMSNorm(config.hidden_size, eps=1e-6)
        self.norm2 = Qwen2RMSNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen2_5_VLCrossAttention(config)
        self.mlp = Qwen2_5_VLMLP(config, bias=True)

    def forward(
        self,
        query,
        target,
        attention_mask,
        query_position_embeddings,
        key_position_embeddings,
    ) -> torch.Tensor:
        query = query + self.attn(
            self.norm1(query),
            self.norm1(target),
            attention_mask,
            query_position_embeddings,
            key_position_embeddings,
        )
        query = query + self.mlp(self.norm2(query))
        return query

class Qwen2_5_VisionTransformerPretrainedModel_Spatial(Qwen2_5_VisionTransformerPretrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.spatial_encoder = dinov2_vitl14_reg(pretrained=False)
        spatial_dim = 1024
        self.spatial_merger = Qwen2_5_VLPatchMerger(
            dim = config.out_hidden_size,
            context_dim = spatial_dim*8,
            spatial_merge_size = 1,
        )
        del self.spatial_encoder.mask_token
        config.hidden_size = config.out_hidden_size
        self.q_former_1 = Qwen2_5_QFormer(config.config_all)
        self.q_former_2 = Qwen2_5_QFormer(config.config_all)
        self.q_former_norm = Qwen2RMSNorm(config.hidden_size)
        self.merge_size = config.spatial_merge_size
        self.rotary_emb = Qwen2_5_VLRotaryEmbedding_Mine(config=config.config_all)
        self.image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.before_mean = torch.tensor([0.48145466,0.4576275,0.40821073]).view(1,3,1,1)
        self.before_std = torch.tensor([0.26862954,0.26130258,0.27577711]).view(1,3,1,1)
        self.q_former_queries = nn.Parameter(torch.randn(32,config.out_hidden_size))
        self.temporal_patch_size = config.temporal_patch_size
        self.spatial_patch_size = config.spatial_patch_size
    def forward(self, hidden_states: torch.Tensor, text_states: torch.Tensor, grid_thw: torch.Tensor, text_length: torch.Tensor, second_per_grid_ts: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        vision_length = grid_thw.prod(-1)
        hidden_states_before_scale_all = []
        begin = 0
        for i in range(vision_length.shape[0]):
            hidden_states_before_scale = hidden_states[begin:(begin+vision_length[i])]
            hidden_states_before_scale = hidden_states_before_scale.view(grid_thw[i][0], grid_thw[i][1]//self.merge_size, grid_thw[i][2]//self.merge_size, self.merge_size, self.merge_size, 3, self.temporal_patch_size, self.spatial_patch_size, self.spatial_patch_size)
            hidden_states_before_scale = hidden_states_before_scale.permute(0,6,5,1,3,7,2,4,8)
            hidden_states_before_scale = hidden_states_before_scale.reshape(math.prod(hidden_states_before_scale.shape[0:2]),hidden_states_before_scale.shape[2],math.prod(hidden_states_before_scale.shape[3:6]),-1)
            hidden_states_before_scale = hidden_states_before_scale*self.before_std.to(device=hidden_states.device)+self.before_mean.to(device=hidden_states.device)
            hidden_states_before_scale = (hidden_states_before_scale-self.image_mean.to(device=hidden_states.device))/self.image_std.to(device=hidden_states.device)
            hidden_states_before_scale_all.append(hidden_states_before_scale)
            begin += vision_length[i]
        hidden_states_before_scale = torch.cat(hidden_states_before_scale_all)
        spatial_states = self.spatial_encoder(hidden_states_before_scale, is_training=True)
        if isinstance(spatial_states, dict):
            spatial_states = spatial_states["x_norm_patchtokens"]
        spatial_states = spatial_states.view(spatial_states.shape[0]//2, 2, grid_thw[0][1]//self.merge_size, self.merge_size, grid_thw[0][2]//self.merge_size, self.merge_size, spatial_states.shape[-1])
        spatial_states = spatial_states.permute(0,2,4,1,3,5,6)
        spatial_states = spatial_states.reshape(spatial_states.shape[0]*spatial_states.shape[1]*spatial_states.shape[2],-1)
        spatial_states = self.spatial_merger(spatial_states)
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens

            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens_now,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.merger(hidden_states)
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]
        vision_position_id = obtain_rotary_pos_id_vision(grid_thw, second_per_grid_ts, self.merge_size)
        text_position_id = obtain_rotary_pos_id_text(text_length)
        text_position_id = text_position_id.to(text_states.device, text_states.dtype)
        query_position_id = obtain_rotary_pos_id_query(len(text_length))
        query_position_id = query_position_id.to(text_states.device, text_states.dtype)
        attention_mask_1 = torch.full((len(text_length)*32, text_states.shape[0]), float('-inf'))
        attention_mask_1 = attention_mask_1.to(hidden_states.device, hidden_states.dtype)
        t_begin=0
        for i,t_length in enumerate(text_length):
            attention_mask_1[i*32:(i+1)*32, t_begin:(t_begin+t_length)] = 0
            t_begin += t_length
        text_position_embed = self.rotary_emb(text_states, text_position_id)
        query_position_embed = self.rotary_emb(self.q_former_queries, query_position_id)
        q_former_res_1 = self.q_former_1(self.q_former_queries, text_states, attention_mask_1, query_position_embed, text_position_embed)
        vision_position_embed = self.rotary_emb(spatial_states, vision_position_id)
        attention_mask_2 = torch.full((len(text_length)*32, spatial_states.shape[0]), float('-inf'))
        attention_mask_2 = attention_mask_2.to(hidden_states.device, hidden_states.dtype)
        v_begin=0
        for i,v_length in enumerate(grid_thw.prod(-1)):
            attention_mask_2[i*32:(i+1)*32, v_begin//4:(v_begin+v_length)//4] = 0
            v_begin += v_length
        q_former_res_2 = self.q_former_2(q_former_res_1, spatial_states, attention_mask_2, query_position_embed, vision_position_embed)
        q_former_res_2 = self.q_former_norm(q_former_res_2)
        hidden_states = torch.cat((hidden_states, q_former_res_2))
        return hidden_states

class Qwen2_5_VLForConditionalGeneration_Spatial(Qwen2_5_VLForConditionalGeneration):
    def __init__(self,config):
        super().__init__(config)
        config.vision_config.config_all = config
        self.visual = Qwen2_5_VisionTransformerPretrainedModel_Spatial._from_config(config.vision_config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
        non_text_tokens = [151652, 151653, 151654, 151655, 151656, 151643, 151644, 151645]
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            text_length = []
            text_embeds = []
            if attention_mask.dim()==2:
                attention_mask_visual = torch.sum(attention_mask,dim=1)
                attention_mask_visual = torch.cat((torch.tensor([0]).to(device=attention_mask_visual.device),attention_mask_visual))
                attention_mask_visual = torch.cumsum(attention_mask_visual, dim=0)
            else:
                attention_mask_visual = attention_mask
            for i in range(attention_mask_visual.shape[0]-1):
                text_id_tmp = input_ids[0, attention_mask_visual[i]:attention_mask_visual[i+1]]
                mask_id = torch.ones_like(text_id_tmp, dtype=torch.bool)
                if labels != None:
                    label_tmp = labels[0, attention_mask_visual[i]:attention_mask_visual[i+1]]
                    mask_id &= (label_tmp==-100)
                    int_mask = mask_id.to(torch.int)
                    last_position = torch.argmax(int_mask.flip(0))
                    last_position = len(mask_id)-last_position-1
                    mask_id[(last_position-2):(last_position+1)] = False
                for val in non_text_tokens:
                    mask_id &= (text_id_tmp != val)
                text_length.append(sum(mask_id))
                mask_id = mask_id.to(inputs_embeds.device)
                text_embeds.append(inputs_embeds[0,attention_mask_visual[i]:attention_mask_visual[i+1]][mask_id])
            text_embeds = torch.cat(text_embeds,dim=0)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, text_states = text_embeds, text_length = text_length, grid_thw = video_grid_thw, second_per_grid_ts = second_per_grid_ts)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    # In some inference pipelines, extra 32 geometry tokens are appended
                    # to video features without adding extra video tokens. Trim them.
                    if n_video_features == n_video_tokens + 32:
                        video_embeds = video_embeds[:n_video_tokens]
                        n_video_features = n_video_tokens
                    else:
                        raise ValueError(
                            f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                        )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )
