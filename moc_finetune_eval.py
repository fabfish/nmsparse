#!/usr/bin/env python3
"""
Mixture-of-Channels (MoC) Fine-tuning and Zero-Shot Evaluation Script

Based on: "Mixture-of-Channels: Exploiting Sparse FFNs for Efficient LLMs Pre-Training and Inference"
arXiv:2511.09323v1 (https://arxiv.org/abs/2511.09323)

Implementation review vs paper:
- Paper: MoC selectively activates Top-K most relevant channels per token via SwiGLU's gating.
  Reduces FFN activation memory and improves inference (partial weight loading to SRAM).
- This implementation:
  - Top-K on gate logits (largest values, not absolute) before SiLU, matching Section 3.2.
  - MoCSwiGLUFFN: Gate/Up proj -> Top-K mask -> SiLU(G') * U' -> Down proj (Figure 2).
  - Custom TopKMaskSelector autograd passes gradient through mask (straight-through style).
  - Optional forward_memory_efficient + gradient checkpointing for training (GCP-style).
  - replace_mlp_with_moc replaces all Llama MLPs with MoC FFN; K=num_channels (e.g. 2048).
- Baseline comparison: test_rte_sparsity.py (original vs 2:4 sparse activation). This script
  hosts the same task process: original (no MoC) vs MoC on the same benchmarks, with timing,
  results table, JSON export, and visualization.

Usage:
    export HF_ENDPOINT=https://hf-mirror.com
    export HF_TOKEN=your_huggingface_token
    python moc_finetune_eval.py --mode eval --model_path /path/to/model
    python moc_finetune_eval.py --mode finetune --dataset rte --epochs 3
"""

import os
import sys
import json
import copy
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Union
from tqdm import tqdm
from collections import defaultdict
import subprocess
import warnings

# Set HuggingFace mirror endpoint
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Load HF_TOKEN from local file if exists
def load_hf_token(token_file: str = ".hf_token") -> Optional[str]:
    """Load HuggingFace token from local file."""
    token_path = Path(__file__).parent / token_file
    if token_path.exists():
        try:
            with open(token_path, 'r') as f:
                token = f.read().strip()
                if token:
                    return token
        except Exception as e:
            print(f"Warning: Could not read token file {token_path}: {e}")
    return None

hf_token = load_hf_token()
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    default_data_collator
)
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, PeftModel, TaskType


# ============================================================================
# Mixture-of-Channels (MoC) Implementation
# ============================================================================

class MoCConfig:
    """MoC config. sparsity_pattern: 'topk' (global K) | '2:4' | '2:8' (block-wise keep 2 of 4 or 8)."""
    def __init__(
        self,
        hidden_size: int = 4096,
        intermediate_size: int = 14336,
        num_channels: int = 3584,  # used for topk (default 25% of 14336)
        use_gradient_checkpointing: bool = True,
        training_mode: bool = False,
        sparsity_pattern: str = "topk",  # "topk" | "2:4" | "2:8"
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_channels = num_channels
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.training_mode = training_mode
        self.sparsity_pattern = sparsity_pattern
        assert sparsity_pattern in ("topk", "2:4", "2:8")
        if sparsity_pattern == "topk":
            assert self.num_channels <= self.intermediate_size
        elif sparsity_pattern == "2:4":
            assert self.intermediate_size % 4 == 0
        else:  # 2:8
            assert self.intermediate_size % 8 == 0


class TopKMaskSelector(torch.autograd.Function):
    """
    自定义Top-K选择算子（带梯度）
    关键：在SiLU之前选择Top-K，基于gate projection的输出
    对应论文Section 3.2
    """
    @staticmethod
    def forward(ctx, gate_logits: torch.Tensor, k: int):
        """
        Args:
            gate_logits: [batch_size, seq_len, intermediate_size]
            k: 选择的通道数
        Returns:
            mask: 二进制掩码 [batch_size, seq_len, intermediate_size]
        """
        # 获取Top-K最大值的索引（按值选择，不是绝对值）
        # 论文: "retains the top-K largest values (not the largest in absolute value)"
        topk_values, topk_indices = torch.topk(gate_logits, k, dim=-1, sorted=False)
        
        # 创建二进制掩码
        mask = torch.zeros_like(gate_logits)
        mask.scatter_(-1, topk_indices, 1.0)
        
        # 保存用于反向传播
        ctx.save_for_backward(mask)
        ctx.k = k
        
        return mask, topk_indices

    @staticmethod
    def backward(ctx, grad_mask, grad_indices):
        mask, = ctx.saved_tensors
        return grad_mask * mask, None


def _moc_mask_block_wise(gate_logits: torch.Tensor, block_size: int, keep: int = 2) -> torch.Tensor:
    """
    Block-wise mask: for each block of `block_size` channels, keep the top `keep` by value (not abs).
    gate_logits: [B, S, inter], inter must be divisible by block_size.
    Returns mask [B, S, inter].
    """
    B, S, inter = gate_logits.shape
    assert inter % block_size == 0
    g = gate_logits.view(B, S, inter // block_size, block_size)
    _, idx = g.topk(keep, dim=-1, sorted=False)  # [B, S, n_blocks, keep]
    mask = torch.zeros_like(g)
    mask.scatter_(-1, idx, 1.0)
    return mask.view(B, S, inter)


class MoCSwiGLUFFN(nn.Module):
    """
    Mixture-of-Channels SwiGLU FFN
    替换标准Transformer中的FFN层，实现Top-K通道稀疏
    
    对应论文Figure 2的架构：
    - Gate Projection -> Top-K选择 -> SiLU -> 与Up Projection相乘 -> Down Projection
    """
    
    def __init__(self, original_mlp: nn.Module, config: MoCConfig):
        """
        从现有的MLP模块初始化MoC FFN
        
        Args:
            original_mlp: 原始的SwiGLU MLP模块（通常包含gate_proj, up_proj, down_proj）
            config: MoC配置
        """
        super().__init__()
        self.config = config
        
        # 复制原始MLP的投影层
        # 标准LlamaMLP结构: gate_proj, up_proj, down_proj
        self.gate_proj = original_mlp.gate_proj  # [hidden_size, intermediate_size]
        self.up_proj = original_mlp.up_proj      # [hidden_size, intermediate_size]
        self.down_proj = original_mlp.down_proj  # [intermediate_size, hidden_size]
        self.act_fn = F.silu  # SwiGLU使用SiLU激活
        
        # 预激活函数（LlamaMLP通常包含）
        self.pretraining_tp = getattr(original_mlp, 'pretraining_tp', 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        MoC前向传播流程（对应论文Figure 2）：
        
        1. 计算Gate和Up投影: G = X @ W_gate, U = X @ W_up
        2. Top-K选择: 基于G选择Top-K通道，生成掩码M
        3. 稀疏化: G' = G ⊙ M, U' = U ⊙ M
        4. SiLU激活: S = SiLU(G')
        5. 元素乘法: Z = S ⊙ U'
        6. Down投影: Output = Z @ W_down
        
        Args:
            x: [batch_size, seq_len, hidden_size]
        Returns:
            output: [batch_size, seq_len, hidden_size]
        """
        # Step 1: Gate和Up投影
        # G = X @ W_gate, U = X @ W_up
        g = self.gate_proj(x)  # [B, S, intermediate_size]
        u = self.up_proj(x)    # [B, S, intermediate_size]
        
        # Step 2: Channel selection by sparsity_pattern (topk | 2:4 | 2:8)
        pattern = getattr(self.config, "sparsity_pattern", "topk")
        if pattern == "topk":
            k = self.config.num_channels
            if self.training and self.config.training_mode:
                mask, _ = TopKMaskSelector.apply(g, k)
            else:
                with torch.no_grad():
                    _, topk_indices = torch.topk(g, k, dim=-1, sorted=False)
                    mask = torch.zeros_like(g)
                    mask.scatter_(-1, topk_indices, 1.0)
        elif pattern == "2:4":
            mask = _moc_mask_block_wise(g, 4, 2)
        elif pattern == "2:8":
            mask = _moc_mask_block_wise(g, 8, 2)
        else:
            raise ValueError(f"Unknown sparsity_pattern: {pattern}")
        
        # Step 3: 应用掩码 - 只保留Top-K通道（稀疏化）
        # G' = G ⊙ M, U' = U ⊙ M
        g_sparse = g * mask  # [B, S, intermediate_size]
        u_sparse = u * mask  # [B, S, intermediate_size]
        
        # Step 4: SiLU激活（只计算选中通道）
        # S = SiLU(G')
        s = self.act_fn(g_sparse)
        
        # Step 5: 元素乘法（SwiGLU的核心）
        # Z = S ⊙ U'
        z = s * u_sparse  # [B, S, intermediate_size]
        
        # Step 6: Down投影
        # Output = Z @ W_down
        # 注意：这里我们使用完整的down_proj，但z是稀疏的
        output = self.down_proj(z)  # [B, S, hidden_size]
        
        return output
    
    def forward_memory_efficient(self, x: torch.Tensor) -> torch.Tensor:
        """
        内存高效版本（训练时使用，对应论文Table 1的MoC+GCP）
        
        关键优化：
        - 只存储G⊙M, U⊙M, M（而不是完整的G, U, S, Z）
        - S和Z在反向传播时重新计算
        """
        # 前向：计算并稀疏化
        g = self.gate_proj(x)
        u = self.up_proj(x)
        
        pattern = getattr(self.config, "sparsity_pattern", "topk")
        if pattern == "topk":
            mask, _ = TopKMaskSelector.apply(g, self.config.num_channels)
        elif pattern == "2:4":
            mask = _moc_mask_block_wise(g, 4, 2)
        else:
            mask = _moc_mask_block_wise(g, 8, 2)
        
        # 只保存稀疏化的激活（关键内存节省）
        g_sparse = g * mask
        
        # 使用gradient checkpointing重新计算S和Z
        if self.config.use_gradient_checkpointing and self.training:
            output = torch.utils.checkpoint.checkpoint(
                self._forward_from_sparse, g_sparse, u, mask
            )
        else:
            u_sparse = u * mask
            s = self.act_fn(g_sparse)
            z = s * u_sparse
            output = self.down_proj(z)
        
        return output
    
    def _forward_from_sparse(self, g_sparse, u, mask):
        """从稀疏激活重新计算前向传播"""
        u_sparse = u * mask
        s = self.act_fn(g_sparse)
        z = s * u_sparse
        return self.down_proj(z)


def replace_mlp_with_moc(
    model: nn.Module,
    config: MoCConfig,
    target_layers: Optional[List[int]] = None
) -> nn.Module:
    """
    将模型中的MLP层替换为MoC FFN
    
    Args:
        model: 原始模型
        config: MoC配置
        target_layers: 指定要替换的层索引，None表示替换所有层
        
    Returns:
        修改后的模型
    """
    # 对于Llama模型，MLP通常在model.layers[i].mlp
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    elif hasattr(model, 'layers'):
        layers = model.layers
    else:
        raise ValueError("Model structure not recognized. Expected model.model.layers or model.layers")
    
    num_replaced = 0
    for idx, layer in enumerate(layers):
        if target_layers is not None and idx not in target_layers:
            continue
            
        if hasattr(layer, 'mlp'):
            original_mlp = layer.mlp
            # 创建MoC FFN替换原始MLP
            moc_mlp = MoCSwiGLUFFN(original_mlp, config)
            layer.mlp = moc_mlp
            num_replaced += 1
    
    print(f"Replaced {num_replaced} MLP layers with MoC FFN (K={config.num_channels})")
    return model


def count_moc_layers(model: nn.Module) -> Dict[str, int]:
    """统计模型中MoC层和标准MLP层的数量"""
    moc_count = 0
    standard_count = 0
    
    for module in model.modules():
        if isinstance(module, MoCSwiGLUFFN):
            moc_count += 1
        # 检查是否是标准的LlamaMLP（通过属性判断）
        elif hasattr(module, 'gate_proj') and hasattr(module, 'up_proj') and hasattr(module, 'down_proj'):
            if not isinstance(module, MoCSwiGLUFFN):
                standard_count += 1
    
    return {'moc': moc_count, 'standard_mlp': standard_count}


# ============================================================================
# Activation sparsity: 2:4, 2:8, topk (for sparse_all / sparse_ffn)
# ============================================================================

def apply_2_4_sparsity(x: torch.Tensor) -> torch.Tensor:
    """Keep top-2 of every 4 elements by magnitude."""
    original_shape = x.shape
    last_dim = original_shape[-1]
    if last_dim % 4 != 0:
        pad_size = 4 - (last_dim % 4)
        x = F.pad(x, (0, pad_size), mode="constant", value=0)
        padded = True
    else:
        padded = False
    new_shape = x.shape[:-1] + (x.shape[-1] // 4, 4)
    x_reshaped = x.view(new_shape)
    _, top2_indices = x_reshaped.abs().topk(k=2, dim=-1)
    mask = torch.zeros_like(x_reshaped)
    mask.scatter_(-1, top2_indices, 1.0)
    sparse_x = (x_reshaped * mask).view(x.shape)
    if padded:
        sparse_x = sparse_x[..., :last_dim]
    return sparse_x


def apply_2_8_sparsity(x: torch.Tensor) -> torch.Tensor:
    """Keep top-2 of every 8 elements by magnitude."""
    last_dim = x.shape[-1]
    if last_dim % 8 != 0:
        pad_size = 8 - (last_dim % 8)
        x = F.pad(x, (0, pad_size), mode="constant", value=0)
        padded = True
    else:
        padded = False
    new_shape = x.shape[:-1] + (x.shape[-1] // 8, 8)
    x_reshaped = x.view(new_shape)
    _, top2_indices = x_reshaped.abs().topk(k=2, dim=-1)
    mask = torch.zeros_like(x_reshaped)
    mask.scatter_(-1, top2_indices, 1.0)
    sparse_x = (x_reshaped * mask).view(x.shape)
    if padded:
        sparse_x = sparse_x[..., :last_dim]
    return sparse_x


def apply_topk_activation_sparsity(x: torch.Tensor, k: int) -> torch.Tensor:
    """Keep top-k elements by magnitude along last dimension."""
    dim = x.shape[-1]
    k = min(k, dim)
    _, idx = x.abs().topk(k, dim=-1, sorted=False)
    mask = torch.zeros_like(x)
    mask.scatter_(-1, idx, 1.0)
    return x * mask


class SparseActivationLinear(nn.Module):
    """Linear wrapper that applies activation sparsity: pattern '2:4' | '2:8' | 'topk' (topk_ratio for topk)."""
    def __init__(self, original_linear: nn.Linear, pattern: str = "2:4", topk_ratio: float = 0.5):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.weight = original_linear.weight
        self.bias = original_linear.bias
        self.pattern = pattern
        self.topk_ratio = topk_ratio  # for pattern "topk": keep in_features * topk_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pattern == "2:4":
            sparse_x = apply_2_4_sparsity(x)
        elif self.pattern == "2:8":
            sparse_x = apply_2_8_sparsity(x)
        else:  # topk
            k = max(1, int(x.shape[-1] * self.topk_ratio))
            sparse_x = apply_topk_activation_sparsity(x, k)
        return F.linear(sparse_x, self.weight, self.bias)


def replace_linear_with_sparse(
    model: nn.Module,
    exclude_names: Optional[List[str]] = None,
    pattern: str = "2:4",
    topk_ratio: float = 0.5,
) -> nn.Module:
    """Replace nn.Linear with SparseActivationLinear (exclude lm_head, embed). pattern: 2:4 | 2:8 | topk."""
    if exclude_names is None:
        exclude_names = ["lm_head", "embed"]

    def should_replace(name: str) -> bool:
        for p in exclude_names:
            if p in name:
                return False
        return True

    def replace_recursive(module: nn.Module, prefix: str = "") -> None:
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and should_replace(full_name):
                setattr(module, name, SparseActivationLinear(child, pattern=pattern, topk_ratio=topk_ratio))
            else:
                replace_recursive(child, full_name)

    replace_recursive(model)
    return model


def replace_linear_with_sparse_ffn_only(
    model: nn.Module,
    pattern: str = "2:4",
    topk_ratio: float = 0.5,
) -> nn.Module:
    """Replace only FFN Linear layers with SparseActivationLinear. pattern: 2:4 | 2:8 | topk."""
    def replace_recursive(module: nn.Module, prefix: str = "") -> None:
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and "mlp" in full_name and (
                "gate_proj" in full_name or "up_proj" in full_name or "down_proj" in full_name
            ):
                setattr(module, name, SparseActivationLinear(child, pattern=pattern, topk_ratio=topk_ratio))
            else:
                replace_recursive(child, full_name)
    replace_recursive(model)
    return model


# ============================================================================
# PRAC: Principal-Random Activation Compression (arXiv:2602.23111)
#
# Verified against paper (Li, Zhang, Fang 2026):
# - Q1 via truncated SVD (top-r1 right singular vectors)  [Section 4.2]
# - Q2 sampled uniformly from orth. complement of Q1      [Section 4.2]
# - Reconstruction: X_hat = XQ1 Q1^T + k * XQ2 Q2^T      [Section 4.3]
# - Scaling factor k = (n - r1) / r2  (unbiased, min-variance) [Thm + Fig 7]
#
# For zero-shot comparison with element-wise sparsity methods, PRAC's
# compress->decompress pipeline is applied during the forward pass:
#   x  ->  project to subspace  ->  reconstruct x_hat  ->  linear(x_hat)
# ============================================================================

class PRACCompressor:
    """
    PRAC activation compressor (self-contained, from prac/src/prac.py).

    Algorithm:
    1. SVD on activation X -> principal subspace Q1 (top-r1 right singular vectors)
    2. Sample Q2 uniformly from the orthogonal complement of Q1 (via QR)
    3. Compress: store XQ1 [B,S,r1] and XQ2 [B,S,r2]
    4. Reconstruct: X_hat = XQ1 @ Q1^T + k * XQ2 @ Q2^T,  k = (n-r1)/r2
    """

    def __init__(self, feature_dim: int, principal_rank: float = 0.3,
                 random_rank: float = 0.3, scaling_factor: Optional[float] = None,
                 update_freq: int = 1000):
        self.feature_dim = feature_dim
        self.r1 = max(1, int(feature_dim * principal_rank))
        self.r2 = max(1, int(feature_dim * random_rank))
        self.k = scaling_factor if scaling_factor is not None else (feature_dim - self.r1) / self.r2
        self.update_freq = update_freq
        self.Q1: Optional[torch.Tensor] = None
        self.Q2: Optional[torch.Tensor] = None
        self.step_count = 0

    def update_subspaces(self, activations: torch.Tensor) -> None:
        should_update = (self.step_count % self.update_freq == 0
                         or self.Q1 is None)
        if should_update:
            self._update_principal(activations)
            self._update_random()
        self.step_count += 1

    def _update_principal(self, activations: torch.Tensor) -> None:
        X = activations.reshape(-1, activations.shape[-1]) if activations.dim() == 3 else activations
        orig_dtype = activations.dtype
        try:
            _, _, Vh = torch.linalg.svd(X.float(), full_matrices=False)
            self.Q1 = Vh[:self.r1, :].T.contiguous().detach().to(orig_dtype)
        except Exception:
            Q = torch.randn(self.feature_dim, self.r1,
                            device=activations.device, dtype=torch.float32)
            self.Q1, _ = torch.linalg.qr(Q)
            self.Q1 = self.Q1.to(orig_dtype)

    def _update_random(self) -> None:
        device, dtype = self.Q1.device, self.Q1.dtype
        Q1f = self.Q1.float()
        Z = torch.randn(self.feature_dim, self.r2, device=device, dtype=torch.float32)
        Z = Z - Q1f @ (Q1f.T @ Z)
        Q2f, _ = torch.linalg.qr(Z)
        self.Q2 = Q2f.to(dtype)

    def compress_decompress(self, x: torch.Tensor) -> torch.Tensor:
        """Compress then immediately decompress: x -> x_hat (lossy round-trip)."""
        if self.Q1 is None or self.Q2 is None:
            self.update_subspaces(x)
        Q1 = self.Q1.to(device=x.device, dtype=x.dtype)
        Q2 = self.Q2.to(device=x.device, dtype=x.dtype)
        XQ1 = x @ Q1
        XQ2 = x @ Q2
        return XQ1 @ Q1.T + self.k * (XQ2 @ Q2.T)

    def retention_ratio(self) -> float:
        return (self.r1 + self.r2) / self.feature_dim


class PRACActivationLinear(nn.Module):
    """Linear wrapper that applies PRAC subspace compression to input activations.

    Analogous to SparseActivationLinear but uses principal+random subspace projection
    instead of element-wise sparsity. Shares weights from the original linear (no copy).
    """

    def __init__(self, original_linear: nn.Linear, principal_rank: float = 0.25,
                 random_rank: float = 0.25, update_freq: int = 1000,
                 min_dim: int = 64):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.weight = original_linear.weight
        self.bias = original_linear.bias
        self.principal_rank = principal_rank
        self.random_rank = random_rank
        self.compressor: Optional[PRACCompressor] = None
        if self.in_features >= min_dim:
            self.compressor = PRACCompressor(
                self.in_features, principal_rank, random_rank,
                update_freq=update_freq,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.compressor is not None:
            with torch.no_grad():
                self.compressor.update_subspaces(x.detach())
            x = self.compressor.compress_decompress(x)
        return F.linear(x, self.weight, self.bias)


def replace_linear_with_prac(
    model: nn.Module,
    principal_rank: float = 0.25,
    random_rank: float = 0.25,
    exclude_names: Optional[List[str]] = None,
    update_freq: int = 1000,
    min_dim: int = 64,
) -> nn.Module:
    """Replace nn.Linear with PRACActivationLinear (exclude lm_head, embed)."""
    if exclude_names is None:
        exclude_names = ["lm_head", "embed"]

    def should_replace(name: str) -> bool:
        return not any(p in name for p in exclude_names)

    def replace_recursive(module: nn.Module, prefix: str = "") -> None:
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and should_replace(full_name):
                setattr(module, name, PRACActivationLinear(
                    child, principal_rank, random_rank, update_freq, min_dim))
            else:
                replace_recursive(child, full_name)

    replace_recursive(model)
    n = sum(1 for m in model.modules() if isinstance(m, PRACActivationLinear))
    print(f"Replaced {n} Linear layers with PRACActivationLinear "
          f"(r1={principal_rank}, r2={random_rank}, retain {(principal_rank+random_rank)*100:.0f}%)")
    return model


def replace_linear_with_prac_ffn_only(
    model: nn.Module,
    principal_rank: float = 0.25,
    random_rank: float = 0.25,
    update_freq: int = 1000,
    min_dim: int = 64,
) -> nn.Module:
    """Replace only FFN Linear layers (gate_proj, up_proj, down_proj) with PRACActivationLinear."""
    def replace_recursive(module: nn.Module, prefix: str = "") -> None:
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if (isinstance(child, nn.Linear) and "mlp" in full_name
                    and any(k in full_name for k in ("gate_proj", "up_proj", "down_proj"))):
                setattr(module, name, PRACActivationLinear(
                    child, principal_rank, random_rank, update_freq, min_dim))
            else:
                replace_recursive(child, full_name)

    replace_recursive(model)
    n = sum(1 for m in model.modules() if isinstance(m, PRACActivationLinear))
    print(f"Replaced {n} FFN Linear layers with PRACActivationLinear "
          f"(r1={principal_rank}, r2={random_rank}, retain {(principal_rank+random_rank)*100:.0f}%)")
    return model


# ============================================================================
# Sparsity logic table: pattern (topk, topk_50, 2:4, 2:8) x method (base, sparse_all, sparse_ffn, moc, moc+sparse, prac)
# ============================================================================

# topk default: 25% of inter (e.g. K=3584 for 14336); topk_50: 50% channels
def _moc_topk_sparsity_str(num_channels: int, intermediate_size: int) -> str:
    pct = 100.0 * num_channels / intermediate_size if intermediate_size else 0
    return f"K={num_channels} ({pct:.1f}%)"

SPARSITY_LOGIC_TABLE: Dict[str, Dict[str, Any]] = {
    "original":          {"desc": "Base (no sparsity)",              "sparse_all": False, "sparse_ffn": False, "sparse_pattern": None, "sparse_topk_ratio": None, "moc_pattern": None, "moc_topk_ratio": None},
    "sparse_all_topk":    {"desc": "sparse_all Top-K 25%",            "sparse_all": True,  "sparse_ffn": False, "sparse_pattern": "topk", "sparse_topk_ratio": 0.25, "moc_pattern": None, "moc_topk_ratio": None},
    "sparse_all_topk_50": {"desc": "sparse_all Top-K 50%",            "sparse_all": True,  "sparse_ffn": False, "sparse_pattern": "topk", "sparse_topk_ratio": 0.5,  "moc_pattern": None, "moc_topk_ratio": None},
    "sparse_all_2_4":     {"desc": "sparse_all 2:4",                  "sparse_all": True,  "sparse_ffn": False, "sparse_pattern": "2:4",  "sparse_topk_ratio": None, "moc_pattern": None, "moc_topk_ratio": None},
    "sparse_all_2_8":     {"desc": "sparse_all 2:8",                  "sparse_all": True,  "sparse_ffn": False, "sparse_pattern": "2:8",  "sparse_topk_ratio": None, "moc_pattern": None, "moc_topk_ratio": None},
    "sparse_ffn_topk":    {"desc": "sparse_ffn Top-K 25%",            "sparse_all": False, "sparse_ffn": True,  "sparse_pattern": "topk", "sparse_topk_ratio": 0.25, "moc_pattern": None, "moc_topk_ratio": None},
    "sparse_ffn_topk_50": {"desc": "sparse_ffn Top-K 50%",            "sparse_all": False, "sparse_ffn": True,  "sparse_pattern": "topk", "sparse_topk_ratio": 0.5,  "moc_pattern": None, "moc_topk_ratio": None},
    "sparse_ffn_2_4":    {"desc": "sparse_ffn 2:4",                  "sparse_all": False, "sparse_ffn": True,  "sparse_pattern": "2:4",  "sparse_topk_ratio": None, "moc_pattern": None, "moc_topk_ratio": None},
    "sparse_ffn_2_8":    {"desc": "sparse_ffn 2:8",                  "sparse_all": False, "sparse_ffn": True,  "sparse_pattern": "2:8",  "sparse_topk_ratio": None, "moc_pattern": None, "moc_topk_ratio": None},
    "moc_topk":           {"desc": "MoC Top-K 25% (FFN)",             "sparse_all": False, "sparse_ffn": False, "sparse_pattern": None, "sparse_topk_ratio": None, "moc_pattern": "topk", "moc_topk_ratio": 0.25},
    "moc_topk_50":        {"desc": "MoC Top-K 50% (FFN)",             "sparse_all": False, "sparse_ffn": False, "sparse_pattern": None, "sparse_topk_ratio": None, "moc_pattern": "topk", "moc_topk_ratio": 0.5},
    "moc_2_4":            {"desc": "MoC 2:4 (FFN)",                  "sparse_all": False, "sparse_ffn": False, "sparse_pattern": None, "sparse_topk_ratio": None, "moc_pattern": "2:4", "moc_topk_ratio": None},
    "moc_2_8":            {"desc": "MoC 2:8 (FFN)",                  "sparse_all": False, "sparse_ffn": False, "sparse_pattern": None, "sparse_topk_ratio": None, "moc_pattern": "2:8", "moc_topk_ratio": None},
    "moc_topk_sparse":    {"desc": "MoC Top-K 25% + sparse_all",     "sparse_all": True,  "sparse_ffn": False, "sparse_pattern": "2:4", "sparse_topk_ratio": None, "moc_pattern": "topk", "moc_topk_ratio": 0.25},
    "moc_topk_50_sparse": {"desc": "MoC Top-K 50% + sparse_all",     "sparse_all": True,  "sparse_ffn": False, "sparse_pattern": "2:4", "sparse_topk_ratio": None, "moc_pattern": "topk", "moc_topk_ratio": 0.5},
    "moc_2_4_sparse":     {"desc": "MoC 2:4 + sparse_all",           "sparse_all": True,  "sparse_ffn": False, "sparse_pattern": "2:4", "sparse_topk_ratio": None, "moc_pattern": "2:4", "moc_topk_ratio": None},
    "moc_2_8_sparse":     {"desc": "MoC 2:8 + sparse_all",           "sparse_all": True,  "sparse_ffn": False, "sparse_pattern": "2:4", "sparse_topk_ratio": None, "moc_pattern": "2:8", "moc_topk_ratio": None},
    # PRAC variants (arXiv:2602.23111): principal+random subspace compression on activations
    "prac_all_60":        {"desc": "PRAC all 60% retain",             "sparse_all": False, "sparse_ffn": False, "sparse_pattern": None, "sparse_topk_ratio": None, "moc_pattern": None, "moc_topk_ratio": None, "prac_all": True,  "prac_ffn": False, "prac_rank": 0.30},
    "prac_all_50":        {"desc": "PRAC all 50% retain",             "sparse_all": False, "sparse_ffn": False, "sparse_pattern": None, "sparse_topk_ratio": None, "moc_pattern": None, "moc_topk_ratio": None, "prac_all": True,  "prac_ffn": False, "prac_rank": 0.25},
    "prac_ffn_60":        {"desc": "PRAC FFN 60% retain",             "sparse_all": False, "sparse_ffn": False, "sparse_pattern": None, "sparse_topk_ratio": None, "moc_pattern": None, "moc_topk_ratio": None, "prac_all": False, "prac_ffn": True,  "prac_rank": 0.30},
    "prac_ffn_50":        {"desc": "PRAC FFN 50% retain",             "sparse_all": False, "sparse_ffn": False, "sparse_pattern": None, "sparse_topk_ratio": None, "moc_pattern": None, "moc_topk_ratio": None, "prac_all": False, "prac_ffn": True,  "prac_rank": 0.25},
}
VARIANT_KEYS = list(SPARSITY_LOGIC_TABLE.keys())

# Short column labels so table fits one line (e.g. 80-col terminal)
VARIANT_SHORT_LABELS = {
    "original": "Orig",
    "sparse_all_topk": "S_T25",
    "sparse_all_topk_50": "S_T50",
    "sparse_all_2_4": "S_24",
    "sparse_all_2_8": "S_28",
    "sparse_ffn_topk": "F_T25",
    "sparse_ffn_topk_50": "F_T50",
    "sparse_ffn_2_4": "F_24",
    "sparse_ffn_2_8": "F_28",
    "moc_topk": "MoC_T25",
    "moc_topk_50": "MoC_T50",
    "moc_2_4": "MoC_24",
    "moc_2_8": "MoC_28",
    "moc_topk_sparse": "T25+S",
    "moc_topk_50_sparse": "T50+S",
    "moc_2_4_sparse": "24+S",
    "moc_2_8_sparse": "28+S",
    "prac_all_60": "P_60",
    "prac_all_50": "P_50",
    "prac_ffn_60": "PF_60",
    "prac_ffn_50": "PF_50",
}


def print_sparsity_logic_table(intermediate_size: int = 14336, num_channels: int = None) -> None:
    """Print logic table: pattern x method. For Top-K show sparsity (K and %)."""
    if num_channels is None:
        num_channels = (intermediate_size * 25) // 100  # 25% default
    k50 = (intermediate_size * 50) // 100
    topk_str = _moc_topk_sparsity_str(num_channels, intermediate_size)
    topk50_str = _moc_topk_sparsity_str(k50, intermediate_size)
    print("\n" + "=" * 140)
    print("SPARSITY LOGIC TABLE (pattern x method → variant)")
    print("=" * 140)
    print("\n  Cols: base | sparse_all | sparse_ffn | moc (FFN) | moc+sparse | prac_all | prac_ffn")
    print("  Rows: pattern. Top-K sparsity: " + topk_str + "; other: " + topk50_str + ".")
    print()
    print(f"  {'Pattern':<14} | {'base':<12} | {'sparse_all':<18} | {'sparse_ffn':<18} | {'moc (FFN)':<16} | {'moc+sparse':<16} | {'prac_all':<14} | {'prac_ffn':<14}")
    print("  " + "-" * 140)
    print(f"  {'(none)':<14} | {'original':<12} | {'-':<18} | {'-':<18} | {'-':<16} | {'-':<16} | {'-':<14} | {'-':<14}")
    print(f"  {'Top-K 25%':<14} | {'-':<12} | {'sparse_all_topk':<18} | {'sparse_ffn_topk':<18} | {'moc_topk':<16} | {'moc_topk_sparse':<16} | {'-':<14} | {'-':<14}")
    print(f"  {'Top-K 50%':<14} | {'-':<12} | {'sparse_all_topk_50':<18} | {'sparse_ffn_topk_50':<18} | {'moc_topk_50':<16} | {'moc_topk_50_sparse':<16} | {'-':<14} | {'-':<14}")
    print(f"  {'2:4':<14} | {'-':<12} | {'sparse_all_2_4':<18} | {'sparse_ffn_2_4':<18} | {'moc_2_4':<16} | {'moc_2_4_sparse':<16} | {'-':<14} | {'-':<14}")
    print(f"  {'2:8':<14} | {'-':<12} | {'sparse_all_2_8':<18} | {'sparse_ffn_2_8':<18} | {'moc_2_8':<16} | {'moc_2_8_sparse':<16} | {'-':<14} | {'-':<14}")
    print(f"  {'PRAC 60%':<14} | {'-':<12} | {'-':<18} | {'-':<18} | {'-':<16} | {'-':<16} | {'prac_all_60':<14} | {'prac_ffn_60':<14}")
    print(f"  {'PRAC 50%':<14} | {'-':<12} | {'-':<18} | {'-':<18} | {'-':<16} | {'-':<16} | {'prac_all_50':<14} | {'prac_ffn_50':<14}")
    print("  " + "-" * 140)
    print("\n  Top-K sparsity (MoC / sparse): default " + topk_str + "; other " + topk50_str + ".")
    print("  PRAC: principal+random subspace activation compression (arXiv:2602.23111). 60%=r1=r2=0.3, 50%=r1=r2=0.25.")
    print("  Variants run in eval:", VARIANT_KEYS)
    print("=" * 140 + "\n")


# ============================================================================
# LoRA Fine-tuning Setup
# ============================================================================

def setup_lora_for_moc(
    model: nn.Module,
    r: int = 16,
    lora_alpha: int = 32,
    target_modules: List[str] = ["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout: float = 0.05
) -> PeftModel:
    """
    为MoC模型设置LoRA微调
    
    Args:
        model: 基础模型（已替换为MoC）
        r: LoRA秩
        lora_alpha: LoRA alpha参数
        target_modules: 应用LoRA的模块
        lora_dropout: Dropout率
        
    Returns:
        PEFT模型
    """
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ============================================================================
# Dataset Loading (保持原有实现)
# ============================================================================

def get_local_cache_path(cache_dir: str, dataset_name: str, subset_name: Optional[str] = None) -> Path:
    """Get the local cache path for a dataset."""
    if subset_name:
        return Path(cache_dir) / "local_cache" / f"{dataset_name}_{subset_name}.json"
    return Path(cache_dir) / "local_cache" / f"{dataset_name}.json"


def save_dataset_to_local(dataset: Any, cache_path: Path) -> None:
    """Save dataset to local JSON file."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data = [dict(item) for item in dataset]
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  Saved to local cache: {cache_path}")


def load_dataset_from_local(cache_path: Path) -> List[Dict]:
    """Load dataset from local JSON file."""
    with open(cache_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _try_load_from_disk(cache_dir: str, dataset_name: str, subset_name: Optional[str],
                        split: str) -> Any:
    """Try loading a dataset saved via save_to_disk under cache_dir."""
    from datasets import load_from_disk
    candidates = []
    base = dataset_name.replace("/", "_")
    if subset_name:
        candidates.append(os.path.join(cache_dir, subset_name.lower().replace("-", "_")))
        candidates.append(os.path.join(cache_dir, subset_name))
        candidates.append(os.path.join(cache_dir, dataset_name.split("/")[-1], subset_name))
        candidates.append(os.path.join(cache_dir, base, subset_name))
    candidates.append(os.path.join(cache_dir, base))
    candidates.append(os.path.join(cache_dir, dataset_name.split("/")[-1]))
    for p in candidates:
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "dataset_dict.json")):
            ds = load_from_disk(p)
            if split in ds:
                return ds[split]
            if "test" in ds and split == "validation":
                return ds["test"]
            return ds[list(ds.keys())[0]]
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "dataset_info.json")):
            return load_from_disk(p)
    return None


def load_dataset_with_cache(
    dataset_name: str,
    subset_name: Optional[str] = None,
    cache_dir: Optional[str] = "/data/datasets/",
    split: str = "validation",
    trust_remote_code: bool = False
) -> Any:
    """Load dataset with local cache support."""
    full_name = f"{dataset_name}/{subset_name}" if subset_name else dataset_name
    
    if cache_dir:
        local_cache_path = get_local_cache_path(cache_dir, dataset_name, subset_name)
        local_cache_path = Path(str(local_cache_path).replace(".json", f"_{split}.json"))
        
        if local_cache_path.exists():
            print(f"Loading {full_name} ({split}) from local cache: {local_cache_path}")
            return load_dataset_from_local(local_cache_path)
    
    hf_token = os.environ.get("HF_TOKEN", None)
    
    print(f"Downloading {full_name} ({split}) from HuggingFace...")
    
    try:
        if subset_name:
            dataset = load_dataset(
                dataset_name, 
                subset_name,
                cache_dir=cache_dir,
                token=hf_token,
                trust_remote_code=trust_remote_code
            )
        else:
            dataset = load_dataset(
                dataset_name,
                cache_dir=cache_dir,
                token=hf_token,
                trust_remote_code=trust_remote_code
            )
        
        if split in dataset:
            data = dataset[split]
        elif "test" in dataset and split == "validation":
            print(f"  Note: '{split}' not found, using 'test' split instead")
            data = dataset["test"]
        else:
            available_splits = list(dataset.keys())
            print(f"  Warning: '{split}' not found. Available: {available_splits}")
            data = dataset[available_splits[0]]
        
        if cache_dir:
            try:
                save_dataset_to_local(data, local_cache_path)
            except Exception as e:
                print(f"  Warning: Could not save to local cache: {e}")
        
        return data
        
    except Exception as e:
        if cache_dir:
            disk_data = _try_load_from_disk(cache_dir, dataset_name, subset_name, split)
            if disk_data is not None:
                print(f"  Loaded {full_name} ({split}) from disk cache (load_from_disk fallback)")
                return disk_data
        print(f"Error loading {full_name}: {e}")
        raise


# Dataset-specific loaders
def load_rte_dataset(cache_dir: str = "/data/datasets/") -> Any:
    try:
        return load_dataset_with_cache("glue", "rte", cache_dir, split="validation")
    except Exception:
        return load_dataset_with_cache("super_glue", "rte", cache_dir, split="validation")

def load_boolq_dataset(cache_dir: str = "/data/datasets/") -> Any:
    return load_dataset_with_cache("super_glue", "boolq", cache_dir, split="validation")

def load_winogrande_dataset(cache_dir: str = "/data/datasets/") -> Any:
    return load_dataset_with_cache("winogrande", "winogrande_xl", cache_dir, split="validation")

def load_arc_easy_dataset(cache_dir: str = "/data/datasets/") -> Any:
    return load_dataset_with_cache("allenai/ai2_arc", "ARC-Easy", cache_dir, split="validation")

def load_arc_challenge_dataset(cache_dir: str = "/data/datasets/") -> Any:
    return load_dataset_with_cache("allenai/ai2_arc", "ARC-Challenge", cache_dir, split="validation")

def load_openbookqa_dataset(cache_dir: str = "/data/datasets/") -> Any:
    return load_dataset_with_cache("allenai/openbookqa", "main", cache_dir, split="validation")

def load_piqa_dataset(cache_dir: str = "/data/datasets/") -> Any:
    return load_dataset_with_cache("piqa", None, cache_dir, split="validation", trust_remote_code=True)

def load_mmlu_dataset(cache_dir: str = "/data/datasets/", subject: str = "all") -> Any:
    if subject == "all":
        return load_dataset_with_cache("cais/mmlu", "all", cache_dir, split="validation")
    else:
        return load_dataset_with_cache("cais/mmlu", subject, cache_dir, split="validation")


def load_longbench_dataset(cache_dir: str = "/data/datasets/", task: str = "qasper") -> Any:
    """Load LongBench dataset (same as baseline test_rte_sparsity.py)."""
    return load_dataset_with_cache("THUDM/LongBench", task, cache_dir, split="test", trust_remote_code=True)


# ============================================================================
# Prompt Templates (保持原有实现)
# ============================================================================

def create_rte_prompt(premise: str, hypothesis: str) -> str:
    return f'''Given the premise: "{premise}"

Question: Does this imply the following hypothesis: "{hypothesis}"?

Answer (Yes or No):'''

def create_boolq_prompt(passage: str, question: str) -> str:
    return f'''Passage: "{passage}"

Question: {question}

Answer (Yes or No):'''

def create_winogrande_prompt(sentence: str, option1: str, option2: str) -> str:
    return f'''Complete the sentence by choosing the correct option.

Sentence: {sentence}

Option 1: {option1}
Option 2: {option2}

Which option correctly fills the blank? Answer with just the number (1 or 2):'''

def create_arc_prompt(question: str, choices: List[str], choice_labels: List[str]) -> str:
    choices_text = "\n".join([f"{label}. {text}" for label, text in zip(choice_labels, choices)])
    return f'''Question: {question}

{choices_text}

Answer with just the letter:'''

def create_openbookqa_prompt(question: str, choices: List[str], choice_labels: List[str]) -> str:
    choices_text = "\n".join([f"{label}. {text}" for label, text in zip(choice_labels, choices)])
    return f'''Question: {question}

{choices_text}

Answer with just the letter:'''

def create_piqa_prompt(goal: str, sol1: str, sol2: str) -> str:
    return f'''Goal: {goal}

Solution 1: {sol1}
Solution 2: {sol2}

Which solution is better? Answer with just the number (1 or 2):'''

def create_mmlu_prompt(question: str, choices: List[str]) -> str:
    choice_labels = ['A', 'B', 'C', 'D']
    choices_text = "\n".join([f"{label}. {text}" for label, text in zip(choice_labels, choices)])
    return f'''Question: {question}

{choices_text}

Answer with just the letter (A, B, C, or D):'''


def create_longbench_prompt(context: str, question: str, max_context_len: int = 4000) -> str:
    """Create prompt for LongBench (same as baseline)."""
    if len(context) > max_context_len:
        context = context[:max_context_len] + "..."
    return f'''Context: {context}

Question: {question}

Answer:'''


# ============================================================================
# Evaluation Helpers (align with baseline test_rte_sparsity.py)
# ============================================================================

def _get_samples(dataset: Any, max_samples: Optional[int]) -> Any:
    """Return dataset or sliced/selected subset; works for list or HF Dataset."""
    if max_samples is None:
        return dataset
    if isinstance(dataset, list):
        return dataset[:max_samples]
    return dataset.select(range(min(max_samples, len(dataset))))


# ============================================================================
# Evaluation Functions (保持原有实现，适配MoC)
# ============================================================================

def get_token_logprob(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    target_token: str,
    device: str = "cuda"
) -> float:
    """Get the log probability of a target token given a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    target_ids = tokenizer.encode(target_token, add_special_tokens=False)
    if len(target_ids) > 0:
        target_id = target_ids[0]
    else:
        target_id = tokenizer.encode(" " + target_token, add_special_tokens=False)[0]
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    last_logits = logits[0, -1, :]
    log_probs = F.log_softmax(last_logits, dim=-1)
    
    return log_probs[target_id].item()


def get_choice_logprobs(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    choices: List[str],
    device: str = "cuda"
) -> List[float]:
    """Get log probabilities for multiple choice options."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    last_logits = logits[0, -1, :]
    log_probs = F.log_softmax(last_logits, dim=-1)
    
    choice_logprobs = []
    for choice in choices:
        target_ids = tokenizer.encode(choice, add_special_tokens=False)
        if len(target_ids) > 0:
            target_id = target_ids[0]
        else:
            target_id = tokenizer.encode(" " + choice, add_special_tokens=False)[0]
        choice_logprobs.append(log_probs[target_id].item())
    
    return choice_logprobs


# 评估函数 (保持原有实现，略作简化)
def evaluate_rte_zero_shot(model, tokenizer, dataset, device, max_samples=None):
    model.eval()
    correct = total = 0
    samples = _get_samples(dataset, max_samples)

    for sample in tqdm(samples, desc="RTE"):
        s1 = sample.get("sentence1") or sample.get("premise", "")
        s2 = sample.get("sentence2") or sample.get("hypothesis", "")
        prompt = create_rte_prompt(s1, s2)
        yes_logprob = get_token_logprob(model, tokenizer, prompt, "Yes", device)
        no_logprob = get_token_logprob(model, tokenizer, prompt, "No", device)
        predicted = 0 if yes_logprob > no_logprob else 1
        
        if predicted == sample["label"]:
            correct += 1
        total += 1
    
    return {"accuracy": correct/total, "correct": correct, "total": total}


def evaluate_boolq_zero_shot(model, tokenizer, dataset, device, max_samples=None):
    model.eval()
    correct = total = 0
    samples = _get_samples(dataset, max_samples)

    for sample in tqdm(samples, desc="BoolQ"):
        prompt = create_boolq_prompt(sample["passage"], sample["question"])
        yes_logprob = get_token_logprob(model, tokenizer, prompt, "Yes", device)
        no_logprob = get_token_logprob(model, tokenizer, prompt, "No", device)
        predicted = yes_logprob > no_logprob
        label = sample.get("label", sample.get("answer"))
        
        if predicted == label:
            correct += 1
        total += 1
    
    return {"accuracy": correct/total, "correct": correct, "total": total}


def evaluate_winogrande_zero_shot(model, tokenizer, dataset, device, max_samples=None):
    model.eval()
    correct = total = 0
    samples = _get_samples(dataset, max_samples)

    for sample in tqdm(samples, desc="WinoGrande"):
        prompt = create_winogrande_prompt(sample["sentence"], sample["option1"], sample["option2"])
        logprobs = get_choice_logprobs(model, tokenizer, prompt, ["1", "2"], device)
        predicted = "1" if logprobs[0] > logprobs[1] else "2"
        
        if predicted == str(sample["answer"]):
            correct += 1
        total += 1
    
    return {"accuracy": correct/total, "correct": correct, "total": total}


def evaluate_arc_zero_shot(model, tokenizer, dataset, device, max_samples=None, task_name="ARC"):
    model.eval()
    correct = total = 0
    samples = _get_samples(dataset, max_samples)

    for sample in tqdm(samples, desc=task_name):
        choices_data = sample["choices"]
        if isinstance(choices_data, dict):
            choice_labels = choices_data["label"]
            choice_texts = choices_data["text"]
        else:
            choice_labels = [c["label"] for c in choices_data]
            choice_texts = [c["text"] for c in choices_data]
        
        prompt = create_arc_prompt(sample["question"], choice_texts, choice_labels)
        logprobs = get_choice_logprobs(model, tokenizer, prompt, choice_labels, device)
        predicted = choice_labels[logprobs.index(max(logprobs))]
        
        if predicted == sample["answerKey"]:
            correct += 1
        total += 1
    
    return {"accuracy": correct/total, "correct": correct, "total": total}


def evaluate_openbookqa_zero_shot(model, tokenizer, dataset, device, max_samples=None):
    model.eval()
    correct = total = 0
    samples = _get_samples(dataset, max_samples)

    for sample in tqdm(samples, desc="OpenBookQA"):
        choices_data = sample["choices"]
        if isinstance(choices_data, dict):
            choice_labels = choices_data["label"]
            choice_texts = choices_data["text"]
        else:
            choice_labels = [c["label"] for c in choices_data]
            choice_texts = [c["text"] for c in choices_data]
        
        prompt = create_openbookqa_prompt(sample["question_stem"], choice_texts, choice_labels)
        logprobs = get_choice_logprobs(model, tokenizer, prompt, choice_labels, device)
        predicted = choice_labels[logprobs.index(max(logprobs))]
        
        if predicted == sample["answerKey"]:
            correct += 1
        total += 1
    
    return {"accuracy": correct/total, "correct": correct, "total": total}


def evaluate_piqa_zero_shot(model, tokenizer, dataset, device, max_samples=None):
    model.eval()
    correct = total = 0
    samples = _get_samples(dataset, max_samples)

    for sample in tqdm(samples, desc="PIQA"):
        prompt = create_piqa_prompt(sample["goal"], sample["sol1"], sample["sol2"])
        logprobs = get_choice_logprobs(model, tokenizer, prompt, ["1", "2"], device)
        predicted = 0 if logprobs[0] > logprobs[1] else 1
        
        if predicted == sample["label"]:
            correct += 1
        total += 1
    
    return {"accuracy": correct/total, "correct": correct, "total": total}


def evaluate_mmlu_zero_shot(model, tokenizer, dataset, device, max_samples=None):
    model.eval()
    correct = total = 0
    samples = _get_samples(dataset, max_samples)

    for sample in tqdm(samples, desc="MMLU"):
        prompt = create_mmlu_prompt(sample["question"], sample["choices"])
        choice_labels = ["A", "B", "C", "D"]
        logprobs = get_choice_logprobs(model, tokenizer, prompt, choice_labels, device)
        predicted = logprobs.index(max(logprobs))
        
        if predicted == sample["answer"]:
            correct += 1
        total += 1
    
    return {"accuracy": correct/total, "correct": correct, "total": total}


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 between prediction and ground truth (same as baseline)."""
    pred_tokens = prediction.lower().split()
    truth_tokens = ground_truth.lower().split()
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return float(pred_tokens == truth_tokens)
    common = set(pred_tokens) & set(truth_tokens)
    num_common = len(common)
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate_longbench_zero_shot(
    model: nn.Module,
    tokenizer: Any,
    dataset: Any,
    device: str = "cuda",
    max_samples: Optional[int] = None,
    max_length: int = 4096
) -> Dict[str, float]:
    """Evaluate on LongBench (generation task, F1); same as baseline."""
    model.eval()
    samples = _get_samples(dataset, max_samples)
    total_f1 = 0.0
    total = 0
    for sample in tqdm(samples, desc="LongBench"):
        context = sample["context"]
        question = sample["input"]
        answers = sample["answers"]
        prompt = create_longbench_prompt(context, question)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        best_f1 = 0.0
        for answer in answers:
            best_f1 = max(best_f1, compute_f1(generated, answer))
        total_f1 += best_f1
        total += 1
    avg_f1 = total_f1 / total if total > 0 else 0.0
    return {"accuracy": avg_f1, "f1_score": avg_f1, "correct": int(total_f1), "total": total}


# ============================================================================
# GPU and timing (align with baseline test_rte_sparsity.py)
# ============================================================================

def get_all_gpu_info() -> List[Dict[str, Any]]:
    """Get info for all GPUs (nvidia-smi or PyTorch fallback)."""
    gpus = []
    if not torch.cuda.is_available():
        return gpus
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,memory.free,utilization.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True
        )
        for line in result.stdout.strip().split('\n'):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 6:
                gpus.append({
                    'index': int(parts[0]), 'name': parts[1],
                    'memory_used': int(parts[2]), 'memory_total': int(parts[3]),
                    'memory_free': int(parts[4]), 'utilization': int(parts[5])
                })
    except Exception:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append({
                'index': i, 'name': props.name,
                'memory_total': props.total_memory // (1024**2), 'memory_used': 0,
                'memory_free': props.total_memory // (1024**2), 'utilization': 0
            })
    return gpus


def get_available_gpus(min_free_memory_mb: int = 20000) -> List[int]:
    """GPUs with at least min_free_memory_mb free."""
    gpus = get_all_gpu_info()
    return [g['index'] for g in gpus if g['memory_free'] >= min_free_memory_mb] or ([0] if gpus else [])


def print_gpu_info(gpu_ids: Optional[List[int]] = None) -> None:
    """Print GPU info; gpu_ids = which GPUs are selected for use."""
    if not torch.cuda.is_available():
        print("CUDA is not available!")
        return
    print("\n" + "=" * 80)
    print("GPU INFORMATION")
    print("=" * 80)
    gpus = get_all_gpu_info()
    print(f"\nTotal GPUs detected: {len(gpus)}")
    if gpu_ids:
        print(f"GPUs to be used: {gpu_ids}")
    print("-" * 80)
    print(f"{'Index':<6} {'Name':<40} {'Used':<12} {'Free':<12} {'Total':<12} {'Util':<8}")
    print("-" * 80)
    for gpu in gpus:
        marker = " *" if gpu_ids and gpu['index'] in gpu_ids else ""
        print(f"{gpu['index']:<6} {gpu['name']:<40} {gpu['memory_used']:<12} {gpu['memory_free']:<12} {gpu['memory_total']:<12} {gpu['utilization']}%{marker}")
    print("=" * 80 + "\n")


class Timer:
    """Simple timer for sections (same as baseline)."""
    def __init__(self):
        self.times: Dict[str, float] = {}
        self.start_times: Dict[str, float] = {}

    def start(self, name: str) -> None:
        self.start_times[name] = time.time()

    def stop(self, name: str) -> float:
        if name not in self.start_times:
            return 0.0
        elapsed = time.time() - self.start_times[name]
        self.times[name] = elapsed
        return elapsed

    def get(self, name: str) -> float:
        return self.times.get(name, 0.0)

    def get_elapsed(self, name: str) -> float:
        """Current elapsed time (if still running) or stored time (if stopped)."""
        if name in self.times:
            return self.times[name]
        if name in self.start_times:
            return time.time() - self.start_times[name]
        return 0.0

    def format_time(self, seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        if seconds < 3600:
            return f"{int(seconds // 60)}m {seconds % 60:.1f}s"
        return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m {seconds % 60:.0f}s"


# ============================================================================
# Task orchestration: evaluate_all_tasks, table, save, viz (baseline-style)
# ============================================================================

def evaluate_all_tasks(
    model: nn.Module,
    tokenizer: Any,
    datasets: Dict[str, Any],
    device: str,
    max_samples: Optional[int] = None,
    timer: Optional[Timer] = None,
    prefix: str = ""
) -> Dict[str, Dict[str, float]]:
    """Run all loaded tasks with optional timing; same contract as baseline."""
    results = {}
    for task_name, dataset in datasets.items():
        print(f"\n--- Evaluating {task_name} ---")
        if timer:
            timer.start(f"{prefix}{task_name}")
        if task_name == "rte":
            results[task_name] = evaluate_rte_zero_shot(model, tokenizer, dataset, device, max_samples)
        elif task_name == "boolq":
            results[task_name] = evaluate_boolq_zero_shot(model, tokenizer, dataset, device, max_samples)
        elif task_name == "winogrande":
            results[task_name] = evaluate_winogrande_zero_shot(model, tokenizer, dataset, device, max_samples)
        elif task_name == "arc_easy":
            results[task_name] = evaluate_arc_zero_shot(model, tokenizer, dataset, device, max_samples, "ARC-Easy")
        elif task_name == "arc_challenge":
            results[task_name] = evaluate_arc_zero_shot(model, tokenizer, dataset, device, max_samples, "ARC-Challenge")
        elif task_name == "openbookqa":
            results[task_name] = evaluate_openbookqa_zero_shot(model, tokenizer, dataset, device, max_samples)
        elif task_name == "piqa":
            results[task_name] = evaluate_piqa_zero_shot(model, tokenizer, dataset, device, max_samples)
        elif task_name == "mmlu":
            results[task_name] = evaluate_mmlu_zero_shot(model, tokenizer, dataset, device, max_samples)
        elif task_name == "longbench":
            results[task_name] = evaluate_longbench_zero_shot(model, tokenizer, dataset, device, max_samples)
        else:
            print(f"  Unknown task: {task_name}, skipping...")
            continue
        if timer:
            elapsed = timer.stop(f"{prefix}{task_name}")
            results[task_name]["time"] = elapsed
            print(f"  {task_name}: {results[task_name]['accuracy']:.4f} (time: {timer.format_time(elapsed)})")
        else:
            print(f"  {task_name}: {results[task_name]['accuracy']:.4f}")
    return results


# ANSI: highlight new result (green bold); no-op if not a tty
def _hl(s: str, on: bool = True) -> str:
    if not on:
        return s
    return f"\033[1;32m{s}\033[0m"


def print_results_table(
    results_by_variant: Dict[str, Dict[str, Dict[str, float]]],
    tasks_to_run: List[str],
    timer: Timer,
    highlight_variant: Optional[str] = None,
    variant_order: Optional[List[str]] = None,
) -> None:
    """Print results table; short labels so one line fits. Overall = current elapsed."""
    order = variant_order if variant_order is not None else VARIANT_KEYS
    variants = [k for k in order if k in results_by_variant]
    if not variants:
        return
    use_color = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    col_w = 7   # accuracy column width
    time_w = 10 # time column (e.g. "5m 3.1s")
    sep_w = 10 + len(variants) * (col_w + 1) + 8
    sep_w = min(sep_w, 120)
    print("\n" + "=" * sep_w)
    title = "RESULTS"
    if highlight_variant:
        title += " [*new]"
    print(title)
    print("=" * sep_w)
    header = f"{'Task':<10}"
    for v in variants:
        label = VARIANT_SHORT_LABELS.get(v, v[:col_w])
        if v == highlight_variant:
            label = label + "*"
        header += f" {label:<{col_w}}"
    header += " N"
    print(header)
    print("-" * sep_w)
    totals = {v: {"acc": 0.0, "time": 0.0} for v in variants}
    num_tasks = 0
    for task_name in tasks_to_run:
        row_vals = []
        ok = True
        for v in variants:
            if task_name not in results_by_variant[v]:
                ok = False
                break
            r = results_by_variant[v][task_name]
            row_vals.append(r["accuracy"])
            totals[v]["acc"] += r["accuracy"]
            totals[v]["time"] += timer.get(f"{v}_{task_name}")
        if not ok:
            continue
        num_tasks += 1
        line = f"{task_name:<10}"
        for i, v in enumerate(variants):
            acc_str = f"{row_vals[i]:<{col_w}.3f}"
            if use_color and v == highlight_variant:
                acc_str = _hl(acc_str, use_color)
            line += f" {acc_str}"
        line += f" {results_by_variant[variants[0]][task_name].get('total', ''):<6}"
        print(line)
    print("-" * sep_w)
    if num_tasks > 0:
        line = f"{'AVG':<10}"
        for v in variants:
            avg_str = f"{totals[v]['acc']/num_tasks:<{col_w}.3f}"
            if use_color and v == highlight_variant:
                avg_str = _hl(avg_str, use_color)
            line += f" {avg_str}"
        print(line)
        time_line = f"{'Time':<10}"
        for v in variants:
            t_str = timer.format_time(totals[v]["time"])
            if use_color and v == highlight_variant:
                t_str = _hl(t_str, use_color)
            time_line += f" {t_str:<{time_w}}"
        print(time_line)
    print("=" * sep_w)
    overall_elapsed = timer.get_elapsed("total")
    print(f"Overall: {timer.format_time(overall_elapsed)}")
    legend = "Legend: Orig=base | S_*=sparse_all, F_*=sparse_ffn | MoC_*=MoC | *+S=MoC+sparse | P_*=PRAC_all, PF_*=PRAC_ffn | T25/T50=TopK, 24/28=2:4/2:8"
    if len(legend) <= sep_w:
        print(legend)
    else:
        print("Legend: Orig S_* F_* MoC_* *+S P_* PF_* T25 T50 24 28")


def save_results_to_json(
    results: Dict[str, Dict[str, Dict[str, float]]],
    timing_info: Dict[str, float],
    config: Dict[str, Any],
    output_dir: str = "moc_results"
) -> str:
    """Save results (original, sparse24, moc, moc24) and config to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "results": results,
        "timing": timing_info,
        "summary": {}
    }
    tasks = list(results.get("original", {}).keys()) if "original" in results else list(next(iter(results.values()), {}).keys())
    if tasks:
        out["summary"] = {"num_tasks": len(tasks)}
        for v in (list(results.keys()) if results else []):
            if v in results and all(t in results[v] for t in tasks):
                accs = [results[v][t]["accuracy"] for t in tasks]
                out["summary"][f"avg_{v}_accuracy"] = sum(accs) / len(accs)
    path = os.path.join(output_dir, f"moc_eval_results_{timestamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {path}")
    return path


def create_visualization(
    results: Dict[str, Dict[str, Dict[str, float]]],
    timing_info: Dict[str, float],
    output_dir: str = "moc_results"
) -> str:
    """Bar chart: Original | 2:4 Sparse | MoC | 2:4 MoC."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Warning: matplotlib not installed. Skipping visualization.")
        return ""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tasks = list(results.get("original", {}).keys()) if "original" in results else list(next(iter(results.values()), {}).keys())
    variants = [k for k in VARIANT_KEYS if k in results] if any(k in VARIANT_KEYS for k in results) else list(results.keys())
    if not tasks or not variants:
        return ""
    labels = {v: SPARSITY_LOGIC_TABLE.get(v, {}).get("desc", v)[:12] for v in variants}
    colors = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c", "#1abc9c", "#e67e22", "#34495e", "#8e44ad", "#27ae60"]
    color_map = {v: colors[i % len(colors)] for i, v in enumerate(variants)}
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Zero-Shot Benchmark: Sparsity pattern x method (all variants)", fontsize=14, fontweight="bold")
    x = np.arange(len(tasks))
    nv = len(variants)
    w = 0.8 / max(nv, 1)
    ax1 = axes[0, 0]
    for i, v in enumerate(variants):
        accs = [results[v][t]["accuracy"] * 100 for t in tasks]
        offset = (i - (nv - 1) / 2) * w
        ax1.bar(x + offset, accs, w, label=labels.get(v, v), color=color_map.get(v, "gray"), alpha=0.8)
    ax1.set_xlabel("Task")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Accuracy by Task")
    ax1.set_xticks(x)
    ax1.set_xticklabels(tasks, rotation=45, ha="right")
    ax1.legend()
    ax1.set_ylim(0, 100)
    ax1.grid(axis="y", alpha=0.3)
    ax2 = axes[0, 1]
    orig_accs = [results["original"][t]["accuracy"] * 100 for t in tasks] if "original" in results else []
    diff_variants = [v for v in variants if v != "original"]
    nv2 = len(diff_variants)
    w2 = 0.8 / max(nv2, 1)
    for i, v in enumerate(diff_variants):
        accs = [results[v][t]["accuracy"] * 100 for t in tasks]
        diffs = [a - (orig_accs[j] if orig_accs else 0) for j, a in enumerate(accs)]
        offset = (i - (nv2 - 1) / 2) * w2
        ax2.bar(x + offset, diffs, w2, label=f"{labels.get(v, v)} - Orig", color=color_map.get(v, "gray"), alpha=0.8)
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax2.set_ylabel("Accuracy diff vs Original (%)")
    ax2.set_title("Difference vs Original")
    ax2.set_xticks(x)
    ax2.set_xticklabels(tasks, rotation=45, ha="right")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)
    ax3 = axes[1, 0]
    nv = len(variants)
    w = 0.8 / max(nv, 1)
    for i, v in enumerate(variants):
        times = [timing_info.get(f"{v}_{t}", 0) for t in tasks]
        offset = (i - (nv - 1) / 2) * w
        ax3.bar(x + offset, times, w, label=labels.get(v, v), alpha=0.8)
    ax3.set_ylabel("Time (s)")
    ax3.set_title("Evaluation Time by Task")
    ax3.set_xticks(x)
    ax3.set_xticklabels(tasks, rotation=45, ha="right")
    ax3.legend()
    ax3.grid(axis="y", alpha=0.3)
    ax4 = axes[1, 1]
    ax4.axis("off")
    lines = [f"Tasks: {len(tasks)}", f"Total time: {timing_info.get('total', 0)/60:.1f} min"]
    for v in variants:
        accs = [results[v][t]["accuracy"] * 100 for t in tasks]
        lines.append(f"Avg {labels.get(v, v)}: {np.mean(accs):.2f}%")
    ax4.text(0.1, 0.5, "\n".join(lines), transform=ax4.transAxes, fontsize=11, verticalalignment="center",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.tight_layout()
    path = os.path.join(output_dir, f"moc_eval_results_{timestamp}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Visualization saved to: {path}")
    return path


# ============================================================================
# Fine-tuning Functions (incl. finetune_eval: only FFN trainable, then eval)
# ============================================================================

# Variants to run for finetune_eval: all F (sparse_ffn) + all MoC (moc only, no moc+sparse)
FINETUNE_EVAL_VARIANTS = [
    "sparse_ffn_topk",
    "sparse_ffn_topk_50",
    "sparse_ffn_2_4",
    "sparse_ffn_2_8",
    "moc_topk",
    "moc_topk_50",
    "moc_2_4",
    "moc_2_8",
]


def freeze_all_except_ffn(model: nn.Module) -> None:
    """Freeze all parameters except those in FFN (mlp). Only FFN is trainable."""
    for name, param in model.named_parameters():
        if "mlp" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {n_trainable:,} / {n_total:,} ({100.0 * n_trainable / n_total:.2f}%)")


def build_model_for_variant(
    model_path: str,
    vid: str,
    load_kwargs: Dict[str, Any],
    hidden_size: int,
    intermediate_size: int,
    num_channels: Optional[int],
    training_mode: bool = False,
) -> nn.Module:
    """Build model with the given variant (same logic as eval loop). training_mode=True for finetune."""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    logic = SPARSITY_LOGIC_TABLE[vid]
    if logic["sparse_all"]:
        pat = logic.get("sparse_pattern") or "2:4"
        ratio = logic.get("sparse_topk_ratio") if pat == "topk" else 0.5
        replace_linear_with_sparse(model, pattern=pat, topk_ratio=ratio)
    elif logic["sparse_ffn"]:
        pat = logic.get("sparse_pattern") or "2:4"
        ratio = logic.get("sparse_topk_ratio") if pat == "topk" else 0.5
        replace_linear_with_sparse_ffn_only(model, pattern=pat, topk_ratio=ratio)
    if logic["moc_pattern"] is not None:
        if logic["moc_pattern"] == "topk" and logic.get("moc_topk_ratio") is not None:
            moc_k = int(intermediate_size * logic["moc_topk_ratio"])
        else:
            moc_k = num_channels or (intermediate_size * 25) // 100
        moc_config = MoCConfig(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_channels=moc_k,
            training_mode=training_mode,
            use_gradient_checkpointing=True,
            sparsity_pattern=logic["moc_pattern"],
        )
        replace_mlp_with_moc(model, moc_config)
    prac_rank = logic.get("prac_rank")
    if prac_rank is not None:
        if logic.get("prac_all"):
            replace_linear_with_prac(model, principal_rank=prac_rank, random_rank=prac_rank)
        elif logic.get("prac_ffn"):
            replace_linear_with_prac_ffn_only(model, principal_rank=prac_rank, random_rank=prac_rank)
    return model


def load_state_dict_from_checkpoint(ckpt_dir: str) -> Dict[str, torch.Tensor]:
    """Load state dict from a saved checkpoint (safetensors or pytorch_model.bin)."""
    import os
    sf_path = os.path.join(ckpt_dir, "model.safetensors")
    pt_path = os.path.join(ckpt_dir, "pytorch_model.bin")
    if os.path.isfile(sf_path):
        from safetensors.torch import load_file
        return load_file(sf_path)
    if os.path.isfile(pt_path):
        return torch.load(pt_path, map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin in {ckpt_dir}")


def prepare_finetuning_data(
    dataset_name: str,
    tokenizer: Any,
    max_length: int = 512,
    cache_dir: str = "/data/datasets/"
) -> Dataset:
    """
    准备微调数据
    
    Args:
        dataset_name: 数据集名称 (rte, boolq, etc.)
        tokenizer: 分词器
        max_length: 最大序列长度
        cache_dir: 缓存目录
        
    Returns:
        处理后的数据集
    """
    # 加载数据集
    if dataset_name == "rte":
        raw_data = load_rte_dataset(cache_dir)
        def format_example(ex):
            prompt = create_rte_prompt(ex["sentence1"], ex["sentence2"])
            label = "Yes" if ex["label"] == 0 else "No"
            return {"text": f"{prompt} {label}"}
    elif dataset_name == "boolq":
        raw_data = load_boolq_dataset(cache_dir)
        def format_example(ex):
            prompt = create_boolq_prompt(ex["passage"], ex["question"])
            label = "Yes" if ex["label"] else "No"
            return {"text": f"{prompt} {label}"}
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # 转换为HuggingFace Dataset
    if isinstance(raw_data, list):
        dataset = Dataset.from_list(raw_data)
    else:
        dataset = raw_data
    
    # 格式化
    dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    return tokenized_dataset


class TrainerWithDeviceMap(Trainer):
    """Trainer that moves labels to the output device when using device_map='auto' (multi-GPU).
    Avoids 'Expected all tensors to be on the same device' in loss computation."""
    def compute_loss(self, model, inputs, return_outputs=False):
        if "labels" in inputs and hasattr(model, "lm_head"):
            out_device = next(model.lm_head.parameters()).device
            labels = inputs["labels"]
            if labels is not None and labels.device != out_device:
                inputs = dict(inputs)
                inputs["labels"] = labels.to(out_device)
        return super().compute_loss(model, inputs, return_outputs=return_outputs)


def finetune_moc_model(
    model: nn.Module,
    tokenizer: Any,
    dataset_name: str,
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    cache_dir: str = "/data/datasets/"
):
    """
    微调MoC模型
    
    Args:
        model: MoC模型（已应用LoRA）
        tokenizer: 分词器
        dataset_name: 数据集名称
        output_dir: 输出目录
        num_epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
    """
    # 准备数据
    train_dataset = prepare_finetuning_data(dataset_name, tokenizer, cache_dir=cache_dir)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        warmup_steps=100,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        report_to="none",
    )
    
    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # 训练器（支持 device_map="auto" 多卡：labels 会搬到 logits 所在设备再算 loss）
    trainer = TrainerWithDeviceMap(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # 训练
    print(f"Starting fine-tuning on {dataset_name}...")
    trainer.train()
    
    # 保存
    model.save_pretrained(os.path.join(output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))
    print(f"Model saved to {output_dir}/final_model")
    
    return model


# ============================================================================
# Main Function (hosts task process: Original vs MoC, aligned with test_rte_sparsity.py)
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="MoC Fine-tuning and Evaluation (baseline-style task process)")
    parser.add_argument("--mode", choices=["eval", "finetune", "both", "finetune_eval"], default="eval",
                        help="eval=Original vs variants; finetune=MoC+LoRA only; both=eval then finetune; finetune_eval=per-variant FFN-only finetune then full eval")
    parser.add_argument("--model_path", type=str, default="/data/models/Llama-3.1-8B-Instruct")
    parser.add_argument("--dataset", type=str, default="rte", help="Finetune dataset (rte, boolq)")
    parser.add_argument("--tasks", nargs="+",
                        default=["rte", "boolq", "winogrande", "arc_easy", "arc_challenge", "openbookqa", "piqa", "mmlu", "longbench"],
                        help="Evaluation tasks (same as baseline test_rte_sparsity.py)")
    parser.add_argument("--quick_test", action="store_true",
                        help="Quick test: only run tasks that typically finish in <30s (rte, arc_easy, arc_challenge, openbookqa)")
    parser.add_argument("--moc_channels", type=int, default=None, help="MoC Top-K channels (default: 25%% of intermediate_size)")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples per task (None=all)")
    parser.add_argument("--output_dir", type=str, default="./moc_results")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for finetune")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--cache_dir", type=str, default="/data/datasets/")
    # GPU (align with baseline)
    parser.add_argument("--use_gpus", nargs="*", type=int, default=None,
                        help="GPU IDs to use (e.g. 1 2 3). None=auto-detect.")
    parser.add_argument("--exclude_gpus", nargs="*", type=int, default=None,
                        help="GPU IDs to exclude (default: none; use all GPUs including 0). Example: --exclude_gpus 0")
    parser.add_argument("--min_free_memory_mb", type=int, default=20000)
    parser.add_argument("--variants", nargs="*", default=None,
                        help="Variant keys to run (e.g. original prac_all_50). None=all variants.")

    args = parser.parse_args()
    tasks_to_run = ["rte", "arc_easy", "arc_challenge", "openbookqa"] if args.quick_test else args.tasks
    if args.quick_test:
        print("Quick test mode: only tasks with typical run time <30s (rte, arc_easy, arc_challenge, openbookqa).")
    os.makedirs(args.output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # GPU detection and device_map (same as test_rte_sparsity.py)
    # -------------------------------------------------------------------------
    if torch.cuda.is_available():
        available_gpus = get_available_gpus(min_free_memory_mb=args.min_free_memory_mb)
        exclude = list(args.exclude_gpus or [])
        available_gpus = [g for g in available_gpus if g not in exclude]
        if args.use_gpus is not None and len(args.use_gpus) > 0:
            gpu_ids = [g for g in args.use_gpus if g in available_gpus]
            if not gpu_ids:
                gpu_ids = available_gpus
        else:
            gpu_ids = available_gpus
        if not gpu_ids:
            print("ERROR: No available GPUs.")
            return
        if len(gpu_ids) >= 2:
            max_memory = {i: "90GiB" for i in gpu_ids}
            max_memory["cpu"] = "30GiB"
            device_map = "auto"
        else:
            gpu_id = gpu_ids[0]
            max_memory = {gpu_id: "90GiB", "cpu": "30GiB"}
            device_map = {"": gpu_id}
        primary_device = f"cuda:{gpu_ids[0]}"
        print(f"Using GPUs: {gpu_ids}")
        print_gpu_info(gpu_ids)
    else:
        gpu_ids = []
        device_map = "cpu"
        primary_device = "cpu"
        max_memory = None

    config = {
        "model_path": args.model_path,
        "cache_dir": args.cache_dir,
        "max_samples": args.max_samples,
        "tasks": tasks_to_run,
        "gpu_ids": gpu_ids,
        "moc_channels": args.moc_channels,
        "timestamp": datetime.now().isoformat(),
    }

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "torch_dtype": torch.float16,
        "device_map": device_map,
        "local_files_only": True,
    }
    if max_memory is not None:
        load_kwargs["max_memory"] = max_memory

    # -------------------------------------------------------------------------
    # Finetune-Eval: per-variant FFN-only finetune then full eval (F + MoC)
    # -------------------------------------------------------------------------
    if args.mode == "finetune_eval":
        print("\n" + "=" * 80)
        print("MODE: finetune_eval — FFN-only finetune then eval on trained task only")
        print("Variants:", FINETUNE_EVAL_VARIANTS)
        print("Train & Eval task:", args.dataset)
        print("=" * 80)
        # Only load the dataset used for training; eval only on that task
        eval_task = args.dataset
        datasets = {}
        if eval_task == "rte":
            d = load_rte_dataset(args.cache_dir)
            if d is not None:
                datasets["rte"] = d
                print(f"  ✓ rte: {len(d) if hasattr(d, '__len__') else '?'} samples")
        elif eval_task == "boolq":
            d = load_boolq_dataset(args.cache_dir)
            if d is not None:
                datasets["boolq"] = d
                print(f"  ✓ boolq: {len(d) if hasattr(d, '__len__') else '?'} samples")
        else:
            print(f"Unsupported --dataset for finetune_eval: {eval_task} (use rte or boolq).")
            return

        if not datasets:
            print("No dataset loaded. Exiting.")
            return
        eval_tasks = [eval_task]

        _config = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
        hidden_size = getattr(_config, "hidden_size", 4096)
        intermediate_size = getattr(_config, "intermediate_size", 14336)
        num_channels = args.moc_channels if args.moc_channels is not None else (intermediate_size * 25) // 100

        config["finetune_eval"] = True
        config["finetune_dataset"] = args.dataset
        config["finetune_epochs"] = args.epochs
        config["finetune_batch_size"] = args.batch_size
        config["variants"] = FINETUNE_EVAL_VARIANTS

        timer = Timer()
        timer.start("total")
        finetuned_results = {}

        for vid in FINETUNE_EVAL_VARIANTS:
            print("\n" + "=" * 80)
            print(f"VARIANT: {vid} — build → freeze non-FFN → finetune → save")
            print("=" * 80)
            model = build_model_for_variant(
                args.model_path, vid, load_kwargs,
                hidden_size, intermediate_size, num_channels,
                training_mode=True,
            )
            print(f"  Freezing all except FFN for {vid}...")
            freeze_all_except_ffn(model)
            out_dir = os.path.join(args.output_dir, f"finetuned_{vid}")
            os.makedirs(out_dir, exist_ok=True)
            finetune_moc_model(
                model=model,
                tokenizer=tokenizer,
                dataset_name=args.dataset,
                output_dir=out_dir,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                cache_dir=args.cache_dir,
            )
            del model
            torch.cuda.empty_cache()

            print(f"\n--- Eval after finetune: {vid} ---")
            model = build_model_for_variant(
                args.model_path, vid, load_kwargs,
                hidden_size, intermediate_size, num_channels,
                training_mode=False,
            )
            ckpt_dir = os.path.join(out_dir, "final_model")
            state = load_state_dict_from_checkpoint(ckpt_dir)
            model.load_state_dict(state, strict=True)
            model.eval()
            finetuned_results[vid] = evaluate_all_tasks(
                model, tokenizer, datasets,
                device=primary_device, max_samples=args.max_samples, timer=timer, prefix=f"{vid}_"
            )
            del model
            torch.cuda.empty_cache()

        timer.stop("total")
        print("\n" + "=" * 80)
        print("FINETUNED EVAL RESULTS (trained task only:", eval_task, ")")
        print("=" * 80)
        print_results_table(finetuned_results, eval_tasks, timer, variant_order=FINETUNE_EVAL_VARIANTS)
        save_results_to_json(finetuned_results, timer.times, config, args.output_dir)
        create_visualization(finetuned_results, timer.times, args.output_dir)
        num_tasks = len(eval_tasks)
        if num_tasks > 0:
            for v in FINETUNE_EVAL_VARIANTS:
                if v in finetuned_results:
                    task_accs = [finetuned_results[v][t]["accuracy"] for t in eval_tasks if t in finetuned_results[v]]
                    avg = sum(task_accs) / len(task_accs) if task_accs else 0.0
                    print(f"  Avg {v}: {avg*100:.2f}%")
        print(f"Total time: {timer.format_time(timer.get('total'))}")
        print("\nDone.")
        return

    # -------------------------------------------------------------------------
    # Eval (and optionally both): host task process — Original vs MoC
    # -------------------------------------------------------------------------
    if args.mode in ["eval", "both"]:
        print("\n" + "=" * 80)
        print("LOADING DATASETS")
        print("=" * 80)
        skipped = []
        datasets = {}

        def safe_load(name: str, loader, *a, **kw):
            try:
                data = loader(*a, **kw)
                n = len(data) if hasattr(data, "__len__") else "?"
                print(f"  ✓ {name}: {n} samples")
                return data
            except Exception as e:
                print(f"  ✗ {name}: {str(e)[:80]}")
                skipped.append(name)
                return None

        if "rte" in tasks_to_run:
            d = safe_load("rte", load_rte_dataset, args.cache_dir)
            if d is not None:
                datasets["rte"] = d
        if "boolq" in tasks_to_run:
            d = safe_load("boolq", load_boolq_dataset, args.cache_dir)
            if d is not None:
                datasets["boolq"] = d
        if "winogrande" in tasks_to_run:
            d = safe_load("winogrande", load_winogrande_dataset, args.cache_dir)
            if d is not None:
                datasets["winogrande"] = d
        if "arc_easy" in tasks_to_run:
            d = safe_load("arc_easy", load_arc_easy_dataset, args.cache_dir)
            if d is not None:
                datasets["arc_easy"] = d
        if "arc_challenge" in tasks_to_run:
            d = safe_load("arc_challenge", load_arc_challenge_dataset, args.cache_dir)
            if d is not None:
                datasets["arc_challenge"] = d
        if "openbookqa" in tasks_to_run:
            d = safe_load("openbookqa", load_openbookqa_dataset, args.cache_dir)
            if d is not None:
                datasets["openbookqa"] = d
        if "piqa" in tasks_to_run:
            d = safe_load("piqa", load_piqa_dataset, args.cache_dir)
            if d is not None:
                datasets["piqa"] = d
        if "mmlu" in tasks_to_run:
            d = safe_load("mmlu", load_mmlu_dataset, args.cache_dir)
            if d is not None:
                datasets["mmlu"] = d
        if "longbench" in tasks_to_run:
            d = safe_load("longbench", load_longbench_dataset, args.cache_dir, "qasper")
            if d is not None:
                datasets["longbench"] = d

        if skipped:
            print(f"Skipped: {skipped}")
        if not datasets:
            print("No datasets loaded. Exiting.")
            return

        # Get model dimensions for logic table (Top-K sparsity display)
        _config = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
        _intermediate_size = getattr(_config, "intermediate_size", 14336)
        _num_channels = args.moc_channels if args.moc_channels is not None else ((getattr(_config, "intermediate_size", 14336) * 25) // 100)
        print_sparsity_logic_table(intermediate_size=_intermediate_size, num_channels=_num_channels)

        timer = Timer()
        timer.start("total")
        all_results = {}
        hidden_size = None
        intermediate_size = None
        num_channels = args.moc_channels  # set below from intermediate_size if None

        variants_to_run = args.variants if args.variants else VARIANT_KEYS
        for vid in variants_to_run:
            if vid not in SPARSITY_LOGIC_TABLE:
                print(f"WARNING: unknown variant '{vid}', skipping.")
                continue
            logic = SPARSITY_LOGIC_TABLE[vid]
            print("\n" + "=" * 80)
            print(f"EVALUATING: {logic['desc']} ({vid})")
            print("=" * 80)
            model = AutoModelForCausalLM.from_pretrained(args.model_path, **load_kwargs)
            if hidden_size is None:
                hidden_size = model.config.hidden_size
                intermediate_size = model.config.intermediate_size
                if num_channels is None:
                    num_channels = (intermediate_size * 25) // 100  # 25% default
            if logic["sparse_all"]:
                pat = logic.get("sparse_pattern") or "2:4"
                ratio = logic.get("sparse_topk_ratio") if pat == "topk" else 0.5
                replace_linear_with_sparse(model, pattern=pat, topk_ratio=ratio)
            elif logic["sparse_ffn"]:
                pat = logic.get("sparse_pattern") or "2:4"
                ratio = logic.get("sparse_topk_ratio") if pat == "topk" else 0.5
                replace_linear_with_sparse_ffn_only(model, pattern=pat, topk_ratio=ratio)
            if logic["moc_pattern"] is not None:
                # moc_topk_ratio: 0.25 = 25%, 0.5 = 50%; None/1.0 = use num_channels
                if logic["moc_pattern"] == "topk" and logic.get("moc_topk_ratio") is not None:
                    moc_k = int(intermediate_size * logic["moc_topk_ratio"])
                else:
                    moc_k = num_channels
                moc_config = MoCConfig(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_channels=moc_k,
                    training_mode=False,
                    use_gradient_checkpointing=True,
                    sparsity_pattern=logic["moc_pattern"],
                )
                replace_mlp_with_moc(model, moc_config)
            prac_rank = logic.get("prac_rank")
            if prac_rank is not None:
                if logic.get("prac_all"):
                    replace_linear_with_prac(model, principal_rank=prac_rank, random_rank=prac_rank)
                elif logic.get("prac_ffn"):
                    replace_linear_with_prac_ffn_only(model, principal_rank=prac_rank, random_rank=prac_rank)
            all_results[vid] = evaluate_all_tasks(
                model, tokenizer, datasets,
                device=primary_device, max_samples=args.max_samples, timer=timer, prefix=f"{vid}_"
            )
            del model
            torch.cuda.empty_cache()

            print_results_table(all_results, tasks_to_run, timer, highlight_variant=vid, variant_order=variants_to_run)

        timer.stop("total")

        print_results_table(all_results, tasks_to_run, timer, variant_order=variants_to_run)
        save_results_to_json(all_results, timer.times, config, args.output_dir)
        create_visualization(all_results, timer.times, args.output_dir)

        first_variant = next((v for v in variants_to_run if v in all_results), None)
        num_tasks = len(all_results[first_variant]) if first_variant else 0
        if num_tasks > 0:
            for v in variants_to_run:
                if v in all_results:
                    avg = sum(all_results[v][t]["accuracy"] for t in all_results[v]) / num_tasks
                    print(f"  Avg {v}: {avg*100:.2f}%")
            print(f"Total time: {timer.format_time(timer.get('total'))}")

    # -------------------------------------------------------------------------
    # Finetune (and optionally after "both" comparison)
    # -------------------------------------------------------------------------
    if args.mode in ["finetune", "both"]:
        print("\n" + "=" * 80)
        print("FINE-TUNING MoC MODEL")
        print("=" * 80)
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **load_kwargs)
        hidden_size = model.config.hidden_size
        intermediate_size = model.config.intermediate_size
        num_channels = args.moc_channels or (model.config.intermediate_size * 25) // 100
        moc_config = MoCConfig(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_channels=num_channels,
            training_mode=True,
            use_gradient_checkpointing=True,
        )
        model = replace_mlp_with_moc(model, moc_config)
        if args.use_lora:
            model = setup_lora_for_moc(model, r=args.lora_r)
        finetune_moc_model(
            model=model,
            tokenizer=tokenizer,
            dataset_name=args.dataset,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            cache_dir=args.cache_dir,
        )
        del model
        torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()