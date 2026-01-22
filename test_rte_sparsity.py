#!/usr/bin/env python3
"""
Test script to compare original Llama-3.1-8B-Instruct model with 
2:4 semi-structured activation sparsity version on multiple benchmarks.

Supports multi-GPU parallel evaluation with timing and visualization.

Supported datasets:
- RTE (GLUE)
- BoolQ (SuperGLUE)
- WinoGrande
- ARC-Easy, ARC-Challenge
- OpenBookQA
- PIQA
- MMLU
- LongBench

Usage:
    export HF_ENDPOINT=https://hf-mirror.com
    export HF_TOKEN=your_huggingface_token  # Optional, helps avoid rate limiting
    python test_rte_sparsity.py
"""

import os
import json
import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from tqdm import tqdm
from collections import defaultdict
import subprocess

# Set HuggingFace mirror endpoint
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Load HF_TOKEN from local file if exists
def load_hf_token(token_file: str = ".hf_token") -> Optional[str]:
    """
    Load HuggingFace token from local file.
    
    Args:
        token_file: Path to the token file (default: .hf_token)
        
    Returns:
        Token string if file exists, None otherwise
    """
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

# Try to load token from file, fallback to environment variable
hf_token = load_hf_token()
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


# ============================================================================
# 2:4 Semi-Structured Sparsity Implementation
# ============================================================================

def apply_2_4_sparsity(x: torch.Tensor) -> torch.Tensor:
    """
    Apply 2:4 semi-structured sparsity to the input tensor.
    
    For every 4 consecutive elements along the last dimension,
    keep the 2 elements with the largest absolute values and zero out the rest.
    
    Args:
        x: Input tensor of shape (..., N) where N must be divisible by 4
        
    Returns:
        Sparse tensor with same shape, where 2 out of every 4 elements are zeroed
    """
    original_shape = x.shape
    last_dim = original_shape[-1]
    
    # Handle case where last dimension is not divisible by 4
    if last_dim % 4 != 0:
        # Pad to make it divisible by 4
        pad_size = 4 - (last_dim % 4)
        x = F.pad(x, (0, pad_size), mode='constant', value=0)
        padded = True
    else:
        pad_size = 0
        padded = False
    
    # Reshape to (..., N/4, 4)
    new_shape = x.shape[:-1] + (x.shape[-1] // 4, 4)
    x_reshaped = x.view(new_shape)
    
    # Get absolute values and find top-2 indices in each group of 4
    abs_vals = x_reshaped.abs()
    
    # Get indices of top-2 values in each group of 4
    _, top2_indices = abs_vals.topk(k=2, dim=-1)
    
    # Create mask: 1 for top-2 positions, 0 for others
    mask = torch.zeros_like(x_reshaped)
    mask.scatter_(-1, top2_indices, 1.0)
    
    # Apply mask
    sparse_x = x_reshaped * mask
    
    # Reshape back to original shape
    sparse_x = sparse_x.view(x.shape)
    
    # Remove padding if added
    if padded:
        sparse_x = sparse_x[..., :last_dim]
    
    return sparse_x


class SparseActivationLinear(nn.Module):
    """
    Linear layer wrapper that applies 2:4 semi-structured sparsity to input activations.
    
    This module wraps a standard nn.Linear and applies 2:4 sparsity to the input
    activations before the linear transformation.
    """
    
    def __init__(self, original_linear: nn.Linear):
        """
        Initialize from an existing nn.Linear module.
        
        Args:
            original_linear: The original nn.Linear module to wrap
        """
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        
        # Copy weights and bias from original linear layer
        self.weight = original_linear.weight
        self.bias = original_linear.bias
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with 2:4 sparse activations.
        
        Args:
            x: Input tensor of shape (..., in_features)
            
        Returns:
            Output tensor of shape (..., out_features)
        """
        # Apply 2:4 sparsity to input activations
        sparse_x = apply_2_4_sparsity(x)
        
        # Perform linear transformation
        return F.linear(sparse_x, self.weight, self.bias)
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, sparse_activation=2:4'


# ============================================================================
# Model Reconstruction
# ============================================================================

def replace_linear_with_sparse(
    model: nn.Module,
    exclude_names: Optional[list] = None
) -> nn.Module:
    """
    Recursively replace all nn.Linear layers with SparseActivationLinear.
    
    Args:
        model: The model to modify (modified in-place)
        exclude_names: List of module name patterns to exclude from replacement
        
    Returns:
        The modified model
    """
    if exclude_names is None:
        exclude_names = ['lm_head', 'embed']  # Usually don't sparsify these
    
    def should_replace(name: str) -> bool:
        """Check if module should be replaced based on name."""
        for pattern in exclude_names:
            if pattern in name:
                return False
        return True
    
    def replace_recursive(module: nn.Module, prefix: str = '') -> None:
        """Recursively replace Linear layers."""
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            
            if isinstance(child, nn.Linear) and should_replace(full_name):
                # Replace with SparseActivationLinear
                sparse_linear = SparseActivationLinear(child)
                setattr(module, name, sparse_linear)
            else:
                # Recurse into child modules
                replace_recursive(child, full_name)
    
    replace_recursive(model)
    return model


def list_linear_layers(model: nn.Module) -> Dict[str, List[str]]:
    """
    List all Linear and SparseActivationLinear layer names in the model.
    
    Args:
        model: The model to analyze
        
    Returns:
        Dictionary with lists of layer names
    """
    linear_names = []
    sparse_names = []
    
    for name, module in model.named_modules():
        if isinstance(module, SparseActivationLinear):
            sparse_names.append(name)
        elif isinstance(module, nn.Linear):
            linear_names.append(name)
    
    return {'linear': linear_names, 'sparse_activation_linear': sparse_names}


def count_linear_layers(model: nn.Module) -> Dict[str, int]:
    """
    Count the number of Linear and SparseActivationLinear layers.
    
    Args:
        model: The model to analyze
        
    Returns:
        Dictionary with counts
    """
    linear_count = 0
    sparse_count = 0
    
    for module in model.modules():
        if isinstance(module, SparseActivationLinear):
            sparse_count += 1
        elif isinstance(module, nn.Linear):
            linear_count += 1
    
    return {'linear': linear_count, 'sparse_activation_linear': sparse_count}


# ============================================================================
# Dataset Loading with Local Cache
# ============================================================================

def get_local_cache_path(cache_dir: str, dataset_name: str, subset_name: Optional[str] = None) -> Path:
    """Get the local cache path for a dataset."""
    if subset_name:
        return Path(cache_dir) / "local_cache" / f"{dataset_name}_{subset_name}.json"
    return Path(cache_dir) / "local_cache" / f"{dataset_name}.json"


def save_dataset_to_local(dataset: Any, cache_path: Path) -> None:
    """Save dataset to local JSON file."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Convert to list of dicts
    data = [dict(item) for item in dataset]
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  Saved to local cache: {cache_path}")


def load_dataset_from_local(cache_path: Path) -> List[Dict]:
    """Load dataset from local JSON file."""
    with open(cache_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_dataset_with_cache(
    dataset_name: str,
    subset_name: Optional[str] = None,
    cache_dir: Optional[str] = "/data/datasets/",
    split: str = "validation",
    trust_remote_code: bool = False
) -> Any:
    """
    Load dataset with local cache support. First download saves to local JSON,
    subsequent loads read from local cache.
    
    Args:
        dataset_name: Name of the dataset (e.g., "glue", "super_glue")
        subset_name: Subset name (e.g., "rte", "boolq")
        cache_dir: Directory to cache the dataset.
        split: Dataset split to load (default: "validation")
        trust_remote_code: Whether to trust remote code for dataset loading
        
    Returns:
        The dataset (as list of dicts if from local cache, or HF dataset)
    """
    full_name = f"{dataset_name}/{subset_name}" if subset_name else dataset_name
    
    # Check local cache first
    if cache_dir:
        local_cache_path = get_local_cache_path(cache_dir, dataset_name, subset_name)
        local_cache_path = Path(str(local_cache_path).replace(".json", f"_{split}.json"))
        
        if local_cache_path.exists():
            print(f"Loading {full_name} ({split}) from local cache: {local_cache_path}")
            return load_dataset_from_local(local_cache_path)
    
    # Download from HuggingFace
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
        
        # Get the requested split
        if split in dataset:
            data = dataset[split]
        elif "test" in dataset and split == "validation":
            print(f"  Note: '{split}' not found, using 'test' split instead")
            data = dataset["test"]
        else:
            available_splits = list(dataset.keys())
            print(f"  Warning: '{split}' not found. Available: {available_splits}")
            data = dataset[available_splits[0]]
        
        # Save to local cache
        if cache_dir:
            try:
                save_dataset_to_local(data, local_cache_path)
            except Exception as e:
                print(f"  Warning: Could not save to local cache: {e}")
        
        return data
        
    except Exception as e:
        print(f"Error loading {full_name}: {e}")
        raise


# Dataset-specific loaders
def load_rte_dataset(cache_dir: str = "/data/datasets/") -> Any:
    """Load RTE dataset from GLUE benchmark."""
    return load_dataset_with_cache("glue", "rte", cache_dir, split="validation")


def load_boolq_dataset(cache_dir: str = "/data/datasets/") -> Any:
    """Load BoolQ dataset from SuperGLUE benchmark."""
    return load_dataset_with_cache("super_glue", "boolq", cache_dir, split="validation")


def load_winogrande_dataset(cache_dir: str = "/data/datasets/") -> Any:
    """Load WinoGrande dataset."""
    return load_dataset_with_cache("winogrande", "winogrande_xl", cache_dir, split="validation")


def load_arc_easy_dataset(cache_dir: str = "/data/datasets/") -> Any:
    """Load ARC-Easy dataset."""
    return load_dataset_with_cache("allenai/ai2_arc", "ARC-Easy", cache_dir, split="validation")


def load_arc_challenge_dataset(cache_dir: str = "/data/datasets/") -> Any:
    """Load ARC-Challenge dataset."""
    return load_dataset_with_cache("allenai/ai2_arc", "ARC-Challenge", cache_dir, split="validation")


def load_openbookqa_dataset(cache_dir: str = "/data/datasets/") -> Any:
    """Load OpenBookQA dataset."""
    return load_dataset_with_cache("allenai/openbookqa", "main", cache_dir, split="validation")


def load_piqa_dataset(cache_dir: str = "/data/datasets/") -> Any:
    """Load PIQA dataset."""
    # Use piqa from the datasets hub directly (not ybisk/piqa which has script issues)
    return load_dataset_with_cache("piqa", None, cache_dir, split="validation", trust_remote_code=True)


def load_mmlu_dataset(cache_dir: str = "/data/datasets/", subject: str = "all") -> Any:
    """
    Load MMLU dataset.
    
    Args:
        cache_dir: Cache directory
        subject: MMLU subject to load, or "all" for all subjects
        
    Returns:
        Dataset samples
    """
    if subject == "all":
        return load_dataset_with_cache("cais/mmlu", "all", cache_dir, split="validation")
    else:
        return load_dataset_with_cache("cais/mmlu", subject, cache_dir, split="validation")


def load_longbench_dataset(cache_dir: str = "/data/datasets/", task: str = "qasper") -> Any:
    """
    Load LongBench dataset.
    
    Args:
        cache_dir: Cache directory
        task: LongBench task name (e.g., "qasper", "multifieldqa_en", "narrativeqa")
        
    Returns:
        Dataset samples
    """
    return load_dataset_with_cache("THUDM/LongBench", task, cache_dir, split="test", trust_remote_code=True)


# ============================================================================
# Prompt Templates for Each Dataset
# ============================================================================

def create_rte_prompt(premise: str, hypothesis: str) -> str:
    """Create a zero-shot prompt for RTE task."""
    return f'''Given the premise: "{premise}"

Question: Does this imply the following hypothesis: "{hypothesis}"?

Answer (Yes or No):'''


def create_boolq_prompt(passage: str, question: str) -> str:
    """Create a zero-shot prompt for BoolQ task."""
    return f'''Passage: "{passage}"

Question: {question}

Answer (Yes or No):'''


def create_winogrande_prompt(sentence: str, option1: str, option2: str) -> str:
    """Create a zero-shot prompt for WinoGrande task."""
    return f'''Complete the sentence by choosing the correct option.

Sentence: {sentence}

Option 1: {option1}
Option 2: {option2}

Which option correctly fills the blank? Answer with just the number (1 or 2):'''


def create_arc_prompt(question: str, choices: List[str], choice_labels: List[str]) -> str:
    """Create a zero-shot prompt for ARC task."""
    choices_text = "\n".join([f"{label}. {text}" for label, text in zip(choice_labels, choices)])
    return f'''Question: {question}

{choices_text}

Answer with just the letter:'''


def create_openbookqa_prompt(question: str, choices: List[str], choice_labels: List[str]) -> str:
    """Create a zero-shot prompt for OpenBookQA task."""
    choices_text = "\n".join([f"{label}. {text}" for label, text in zip(choice_labels, choices)])
    return f'''Question: {question}

{choices_text}

Answer with just the letter:'''


def create_piqa_prompt(goal: str, sol1: str, sol2: str) -> str:
    """Create a zero-shot prompt for PIQA task."""
    return f'''Goal: {goal}

Solution 1: {sol1}
Solution 2: {sol2}

Which solution is better? Answer with just the number (1 or 2):'''


def create_mmlu_prompt(question: str, choices: List[str]) -> str:
    """Create a zero-shot prompt for MMLU task."""
    choice_labels = ['A', 'B', 'C', 'D']
    choices_text = "\n".join([f"{label}. {text}" for label, text in zip(choice_labels, choices)])
    return f'''Question: {question}

{choices_text}

Answer with just the letter (A, B, C, or D):'''


def create_longbench_prompt(context: str, question: str) -> str:
    """Create a prompt for LongBench task."""
    # Truncate context if too long (for display purposes)
    max_context_len = 4000
    if len(context) > max_context_len:
        context = context[:max_context_len] + "..."
    return f'''Context: {context}

Question: {question}

Answer:'''


def get_token_logprob(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    target_token: str,
    device: str = "cuda"
) -> float:
    """
    Get the log probability of a target token given a prompt.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: The input prompt
        target_token: The token to get probability for (e.g., "Yes" or "No")
        device: Device to run on
        
    Returns:
        Log probability of the target token
    """
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Get target token id
    target_ids = tokenizer.encode(target_token, add_special_tokens=False)
    if len(target_ids) > 0:
        target_id = target_ids[0]
    else:
        target_id = tokenizer.encode(" " + target_token, add_special_tokens=False)[0]
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Get log probabilities for the last position
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
    """
    Get log probabilities for multiple choice options.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: The input prompt
        choices: List of choice tokens (e.g., ["A", "B", "C", "D"])
        device: Device to run on
        
    Returns:
        List of log probabilities for each choice
    """
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Get log probabilities for the last position
    last_logits = logits[0, -1, :]
    log_probs = F.log_softmax(last_logits, dim=-1)
    
    # Get log prob for each choice
    choice_logprobs = []
    for choice in choices:
        target_ids = tokenizer.encode(choice, add_special_tokens=False)
        if len(target_ids) > 0:
            target_id = target_ids[0]
        else:
            target_id = tokenizer.encode(" " + choice, add_special_tokens=False)[0]
        choice_logprobs.append(log_probs[target_id].item())
    
    return choice_logprobs


def evaluate_rte_zero_shot(
    model: nn.Module,
    tokenizer: Any,
    dataset: Any,
    device: str = "cuda",
    max_samples: Optional[int] = None
) -> Dict[str, float]:
    """
    Evaluate model on RTE dataset using zero-shot classification.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        dataset: The RTE validation dataset
        device: Device to run on
        max_samples: Maximum number of samples to evaluate (None for all)
        
    Returns:
        Dictionary containing accuracy and other metrics
    """
    model.eval()
    
    correct = 0
    total = 0
    
    # RTE labels: 0 = entailment, 1 = not_entailment
    label_map = {0: "Yes", 1: "No"}
    
    samples = dataset if max_samples is None else dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"Evaluating RTE on {len(samples)} samples...")
    
    for sample in tqdm(samples, desc="Evaluating RTE"):
        premise = sample["sentence1"]
        hypothesis = sample["sentence2"]
        true_label = sample["label"]
        
        # Create prompt
        prompt = create_rte_prompt(premise, hypothesis)
        
        # Get log probabilities for "Yes" and "No"
        yes_logprob = get_token_logprob(model, tokenizer, prompt, "Yes", device)
        no_logprob = get_token_logprob(model, tokenizer, prompt, "No", device)
        
        # Predict based on higher probability
        predicted = 0 if yes_logprob > no_logprob else 1
        
        if predicted == true_label:
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total
    }


def evaluate_boolq_zero_shot(
    model: nn.Module,
    tokenizer: Any,
    dataset: Any,
    device: str = "cuda",
    max_samples: Optional[int] = None
) -> Dict[str, float]:
    """Evaluate model on BoolQ dataset using zero-shot classification."""
    model.eval()
    correct = 0
    total = 0
    
    samples = dataset if max_samples is None else (
        dataset[:max_samples] if isinstance(dataset, list) else dataset.select(range(min(max_samples, len(dataset))))
    )
    
    print(f"Evaluating BoolQ on {len(samples)} samples...")
    
    for sample in tqdm(samples, desc="Evaluating BoolQ"):
        passage = sample["passage"]
        question = sample["question"]
        true_label = sample["label"]
        
        prompt = create_boolq_prompt(passage, question)
        yes_logprob = get_token_logprob(model, tokenizer, prompt, "Yes", device)
        no_logprob = get_token_logprob(model, tokenizer, prompt, "No", device)
        
        predicted = True if yes_logprob > no_logprob else False
        if predicted == true_label:
            correct += 1
        total += 1
    
    return {"accuracy": correct / total if total > 0 else 0.0, "correct": correct, "total": total}


def evaluate_winogrande_zero_shot(
    model: nn.Module,
    tokenizer: Any,
    dataset: Any,
    device: str = "cuda",
    max_samples: Optional[int] = None
) -> Dict[str, float]:
    """Evaluate model on WinoGrande dataset."""
    model.eval()
    correct = 0
    total = 0
    
    samples = dataset if max_samples is None else (
        dataset[:max_samples] if isinstance(dataset, list) else dataset.select(range(min(max_samples, len(dataset))))
    )
    
    print(f"Evaluating WinoGrande on {len(samples)} samples...")
    
    for sample in tqdm(samples, desc="Evaluating WinoGrande"):
        sentence = sample["sentence"]
        option1 = sample["option1"]
        option2 = sample["option2"]
        answer = sample["answer"]  # "1" or "2"
        
        prompt = create_winogrande_prompt(sentence, option1, option2)
        logprobs = get_choice_logprobs(model, tokenizer, prompt, ["1", "2"], device)
        
        predicted = "1" if logprobs[0] > logprobs[1] else "2"
        if predicted == answer:
            correct += 1
        total += 1
    
    return {"accuracy": correct / total if total > 0 else 0.0, "correct": correct, "total": total}


def evaluate_arc_zero_shot(
    model: nn.Module,
    tokenizer: Any,
    dataset: Any,
    device: str = "cuda",
    max_samples: Optional[int] = None,
    task_name: str = "ARC"
) -> Dict[str, float]:
    """Evaluate model on ARC dataset (Easy or Challenge)."""
    model.eval()
    correct = 0
    total = 0
    
    samples = dataset if max_samples is None else (
        dataset[:max_samples] if isinstance(dataset, list) else dataset.select(range(min(max_samples, len(dataset))))
    )
    
    print(f"Evaluating {task_name} on {len(samples)} samples...")
    
    for sample in tqdm(samples, desc=f"Evaluating {task_name}"):
        question = sample["question"]
        choices_data = sample["choices"]
        answer_key = sample["answerKey"]
        
        # Handle both dict and list formats
        if isinstance(choices_data, dict):
            choice_labels = choices_data["label"]
            choice_texts = choices_data["text"]
        else:
            choice_labels = [c["label"] for c in choices_data]
            choice_texts = [c["text"] for c in choices_data]
        
        prompt = create_arc_prompt(question, choice_texts, choice_labels)
        logprobs = get_choice_logprobs(model, tokenizer, prompt, choice_labels, device)
        
        predicted_idx = logprobs.index(max(logprobs))
        predicted = choice_labels[predicted_idx]
        
        if predicted == answer_key:
            correct += 1
        total += 1
    
    return {"accuracy": correct / total if total > 0 else 0.0, "correct": correct, "total": total}


def evaluate_openbookqa_zero_shot(
    model: nn.Module,
    tokenizer: Any,
    dataset: Any,
    device: str = "cuda",
    max_samples: Optional[int] = None
) -> Dict[str, float]:
    """Evaluate model on OpenBookQA dataset."""
    model.eval()
    correct = 0
    total = 0
    
    samples = dataset if max_samples is None else (
        dataset[:max_samples] if isinstance(dataset, list) else dataset.select(range(min(max_samples, len(dataset))))
    )
    
    print(f"Evaluating OpenBookQA on {len(samples)} samples...")
    
    for sample in tqdm(samples, desc="Evaluating OpenBookQA"):
        question = sample["question_stem"]
        choices_data = sample["choices"]
        answer_key = sample["answerKey"]
        
        if isinstance(choices_data, dict):
            choice_labels = choices_data["label"]
            choice_texts = choices_data["text"]
        else:
            choice_labels = [c["label"] for c in choices_data]
            choice_texts = [c["text"] for c in choices_data]
        
        prompt = create_openbookqa_prompt(question, choice_texts, choice_labels)
        logprobs = get_choice_logprobs(model, tokenizer, prompt, choice_labels, device)
        
        predicted_idx = logprobs.index(max(logprobs))
        predicted = choice_labels[predicted_idx]
        
        if predicted == answer_key:
            correct += 1
        total += 1
    
    return {"accuracy": correct / total if total > 0 else 0.0, "correct": correct, "total": total}


def evaluate_piqa_zero_shot(
    model: nn.Module,
    tokenizer: Any,
    dataset: Any,
    device: str = "cuda",
    max_samples: Optional[int] = None
) -> Dict[str, float]:
    """Evaluate model on PIQA dataset."""
    model.eval()
    correct = 0
    total = 0
    
    samples = dataset if max_samples is None else (
        dataset[:max_samples] if isinstance(dataset, list) else dataset.select(range(min(max_samples, len(dataset))))
    )
    
    print(f"Evaluating PIQA on {len(samples)} samples...")
    
    for sample in tqdm(samples, desc="Evaluating PIQA"):
        goal = sample["goal"]
        sol1 = sample["sol1"]
        sol2 = sample["sol2"]
        label = sample["label"]  # 0 or 1
        
        prompt = create_piqa_prompt(goal, sol1, sol2)
        logprobs = get_choice_logprobs(model, tokenizer, prompt, ["1", "2"], device)
        
        predicted = 0 if logprobs[0] > logprobs[1] else 1
        if predicted == label:
            correct += 1
        total += 1
    
    return {"accuracy": correct / total if total > 0 else 0.0, "correct": correct, "total": total}


def evaluate_mmlu_zero_shot(
    model: nn.Module,
    tokenizer: Any,
    dataset: Any,
    device: str = "cuda",
    max_samples: Optional[int] = None
) -> Dict[str, float]:
    """Evaluate model on MMLU dataset."""
    model.eval()
    correct = 0
    total = 0
    
    samples = dataset if max_samples is None else (
        dataset[:max_samples] if isinstance(dataset, list) else dataset.select(range(min(max_samples, len(dataset))))
    )
    
    print(f"Evaluating MMLU on {len(samples)} samples...")
    
    for sample in tqdm(samples, desc="Evaluating MMLU"):
        question = sample["question"]
        choices = sample["choices"]
        answer = sample["answer"]  # 0, 1, 2, or 3
        
        prompt = create_mmlu_prompt(question, choices)
        choice_labels = ["A", "B", "C", "D"]
        logprobs = get_choice_logprobs(model, tokenizer, prompt, choice_labels, device)
        
        predicted = logprobs.index(max(logprobs))
        if predicted == answer:
            correct += 1
        total += 1
    
    return {"accuracy": correct / total if total > 0 else 0.0, "correct": correct, "total": total}


def evaluate_longbench_zero_shot(
    model: nn.Module,
    tokenizer: Any,
    dataset: Any,
    device: str = "cuda",
    max_samples: Optional[int] = None,
    max_length: int = 4096
) -> Dict[str, float]:
    """
    Evaluate model on LongBench dataset.
    Uses F1 score for evaluation since LongBench is a generation task.
    """
    model.eval()
    
    samples = dataset if max_samples is None else (
        dataset[:max_samples] if isinstance(dataset, list) else dataset.select(range(min(max_samples, len(dataset))))
    )
    
    print(f"Evaluating LongBench on {len(samples)} samples...")
    
    total_f1 = 0.0
    total = 0
    
    for sample in tqdm(samples, desc="Evaluating LongBench"):
        context = sample["context"]
        question = sample["input"]
        answers = sample["answers"]  # List of acceptable answers
        
        # Truncate context to fit in model's context window
        prompt = create_longbench_prompt(context, question)
        
        # Tokenize and truncate
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode response
        generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # Calculate F1 score against all acceptable answers
        best_f1 = 0.0
        for answer in answers:
            f1 = compute_f1(generated, answer)
            best_f1 = max(best_f1, f1)
        
        total_f1 += best_f1
        total += 1
    
    avg_f1 = total_f1 / total if total > 0 else 0.0
    
    return {"accuracy": avg_f1, "f1_score": avg_f1, "total": total}


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score between prediction and ground truth."""
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
    f1 = 2 * precision * recall / (precision + recall)
    
    return f1


# ============================================================================
# GPU Utilities
# ============================================================================

def get_all_gpu_info() -> List[Dict[str, Any]]:
    """
    Get information about all available GPUs.
    
    Returns:
        List of dictionaries with GPU info
    """
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
                    'index': int(parts[0]),
                    'name': parts[1],
                    'memory_used': int(parts[2]),
                    'memory_total': int(parts[3]),
                    'memory_free': int(parts[4]),
                    'utilization': int(parts[5])
                })
    except Exception as e:
        print(f"Warning: Could not query GPU status: {e}")
        # Fallback to PyTorch detection
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append({
                'index': i,
                'name': props.name,
                'memory_total': props.total_memory // (1024**2),
                'memory_used': 0,
                'memory_free': props.total_memory // (1024**2),
                'utilization': 0
            })
    
    return gpus


def get_available_gpus(min_free_memory_mb: int = 20000) -> List[int]:
    """
    Get list of available GPUs with sufficient free memory.
    
    Args:
        min_free_memory_mb: Minimum free memory in MB required
        
    Returns:
        List of GPU indices that are available
    """
    gpus = get_all_gpu_info()
    available = []
    
    for gpu in gpus:
        if gpu['memory_free'] >= min_free_memory_mb:
            available.append(gpu['index'])
    
    return available if available else [0] if gpus else []


def get_free_gpu() -> int:
    """
    Find the GPU with the most free memory.
    
    Returns:
        GPU index with the most free memory
    """
    if not torch.cuda.is_available():
        return -1
    
    gpus = get_all_gpu_info()
    if not gpus:
        return 0
    
    best_gpu = max(gpus, key=lambda x: x['memory_free'])
    return best_gpu['index']


def print_gpu_info(gpu_ids: Optional[List[int]] = None) -> None:
    """
    Print detailed information about GPUs.
    
    Args:
        gpu_ids: List of GPU indices to print info for. If None, prints all.
    """
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
    
    print("\n" + "-" * 80)
    print(f"{'Index':<6} {'Name':<40} {'Used':<12} {'Free':<12} {'Total':<12} {'Util':<8}")
    print("-" * 80)
    
    for gpu in gpus:
        marker = " *" if gpu_ids and gpu['index'] in gpu_ids else ""
        print(f"{gpu['index']:<6} {gpu['name']:<40} {gpu['memory_used']:<12} {gpu['memory_free']:<12} {gpu['memory_total']:<12} {gpu['utilization']}%{marker}")
    
    print("-" * 80)
    print("(* = selected for evaluation)")
    print("=" * 80 + "\n")


# ============================================================================
# Timing Utilities
# ============================================================================

class Timer:
    """Simple timer for measuring execution time."""
    
    def __init__(self):
        self.times = {}
        self.start_times = {}
    
    def start(self, name: str):
        """Start timing a named section."""
        self.start_times[name] = time.time()
    
    def stop(self, name: str) -> float:
        """Stop timing and return elapsed time in seconds."""
        if name not in self.start_times:
            return 0.0
        elapsed = time.time() - self.start_times[name]
        self.times[name] = elapsed
        return elapsed
    
    def get(self, name: str) -> float:
        """Get recorded time for a section."""
        return self.times.get(name, 0.0)
    
    def format_time(self, seconds: float) -> str:
        """Format time in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = seconds % 60
            return f"{mins}m {secs:.1f}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours}h {mins}m {secs:.0f}s"


# ============================================================================
# Visualization
# ============================================================================

def create_visualization(
    results: Dict[str, Dict[str, Dict[str, float]]],
    timing_info: Dict[str, float],
    output_dir: str = "results"
) -> str:
    """
    Create visualization of evaluation results.
    
    Args:
        results: Dictionary with 'original' and 'sparse' results for each task
        timing_info: Dictionary with timing information
        output_dir: Directory to save visualizations
        
    Returns:
        Path to the saved visualization
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Warning: matplotlib not installed. Skipping visualization.")
        return ""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get task names and accuracies
    tasks = list(results['original'].keys())
    original_accs = [results['original'][t]['accuracy'] * 100 for t in tasks]
    sparse_accs = [results['sparse'][t]['accuracy'] * 100 for t in tasks]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Evaluation: Original vs 2:4 Sparse Activation', fontsize=14, fontweight='bold')
    
    # 1. Bar chart comparison
    ax1 = axes[0, 0]
    x = np.arange(len(tasks))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, original_accs, width, label='Original', color='#2ecc71', alpha=0.8)
    bars2 = ax1.bar(x + width/2, sparse_accs, width, label='2:4 Sparse', color='#e74c3c', alpha=0.8)
    
    ax1.set_xlabel('Task')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Accuracy Comparison by Task')
    ax1.set_xticks(x)
    ax1.set_xticklabels(tasks, rotation=45, ha='right')
    ax1.legend()
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
    # 2. Difference chart
    ax2 = axes[0, 1]
    diffs = [s - o for o, s in zip(original_accs, sparse_accs)]
    colors = ['#e74c3c' if d < 0 else '#2ecc71' for d in diffs]
    bars = ax2.bar(tasks, diffs, color=colors, alpha=0.8)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Task')
    ax2.set_ylabel('Accuracy Difference (%)')
    ax2.set_title('Sparse - Original Accuracy Difference')
    ax2.set_xticklabels(tasks, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, diff in zip(bars, diffs):
        height = bar.get_height()
        ax2.annotate(f'{diff:+.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3 if height >= 0 else -12), textcoords="offset points", 
                    ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
    
    # 3. Timing chart
    ax3 = axes[1, 0]
    timing_tasks = [k for k in timing_info.keys() if k.startswith('original_') or k.startswith('sparse_')]
    
    if timing_tasks:
        orig_times = {k.replace('original_', ''): v for k, v in timing_info.items() if k.startswith('original_')}
        sparse_times = {k.replace('sparse_', ''): v for k, v in timing_info.items() if k.startswith('sparse_')}
        
        common_tasks = sorted(set(orig_times.keys()) & set(sparse_times.keys()))
        if common_tasks:
            x = np.arange(len(common_tasks))
            orig_t = [orig_times[t] for t in common_tasks]
            sparse_t = [sparse_times[t] for t in common_tasks]
            
            bars1 = ax3.bar(x - width/2, orig_t, width, label='Original', color='#3498db', alpha=0.8)
            bars2 = ax3.bar(x + width/2, sparse_t, width, label='2:4 Sparse', color='#9b59b6', alpha=0.8)
            
            ax3.set_xlabel('Task')
            ax3.set_ylabel('Time (seconds)')
            ax3.set_title('Evaluation Time by Task')
            ax3.set_xticks(x)
            ax3.set_xticklabels(common_tasks, rotation=45, ha='right')
            ax3.legend()
            ax3.grid(axis='y', alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No timing data available', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Evaluation Time by Task')
    
    # 4. Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create summary data
    avg_orig = np.mean(original_accs)
    avg_sparse = np.mean(sparse_accs)
    total_time = timing_info.get('total', 0)
    
    summary_text = f"""
    ╔══════════════════════════════════════════════════════╗
    ║                  EVALUATION SUMMARY                   ║
    ╠══════════════════════════════════════════════════════╣
    ║  Tasks Evaluated: {len(tasks):<35} ║
    ║  Average Original Accuracy: {avg_orig:>6.2f}%                 ║
    ║  Average Sparse Accuracy:   {avg_sparse:>6.2f}%                 ║
    ║  Average Difference:        {avg_sparse - avg_orig:>+6.2f}%                 ║
    ║  Total Evaluation Time:     {total_time/60:>6.1f} minutes           ║
    ╚══════════════════════════════════════════════════════╝
    """
    
    ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, f"evaluation_results_{timestamp}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved to: {output_path}")
    return output_path


def save_results_to_json(
    results: Dict[str, Dict[str, Dict[str, float]]],
    timing_info: Dict[str, float],
    config: Dict[str, Any],
    output_dir: str = "results"
) -> str:
    """
    Save evaluation results to JSON file.
    
    Args:
        results: Dictionary with 'original' and 'sparse' results
        timing_info: Dictionary with timing information
        config: Configuration dictionary
        output_dir: Directory to save results
        
    Returns:
        Path to saved JSON file
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "results": results,
        "timing": timing_info,
        "summary": {}
    }
    
    # Calculate summary
    tasks = list(results.get('original', {}).keys())
    if tasks:
        orig_accs = [results['original'][t]['accuracy'] for t in tasks]
        sparse_accs = [results['sparse'][t]['accuracy'] for t in tasks]
        output_data["summary"] = {
            "num_tasks": len(tasks),
            "avg_original_accuracy": sum(orig_accs) / len(orig_accs),
            "avg_sparse_accuracy": sum(sparse_accs) / len(sparse_accs),
            "avg_difference": (sum(sparse_accs) - sum(orig_accs)) / len(tasks)
        }
    
    output_path = os.path.join(output_dir, f"evaluation_results_{timestamp}.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {output_path}")
    return output_path


# ============================================================================
# Main Function
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
    """
    Evaluate model on all loaded datasets with timing.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        datasets: Dictionary of dataset name -> dataset
        device: Device to run on
        max_samples: Maximum samples per dataset
        timer: Timer object for timing (optional)
        prefix: Prefix for timing keys (e.g., "original_" or "sparse_")
        
    Returns:
        Dictionary of task name -> results
    """
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
            results[task_name]['time'] = elapsed
            print(f"  {task_name}: {results[task_name]['accuracy']:.4f} (time: {timer.format_time(elapsed)})")
        else:
            print(f"  {task_name}: {results[task_name]['accuracy']:.4f}")
    
    return results


def print_results_table(
    original_results: Dict[str, Dict[str, float]],
    sparse_results: Dict[str, Dict[str, float]],
    tasks_to_run: List[str],
    timer: Timer
) -> None:
    """Print a detailed results table with timing information."""
    
    print("\n" + "=" * 100)
    print("DETAILED RESULTS TABLE")
    print("=" * 100)
    
    # Header
    print(f"\n{'Task':<15} {'Original':<10} {'Sparse':<10} {'Diff':<10} {'Orig Time':<12} {'Sparse Time':<12} {'Samples':<8}")
    print("-" * 100)
    
    total_original = 0.0
    total_sparse = 0.0
    total_orig_time = 0.0
    total_sparse_time = 0.0
    num_tasks = 0
    
    for task_name in tasks_to_run:
        if task_name in original_results and task_name in sparse_results:
            orig_acc = original_results[task_name]['accuracy']
            sparse_acc = sparse_results[task_name]['accuracy']
            diff = sparse_acc - orig_acc
            total = original_results[task_name]['total']
            
            orig_time = timer.get(f"original_{task_name}")
            sparse_time = timer.get(f"sparse_{task_name}")
            
            print(f"{task_name:<15} {orig_acc:<10.4f} {sparse_acc:<10.4f} {diff:+10.4f} "
                  f"{timer.format_time(orig_time):<12} {timer.format_time(sparse_time):<12} {total:<8}")
            
            total_original += orig_acc
            total_sparse += sparse_acc
            total_orig_time += orig_time
            total_sparse_time += sparse_time
            num_tasks += 1
    
    print("-" * 100)
    
    if num_tasks > 0:
        avg_original = total_original / num_tasks
        avg_sparse = total_sparse / num_tasks
        avg_diff = avg_sparse - avg_original
        print(f"{'AVERAGE':<15} {avg_original:<10.4f} {avg_sparse:<10.4f} {avg_diff:+10.4f} "
              f"{timer.format_time(total_orig_time):<12} {timer.format_time(total_sparse_time):<12}")
    
    print("=" * 100)
    
    # Print total times
    print(f"\nTotal evaluation time:")
    print(f"  Original model: {timer.format_time(total_orig_time)}")
    print(f"  Sparse model:   {timer.format_time(total_sparse_time)}")
    print(f"  Overall:        {timer.format_time(timer.get('total'))}")


def main():
    """Main function to run the comparison experiment."""
    
    # ========================================================================
    # Configuration
    # ========================================================================
    model_path = "/data/models/Llama-3.1-8B-Instruct"
    dataset_cache_dir = "/data/datasets/"
    output_dir = "results"
    max_samples = None  # Set to a number for quick testing, None for full evaluation
    
    # GPU Configuration
    # Option 1: Specify which GPUs to use (e.g., [1, 2, 3] to use GPUs 1, 2, 3)
    # Option 2: Set to None to auto-detect all available GPUs
    use_gpus = [1, 2, 3]  # Set to None for auto-detect, or specify GPU IDs like [1, 2, 3]
    
    # GPUs to exclude (e.g., [0] to exclude GPU 0 which is being used by someone else)
    exclude_gpus = [0]  # GPUs to never use
    
    # Minimum free memory (MB) required for a GPU to be considered available
    min_free_memory_mb = 20000
    
    # Tasks to evaluate (comment out tasks you don't want to run)
    tasks_to_run = [
        "rte",
        "boolq",
        "winogrande",
        "arc_easy",
        "arc_challenge",
        "openbookqa",
        "piqa",
        "mmlu",
        "longbench",
    ]
    
    # Initialize timer
    timer = Timer()
    timer.start('total')
    
    # ========================================================================
    # GPU Detection and Multi-GPU Setup
    # ========================================================================
    print("\n" + "=" * 80)
    print("MULTI-GPU DETECTION AND CONFIGURATION")
    print("=" * 80)
    
    if torch.cuda.is_available():
        all_gpus = get_all_gpu_info()
        print(f"\nTotal GPUs detected: {len(all_gpus)}")
        
        # Get GPUs with sufficient free memory
        available_gpus = get_available_gpus(min_free_memory_mb=min_free_memory_mb)
        print(f"GPUs with >{min_free_memory_mb}MB free: {available_gpus}")
        
        # Apply exclusion list
        if exclude_gpus:
            available_gpus = [g for g in available_gpus if g not in exclude_gpus]
            print(f"After excluding {exclude_gpus}: {available_gpus}")
        
        # Apply user-specified GPU list if provided
        if use_gpus is not None:
            # Filter to only use specified GPUs that are also available
            gpu_ids = [g for g in use_gpus if g in available_gpus]
            if not gpu_ids:
                print(f"Warning: None of specified GPUs {use_gpus} are available. Using all available GPUs.")
                gpu_ids = available_gpus
            else:
                print(f"Using specified GPUs (filtered by availability): {gpu_ids}")
        else:
            # Auto-detect: use all available GPUs
            gpu_ids = available_gpus
            print(f"Auto-detected available GPUs: {gpu_ids}")
        
        if not gpu_ids:
            print("ERROR: No available GPUs found! Check GPU memory or exclusion settings.")
            return
        
        # Setup device map for multi-GPU using max_memory to control which GPUs to use
        if len(gpu_ids) >= 2:
            # Multi-GPU: use max_memory to specify which GPUs can be used
            # This is more reliable than CUDA_VISIBLE_DEVICES after torch import
            max_memory = {i: "90GiB" for i in gpu_ids}  # Allow up to 90GB per GPU
            max_memory["cpu"] = "30GiB"  # Fallback to CPU if needed
            
            device_map = "auto"  # Let accelerate distribute across specified GPUs
            primary_device = f"cuda:{gpu_ids[0]}"
            print(f"Using {len(gpu_ids)} GPUs: {gpu_ids}")
            print(f"Max memory config: {max_memory}")
        else:
            # Single GPU
            gpu_id = gpu_ids[0]
            max_memory = {gpu_id: "90GiB", "cpu": "30GiB"}
            device_map = {"": gpu_id}
            primary_device = f"cuda:{gpu_id}"
            print(f"Using single GPU: {gpu_id}")
        
        print_gpu_info(gpu_ids)
    else:
        gpu_ids = []
        device_map = "cpu"
        primary_device = "cpu"
        max_memory = None
        print("CUDA not available. Using CPU.")
    
    # Configuration for model loading
    config = {
        "model_path": model_path,
        "dataset_cache_dir": dataset_cache_dir,
        "max_samples": max_samples,
        "tasks": tasks_to_run,
        "gpu_ids": gpu_ids,
        "excluded_gpus": exclude_gpus,
        "specified_gpus": use_gpus,
        "device_map": str(device_map),
        "timestamp": datetime.now().isoformat()
    }
    
    print("\n" + "=" * 80)
    print("EVALUATION CONFIGURATION")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Specified GPUs: {use_gpus if use_gpus else 'Auto-detect'}")
    print(f"Excluded GPUs: {exclude_gpus}")
    print(f"Final GPUs to use: {gpu_ids if gpu_ids else 'CPU'}")
    print(f"Device map: {device_map}")
    print(f"Max samples per task: {max_samples if max_samples else 'All'}")
    print(f"Tasks: {', '.join(tasks_to_run)}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load all datasets with error handling
    print("\n" + "=" * 80)
    print("LOADING DATASETS")
    print("=" * 80)
    
    timer.start('load_datasets')
    datasets = {}
    skipped_tasks = []
    
    def safe_load_dataset(name: str, loader_func, *args, **kwargs):
        """Safely load a dataset, return None if failed."""
        try:
            data = loader_func(*args, **kwargs)
            print(f"  ✓ {name}: {len(data)} samples")
            return data
        except Exception as e:
            print(f"  ✗ {name}: Failed to load - {str(e)[:100]}")
            skipped_tasks.append(name)
            return None
    
    if "rte" in tasks_to_run:
        data = safe_load_dataset("RTE", load_rte_dataset, dataset_cache_dir)
        if data is not None:
            datasets["rte"] = data
    
    if "boolq" in tasks_to_run:
        data = safe_load_dataset("BoolQ", load_boolq_dataset, dataset_cache_dir)
        if data is not None:
            datasets["boolq"] = data
    
    if "winogrande" in tasks_to_run:
        data = safe_load_dataset("WinoGrande", load_winogrande_dataset, dataset_cache_dir)
        if data is not None:
            datasets["winogrande"] = data
    
    if "arc_easy" in tasks_to_run:
        data = safe_load_dataset("ARC-Easy", load_arc_easy_dataset, dataset_cache_dir)
        if data is not None:
            datasets["arc_easy"] = data
    
    if "arc_challenge" in tasks_to_run:
        data = safe_load_dataset("ARC-Challenge", load_arc_challenge_dataset, dataset_cache_dir)
        if data is not None:
            datasets["arc_challenge"] = data
    
    if "openbookqa" in tasks_to_run:
        data = safe_load_dataset("OpenBookQA", load_openbookqa_dataset, dataset_cache_dir)
        if data is not None:
            datasets["openbookqa"] = data
    
    if "piqa" in tasks_to_run:
        data = safe_load_dataset("PIQA", load_piqa_dataset, dataset_cache_dir)
        if data is not None:
            datasets["piqa"] = data
    
    if "mmlu" in tasks_to_run:
        data = safe_load_dataset("MMLU", load_mmlu_dataset, dataset_cache_dir)
        if data is not None:
            datasets["mmlu"] = data
    
    if "longbench" in tasks_to_run:
        data = safe_load_dataset("LongBench (qasper)", load_longbench_dataset, dataset_cache_dir, task="qasper")
        if data is not None:
            datasets["longbench"] = data
    
    timer.stop('load_datasets')
    print(f"\nDatasets loaded in {timer.format_time(timer.get('load_datasets'))}")
    print(f"Successfully loaded: {len(datasets)} datasets")
    if skipped_tasks:
        print(f"Skipped (failed to load): {skipped_tasks}")
    
    if not datasets:
        print("\nERROR: No datasets were successfully loaded. Exiting.")
        return
    
    # ========================================================================
    # Evaluate Original Model
    # ========================================================================
    print("\n" + "=" * 80)
    print("EVALUATING ORIGINAL MODEL")
    print("=" * 80)
    
    timer.start('load_original_model')
    load_kwargs = {
        "torch_dtype": torch.float16,
        "device_map": device_map,
        "local_files_only": True
    }
    # Add max_memory for multi-GPU distribution
    if max_memory is not None:
        load_kwargs["max_memory"] = max_memory
    
    original_model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    timer.stop('load_original_model')
    
    layer_counts = count_linear_layers(original_model)
    print(f"Original model layer counts: {layer_counts}")
    print(f"Model loaded in {timer.format_time(timer.get('load_original_model'))}")
    
    original_results = evaluate_all_tasks(
        original_model,
        tokenizer,
        datasets,
        device=primary_device,
        max_samples=max_samples,
        timer=timer,
        prefix="original_"
    )
    
    # Free up memory
    del original_model
    torch.cuda.empty_cache()
    
    # ========================================================================
    # Evaluate Sparse Activation Model
    # ========================================================================
    print("\n" + "=" * 80)
    print("EVALUATING 2:4 SPARSE ACTIVATION MODEL")
    print("=" * 80)
    
    timer.start('load_sparse_model')
    sparse_model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    
    # Debug: print all Linear layers before replacement
    print("\nLinear layers before replacement:")
    before_layers = list_linear_layers(sparse_model)
    print(f"  Total Linear layers: {len(before_layers['linear'])}")
    if len(before_layers['linear']) <= 20:
        for name in before_layers['linear']:
            print(f"    - {name}")
    else:
        print(f"    First 10: {before_layers['linear'][:10]}")
        print(f"    Last 10: {before_layers['linear'][-10:]}")
    
    print("\nReplacing Linear layers with SparseActivationLinear...")
    sparse_model = replace_linear_with_sparse(sparse_model)
    timer.stop('load_sparse_model')
    
    # Debug: print remaining Linear layers after replacement
    layer_counts = count_linear_layers(sparse_model)
    after_layers = list_linear_layers(sparse_model)
    print(f"\nLinear layers after replacement (not replaced):")
    for name in after_layers['linear']:
        print(f"    - {name}")
    
    print(f"\nSparse model layer counts: {layer_counts}")
    print(f"Sparse model loaded and converted in {timer.format_time(timer.get('load_sparse_model'))}")
    
    sparse_results = evaluate_all_tasks(
        sparse_model,
        tokenizer,
        datasets,
        device=primary_device,
        max_samples=max_samples,
        timer=timer,
        prefix="sparse_"
    )
    
    # Stop total timer
    timer.stop('total')
    
    # ========================================================================
    # Results Summary
    # ========================================================================
    print_results_table(original_results, sparse_results, tasks_to_run, timer)
    
    # ========================================================================
    # Save Results and Visualization
    # ========================================================================
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    # Prepare results for saving
    all_results = {
        'original': original_results,
        'sparse': sparse_results
    }
    
    # Save JSON results
    json_path = save_results_to_json(all_results, timer.times, config, output_dir)
    
    # Create visualization
    viz_path = create_visualization(all_results, timer.times, output_dir)
    
    # ========================================================================
    # Final Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    
    num_tasks = len([t for t in tasks_to_run if t in original_results])
    avg_orig = sum(original_results[t]['accuracy'] for t in original_results) / num_tasks if num_tasks > 0 else 0
    avg_sparse = sum(sparse_results[t]['accuracy'] for t in sparse_results) / num_tasks if num_tasks > 0 else 0
    
    print(f"\nSummary:")
    print(f"  Tasks evaluated: {num_tasks}")
    print(f"  GPUs used: {gpu_ids if gpu_ids else 'CPU'}")
    print(f"  Average Original Accuracy: {avg_orig*100:.2f}%")
    print(f"  Average Sparse Accuracy:   {avg_sparse*100:.2f}%")
    print(f"  Average Difference:        {(avg_sparse-avg_orig)*100:+.2f}%")
    print(f"  Total Time: {timer.format_time(timer.get('total'))}")
    print(f"\nResults saved to: {output_dir}/")
    
    # Clean up
    del sparse_model
    torch.cuda.empty_cache()
    
    print("\nDone!")
    

if __name__ == "__main__":
    main()
