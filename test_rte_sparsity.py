#!/usr/bin/env python3
"""
Test script to compare original Llama-3.1-8B-Instruct model with 
2:4 semi-structured activation sparsity version on RTE and BoolQ datasets.

Usage:
    export HF_ENDPOINT=https://hf-mirror.com
    export HF_TOKEN=your_huggingface_token  # Optional, helps avoid rate limiting
    python test_rte_sparsity.py
"""

import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from tqdm import tqdm

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
# RTE Zero-Shot Evaluation
# ============================================================================

def load_dataset_with_cache(
    dataset_name: str,
    subset_name: Optional[str] = None,
    cache_dir: Optional[str] = "/data/datasets/"
) -> Any:
    """
    Load dataset with cache directory handling.
    
    Args:
        dataset_name: Name of the dataset (e.g., "glue", "super_glue")
        subset_name: Subset name (e.g., "rte", "boolq")
        cache_dir: Directory to cache the dataset. If None or not writable, uses default cache.
        
    Returns:
        The validation dataset
    """
    # Get HF token from environment if available
    hf_token = os.environ.get("HF_TOKEN", None)
    
    full_name = f"{dataset_name}/{subset_name}" if subset_name else dataset_name
    
    # Check if cache_dir is writable, if not use default cache
    if cache_dir:
        try:
            # Try to create directory if it doesn't exist
            os.makedirs(cache_dir, exist_ok=True)
            # Try to create a test file to check write permissions
            test_file = os.path.join(cache_dir, ".write_test")
            try:
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
                print(f"Loading {full_name} dataset to {cache_dir}...")
            except (PermissionError, OSError):
                print(f"Warning: {cache_dir} is not writable, using default cache directory...")
                cache_dir = None
        except (PermissionError, OSError):
            print(f"Warning: Cannot create {cache_dir}, using default cache directory...")
            cache_dir = None
    
    if cache_dir is None:
        print(f"Loading {full_name} dataset (using default cache directory)...")
    
    if subset_name:
        dataset = load_dataset(
            dataset_name, 
            subset_name,
            cache_dir=cache_dir,
            token=hf_token
        )
    else:
        dataset = load_dataset(
            dataset_name,
            cache_dir=cache_dir,
            token=hf_token
        )
    return dataset["validation"]


def load_rte_dataset(cache_dir: Optional[str] = "/data/datasets/") -> Any:
    """
    Load RTE dataset from GLUE benchmark.
    
    Args:
        cache_dir: Directory to cache the dataset. If None or not writable, uses default cache.
        
    Returns:
        The RTE validation dataset
    """
    return load_dataset_with_cache("glue", "rte", cache_dir)


def load_boolq_dataset(cache_dir: Optional[str] = "/data/datasets/") -> Any:
    """
    Load BoolQ dataset from SuperGLUE benchmark.
    
    Args:
        cache_dir: Directory to cache the dataset. If None or not writable, uses default cache.
        
    Returns:
        The BoolQ validation dataset
    """
    return load_dataset_with_cache("super_glue", "boolq", cache_dir)


def create_rte_prompt(premise: str, hypothesis: str) -> str:
    """
    Create a zero-shot prompt for RTE task.
    
    Args:
        premise: The premise text
        hypothesis: The hypothesis text
        
    Returns:
        Formatted prompt string
    """
    prompt = f'''Given the premise: "{premise}"

Question: Does this imply the following hypothesis: "{hypothesis}"?

Answer (Yes or No):'''
    return prompt


def create_boolq_prompt(passage: str, question: str) -> str:
    """
    Create a zero-shot prompt for BoolQ task.
    
    Args:
        passage: The passage text
        question: The question about the passage
        
    Returns:
        Formatted prompt string
    """
    prompt = f'''Passage: "{passage}"

Question: {question}

Answer (Yes or No):'''
    return prompt


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
    """
    Evaluate model on BoolQ dataset using zero-shot classification.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        dataset: The BoolQ validation dataset
        device: Device to run on
        max_samples: Maximum number of samples to evaluate (None for all)
        
    Returns:
        Dictionary containing accuracy and other metrics
    """
    model.eval()
    
    correct = 0
    total = 0
    
    # BoolQ labels: True = Yes, False = No
    samples = dataset if max_samples is None else dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"Evaluating BoolQ on {len(samples)} samples...")
    
    for sample in tqdm(samples, desc="Evaluating BoolQ"):
        passage = sample["passage"]
        question = sample["question"]
        true_label = sample["label"]  # Boolean: True or False
        
        # Create prompt
        prompt = create_boolq_prompt(passage, question)
        
        # Get log probabilities for "Yes" and "No"
        yes_logprob = get_token_logprob(model, tokenizer, prompt, "Yes", device)
        no_logprob = get_token_logprob(model, tokenizer, prompt, "No", device)
        
        # Predict based on higher probability
        predicted = True if yes_logprob > no_logprob else False
        
        if predicted == true_label:
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total
    }


# ============================================================================
# GPU Utilities
# ============================================================================

def get_free_gpu() -> int:
    """
    Find the GPU with the most free memory.
    
    Returns:
        GPU index with the most free memory
    """
    if not torch.cuda.is_available():
        return -1
    
    import subprocess
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True
        )
        
        best_gpu = 0
        min_used = float('inf')
        
        for line in result.stdout.strip().split('\n'):
            parts = line.split(',')
            if len(parts) >= 3:
                gpu_idx = int(parts[0].strip())
                memory_used = int(parts[1].strip())
                
                if memory_used < min_used:
                    min_used = memory_used
                    best_gpu = gpu_idx
        
        return best_gpu
    except Exception as e:
        print(f"Warning: Could not query GPU status: {e}")
        return 0


def print_gpu_info(gpu_id: int) -> None:
    """
    Print detailed information about the selected GPU.
    
    Args:
        gpu_id: The GPU index to print info for
    """
    if not torch.cuda.is_available():
        print("CUDA is not available!")
        return
    
    print("\n" + "=" * 70)
    print("GPU INFORMATION")
    print("=" * 70)
    
    # Get GPU properties
    props = torch.cuda.get_device_properties(gpu_id)
    
    print(f"Selected GPU: {gpu_id}")
    print(f"GPU Name: {props.name}")
    print(f"GPU Memory Total: {props.total_memory / 1024**3:.2f} GB")
    print(f"CUDA Capability: {props.major}.{props.minor}")
    print(f"Multi-Processor Count: {props.multi_processor_count}")
    
    # Current memory usage
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
        reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
        print(f"Memory Allocated: {allocated:.2f} GB")
        print(f"Memory Reserved: {reserved:.2f} GB")
    
    # Also print all GPU status
    print("\nAll GPU Status:")
    print("-" * 70)
    import subprocess
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu', 
             '--format=csv,noheader'],
            capture_output=True, text=True, check=True
        )
        print(f"{'Index':<6} {'Name':<45} {'Used':<12} {'Total':<12} {'Util':<8}")
        print("-" * 70)
        for line in result.stdout.strip().split('\n'):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 5:
                print(f"{parts[0]:<6} {parts[1]:<45} {parts[2]:<12} {parts[3]:<12} {parts[4]:<8}")
    except Exception as e:
        print(f"Could not query nvidia-smi: {e}")
    
    print("=" * 70 + "\n")


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main function to run the comparison experiment."""
    
    # Configuration
    model_path = "/data/models/Llama-3.1-8B-Instruct"
    dataset_cache_dir = "/data/datasets/"
    max_samples = None  # Set to a number for quick testing, None for full evaluation
    
    # Find the best available GPU (most free memory)
    if torch.cuda.is_available():
        gpu_id = get_free_gpu()
        device = f"cuda:{gpu_id}"
        print_gpu_info(gpu_id)
    else:
        device = "cpu"
        gpu_id = -1
    
    print("=" * 70)
    print("RTE & BoolQ Zero-Shot Evaluation: Original vs 2:4 Activation Sparse Model")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Device: {device}")
    print(f"Max samples: {max_samples if max_samples else 'All'}")
    print()
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True  # Model is local, no need to download
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load datasets
    print("\nLoading datasets...")
    rte_dataset = load_rte_dataset(dataset_cache_dir)
    print(f"Loaded RTE: {len(rte_dataset)} validation samples")
    
    boolq_dataset = load_boolq_dataset(dataset_cache_dir)
    print(f"Loaded BoolQ: {len(boolq_dataset)} validation samples")
    print()
    
    # ========================================================================
    # Evaluate Original Model
    # ========================================================================
    print("-" * 70)
    print("Loading and evaluating ORIGINAL model...")
    print("-" * 70)
    
    original_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map={"": gpu_id} if gpu_id >= 0 else "cpu",  # Load entirely to selected GPU
        local_files_only=True  # Model is local, no need to download
    )
    
    layer_counts = count_linear_layers(original_model)
    print(f"Original model layer counts: {layer_counts}")
    
    # Evaluate on RTE
    original_rte_results = evaluate_rte_zero_shot(
        original_model,
        tokenizer,
        rte_dataset,
        device=device,
        max_samples=max_samples
    )
    print(f"\nOriginal Model RTE Results:")
    print(f"  Accuracy: {original_rte_results['accuracy']:.4f} ({original_rte_results['correct']}/{original_rte_results['total']})")
    
    # Evaluate on BoolQ
    original_boolq_results = evaluate_boolq_zero_shot(
        original_model,
        tokenizer,
        boolq_dataset,
        device=device,
        max_samples=max_samples
    )
    print(f"\nOriginal Model BoolQ Results:")
    print(f"  Accuracy: {original_boolq_results['accuracy']:.4f} ({original_boolq_results['correct']}/{original_boolq_results['total']})")
    
    # Free up memory
    del original_model
    torch.cuda.empty_cache()
    
    # ========================================================================
    # Evaluate Sparse Activation Model
    # ========================================================================
    print()
    print("-" * 70)
    print("Loading and evaluating 2:4 SPARSE ACTIVATION model...")
    print("-" * 70)
    
    # Load model again
    sparse_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map={"": gpu_id} if gpu_id >= 0 else "cpu",  # Load entirely to selected GPU
        local_files_only=True  # Model is local, no need to download
    )
    
    # Replace Linear layers with SparseActivationLinear
    print("Replacing Linear layers with SparseActivationLinear...")
    sparse_model = replace_linear_with_sparse(sparse_model)
    
    layer_counts = count_linear_layers(sparse_model)
    print(f"Sparse model layer counts: {layer_counts}")
    
    # Evaluate on RTE
    sparse_rte_results = evaluate_rte_zero_shot(
        sparse_model,
        tokenizer,
        rte_dataset,
        device=device,
        max_samples=max_samples
    )
    print(f"\nSparse Activation Model RTE Results:")
    print(f"  Accuracy: {sparse_rte_results['accuracy']:.4f} ({sparse_rte_results['correct']}/{sparse_rte_results['total']})")
    
    # Evaluate on BoolQ
    sparse_boolq_results = evaluate_boolq_zero_shot(
        sparse_model,
        tokenizer,
        boolq_dataset,
        device=device,
        max_samples=max_samples
    )
    print(f"\nSparse Activation Model BoolQ Results:")
    print(f"  Accuracy: {sparse_boolq_results['accuracy']:.4f} ({sparse_boolq_results['correct']}/{sparse_boolq_results['total']})")
    
    # ========================================================================
    # Comparison Summary
    # ========================================================================
    print()
    print("=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    
    # RTE Results
    print("\n[RTE Task]")
    print(f"{'Model':<30} {'Accuracy':<15} {'Correct/Total':<15}")
    print("-" * 70)
    print(f"{'Original':<30} {original_rte_results['accuracy']:<15.4f} {original_rte_results['correct']}/{original_rte_results['total']}")
    print(f"{'2:4 Sparse Activation':<30} {sparse_rte_results['accuracy']:<15.4f} {sparse_rte_results['correct']}/{sparse_rte_results['total']}")
    rte_diff = sparse_rte_results['accuracy'] - original_rte_results['accuracy']
    print(f"Accuracy Difference: {rte_diff:+.4f} ({rte_diff*100:+.2f}%)")
    
    # BoolQ Results
    print("\n[BoolQ Task]")
    print(f"{'Model':<30} {'Accuracy':<15} {'Correct/Total':<15}")
    print("-" * 70)
    print(f"{'Original':<30} {original_boolq_results['accuracy']:<15.4f} {original_boolq_results['correct']}/{original_boolq_results['total']}")
    print(f"{'2:4 Sparse Activation':<30} {sparse_boolq_results['accuracy']:<15.4f} {sparse_boolq_results['correct']}/{sparse_boolq_results['total']}")
    boolq_diff = sparse_boolq_results['accuracy'] - original_boolq_results['accuracy']
    print(f"Accuracy Difference: {boolq_diff:+.4f} ({boolq_diff*100:+.2f}%)")
    
    # Overall Summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    print(f"{'Task':<15} {'Original':<15} {'Sparse':<15} {'Difference':<15}")
    print("-" * 70)
    print(f"{'RTE':<15} {original_rte_results['accuracy']:<15.4f} {sparse_rte_results['accuracy']:<15.4f} {rte_diff:+.4f}")
    print(f"{'BoolQ':<15} {original_boolq_results['accuracy']:<15.4f} {sparse_boolq_results['accuracy']:<15.4f} {boolq_diff:+.4f}")
    
    avg_original = (original_rte_results['accuracy'] + original_boolq_results['accuracy']) / 2
    avg_sparse = (sparse_rte_results['accuracy'] + sparse_boolq_results['accuracy']) / 2
    avg_diff = avg_sparse - avg_original
    print("-" * 70)
    print(f"{'Average':<15} {avg_original:<15.4f} {avg_sparse:<15.4f} {avg_diff:+.4f}")
    print()
    
    # Clean up
    del sparse_model
    torch.cuda.empty_cache()
    
    print("Evaluation complete!")
    

if __name__ == "__main__":
    main()
