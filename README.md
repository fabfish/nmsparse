# nmsparse

Sparsity experiments for LLMs: **Mixture-of-Channels (MoC)** and **activation sparsity** (2:4, 2:8, Top-K) on Llama, with zero-shot evaluation.

- **MoC** (arXiv 2511.09323): Top-K or block-wise (2:4, 2:8) channel selection in FFN via gate.
- **sparse_all**: activation sparsity (2:4 / 2:8 / Top-K) on **all** linear layers.
- **sparse_ffn**: same activation sparsity only on FFN linears (gate_proj, up_proj, down_proj).
- **moc+sparse**: sparse_all (2:4) + MoC in FFN.

Baseline: `test_rte_sparsity.py` (Original vs 2:4 sparse). Full benchmark: `moc_finetune_eval.py` (all variants, multi-task).

---

## Evaluation Results (Full Run)

**Setup**: Llama-3.1-8B-Instruct, 10 tasks (rte, boolq, winogrande, arc_easy, arc_challenge, openbookqa, piqa, mmlu, longbench), zero-shot. **Total time: 10h 36m 24s**.

### Accuracy by task (10 tasks)

| Task | Orig | S_T25 | S_T50 | S_24 | S_28 | F_T25 | F_T50 | F_24 | F_28 | MoC_T25 | MoC_T50 | MoC_24 | MoC_28 | T25+S | T50+S | 24+S | 28+S | N |
|------|------|-------|-------|------|------|-------|-------|------|------|---------|---------|--------|--------|-------|-------|------|------|---|
| rte | **81.2** | 57.0 | 70.4 | 63.9 | 47.3 | 61.0 | **76.2** | 69.7 | 55.2 | 46.9 | 46.6 | 47.7 | 46.9 | 50.2 | 48.0 | 45.9 | 46.9 | 277 |
| boolq | **85.4** | 51.5 | 80.5 | 69.5 | 37.8 | 66.5 | **82.1** | 79.9 | 55.0 | 41.0 | 40.7 | 44.1 | 39.8 | 46.9 | 45.5 | 49.1 | 44.7 | 3270 |
| winogrande | 50.5 | 49.6 | 49.5 | 49.6 | 49.6 | 49.4 | 49.3 | **51.8** | 49.2 | 51.6 | 51.1 | 50.4 | **52.2** | 51.5 | 49.6 | 49.4 | 49.7 | 1267 |
| arc_easy | **89.3** | 24.0 | 69.8 | 35.6 | 26.8 | 39.3 | **82.5** | 71.2 | 27.2 | 24.6 | 23.3 | 27.4 | 24.7 | 24.4 | 26.3 | 25.1 | 24.9 | 570 |
| arc_challenge | **75.3** | 24.1 | 59.2 | 31.8 | 21.7 | 24.8 | **67.6** | 56.2 | 27.1 | 22.7 | 24.4 | 26.8 | 23.4 | 24.7 | 23.1 | 26.1 | 25.8 | 299 |
| openbookqa | **74.2** | 25.8 | 53.2 | 30.4 | 24.4 | 28.0 | **64.4** | 54.0 | 24.4 | 22.4 | 21.4 | 20.2 | 22.6 | 23.4 | 24.2 | 23.4 | 22.0 | 500 |
| piqa | **64.7** | 49.4 | 53.8 | 49.3 | 49.6 | 50.3 | 56.6 | 51.4 | 49.5 | 49.6 | 49.8 | 49.4 | 49.7 | 49.9 | 49.4 | 49.9 | 49.9 | 1838 |
| mmlu | **58.0** | 25.0 | 47.7 | 33.3 | 23.8 | 26.1 | 50.4 | 43.9 | 25.3 | 24.3 | 24.8 | 24.8 | 23.6 | 24.8 | 25.0 | 24.8 | 28.1 | 1531 |
| longbench | **8.7** | 3.2 | 8.6 | 8.1 | 2.2 | 6.7 | **9.2** | 8.6 | 5.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 200 |
| **AVG** | **65.3** | 34.4 | 54.7 | 41.3 | 31.5 | 39.1 | **59.8** | 54.1 | 35.3 | 31.5 | 31.4 | 32.3 | 31.4 | 32.9 | 32.3 | 32.6 | 32.4 | |

**Legend**: Orig=base | S_*=sparse_all, F_*=sparse_ffn | MoC_*=MoC (FFN only) | *+S=MoC+sparse | T25/T50=Top-K 25%/50%, 24/28=2:4 2:8.

### Average accuracy by variant

| Variant | Avg Acc |
|---------|---------|
| original | **65.25%** |
| sparse_ffn_topk_50 | **59.81%** |
| sparse_ffn_2_4 | 54.08% |
| sparse_all_topk_50 | 54.74% |
| sparse_ffn_2_8 | 35.32% |
| sparse_all_2_4 | 41.28% |
| sparse_ffn_topk | 39.11% |
| moc_topk_sparse | 32.87% |
| moc_2_4_sparse | 32.62% |
| moc_2_8_sparse | 32.45% |
| moc_topk_50_sparse | 32.35% |
| moc_2_4 | 32.29% |
| moc_2_8 | 31.44% |
| moc_topk | 31.46% |
| moc_topk_50 | 31.36% |
| sparse_all_2_8 | 31.47% |
| sparse_all_topk | 34.39% |

### Summary

- **Original** (no sparsity) is best on average (**65.3%** over 10 tasks).
- **sparse_ffn** retains the most accuracy among sparsity variants: **sparse_ffn_topk_50** (59.8% avg) and **sparse_ffn_2_4** (54.1%); **sparse_all_topk_50** (54.7%) is best among sparse_all.
- **MoC-only** and **MoC+sparse** sit around **31–33%** avg (large drop without fine-tuning). **longbench** (generation) collapses to 0% for all MoC variants.
- **winogrande** and **piqa** show smaller gaps; **rte / boolq / arc_*** and **mmlu** show larger drops under sparsity/MoC.

---

## Usage

```bash
# Quick test (4 tasks, <30s each)
python moc_finetune_eval.py --mode eval --quick_test

# Full eval (all tasks, all 17 variants)
python moc_finetune_eval.py --mode eval

# Optional: GPU selection
python moc_finetune_eval.py --mode eval --use_gpus 0 1 --exclude_gpus 2
```

See `moc_finetune_eval.py` for `--tasks`, `--moc_channels`, `--max_samples`, etc.
