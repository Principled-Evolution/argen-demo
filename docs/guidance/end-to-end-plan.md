Belom is a **work-plan you can follow literally line-by-line** to turn the repo into a clean, reproducible GRPO demo for the paper.  
I break it into six phases, each with concrete deliverables and CLI commands. Skip nothing—every step plugs a hole we uncovered above.

## Phase 1 — Config & reward plumbing (1 day)

| Step | File | What to change |
|------|------|----------------|
| 1.1 | `src/config.py` | `REWARD_WEIGHTS = {"ahimsa": 0.33, "dharma": 0.67}` |
| 1.2 | `src/config.py` | `GRPO_CONFIG["num_generations"] = 6` (was 2) |
| 1.3 | `train_grpo.py` | Expose `--num_generations` and forward to `GRPOConfig`. |
| 1.4 | `trl_rewards.py` | *Log rewards/penalties*: At the end of reward calculation, add `wandb.log` calls for raw Ahimsa/Dharma scores, penalty applied, and final combined reward for traceability. |
| 1.5 | `openai_rewards.py` | Confirm `severity` ('none'/'minor'/'major') is requested in OpenAI prompts (verified complete). |
| 1.6 | **Test** | `pytest` or quick script on five prompts to confirm reward float is in [-1,1] and logged components appear correctly in W&B. |

---

## Phase 2 — GRPO training run (≈6 h wall-clock)

| Step | Command | Notes |
|------|---------|-------|
| 2.1 | `python train_grpo.py \`<br>`  --scenarios data/grpo_train_20250430.jsonl \`<br>`  --model unsloth/Llama-3.2-1B-Instruct \`<br>`  --num_train_epochs 8 \`<br>`  --num_iterations 40 \`<br>`  --num_generations 4 \`<br>`  --use_separate_rewards` | GPU ≥ 24 GB recommended. Uses separate reward functions with weights from `config.py`. Costs scale with 40 × 4 × 150 OpenAI eval calls. |
| 2.2 | **Monitor W&B.** Abort if `raw_openai_reward` (or similar logged metric) is flat ≤0.01 after 3 iterations (means evaluator/API broken). | N/A |
| 2.3 | **Checkpoint.** Ensure `save_strategy="epoch"` is in config so you keep 8 checkpoints. | `runs/grpo_e8_i40/checkpoint-*/` |

---

## Phase 3 — Baseline + trained evaluation (2 h)

| Step | Command |
|------|---------|
| 3.1 | **Baseline smoke-test** (unchanged model):<br>`python evaluate_baseline.py --scenarios data/combined_predibase_updated.jsonl --out baseline20.json` |
| 3.2 | **Baseline on Core-60:**<br>`python evaluate_baseline.py --scenarios data/eval_core_60_20250430.jsonl --out baseline_core60.json` |
| 3.3 | **Baseline on Adv-40:**<br>`python evaluate_baseline.py --scenarios data/eval_adv_40_20250430.jsonl --out baseline_adv40.json` |
| 3.4 | **Trained model on three splits:** repeat 3.1–3.3 but call `evaluate_trained_model.py --model runs/grpo_e8_i40/checkpoint-best` and pass `--baseline_results` to calculate deltas. |

Deliverables: six JSON result files plus `metrics_summary.csv` auto-written by the evaluator scripts.

---

## Phase 4 — Analysis & plots (½ day)

| Step | Action |
|------|--------|
| 4.1 | Use the helper notebook (create if absent): load the six JSONs, compute mean Ahimsa, mean Dharma, combined, and ∆ vs baseline for each split. |
| 4.2 | Generate two matplotlib plots via `python_user_visible` if you want in-paper figs:<br>• Reward vs iteration (from W&B CSV).<br>• Bar chart of baseline vs trained scores on Core-60 & Adv-40. |
| 4.3 | Save plots to `fig/` and commit. |

---

## Phase 5 — Paper integration (½ day)

| Step | File | Insert |
|------|------|--------|
| 5.1 | `ArGen_3_2_for_arXiv.pdf` source or LaTeX | Replace Table 2 with fresh Core-60 numbers; add new Table 3 for Adv-40. |
| 5.2 | Methods section | Add one paragraph: "Training employed 150 prompts, 4-way GRPO, β-annealed 0.04 → 0.2 …" |
| 5.3 | Appendix | Include link to Zenodo snapshot of the four JSONLs + inference notebook. |

---

## Phase 6 — Repro & release checklist (≤1 day)

| Step | Action | Done when… |
|------|--------|-----------|
| 6.1 | `scripts/download_models.sh` | Automates pulling UnsLoTH model & checkpoints. |
| 6.2 | `README.md` | Update quick-start commands with new scenario paths. |
| 6.3 | Tag Git | `git tag v0.3-demo-paper && git push --tags` |
| 6.4 | Zenodo / HuggingFace Space | Upload dataset files + best checkpoint; get DOI. |
| 6.5 | **Dry-run** on fresh machine | One-command script reproduces Table 2 numbers within ±0.01. |

---

### Time & cost estimate

| Phase | GPU hrs | OpenAI $ | Wall-clock |
|-------|---------|----------|------------|
| 0–1 | 0 | 0 | 1.5 d |
| 2 | ~6 (A100) | ~$45 (150 × 40 × 4 evals @ $0.0026/1K) | 1 d |
| 3 | ≤1 | ~$4 | 0.5 d |
| 4–6 | 0 | 0 | 1 d |
| **Total** | **7 GPU-hrs** | **≈$50** | **~4 work-days** |

---

### Final deliverables checklist (pin this to GitHub Issues)

- [ ] Four JSONL datasets committed under `data/`.
- [ ] Updated config, reward code, CLI flags.
- [ ] W&B run link with rising reward curve.
- [ ] Six evaluator result JSONs + plots in `fig/`.
- [ ] Paper draft updated, DOI links live.
- [ ] `v0.3-demo-paper` tag reproducible on fresh box.

Stay disciplined on the phases; if you hit a red flag (reward flat, exploding KL), **stop and debug before moving on**—otherwise you'll waste OpenAI calls. Once the numbers move by ≥ +0.05 on Core-60, you're golden for submission.