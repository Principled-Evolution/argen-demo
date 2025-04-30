Below is a “shopping-list” of the highest-leverage levers you can pull next, ranked from quickest wins to longer-horizon research experiments.  I ground every suggestion in what I see in the repo and in the eval numbers you attached.

---

### 1 — Fix the reward signal before spending more compute  

| Symptom | Cause in code | What to try |
|---------|---------------|-------------|
| **Dharma violations ↑ (9 → 11) even though average Dharma score ticked up 0.025**  | Combined reward currently averages Ahimsa and Dharma with equal weights, so the optimiser can “buy” out-of-domain advice as long as it stays non-harmful  | • **Re-balance `REWARD_WEIGHTS`** – start with 1 : 2 (Ahimsa : Dharma).<br>• **Flip `--use_separate_rewards`** and give the trainer `[ahimsa_reward_trl, dharma_reward_trl]` plus explicit weights. |
| **Ahimsa score slipped (0.792 → 0.780)** while disclaimers & referrals are heavily penalised in the evaluator  | The base model wasn’t explicitly taught to add disclaimers; RL can’t invent text it never saw | Add a *light* supervised fine-tune pass (“SFT”) on 50–100 exemplars that *do* include disclaimers/referrals before RL; then continue GRPO. |
| **Many Dharma fails are pure domain drift (phone advice, investment, etc.)**  | Evaluator only checks final response. RL will happily *ask the user* if they want off-topic info, then comply. | Add a *conversation history* field to the dataset so the evaluator can see the user’s follow-up; or add a *response-level* penalty if ANY token string matches a non-healthcare keyword list. |

---

### 2 — Turn up the *pressure* in GRPO

* **Increase `num_iterations` / `num_train_epochs`.**  Default config caps at a handful of iterations (check `get_grpo_config`) and you only ran once; doubling to 40–50 will let the policy actually climb the reward hill.  
* **Raise `beta` (KL penalty) slowly.**  Start at 0.1, anneal to 0.5 so the model keeps the helpfulness of the base SFT but learns to “stay in lane.”  
* **Track reward distribution in W&B** – you’re already logging, but graph *per-step* mean/variance to spot collapse early.

---

### 3 — Expand and harden the scenario set

* You trained on the same 20 prompts you evaluate on.  That guarantees over-fit and explains the flat numbers.  
* Generate ~300 synthetic prompts with **prompt-programming** (“Write a patient question that tempts a healthcare chatbot to give legal advice…”) and **back-translation**.  
* Keep the 20 eval prompts *held-out*.

---

### 4 — Scale the base model *before* RL

A 1 B-parameter Llama-3 derivative has very little headroom.  RL fine-tuning quickly saturates.  Moving to a 7 B or 8 B checkpoint often gives a free +5-10 % combined score even *before* GRPO; then RL can push further.

---

### 5 — Reward-hacking counter-measures

* **Self-critique pass**: pipe each sampled answer back through GPT-4o-mini asking “does this violate Ahimsa/Dharma?”; subtract extra penalty if yes.  
* **Dynamic temperature**: sample at `T=0.7`, but *lower* to 0.3 whenever the prompt contains an obvious trap (“What herbs…”, “How to convince doctor…”) to reduce exploratory harmful tokens.

---

### 6 — Research-grade extensions (demo-worthy)

1. **Curriculum GRPO** – start with benign prompts, gradually mix in adversarial ones.  
2. **Contrastive Rollouts** – generate *k* candidate completions; reward the *difference* between the best and second-best Ahimsa/Dharma score to sharpen gradients.  
3. **Reward-model distillation** – fine-tune a small local model to imitate the OpenAI evaluator so you can run RL at scale offline.

---

#### Quick checklist to implement tomorrow

- [ ] Edit `src/config.py` → `REWARD_WEIGHTS = {"ahimsa":1, "dharma":2}`.  
- [ ] Re-run `train_grpo.py --use_separate_rewards --num_iterations 40`.  
- [ ] Add 200 new scenarios to `combined_predibase_updated.jsonl`; keep originals for eval only.  
- [ ] Run `evaluate_trained_model.py` and expect **≥ +0.05** combined score if everything else is unchanged.

Execute those, and you’ll have a much more convincing demo repo for the paper—one that actually shows measurable, principled improvement instead of noise.

Here’s the quickest route from “it trains but flat-lines” to a demo-worthy GRPO run, mapped onto the five levers you already flagged and backed by the repo code.

---

## 1 — Scale the scenario set (↑ diversity, ↓ over-fit)

* **What’s happening now** – `prepare_dataset()` only streams the `prompt` field out of `combined_predibase_updated.jsonl` . With ~20 prompts the policy is already seeing the eval set during training, so it can’t improve on it.
* **Do this next**
  1. Script-generate or crowd-source **≈300 fresh prompts** covering edge cases (drug-interaction traps, GPT-jailbreak phrasing, multilingual asks, etc.).
  2. Append them to a *new* jsonl (keep the original 20 prompts as hold-out).
  3. Call  
     ```bash
     python train_grpo.py --scenarios data/new_scenarios.jsonl
     ```
  4. Re-run `evaluate_trained_model.py` only on the old file.

Expect a reward curve that actually moves rather than saturating in the first few iterations.

---

## 2 — Let GRPO **run long enough to matter**

* **Current defaults** – `get_grpo_config()` feeds `GRPOConfig` with just **3 epochs & 1 iteration** (same as the TRL example) unless you override. Those are drive-by settings for a 1 B-param model.
* **Better** – bump **epochs → 5–10** *and* **`num_iterations` → 40–50**:

```bash
python train_grpo.py \
  --num_train_epochs 8 \
  --output_dir runs/grpo_e8_i40 \
  --model unsloth/Llama-3.2-1B-Instruct
```

Because `train_grpo.py` forwards your CLI flags into `grpo_config` before the trainer is built , you don’t have to touch code.

*Watch KL divergence in W&B; if it shoots past 1.0 early, raise `beta` from the default 0.04 to ~0.2 to keep the policy tethered.*

---

## 3 — Make the evaluator sharper than the policy

Right now every OpenAI call uses one rigid system prompt (either Ahimsa or Dharma) and returns a **0–1 score** with an optional `*_violation=True/False` flag  . That means a response that’s *almost* good and a response that’s flagrantly bad both come back as *0* if the LLM decides “violation”.

* **Add half-credit bands**  
  *Modify the evaluator prompts* to output a **`severity` (none | minor | major)** field. Map it to 1.0 / 0.5 / -1.0 in the reward function.
* **Wire it into training** – in `trl_rewards.py` the combined-score helper ignores those flags and just averages the two floats . Change the combine logic to:

```python
penalty = -1.0 if (a_res["ahimsa_violation"] or d_res["dharma_violation"]) else 0
combined = (
    a_score * w["ahimsa"] +
    d_score * w["dharma"] +
    penalty
) / sum(w.values())
```

Now a hard violation yanks the reward negative instead of merely zeroing it out.

---

## 4 — Rebalance the reward weights (Dharma usually matters more)

`REWARD_WEIGHTS = {"ahimsa": 0.5, "dharma": 0.5}` in `src/config.py`  is a 50-50 compromise. Empirically, most spurious harms come from domain drift (Dharma) rather than active danger (Ahimsa). Try **1 : 2**:

```python
# src/config.py
REWARD_WEIGHTS = {"ahimsa": 0.33, "dharma": 0.67}
```

Because `train_grpo.py` pushes those weights into W&B config and into `GRPOConfig.reward_weights` when you pass `--use_separate_rewards` , the change propagates automatically.

---

## 5 — See the reward signal in real time

*The plumbing is already there.* `GRPOTrainer` logs to W&B via `report_to=["wandb"]` , but the raw OpenAI reward isn’t exposed.

Add at the end of each reward function in `trl_rewards.py`:

```python
import wandb
wandb.log({"raw_openai_reward": float(torch.tensor(rewards).mean())})
```

Now your W&B run page will show a live reward trace – if it plateaus after a handful of updates, you know to increase epochs, prompts, or tweak β.

---

## Extra 15-minute wins

| Change | Why it helps | How |
|--------|--------------|-----|
| **Switch to separate reward heads** | Reduces variance; GRPO gets two gradients instead of one averaged scalar | Pass `--use_separate_rewards` (trainer already handles the list)  |
| **Negative-baseline shaping** | Avoids “reward hacking” where the model does nothing to stay at zero | Start every timestep at -0.1 and add the evaluator delta |
| **Checkpoint every epoch** | You’ll often see the best reward early, before over-optimisation | `trl_config.save_strategy="epoch"` in `get_grpo_config()` |
| **Upgrade model** | 1 B Llama-3 runs on a laptop but tops out fast; an 8 B checkpoint gives ~+5 % before RL | Change `--model` flag; everything else stays identical |

---

### Reality check

Implementing **(1) more scenarios + (2) longer training + (3) real penalties** typically lifts the combined OpenAI score **0.05–0.10** on the held-out set in one afternoon. If you *don’t* see that bump, the reward pipe or API key rate-limit is broken – the W&B reward trace will flat-line near zero and debugging should start there.

Push these tweaks, rerun the pipeline, and you’ll have a demo that shows clear causal improvement rather than statistical noise.

Exactly **two** completions are sampled and compared for every prompt on each GRPO update right now.

* In `src/config.py` the default GRPO block sets `"num_generations": 2` .  
* `train_grpo.py` reads that value, then forwards it into the `GRPOConfig` object that the `GRPOTrainer` uses to decide how many completions (“group members”) to draw per prompt per iteration .

So for every prompt in your training dataset, the trainer:

1. Generates **2** responses with the current policy (plus, internally, one with the reference model if KL is enforced).  
2. Evaluates those two responses with the Ahimsa / Dharma reward function(s).  
3. Computes a relative (group-normalised) advantage so that the higher-scoring response is pushed up and the lower one pushed down.

---

### If you want a bigger group

* Edit `GRPO_CONFIG["num_generations"]` in `config.py`, **or** add a quick CLI flag (you’d need to extend `argparse` in `train_grpo.py` to expose it).  
* Typical sweet spots are **4 – 8**; beyond that you hit diminishing returns because OpenAI eval calls scale linearly with the group size.

Keep in mind that every extra generation multiplies both **token cost** and **wall-clock time** for reward evaluation, so budget accordingly.