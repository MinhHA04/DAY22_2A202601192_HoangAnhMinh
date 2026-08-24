# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # NB6 — SFT-only vs SFT+DPO benchmark (OPTIONAL / BONUS)
#
# Runs IFEval, GSM8K and sampled MMLU with lm-eval 0.4.12, then an optional
# AlpacaEval-lite pairwise judge. Each lm-eval invocation runs in its own process,
# so GPU memory is released between adapters/tasks. The stack uses 4-bit
# bitsandbytes + eager attention and never imports Unsloth/xFormers/FlashAttention.
#
# T4 defaults are deliberately bounded. Override the `NB6_*` environment variables
# for wider coverage.

# %% [markdown]
# ## 0. Install benchmark dependencies and configure limits

# %%
import gc
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import torch

subprocess.run(
    [
        sys.executable, "-m", "pip", "install", "-q",
        "lm-eval[hf,ifeval,math]==0.4.12",
    ],
    check=True,
    timeout=1200,
)

COMPUTE_TIER = os.environ.get("COMPUTE_TIER", "T4").upper()
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
# IFEval allows generations up to 1,280 tokens, so a 512-token context can
# truncate or fail before the answer is complete. 2,048 remains safe at batch 1
# for a 4-bit 3B model on a T4 and can be overridden without editing the file.
EVAL_MAX_LEN = int(os.environ.get("NB6_MAX_LENGTH", "2048"))
BATCH_SIZE = int(os.environ.get("NB6_BATCH_SIZE", "1"))
LIMIT_IFEVAL = int(os.environ.get("NB6_LIMIT_IFEVAL", "100"))
LIMIT_GSM8K = int(os.environ.get("NB6_LIMIT_GSM8K", "100"))
# lm-eval applies group limits per MMLU subject; 10 means at most ~570 questions.
LIMIT_MMLU_PER_SUBJECT = int(os.environ.get("NB6_LIMIT_MMLU_PER_SUBJECT", "10"))
LIMIT_ALPACA = int(os.environ.get("NB6_LIMIT_ALPACA", "20"))

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
SFT_PATH = REPO_ROOT / "adapters" / "sft-mini"
DPO_PATH = REPO_ROOT / "adapters" / "dpo"
EVAL_OUT = REPO_ROOT / "data" / "eval"
SCREENSHOT_DIR = REPO_ROOT / "submission" / "screenshots"
EVAL_OUT.mkdir(parents=True, exist_ok=True)
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

for adapter in (SFT_PATH, DPO_PATH):
    assert adapter.joinpath("adapter_config.json").exists(), f"Missing adapter: {adapter}"
assert torch.cuda.is_available(), "NB6 requires a Colab GPU runtime"

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"IFEval={LIMIT_IFEVAL}, GSM8K={LIMIT_GSM8K}, MMLU≤{57 * LIMIT_MMLU_PER_SUBJECT}")
print(f"AlpacaEval-lite={LIMIT_ALPACA}, batch={BATCH_SIZE}, context={EVAL_MAX_LEN}")

# %% [markdown]
# ## 1. Robust lm-eval runner

# %%
def run_lm_eval(adapter_path: Path, task: str, limit: int, num_fewshot: int, label: str):
    """Evaluate one adapter/task in a child process and return the full result JSON."""
    # A unique directory prevents a retry from accidentally reading stale JSON
    # left by an earlier partial run.
    out_dir = EVAL_OUT / f"lm-{label}-{task}-{time.time_ns()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_args = ",".join([
        f"pretrained={BASE_MODEL}",
        f"peft={adapter_path}",
        "load_in_4bit=True",
        "bnb_4bit_compute_dtype=float16",
        "dtype=float16",
        "attn_implementation=eager",
        f"max_length={EVAL_MAX_LEN}",
    ])
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", model_args,
        "--tasks", task,
        "--num_fewshot", str(num_fewshot),
        "--limit", str(limit),
        "--batch_size", str(BATCH_SIZE),
        "--device", "cuda:0",
        "--apply_chat_template",
        "--output_path", str(out_dir),
    ]
    print(f"\n{'=' * 72}\n{label.upper()} · {task} · limit={limit}\n{'=' * 72}")
    proc = subprocess.run(cmd, check=False, timeout=7200)
    if proc.returncode != 0:
        raise RuntimeError(f"lm-eval failed ({label}/{task}) with exit code {proc.returncode}")
    result_files = sorted(out_dir.glob("**/results*.json"), key=lambda p: p.stat().st_mtime)
    if not result_files:
        raise FileNotFoundError(f"lm-eval wrote no results JSON under {out_dir}")
    payload = json.loads(result_files[-1].read_text(encoding="utf-8"))
    gc.collect()
    torch.cuda.empty_cache()
    return payload


def find_metric(payload, preferred_keys, task_prefix=None):
    """Read a group metric or mean matching task metrics without guessing unrelated numbers."""
    for container_name in ("groups", "results"):
        container = payload.get(container_name, {})
        for task_name, values in container.items():
            if task_prefix and not (task_name == task_prefix or task_name.startswith(task_prefix + "_")):
                continue
            for key in preferred_keys:
                if key in values and isinstance(values[key], (int, float)):
                    return float(values[key])

    if task_prefix:
        per_task = []
        for task_name, values in payload.get("results", {}).items():
            if task_name.startswith(task_prefix + "_"):
                for key in preferred_keys:
                    if key in values and isinstance(values[key], (int, float)):
                        per_task.append(float(values[key]))
                        break
        if per_task:
            return sum(per_task) / len(per_task)
    return float("nan")

# %% [markdown]
# ## 2. IFEval

# %%
sft_ifeval_raw = run_lm_eval(SFT_PATH, "ifeval", LIMIT_IFEVAL, 0, "sft")
dpo_ifeval_raw = run_lm_eval(DPO_PATH, "ifeval", LIMIT_IFEVAL, 0, "dpo")

# %% [markdown]
# ## 3. GSM8K

# %%
sft_gsm8k_raw = run_lm_eval(SFT_PATH, "gsm8k", LIMIT_GSM8K, 8, "sft")
dpo_gsm8k_raw = run_lm_eval(DPO_PATH, "gsm8k", LIMIT_GSM8K, 8, "dpo")

# %% [markdown]
# ## 4. MMLU sampled across all subjects

# %%
sft_mmlu_raw = run_lm_eval(SFT_PATH, "mmlu", LIMIT_MMLU_PER_SUBJECT, 5, "sft")
dpo_mmlu_raw = run_lm_eval(DPO_PATH, "mmlu", LIMIT_MMLU_PER_SUBJECT, 5, "dpo")

# %% [markdown]
# ## 5. AlpacaEval-lite generation

# %%
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_alpaca_prompts(n):
    # The Hub repo uses a legacy dataset script, which conflicts with datasets
    # 5.x. Reading its official JSON directly avoids arbitrary-code loading.
    dataset = load_dataset(
        "json",
        data_files=(
            "https://huggingface.co/datasets/tatsu-lab/alpaca_eval/"
            "resolve/main/alpaca_eval.json"
        ),
        split="train",
    )
    indices = random.Random(42).sample(range(len(dataset)), k=min(n, len(dataset)))
    return [
        {"id": index, "prompt": dataset[index]["instruction"]}
        for index in indices
    ]


def generate_with_adapter(adapter_path: Path, prompts, max_new_tokens=256):
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quantization,
        dtype=torch.float16,
        device_map={"": 0},
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.config.use_cache = True
    model.eval()

    outputs = []
    for item in prompts:
        model_inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": item["prompt"]}],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to("cuda")
        with torch.inference_mode():
            generated = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        prompt_len = model_inputs["input_ids"].shape[1]
        outputs.append(tokenizer.decode(generated[0][prompt_len:], skip_special_tokens=True).strip())

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return outputs


alpaca_prompts = load_alpaca_prompts(LIMIT_ALPACA)
print(f"Loaded {len(alpaca_prompts)} AlpacaEval-lite prompts")
sft_alpaca_outputs = generate_with_adapter(SFT_PATH, alpaca_prompts)
dpo_alpaca_outputs = generate_with_adapter(DPO_PATH, alpaca_prompts)

(EVAL_OUT / "alpaca_lite_generations.json").write_text(
    json.dumps([
        {**prompt, "sft": sft, "dpo": dpo}
        for prompt, sft, dpo in zip(alpaca_prompts, sft_alpaca_outputs, dpo_alpaca_outputs)
    ], ensure_ascii=False, indent=2),
    encoding="utf-8",
)

# %% [markdown]
# ## 6. Optional OpenAI pairwise judge
#
# Add a Colab Secret named `OPENAI_API_KEY`. If Colab Secrets times out, the
# fallback prompt hides the key and keeps it only in kernel memory. Failed API
# calls are marked `error` and are never counted as ties.

# %%
from getpass import getpass


def load_openai_key():
    if os.environ.get("OPENAI_API_KEY"):
        return True
    try:
        from google.colab import userdata

        value = userdata.get("OPENAI_API_KEY")
        if value:
            os.environ["OPENAI_API_KEY"] = value
            return True
    except Exception as exc:
        print(f"Colab Secret unavailable ({type(exc).__name__}).")
    value = getpass("Paste a NEW OPENAI_API_KEY (hidden, Enter to skip): ").strip()
    if value:
        os.environ["OPENAI_API_KEY"] = value
        return True
    return False


# Match the already-reviewed NB4 judge; override with JUDGE_MODEL if required.
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-5-mini")
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "winner": {"type": "string", "enum": ["A", "B", "tie"]},
        "reason": {"type": "string"},
    },
    "required": ["winner", "reason"],
    "additionalProperties": False,
}


def judge_pair(client, prompt, response_a, response_b):
    judge_prompt = f"""Evaluate two assistant responses for helpfulness, correctness and relevance.
Do not prefer a response merely because it is longer.

User prompt: {prompt}

Response A: {response_a}

Response B: {response_b}
"""
    request = dict(
        model=JUDGE_MODEL,
        input=judge_prompt,
        max_output_tokens=1000,
        store=False,
        text={
            "format": {
                "type": "json_schema",
                "name": "pairwise_judgment",
                "strict": True,
                "schema": JUDGE_SCHEMA,
            }
        },
    )
    if JUDGE_MODEL.startswith("gpt-5"):
        request["reasoning"] = {"effort": "minimal"}
    response = client.responses.create(**request)
    if response.status != "completed":
        raise RuntimeError(
            f"Judge status={response.status}; incomplete={response.incomplete_details}"
        )
    if not response.output_text:
        raise RuntimeError("Judge completed without output_text")
    return json.loads(response.output_text)


judgments = []
if load_openai_key():
    from openai import OpenAI

    client = OpenAI()
    rng = random.Random(42)
    for prompt, sft_output, dpo_output in zip(alpaca_prompts, sft_alpaca_outputs, dpo_alpaca_outputs):
        flipped = rng.random() < 0.5
        response_a, response_b = (dpo_output, sft_output) if flipped else (sft_output, dpo_output)
        try:
            result = judge_pair(client, prompt["prompt"], response_a, response_b)
            winner = result["winner"]
            if winner == "tie":
                winner_model = "tie"
            elif flipped:
                winner_model = "dpo" if winner == "A" else "sft"
            else:
                winner_model = "sft" if winner == "A" else "dpo"
            result.update({"id": prompt["id"], "winner_model": winner_model})
        except Exception as exc:
            result = {
                "id": prompt["id"],
                "winner_model": "error",
                "reason": f"{type(exc).__name__}: {str(exc)[:200]}",
            }
        judgments.append(result)
else:
    print("No API key supplied; AlpacaEval-lite judge skipped.")

(EVAL_OUT / "alpaca_lite_judgments.json").write_text(
    json.dumps(judgments, ensure_ascii=False, indent=2), encoding="utf-8"
)

valid_judgments = [item for item in judgments if item.get("winner_model") in {"sft", "dpo", "tie"}]
if valid_judgments:
    dpo_wins = sum(item["winner_model"] == "dpo" for item in valid_judgments)
    sft_wins = sum(item["winner_model"] == "sft" for item in valid_judgments)
    ties = sum(item["winner_model"] == "tie" for item in valid_judgments)
    alpaca_dpo_score = (dpo_wins + 0.5 * ties) / len(valid_judgments)
    alpaca_sft_score = (sft_wins + 0.5 * ties) / len(valid_judgments)
else:
    alpaca_dpo_score = float("nan")
    alpaca_sft_score = float("nan")
print(
    f"Valid Alpaca judgments: {len(valid_judgments)}/{len(alpaca_prompts)}; "
    f"SFT={alpaca_sft_score}, DPO={alpaca_dpo_score}"
)

# %% [markdown]
# ## 7. Aggregate metrics, plot and save

# %%
metrics = {
    "IFEval": {
        "sft": find_metric(sft_ifeval_raw, ["prompt_level_strict_acc,none"], "ifeval"),
        "dpo": find_metric(dpo_ifeval_raw, ["prompt_level_strict_acc,none"], "ifeval"),
    },
    "GSM8K": {
        "sft": find_metric(sft_gsm8k_raw, ["exact_match,strict-match", "exact_match,none"], "gsm8k"),
        "dpo": find_metric(dpo_gsm8k_raw, ["exact_match,strict-match", "exact_match,none"], "gsm8k"),
    },
    "MMLU": {
        "sft": find_metric(sft_mmlu_raw, ["acc,none"], "mmlu"),
        "dpo": find_metric(dpo_mmlu_raw, ["acc,none"], "mmlu"),
    },
    "AlpacaEval-lite": {
        "sft": alpaca_sft_score,
        "dpo": alpaca_dpo_score,
    },
}

print("\n" + "=" * 72 + "\nBENCHMARK RESULTS\n" + "=" * 72)
for benchmark, scores in metrics.items():
    delta = scores["dpo"] - scores["sft"]
    print(f"{benchmark:18s} SFT={scores['sft']:.4f} DPO={scores['dpo']:.4f} Δ={delta:+.4f}")

# %%
import matplotlib.pyplot as plt
import numpy as np

names = list(metrics)
sft_scores = [metrics[name]["sft"] for name in names]
dpo_scores = [metrics[name]["dpo"] for name in names]
x = np.arange(len(names))
width = 0.36

fig, ax = plt.subplots(figsize=(11, 5.5))
sft_bars = ax.bar(x - width / 2, sft_scores, width, label="SFT-only", color="#2e548a")
dpo_bars = ax.bar(x + width / 2, dpo_scores, width, label="SFT+DPO", color="#c83538")
for bars in (sft_bars, dpo_bars):
    for bar in bars:
        height = bar.get_height()
        if not math.isnan(height):
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.012, f"{height:.3f}", ha="center", fontsize=9)
for index, name in enumerate(names):
    sft_score, dpo_score = metrics[name]["sft"], metrics[name]["dpo"]
    if not (math.isnan(sft_score) or math.isnan(dpo_score)):
        ax.text(index, max(sft_score, dpo_score) + 0.07, f"Δ={dpo_score - sft_score:+.3f}", ha="center")
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylabel("Score")
ax.set_ylim(0, 1.12)
ax.set_title("NB6 benchmark · SFT-only vs SFT+DPO · T4")
ax.legend()
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
benchmark_png = SCREENSHOT_DIR / "07-benchmark-comparison.png"
fig.savefig(benchmark_png, dpi=140, bbox_inches="tight")
plt.show()

# %%
benchmark_results = {
    "compute_tier": COMPUTE_TIER,
    "base_model": BASE_MODEL,
    "lm_eval_version": "0.4.12",
    "limits": {
        "ifeval": LIMIT_IFEVAL,
        "gsm8k": LIMIT_GSM8K,
        "mmlu_per_subject": LIMIT_MMLU_PER_SUBJECT,
        "alpaca_lite": LIMIT_ALPACA,
    },
    "metrics": metrics,
    "deltas": {
        name: scores["dpo"] - scores["sft"]
        for name, scores in metrics.items()
        if not (math.isnan(scores["sft"]) or math.isnan(scores["dpo"]))
    },
    "alpaca_valid_judgments": len(valid_judgments),
}


def json_safe(value):
    """Replace non-finite floats so the saved artifact is strict JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


benchmark_path = EVAL_OUT / "benchmark_results.json"
benchmark_path.write_text(
    json.dumps(json_safe(benchmark_results), ensure_ascii=False, indent=2, allow_nan=False),
    encoding="utf-8",
)
print(f"Saved {benchmark_path}")
print(f"Saved {benchmark_png}")

# %% [markdown]
# ## 8. Submission follow-up
#
# Copy the four scores and deltas from `benchmark_results.json` into Reflection §7.
# Explain whether IFEval improved and whether GSM8K/MMLU show an alignment tax.
