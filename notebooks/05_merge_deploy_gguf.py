# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # NB5 — Merge + GGUF + llama.cpp smoke test (OPTIONAL / BONUS)
#
# This notebook is compatible with the conflict-safe T4 core run. It deliberately
# avoids Unsloth, xFormers, FlashAttention and llama-cpp-python. The DPO checkpoint
# produced by NB3 is already the SFT adapter after DPO updates, so the correct
# deployment model is **base + adapters/dpo** (do not stack adapters/sft-mini again).
#
# Outputs:
# - `adapters/merged-fp16/` — merged Hugging Face model (~6 GB)
# - `gguf/lab22-qwen2.5-3b-dpo-Q4_K_M.gguf` (~2 GB)
# - `data/eval/deploy_meta.json`
# - `submission/screenshots/06-gguf-smoke.png`

# %% [markdown]
# ## 0. Setup and resource preflight

# %%
import gc
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

COMPUTE_TIER = os.environ.get("COMPUTE_TIER", "T4").upper()
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
MAX_LEN = 512

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DPO_PATH = REPO_ROOT / "adapters" / "dpo"
MERGED_PATH = REPO_ROOT / "adapters" / "merged-fp16"
GGUF_DIR = REPO_ROOT / "gguf"
EVAL_OUT = REPO_ROOT / "data" / "eval"
SCREENSHOT_DIR = REPO_ROOT / "submission" / "screenshots"
for folder in (MERGED_PATH, GGUF_DIR, EVAL_OUT, SCREENSHOT_DIR):
    folder.mkdir(parents=True, exist_ok=True)

assert DPO_PATH.joinpath("adapter_config.json").exists(), "Run NB3 or restore adapters/dpo first"
assert torch.cuda.is_available(), "Select a Colab T4 GPU runtime"

free_disk = shutil.disk_usage(REPO_ROOT).free / 1024**3
gpu = torch.cuda.get_device_properties(0)
print(f"GPU: {gpu.name} ({gpu.total_memory / 1024**3:.1f} GiB)")
print(f"Free disk: {free_disk:.1f} GiB")
print(f"DPO adapter: {DPO_PATH}")
assert free_disk >= 18, "NB5 needs at least 18 GiB free for base cache + merged weights + temporary F16 GGUF"


def run_checked(cmd, *, timeout=3600, capture=False):
    """Run a command, fail with its stderr tail, and optionally return stdout."""
    print("$", " ".join(map(str, cmd)))
    proc = subprocess.run(
        [str(part) for part in cmd],
        text=True,
        capture_output=capture,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        if capture:
            print(proc.stdout[-2000:])
            print(proc.stderr[-4000:])
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {cmd[0]}")
    return proc.stdout if capture else ""


# %% [markdown]
# ## 1. Merge the single cumulative DPO adapter into the FP16 base

# %%
import importlib

# Colab currently preinstalls torchao 0.10, while PEFT 0.20 rejects every
# installed torchao below 0.16 even for an ordinary FP16 Linear layer. NB5 does
# not use TorchAO, so remove only that optional dispatcher instead of upgrading
# another compiled package inside the proven T4 runtime. This is safe to rerun.
subprocess.run(
    [sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"],
    check=True,
    timeout=300,
)
for module_name in list(sys.modules):
    if module_name == "torchao" or module_name.startswith("torchao."):
        del sys.modules[module_name]
importlib.invalidate_caches()

from peft import PeftModel
from peft.import_utils import is_torchao_available
from transformers import AutoModelForCausalLM, AutoTokenizer

is_torchao_available.cache_clear()
assert not is_torchao_available(), "TorchAO dispatcher is still active; restart this cell once"

# A failed PeftModel load leaves the FP16 base referenced in notebook globals.
# Drop any partial objects so rerunning this cell does not double VRAM usage.
for stale_name in ("merged_model", "peft_model", "base_model"):
    globals().pop(stale_name, None)
gc.collect()
torch.cuda.empty_cache()

tokenizer = AutoTokenizer.from_pretrained(DPO_PATH, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16,
    device_map={"": 0},
    attn_implementation="eager",
    low_cpu_mem_usage=True,
)
peft_model = PeftModel.from_pretrained(base_model, DPO_PATH, is_trainable=False)
assert list(peft_model.peft_config) == ["default"], (
    f"Expected one cumulative DPO adapter, found {list(peft_model.peft_config)}"
)

merged_model = peft_model.merge_and_unload(safe_merge=True, adapter_names=["default"])
merged_model.config.use_cache = True
merged_model.save_pretrained(
    MERGED_PATH,
    safe_serialization=True,
    max_shard_size="4GB",
)
tokenizer.save_pretrained(MERGED_PATH)

weight_files = sorted(MERGED_PATH.glob("*.safetensors"))
assert weight_files, "Merged model did not write safetensors"
merged_size_gib = sum(path.stat().st_size for path in weight_files) / 1024**3
print(f"Merged FP16 model: {MERGED_PATH} ({merged_size_gib:.2f} GiB)")

del merged_model, peft_model, base_model
gc.collect()
torch.cuda.empty_cache()

# %% [markdown]
# ## 2. Pin and build official llama.cpp CPU tools
#
# CPU build is intentional: it avoids introducing another CUDA binary toolchain
# into the already-tested training runtime. Quantization and the short smoke test
# are slower but reproducible on a Colab T4 host.

# %%
LLAMA_CPP_REV = os.environ.get(
    "LLAMA_CPP_REV", "b3c3b96a139d4ef1bdec926ac17aa040981cfc5d"
)
LLAMA_CPP_DIR = Path(f"/content/llama.cpp-{LLAMA_CPP_REV[:12]}")
LLAMA_BUILD_DIR = LLAMA_CPP_DIR / "build"

if not LLAMA_CPP_DIR.exists():
    run_checked([
        "git", "clone", "--filter=blob:none", "--depth", "1", "--no-checkout",
        "https://github.com/ggml-org/llama.cpp.git", str(LLAMA_CPP_DIR),
    ], timeout=600)
run_checked(["git", "-C", str(LLAMA_CPP_DIR), "fetch", "--depth", "1", "origin", LLAMA_CPP_REV])
run_checked(["git", "-C", str(LLAMA_CPP_DIR), "checkout", "--detach", "FETCH_HEAD"])

run_checked([
    "cmake", "-S", str(LLAMA_CPP_DIR), "-B", str(LLAMA_BUILD_DIR),
    "-DGGML_CUDA=OFF", "-DLLAMA_CURL=OFF", "-DBUILD_SHARED_LIBS=OFF",
    "-DCMAKE_BUILD_TYPE=Release",
], timeout=600)
run_checked([
    "cmake", "--build", str(LLAMA_BUILD_DIR), "--config", "Release",
    "-j", "2", "--target", "llama-quantize", "llama-cli",
], timeout=1800)

QUANTIZE_BIN = LLAMA_BUILD_DIR / "bin" / "llama-quantize"
CLI_BIN = LLAMA_BUILD_DIR / "bin" / "llama-cli"
assert QUANTIZE_BIN.exists() and CLI_BIN.exists(), "llama.cpp build did not produce required binaries"
print(f"llama.cpp revision: {LLAMA_CPP_REV}")

# The pinned converter requires Transformers 4.57.6, while the successful core
# runtime uses Transformers 5.15.1. Keep converter dependencies isolated so NB5
# cannot mutate the already-tested training/evaluation environment.
CONVERT_VENV = Path(f"/content/llama-gguf-venv-{LLAMA_CPP_REV[:12]}")
CONVERT_PYTHON = CONVERT_VENV / "bin" / "python"
if not CONVERT_PYTHON.exists():
    run_checked([
        sys.executable, "-m", "venv", "--system-site-packages", str(CONVERT_VENV),
    ], timeout=600)
run_checked([
    str(CONVERT_PYTHON), "-m", "pip", "install", "-q",
    "numpy==1.26.4",
    "sentencepiece>=0.1.98,<0.3.0",
    "transformers==4.57.6",
    "protobuf>=4.21.0,<5.0.0",
], timeout=1200)
run_checked([
    str(CONVERT_PYTHON), "-c",
    "import torch, transformers; "
    "assert torch.__version__.startswith('2.11.'); "
    "assert transformers.__version__ == '4.57.6'; "
    "print('converter stack:', torch.__version__, transformers.__version__)",
], timeout=120)

# %% [markdown]
# ## 3. Convert HF → F16 GGUF → Q4_K_M

# %%
F16_GGUF = GGUF_DIR / "lab22-qwen2.5-3b-dpo-F16.gguf"
Q4_GGUF = GGUF_DIR / "lab22-qwen2.5-3b-dpo-Q4_K_M.gguf"

run_checked([
    str(CONVERT_PYTHON),
    str(LLAMA_CPP_DIR / "convert_hf_to_gguf.py"),
    str(MERGED_PATH),
    "--outfile", str(F16_GGUF),
    "--outtype", "f16",
], timeout=3600)
assert F16_GGUF.exists() and F16_GGUF.stat().st_size > 1_000_000_000

run_checked([
    str(QUANTIZE_BIN), str(F16_GGUF), str(Q4_GGUF), "Q4_K_M"
], timeout=3600)
assert Q4_GGUF.exists(), "Q4_K_M quantization did not write a file"
assert Q4_GGUF.stat().st_size < 5 * 1024**3, "GGUF exceeds the rubric's 5 GiB limit"

# F16 GGUF is only an intermediate; merged HF weights remain reproducible.
F16_GGUF.unlink()
print(f"Q4_K_M GGUF: {Q4_GGUF} ({Q4_GGUF.stat().st_size / 1024**3:.2f} GiB)")

# %% [markdown]
# ## 4. llama.cpp smoke test and screenshot artifact

# %%
SMOKE_PROMPT = "Giải thích ngắn gọn trong 3 câu cách thuật toán Bubble sort hoạt động."
smoke_output = run_checked([
    str(CLI_BIN),
    "-m", str(Q4_GGUF),
    "-cnv",
    "--single-turn",
    "-p", SMOKE_PROMPT,
    "-n", "160",
    "-c", str(MAX_LEN),
    "--temp", "0",
], timeout=900, capture=True)

assert smoke_output.strip(), "llama.cpp returned an empty response"
print(f"GGUF file: {Q4_GGUF.name}")
print(f"PROMPT: {SMOKE_PROMPT}\n")
print(smoke_output[-4000:])

# %%
import matplotlib.pyplot as plt
import textwrap

display_text = (
    f"GGUF: {Q4_GGUF.name}\n"
    f"Prompt: {SMOKE_PROMPT}\n\n"
    + smoke_output[-2500:]
)
wrapped = "\n".join(
    textwrap.fill(line, width=115, replace_whitespace=False)
    for line in display_text.splitlines()
)
fig, ax = plt.subplots(figsize=(14, 7))
ax.axis("off")
ax.text(0.01, 0.99, wrapped, va="top", ha="left", family="monospace", fontsize=9)
fig.tight_layout()
smoke_png = SCREENSHOT_DIR / "06-gguf-smoke.png"
fig.savefig(smoke_png, dpi=140, bbox_inches="tight")
plt.show()
print(f"Saved screenshot artifact: {smoke_png}")

# %% [markdown]
# ## 5. Save deployment metadata

# %%
deploy_meta = {
    "compute_tier": COMPUTE_TIER,
    "base_model": BASE_MODEL,
    "adapter": str(DPO_PATH),
    "merged_path": str(MERGED_PATH),
    "gguf_path": str(Q4_GGUF),
    "gguf_size_gib": round(Q4_GGUF.stat().st_size / 1024**3, 3),
    "quantization": "Q4_K_M",
    "llama_cpp_revision": LLAMA_CPP_REV,
    "smoke_prompt": SMOKE_PROMPT,
    "smoke_output_tail": smoke_output[-4000:],
}
deploy_meta_path = EVAL_OUT / "deploy_meta.json"
deploy_meta_path.write_text(
    json.dumps(deploy_meta, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(f"Saved {deploy_meta_path}")

# %% [markdown]
# ## 6. Optional local download
#
# A Q4 GGUF is around 2 GB. Set `DOWNLOAD_GGUF=True` only when you are ready to
# keep the browser tab open until the download finishes.

# %%
DOWNLOAD_GGUF = False
if DOWNLOAD_GGUF:
    from google.colab import files

    files.download(str(Q4_GGUF))
else:
    print(f"GGUF ready at {Q4_GGUF}; set DOWNLOAD_GGUF=True to download it.")
