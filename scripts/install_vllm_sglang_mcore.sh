#!/bin/bash

USE_MEGATRON=${USE_MEGATRON:-0}
USE_SGLANG=${USE_SGLANG:-0}
FIX_OPENCV=${FIX_OPENCV:-0}

set -euo pipefail

export MAX_JOBS=32

########################################
# Network / mirror acceleration knobs
########################################
# You can override these env vars before running the script:
# - PIP_INDEX_URL: force a specific PyPI index (e.g. https://mirrors.ustc.edu.cn/pypi/web/simple)
# - PIP_EXTRA_INDEX_URL: optional extra index url
# - TORCH_INDEX_URL: torch/vision/audio index (default: PyTorch official wheel index)
# - TORCH_FIND_LINKS: prefer downloading torch wheels from this HTML directory listing (default: Aliyun mirror)
# - GITHUB_PROXY_PREFIXES: space-separated prefixes to accelerate GitHub downloads
#   (default: "https://ghproxy.net/")
# - WHEELHOUSE: directory to cache downloaded wheels
# - USE_ARIA2: 1=use aria2c when available (default 1)
PIP_INDEX_URL="${PIP_INDEX_URL:-}"
PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL:-}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
TORCH_FIND_LINKS="${TORCH_FIND_LINKS:-https://mirrors.aliyun.com/pytorch-wheels/cu124/}"
GITHUB_PROXY_PREFIXES="${GITHUB_PROXY_PREFIXES:-https://ghproxy.net/}"
WHEELHOUSE="${WHEELHOUSE:-/root/.cache/attnrl-wheels}"
USE_ARIA2="${USE_ARIA2:-1}"

mkdir -p "${WHEELHOUSE}"

have_cmd() { command -v "$1" >/dev/null 2>&1; }

have_py_pkg_version() {
  # Usage: have_py_pkg_version DIST_NAME VERSION_PREFIX_OR_EXACT
  # Returns 0 if installed and version startswith the provided string (exact if you pass full version).
  local dist="$1"
  local want="$2"
  python - <<'PY' "${dist}" "${want}" >/dev/null 2>&1
import sys
try:
    import importlib.metadata as m
except Exception:
    import importlib_metadata as m  # type: ignore
dist=sys.argv[1]
want=sys.argv[2]
try:
    v=m.version(dist)
except m.PackageNotFoundError:
    raise SystemExit(2)
if not v.startswith(want):
    raise SystemExit(3)
PY
}

ensure_pip_install() {
  # Usage: ensure_pip_install DIST_NAME VERSION_PREFIX_OR_EXACT PIP_SPEC [extra pip args...]
  # Installs only if missing or version mismatch.
  local dist="$1"
  local want="$2"
  local spec="$3"
  shift 3
  if have_py_pkg_version "${dist}" "${want}"; then
    echo "Already installed: ${dist}==${want}*"
    return 0
  fi
  pip install "${PIP_COMMON_ARGS[@]}" "$@" "${spec}"
}

is_valid_wheel() {
  # Returns 0 if the file looks like a valid wheel (zip), else 1.
  # We validate using Python's zipfile (checks central directory exists).
  local p="$1"
  [[ -s "${p}" ]] || return 1
  python - <<'PY' "${p}" >/dev/null 2>&1
import sys, zipfile
p=sys.argv[1]
try:
    with zipfile.ZipFile(p) as z:
        z.namelist()[:1]  # touch
except Exception:
    raise SystemExit(2)
PY
}

pick_fast_pip_index() {
  # Pick the first reachable index quickly (HEAD a small "simple" page).
  # Note: reachability doesn't guarantee fastest for huge wheels, but avoids broken/blocked mirrors.
  local -a candidates=(
    "https://mirrors.ustc.edu.cn/pypi/web/simple"
    "https://mirrors.aliyun.com/pypi/simple"
    "https://pypi.tuna.tsinghua.edu.cn/simple"
    "https://pypi.org/simple"
  )

  if [[ -n "${PIP_INDEX_URL}" ]]; then
    echo "${PIP_INDEX_URL}"
    return 0
  fi

  # Prefer curl if available; fallback to wget.
  if have_cmd curl; then
    for u in "${candidates[@]}"; do
      if curl -fsSIL --connect-timeout 3 --max-time 5 "${u}/pip/" >/dev/null; then
        echo "${u}"
        return 0
      fi
    done
  else
    for u in "${candidates[@]}"; do
      if wget -q --spider --timeout=5 "${u}/pip/" >/dev/null 2>&1; then
        echo "${u}"
        return 0
      fi
    done
  fi

  # Last resort
  echo "https://pypi.org/simple"
}

download_wheel() {
  # Usage: download_wheel URL [OUT_DIR]
  local url="$1"
  local out_dir="${2:-${WHEELHOUSE}}"
  mkdir -p "${out_dir}"
  local fname="${url##*/}"
  local out_path="${out_dir}/${fname}"
  local tmp_path="${out_path}.part"

  if [[ -s "${out_path}" ]]; then
    if is_valid_wheel "${out_path}"; then
      echo "Already cached: ${out_path}"
      return 0
    fi
    echo "Cached file is invalid, removing: ${out_path}"
    rm -f "${out_path}"
  fi

  # Try GitHub proxies first (direct GitHub often times out), then direct URL.
  local -a candidates=()
  local pfx
  for pfx in ${GITHUB_PROXY_PREFIXES}; do
    [[ -n "${pfx}" ]] || continue
    candidates+=( "${pfx}${url}" )
  done
  candidates+=( "${url}" )

  local c
  for c in "${candidates[@]}"; do
    echo "Downloading: ${c}"
    if [[ "${USE_ARIA2}" == "1" ]] && have_cmd aria2c; then
      # Aggressive timeouts & retries to avoid hanging on bad routes.
      # Download to .part and only move to final name when valid.
      if aria2c -c -x 16 -s 16 -k 1M \
        --connect-timeout=10 --timeout=60 --max-tries=10 --retry-wait=2 \
        --file-allocation=none \
        -o "$(basename "${tmp_path}")" -d "${out_dir}" "${c}"; then
        mv -f "${tmp_path}" "${out_path}"
        if is_valid_wheel "${out_path}"; then
          return 0
        fi
        echo "Downloaded wheel is invalid, retrying..."
        rm -f "${out_path}"
      fi
    else
      # Download to a .part file first; validate before moving into cache.
      # -c for resume; add timeouts/retries to avoid hanging on GitHub.
      if wget -c -nv -O "${tmp_path}" \
        --timeout=20 --read-timeout=20 --tries=10 --waitretry=2 \
        "${c}"; then
        mv -f "${tmp_path}" "${out_path}"
        if is_valid_wheel "${out_path}"; then
          return 0
        fi
        echo "Downloaded wheel is invalid (likely truncated). Removing and retrying..."
        rm -f "${out_path}"
      fi
    fi
    echo "Failed: ${c}"
  done

  echo "ERROR: failed to download ${url}"
  return 1
}

PIP_INDEX_URL="$(pick_fast_pip_index)"

PIP_COMMON_ARGS=(
  --no-cache-dir
  --retries 10
  --timeout 60
  --prefer-binary
  --index-url "${PIP_INDEX_URL}"
)
if [[ -n "${PIP_EXTRA_INDEX_URL}" ]]; then
  PIP_COMMON_ARGS+=( --extra-index-url "${PIP_EXTRA_INDEX_URL}" )
fi

install_torch_stack() {
  # Prefer using a reachable mirror directory for the big wheels, while still resolving deps from PyPI index.
  # If the mirror path is not usable, fall back to official PyTorch wheel index.
  local -a base_args=( --no-cache-dir --retries 10 --timeout 120 --prefer-binary )

  # Skip if already installed (torch wheels include +cu124 suffix in version).
  if python - <<'PY' >/dev/null 2>&1
import torch, torchvision, torchaudio
assert torch.__version__.startswith("2.6.0")
assert torchvision.__version__.startswith("0.21.0")
assert torchaudio.__version__.startswith("2.6.0")
PY
  then
    echo "Torch stack already installed"
    return 0
  fi

  if [[ -n "${TORCH_FIND_LINKS}" ]]; then
    echo "Torch wheels via --find-links: ${TORCH_FIND_LINKS} (deps from ${PIP_INDEX_URL})"
    if pip install "${base_args[@]}" --index-url "${PIP_INDEX_URL}" --find-links "${TORCH_FIND_LINKS}" \
      "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0"; then
      return 0
    fi
    echo "Torch install via --find-links failed, falling back to ${TORCH_INDEX_URL}"
  fi

  pip install "${base_args[@]}" --index-url "${TORCH_INDEX_URL}" \
    "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0"
}

echo "1. install inference frameworks and pytorch they need"
if [ $USE_SGLANG -eq 1 ]; then
    ensure_pip_install "sglang" "0.4.6.post1" "sglang[all]==0.4.6.post1" --find-links https://flashinfer.ai/whl/cu124/torch2.6/flashinfer-python
    if ! have_py_pkg_version "torch-memory-saver" ""; then
      pip install "${PIP_COMMON_ARGS[@]}" torch-memory-saver
    fi
fi

# Install torch/vision/audio (prefer mirror find-links; fallback to official index).
install_torch_stack

# Keep numpy<2 compatibility.
# opencv-python-headless>=4.12 requires numpy>=2 on py>=3.9, so we pin it below 4.12.
ensure_pip_install "numpy" "1.26.4" "numpy==1.26.4"
if have_py_pkg_version "opencv-python-headless" "4.12."; then
  pip install "${PIP_COMMON_ARGS[@]}" --upgrade --force-reinstall --no-deps "opencv-python-headless<4.12"
else
  # If not installed or older than 4.12, just ensure it's <4.12 (no deps to avoid pulling numpy>=2).
  pip install "${PIP_COMMON_ARGS[@]}" --no-deps "opencv-python-headless<4.12"
fi

# Then install vllm and the rest from the chosen PyPI index.
ensure_pip_install "vllm" "0.8.5.post1" "vllm==0.8.5.post1"
ensure_pip_install "tensordict" "0.6.2" "tensordict==0.6.2"
if ! python - <<'PY' >/dev/null 2>&1
import torchdata
PY
then
  pip install "${PIP_COMMON_ARGS[@]}" torchdata
fi

echo "2. install basic packages"
ensure_pip_install "transformers" "4.51.1" "transformers[hf_xet]==4.51.1"
ensure_pip_install "ray" "2.47.1" "ray[default]==2.47.1"
if ! have_py_pkg_version "accelerate" ""; then pip install "${PIP_COMMON_ARGS[@]}" accelerate; fi
if ! have_py_pkg_version "datasets" ""; then pip install "${PIP_COMMON_ARGS[@]}" datasets; fi
if ! have_py_pkg_version "peft" ""; then pip install "${PIP_COMMON_ARGS[@]}" peft; fi
if ! have_py_pkg_version "hf-transfer" ""; then pip install "${PIP_COMMON_ARGS[@]}" hf-transfer; fi
if ! have_py_pkg_version "pyarrow" ""; then pip install "${PIP_COMMON_ARGS[@]}" "pyarrow>=15.0.0"; fi
if ! have_py_pkg_version "pandas" ""; then pip install "${PIP_COMMON_ARGS[@]}" pandas; fi
if ! have_py_pkg_version "codetiming" ""; then pip install "${PIP_COMMON_ARGS[@]}" codetiming; fi
if ! have_py_pkg_version "hydra-core" ""; then pip install "${PIP_COMMON_ARGS[@]}" hydra-core; fi
if ! have_py_pkg_version "pylatexenc" ""; then pip install "${PIP_COMMON_ARGS[@]}" pylatexenc; fi
if ! have_py_pkg_version "qwen-vl-utils" ""; then pip install "${PIP_COMMON_ARGS[@]}" qwen-vl-utils; fi
if ! have_py_pkg_version "wandb" ""; then pip install "${PIP_COMMON_ARGS[@]}" wandb; fi
if ! have_py_pkg_version "pybind11" ""; then pip install "${PIP_COMMON_ARGS[@]}" pybind11; fi
if ! have_py_pkg_version "liger-kernel" ""; then pip install "${PIP_COMMON_ARGS[@]}" liger-kernel; fi
if ! have_py_pkg_version "mathruler" ""; then pip install "${PIP_COMMON_ARGS[@]}" mathruler; fi
if ! have_py_pkg_version "pytest" ""; then pip install "${PIP_COMMON_ARGS[@]}" pytest; fi
if ! have_py_pkg_version "py-spy" ""; then pip install "${PIP_COMMON_ARGS[@]}" py-spy; fi
if ! have_py_pkg_version "pyext" ""; then pip install "${PIP_COMMON_ARGS[@]}" pyext; fi
if ! have_py_pkg_version "pre-commit" ""; then pip install "${PIP_COMMON_ARGS[@]}" pre-commit; fi
if ! have_py_pkg_version "ruff" ""; then pip install "${PIP_COMMON_ARGS[@]}" ruff; fi

pip install "${PIP_COMMON_ARGS[@]}" "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"
if ! have_py_pkg_version "math_verify" ""; then pip install "${PIP_COMMON_ARGS[@]}" math_verify; fi

# ray[default] may pull newer OpenTelemetry (1.39+) which breaks vllm's strict <1.27 requirement.
# Re-pin to vllm-compatible OpenTelemetry versions.
echo "2.5 pin OpenTelemetry versions (vllm compatibility)"
ensure_pip_install "opentelemetry-api" "1.26.0" "opentelemetry-api==1.26.0"
ensure_pip_install "opentelemetry-sdk" "1.26.0" "opentelemetry-sdk==1.26.0"
ensure_pip_install "opentelemetry-proto" "1.26.0" "opentelemetry-proto==1.26.0"
ensure_pip_install "opentelemetry-exporter-otlp" "1.26.0" "opentelemetry-exporter-otlp==1.26.0"
ensure_pip_install "opentelemetry-semantic-conventions" "0.47b0" "opentelemetry-semantic-conventions==0.47b0"
ensure_pip_install "opentelemetry-exporter-prometheus" "0.47b0" "opentelemetry-exporter-prometheus==0.47b0"

echo "3. install FlashAttention and FlashInfer"
# Install flash-attn-2.7.4.post1 (cxx11abi=False)
FLASH_ATTN_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
if ! have_py_pkg_version "flash-attn" "2.7.4.post1"; then
  download_wheel "${FLASH_ATTN_URL}"
  # Avoid pulling huge CUDA Python deps (torch already installed).
  pip install --no-cache-dir --retries 10 --timeout 120 --no-deps "${WHEELHOUSE}/$(basename "${FLASH_ATTN_URL}")"
else
  echo "Already installed: flash-attn==2.7.4.post1"
fi

# Install flashinfer-0.2.2.post1+cu124 (cxx11abi=False)
# vllm-0.8.3 does not support flashinfer>=0.2.3
# see https://github.com/vllm-project/vllm/pull/15777
FLASHINFER_URL="https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.2.post1/flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl"
if ! have_py_pkg_version "flashinfer-python" "0.2.2.post1"; then
  download_wheel "${FLASHINFER_URL}"
  pip install --no-cache-dir --retries 10 --timeout 120 --no-deps "${WHEELHOUSE}/$(basename "${FLASHINFER_URL}")"
else
  echo "Already installed: flashinfer-python==0.2.2.post1"
fi


if [ $USE_MEGATRON -eq 1 ]; then
    echo "4. install TransformerEngine and Megatron"
    echo "Notice that TransformerEngine installation can take very long time, please be patient"
    # Avoid git+https (often blocked); prefer already-installed packages.
    ensure_pip_install "transformer-engine" "2.2.0" "transformer-engine==2.2.0" --no-deps
    ensure_pip_install "transformer-engine-cu12" "2.2.0" "transformer-engine-cu12==2.2.0" --no-deps
    # torch extension may not exist as wheel; allow building if missing
    if ! have_py_pkg_version "transformer-engine-torch" "2.2.0"; then
      NVTE_FRAMEWORK=pytorch pip install "${PIP_COMMON_ARGS[@]}" --no-deps --no-build-isolation "transformer-engine-torch==2.2.0"
    else
      echo "Already installed: transformer-engine-torch==2.2.0"
    fi
    ensure_pip_install "megatron-core" "0.12.0rc3" "megatron-core==0.12.0rc3" --no-deps
fi


echo "5. May need to fix opencv"
if [[ "${FIX_OPENCV}" == "1" ]]; then
  # Default is to skip this step because it may upgrade numpy to >=2 (breaking the numpy<2 pins).
  # If you really need OpenCV GUI bindings, set FIX_OPENCV=1 explicitly.
  pip install "${PIP_COMMON_ARGS[@]}" --no-deps "opencv-python<4.12"
  pip install "${PIP_COMMON_ARGS[@]}" opencv-fixer && \
      python -c "from opencv_fixer import AutoFix; AutoFix()"
else
  echo "Skipping opencv-python/opencv-fixer (set FIX_OPENCV=1 to enable)."
fi


if [ $USE_MEGATRON -eq 1 ]; then
    echo "6. Install cudnn python package (avoid being overridden)"
    ensure_pip_install "nvidia-cudnn-cu12" "9.8.0.87" "nvidia-cudnn-cu12==9.8.0.87"
fi

echo "Successfully installed all packages"
