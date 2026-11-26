#!/usr/bin/env bash
# install.sh -- deploy models with SOL 0.9.1rc2 and install them into src/models/<model>_sol/lib_{cpu,gpu}
#
#   scripts/sol_v09/install.sh [--from-build DIR | --wrapper-only] <cpu|gpu|all> <model...>
#
# --wrapper-only: only step 3 (recompile the wrapper of already installed models, e.g. after editing wrapper.c)
#
# Per model and device:
#   1. sol.deploy with deploy.py in the NEC image sol-deploy:rc2 (or reuse DIR/<nvidia|x86>/<model> from an earlier
#      deploy with --from-build)
#   2. replace the contents of src/models/<model>_sol/lib_<cpu|gpu> (.gitkeep kept) with the SOL output
#   3. compile the generic vAccel wrapper (wrapper.c) in the agent image -> vaccel/libsol_<model>_wrapper.so
#   4. write vaccel/manifest.json: the vAccel LIB resource (the wrapper), the DATA resource (model library,
#      weights, SOL runtime libs), input/output shapes (read by src/sol09.py for SolAdapter / VaccelSolAdapter)
#
# Back up the old libraries first; this replaces them. Run from the repository root.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
HERE="${REPO}/scripts/sol_v09"
SOL_IMAGE="sol-deploy:rc2"
CC_IMAGE="torchvision-app:agent"   # vAccel 0.8 headers + g++, same vAccel as the client/agent images
FROM_BUILD=""; WRAPPER_ONLY=0
case "${1:-}" in
  --from-build)   FROM_BUILD="$(cd "$2" && pwd)"; shift 2 ;;
  --wrapper-only) WRAPPER_ONLY=1; shift ;;
esac
WHICH="${1:-}"; shift || true
case "${WHICH}" in cpu) DEVS=(cpu) ;; gpu) DEVS=(gpu) ;; all) DEVS=(cpu gpu) ;; *) echo "usage: $0 [--from-build DIR | --wrapper-only] <cpu|gpu|all> <model...>"; exit 2 ;; esac
[[ $# -gt 0 ]] || { echo "no models given"; exit 2; }

VIDEO=" r3d_18 r2plus1d_18 swin3d_s swin3d_b "

# cuDNN for GPU runs (gitignored, src/models/runtime): 9.20 for SOL (copied from the SOL image; its GPU libraries run
# with this version only), 9.28 for the PyTorch/AOTInductor agent and runs (pip wheel)
CUDNN_DIR="${REPO}/src/models/runtime/cudnn-9.20"
CUDNN28_DIR="${REPO}/src/models/runtime/cudnn-9.28"
if [[ " ${DEVS[*]} " == *" gpu "* && ! -f "${CUDNN28_DIR}/libcudnn.so.9" ]]; then
  mkdir -p "${CUDNN28_DIR}"
  docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "${CUDNN28_DIR}:/out" --entrypoint sh "${SOL_IMAGE}" -c \
    'pip download --no-deps -q -d /tmp/w nvidia-cudnn-cu12==9.28.1.4 && cd /tmp/w && python3 -m zipfile -e *.whl x && cp -a x/nvidia/cudnn/lib/*.so* /out/'
  echo "[install] cuDNN 9.28 copied to ${CUDNN28_DIR}"
fi
if [[ " ${DEVS[*]} " == *" gpu "* && ! -f "${CUDNN_DIR}/libcudnn.so.9" ]]; then
  mkdir -p "${CUDNN_DIR}"
  cid=$(docker create "${SOL_IMAGE}")
  docker cp "${cid}:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib/." "${CUDNN_DIR}/" >/dev/null
  docker rm "${cid}" >/dev/null
  echo "[install] cuDNN copied to ${CUDNN_DIR}"
fi
TMP="$(mktemp -d)"; trap 'rm -rf "${TMP}"' EXIT

for m in "$@"; do
  for dev in "${DEVS[@]}"; do
    sdev=$([[ $dev == gpu ]] && echo nvidia || echo x86)
    dst="${REPO}/src/models/${m}_sol/lib_${dev}"
    echo "[install] ${m} ${dev}$([[ ${WRAPPER_ONLY} -eq 1 ]] && echo ' (wrapper only)')"

    if [[ ${WRAPPER_ONLY} -eq 0 ]]; then
    # 1. deploy (or reuse)
    if [[ -n "${FROM_BUILD}" ]]; then
      src="${FROM_BUILD}/${sdev}/${m}"
    else
      src="${TMP}/${sdev}/${m}"
      docker run --rm --gpus all --user "$(id -u):$(id -g)" -e HOME=/tmp \
        -v "${HERE}:/work:ro" -v "${TMP}:/out" -v "${REPO}/src/models:/models:ro" -w /work \
        --entrypoint python3 "${SOL_IMAGE}" deploy.py "${m}" "${sdev}" "/out/${sdev}/${m}" > "${TMP}/deploy_${m}_${dev}.log" 2>&1 \
        || { echo "[install] deploy failed, log:"; tail -30 "${TMP}/deploy_${m}_${dev}.log"; exit 1; }
    fi
    [[ -f "${src}/${m}.so" && -f "${src}/${m}.sol_weights" ]] || { echo "[install] missing ${src}/${m}.so/.sol_weights"; exit 1; }

    # 2. replace the library folder
    mkdir -p "${dst}"
    find "${dst}" -mindepth 1 -maxdepth 1 ! -name .gitkeep -exec rm -rf {} +
    cp -a "${src}/." "${dst}/"
    rm -f "${dst}/__init__.py"
    fi
    [[ -f "${dst}/${m}.so" ]] || { echo "[install] ${dst}/${m}.so missing"; exit 1; }

    # 3. wrapper
    mkdir -p "${dst}/vaccel"
    cp "${HERE}/wrapper.c" "${dst}/vaccel/sol_wrapper.c"
    rm -f "${dst}/vaccel/libsol_${m}_wrapper.so"
    gpu_flag=$([[ $dev == gpu ]] && echo "-DSOL_GPU" || true)
    docker run --rm --user "$(id -u):$(id -g)" -v "${dst}:/lib_dir" -w /lib_dir/vaccel --entrypoint sh "${CC_IMAGE}" -c \
      "g++ -std=c++2b -O2 -fPIC -shared ${gpu_flag} -DSOL_NAME='\"${m}\"' \
       -o libsol_${m}_wrapper.so sol_wrapper.c \$(pkg-config --cflags --libs vaccel) -ldl -pthread" \
      2>&1 | grep -v "<info>" || true
    [[ -f "${dst}/vaccel/libsol_${m}_wrapper.so" ]] || { echo "[install] wrapper build failed"; exit 1; }
    [[ ${WRAPPER_ONLY} -eq 1 ]] && continue

    # 4. manifest
    in_shape=$([[ "${VIDEO}" == *" ${m} "* ]] && echo "[1, 3, 16, 112, 112]" || echo "[1, 3, 224, 224]")
    out_shape="[$(grep -oP 'shapes = \[\[\K[0-9, ]+' "${dst}/${m}.py")]"
    deps=$(cd "${dst}" && ls -1 | grep -E '^(libsleef|libsol-x86-dnn)' | LC_ALL=C sort | sed 's|^|"../|; s|$|"|' | paste -sd, -)
    # LIB = wrapper only; the model library is opened by the wrapper (RTLD_LOCAL) from the DATA folder
    sol_ver=$(grep -oP 'Generated with SOL v\K\S+' "${dst}/${m}.py" | head -1)
    cat > "${dst}/vaccel/manifest.json" <<EOF
{
  "format": "sol-0.9",
  "sol_version": "${sol_ver}",
  "model": "${m}",
  "device": "${dev}",
  "libs": ["libsol_${m}_wrapper.so"],
  "data": ["../${m}.so", "../${m}.sol_weights", ${deps}],
  "input_shape": ${in_shape},
  "output_shape": ${out_shape}
}
EOF
    python3 -c "import json,sys; json.load(open(sys.argv[1]))" "${dst}/vaccel/manifest.json"
    # keep only what is used (libraries, weights); drop SOL's python/torch-extension wrapper, header and generated
    # sources (the manifest above was read from them)
    rm -rf "${dst:?}/${m:?}.py" "${dst:?}/${m:?}.h" "${dst:?}/${m:?}.cpp" "${dst:?}/${m:?}.hpp" \
           "${dst:?}/${m:?}_torch.cpp" "${dst:?}/__pycache__"
  done
done
echo "[install] done"
