#!/usr/bin/env bash
# Builds the zero-copy vAccel RPC client and agent (README.md) and installs them into src/models/runtime/vaccel-rpc/:
#   libvaccel-rpc-client-zc.so  client, loaded by src/sol09.py for vaccel-remote-sol (robot side)
#   vaccel-rpc-agent            agent, used by run_agent_sol.sh (edge side)
#   BUILD_INFO                  upstream commits, patch and binary checksums
# Upstream sources are pinned (the revisions the project images run) and patched with the files next to this script.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
OUT="${ROOT}/src/models/runtime/vaccel-rpc"
IMAGE="vaccel-rpc-build:zc"

VACCEL_RUST_URL="https://github.com/nubificus/vaccel-rust.git"
VACCEL_RUST_REV="e559125a59a59742f7ce0179efccccce04f916c1"   # branch feat_genop_optional_send_write (= installed agent)
TTRPC_URL="https://github.com/nubificus/ttrpc-rust.git"
TTRPC_REV="edf554bc6f1ea954b70fc3dda52a8b9dd82f35c5"         # branch 0.8.0+vaccel (pinned by the upstream Cargo.lock)

WORK="$(mktemp -d)"
trap 'docker run --rm -v "${WORK}:/w" --entrypoint rm "${IMAGE}" -rf /w/src /w/target >/dev/null 2>&1 || true; rm -rf "${WORK}"' EXIT

echo "[vaccel-rpc] build image ${IMAGE}"
docker build -q -t "${IMAGE}" "${HERE}" >/dev/null

echo "[vaccel-rpc] sources"
mkdir -p "${WORK}/src"
git clone -q "${VACCEL_RUST_URL}" "${WORK}/src/vaccel-rust"
git -C "${WORK}/src/vaccel-rust" checkout -q "${VACCEL_RUST_REV}"
git clone -q "${TTRPC_URL}" "${WORK}/src/ttrpc-rust"
git -C "${WORK}/src/ttrpc-rust" checkout -q "${TTRPC_REV}"
git -C "${WORK}/src/vaccel-rust" apply "${HERE}/vaccel-rust.patch"
git -C "${WORK}/src/ttrpc-rust" apply "${HERE}/ttrpc.patch"
cp "${HERE}/Cargo.lock" "${WORK}/src/vaccel-rust/Cargo.lock"

echo "[vaccel-rpc] cargo build (client + agent)"
docker run --rm -v "${WORK}:/w" -w /w/src/vaccel-rust "${IMAGE}" \
  bash -c 'CARGO_TARGET_DIR=/w/target cargo build --locked --release -p vaccel-rpc-client -p vaccel-rpc-agent 2>&1 | grep -E "^(error|warning: unused)|Finished" ; test -f /w/target/release/vaccel-rpc-agent && chmod -R a+rX /w/target/release'

mkdir -p "${OUT}"
install -m 0755 "${WORK}/target/release/libvaccel_rpc_client.so" "${OUT}/libvaccel-rpc-client-zc.so"
install -m 0755 "${WORK}/target/release/vaccel-rpc-agent" "${OUT}/vaccel-rpc-agent"
{
  echo "built: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "vaccel-rust: ${VACCEL_RUST_REV}"
  echo "ttrpc-rust: ${TTRPC_REV}"
  (cd "${HERE}" && sha256sum vaccel-rust.patch ttrpc.patch Cargo.lock)
  (cd "${OUT}" && sha256sum libvaccel-rpc-client-zc.so vaccel-rpc-agent)
} > "${OUT}/BUILD_INFO"
cat "${OUT}/BUILD_INFO"
echo "[vaccel-rpc] installed into ${OUT} (copy libvaccel-rpc-client-zc.so to every client host, e.g. the robot)"
