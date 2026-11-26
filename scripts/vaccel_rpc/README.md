# Zero-copy vAccel RPC (vaccel-remote-sol)

vaccel-remote-sol runs SOL through vAccel `exec` (genop RPC) from the robot to the edge agent. The client and agent
we run are nubificus' `vaccel-rust` (branch `feat_genop_optional_send_write`, commit `e559125`) on their ttrpc fork
(branch `0.8.0+vaccel`, commit `edf554b`). Their transport copies every request and response buffer several times:
- **Client, send:** the input is copied 3 times before the first byte leaves (vAccel arg → protobuf arg → genop
  request → ttrpc envelope).
- **Client, receive:** the received data is zero-filled, then copied 4 more times (receive buffer, ttrpc envelope,
  genop response, output buffer). A thread is also started for every request.
- **Agent:** it does the same for the request, and copies the output 3 more times for the reply.

On the robot's N100 this is 3–4 ms per call for video and segmentation inputs/outputs.

The patches here remove those copies. **The protocol and the bytes on the wire are unchanged** (checked against
the stock client), and the stock and patched clients and agents interoperate in every combination.

| Part | Change |
|---|---|
| `ttrpc.patch` | Read without zero-fill or extra copy. Header and body in one `writev`. No thread per request. `Client::request_raw_vectored` (send pre-encoded slices, return the raw response). Server: reply envelope encoded in one pass. |
| `vaccel-rust.patch` | Client `genop` fast path (`ops/genop_fast.rs`): encodes the request around the caller's buffers and writes them with `writev` (no copy), then copies each output once from the received message into the caller's buffer. Agent `genop`: moves the output buffer into the reply instead of copying it. The client is also built as a shared library. |

## Build and install
```bash
scripts/vaccel_rpc/build.sh      # pinned upstream revisions + patches + Cargo.lock, in Docker (FROM torchvision-app:agent)
```
It installs into `src/models/runtime/vaccel-rpc/` (gitignored):
- `libvaccel-rpc-client-zc.so`: the client. Copy it to every client host (the robot).
- `vaccel-rpc-agent`: the agent, used on the edge.
- `BUILD_INFO`: commits and checksums.

## How it is used (defaults)
- **Client** (`src/sol09.py`, `VaccelRunner` with `use_remote`):
  - Loads `libvaccel-rpc-client-zc.so` (`RTLD_GLOBAL`) before its first remote session. The image's RPC plugin then calls the patched client functions, because it binds them lazily at session creation.
  - The first call (`sol_init_unpack`) checks that the patched client really served it. Otherwise the run fails, so a result is never labelled with the wrong client.
  - The summary JSON records `vaccel_rpc_client`.
  - `VACCEL_RPC_CLIENT=stock` (or `evaluate_models_ros_v2.sh --vaccel-rpc stock`) keeps the image's client.
- **Agent** (`run_agent_sol.sh`): mounts the patched `vaccel-rpc-agent` over the image's. `--stock-rpc` keeps the image's agent. `agent_server.py` takes `"rpc": "zero-copy"|"stock"` in `/start`, and `evaluate_models_ros_v2.sh` sends its `--vaccel-rpc` value.
- **Not affected:** PTC (vAccel torch op, harbor image agent) and the ROS 2 path.

## Measured (robot → edge-asus, 2026-11-26, median of two alternating rounds)
| Real SOL GPU models, per call | Stock | Zero-copy |
|---|---|---|
| ResNet-50 | 13.4 ms | 12.8 ms |
| R(2+1)D-18 | 43.0 ms | 39.9 ms |
| FCN-ResNet50 | 62.2 ms | 58.8 ms |

Outputs are bit-identical. For RPC calls with no compute and the same payloads, plain TCP is 31.9 ms (video) and
52.9 ms (segmentation); zero-copy is 33.2 and 55.0 ms, stock 35.8 and 58.6 ms.
