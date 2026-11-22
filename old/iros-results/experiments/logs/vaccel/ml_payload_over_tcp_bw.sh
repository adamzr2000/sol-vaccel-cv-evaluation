#!/usr/bin/env bash
set -euo pipefail

# Purpose: measure network cost (throughput + avg transfer time) of ML tensor payloads over TCP (UE <-> Edge).

EDGE_IP="${EDGE_IP:-10.5.1.20}"   # used for uplink (UE -> EDGE)
UE_IP="${UE_IP:-}"                # required for downlink (EDGE -> UE)
PORT="${PORT:-5001}"

# -----------------------------
# Payload sizes (bytes), float32
# -----------------------------
# Image input (classification/segmentation): (1,3,224,224)
UP_IMG_BS=602112

# Video input (video classification): (1,3,16,112,112)
UP_VID_BS=2408448

# Downlink outputs
DL_CLS_BS=4000        # (1,1000)
DL_SEG_BS=4214784     # (1,21,224,224)
DL_VID_BS=1600        # (1,400)

# -----------------------------
# Default counts (enough data for stable throughput)
# -----------------------------
UP_IMG_COUNT="${UP_IMG_COUNT:-200}"      # ~115 MiB
UP_VID_COUNT="${UP_VID_COUNT:-50}"       # ~115 MiB

DL_CLS_COUNT="${DL_CLS_COUNT:-200000}"   # ~800 MB (small payload => need many)
DL_SEG_COUNT="${DL_SEG_COUNT:-50}"       # ~210 MB
DL_VID_COUNT="${DL_VID_COUNT:-500000}"   # ~800 MB (small payload => need many)

need() { command -v "$1" >/dev/null 2>&1 || { echo "Missing '$1'"; exit 1; }; }
need dd; need nc; need awk; need /usr/bin/time

print_summary() {
  local bytes="$1" sec="$2" count="$3" bs="$4"
  awk -v b="$bytes" -v s="$sec" -v n="$count" -v bs="$bs" '
    BEGIN {
      mib=b/1048576.0
      printf("\nSummary:\n")
      printf("  Total:      %.2f MiB (%d bytes)\n", mib, b)
      printf("  Time:       %.4f s\n", s)
      printf("  Throughput: %.2f MiB/s (%.1f Mbit/s)\n", (mib/s), ((b*8.0/1000000.0)/s))
      printf("  Avg/payload (%d bytes): %.6f s (%.3f ms)\n", bs, (s/n), (s*1000.0/n))
    }
  '
}

run_tcp_stream_test() {
  local label="$1" bs="$2" count="$3" host="$4"
  local total_bytes=$((bs * count))

  echo "=== ${label} ==="
  echo "Target:  ${host}:${PORT}"
  echo "Payload: ${bs} bytes   Count: ${count}"
  echo "Listener on receiver:  nc -lk -p ${PORT} > /dev/null"
  echo

  local t
  t=$(/usr/bin/time -f "%e" sh -c "dd if=/dev/zero bs=${bs} count=${count} status=none | nc -q 0 ${host} ${PORT}" 2>&1)
  t="$(echo "$t" | tr ',' '.')"

  print_summary "$total_bytes" "$t" "$count" "$bs"
}

usage() {
  echo "Usage:"
  echo "  # Run on UE (uplink UE->EDGE):"
  echo "  EDGE_IP=10.5.1.20 PORT=5001 $0 uplink [img|vid|all]"
  echo
  echo "  # Run on EDGE (downlink EDGE->UE):"
  echo "  UE_IP=<ue-ip> PORT=5001 $0 downlink [cls|seg|vid|all]"
}

mode="${1:-}"
kind="${2:-all}"

case "$mode" in
  uplink)
    case "$kind" in
      img)
        run_tcp_stream_test "UPLINK (UE -> EDGE) : image input tensor (1,3,224,224) float32" "$UP_IMG_BS" "$UP_IMG_COUNT" "$EDGE_IP"
        ;;
      vid)
        run_tcp_stream_test "UPLINK (UE -> EDGE) : video input tensor (1,3,16,112,112) float32" "$UP_VID_BS" "$UP_VID_COUNT" "$EDGE_IP"
        ;;
      all)
        run_tcp_stream_test "UPLINK (UE -> EDGE) : image input tensor (1,3,224,224) float32" "$UP_IMG_BS" "$UP_IMG_COUNT" "$EDGE_IP"
        echo
        run_tcp_stream_test "UPLINK (UE -> EDGE) : video input tensor (1,3,16,112,112) float32" "$UP_VID_BS" "$UP_VID_COUNT" "$EDGE_IP"
        ;;
      *) echo "Use: uplink [img|vid|all]"; exit 1 ;;
    esac
    ;;
  downlink)
    [[ -n "$UE_IP" ]] || { echo "UE_IP required for downlink (EDGE -> UE)."; echo; usage; exit 1; }
    case "$kind" in
      cls)
        run_tcp_stream_test "DOWNLINK (EDGE -> UE) : classification logits (1,1000) float32" "$DL_CLS_BS" "$DL_CLS_COUNT" "$UE_IP"
        ;;
      seg)
        run_tcp_stream_test "DOWNLINK (EDGE -> UE) : segmentation logits (1,21,224,224) float32" "$DL_SEG_BS" "$DL_SEG_COUNT" "$UE_IP"
        ;;
      vid)
        run_tcp_stream_test "DOWNLINK (EDGE -> UE) : video classification logits (1,400) float32" "$DL_VID_BS" "$DL_VID_COUNT" "$UE_IP"
        ;;
      all)
        run_tcp_stream_test "DOWNLINK (EDGE -> UE) : classification logits (1,1000) float32" "$DL_CLS_BS" "$DL_CLS_COUNT" "$UE_IP"
        echo
        run_tcp_stream_test "DOWNLINK (EDGE -> UE) : segmentation logits (1,21,224,224) float32" "$DL_SEG_BS" "$DL_SEG_COUNT" "$UE_IP"
        echo
        run_tcp_stream_test "DOWNLINK (EDGE -> UE) : video classification logits (1,400) float32" "$DL_VID_BS" "$DL_VID_COUNT" "$UE_IP"
        ;;
      *) echo "Use: downlink [cls|seg|vid|all]"; exit 1 ;;
    esac
    ;;
  *)
    usage
    exit 1
    ;;
esac

