# Testbed network: robot -> edge-asus (10.5.1.20)

Measured from the robot on 2026-11-22, 11:00–11:05 UTC. Path: robot `ue0` (1 Gbit/s USB-Ethernet)
-> miniPC 192.168.2.1 -> WiFi -> edge. Same path as the vAccel RPC traffic.

| Metric | Mean | Std | Samples |
|---|---|---|---|
| RTT latency | 4.56 ms | 0.34 ms | 300 pings |
| Jitter | 0.30 ms | 0.39 ms | 299 |
| Uplink throughput (robot -> edge) | 648.0 Mbit/s | 9.6 Mbit/s | 90 × 1 s |
| Downlink throughput (edge -> robot) | 743.1 Mbit/s | 70.9 Mbit/s | 90 × 1 s |

Packet loss: 0% (300 pings).

**Method**
- Latency: `ping -c 300 -i 0.2` (60 s). Jitter = |difference of consecutive RTTs|.
- Throughput: `iperf3` TCP, 3 runs × 30 s per direction (alternating; downlink with `-R`),
  first 2 s omitted (`-O 2`); statistics over the 1-s intervals of all runs.

Rerun: `iperf3 -s` on the edge, then on the robot `./measure_network.sh` (writes `raw/`),
then `python3 summarize.py` (writes `summary.csv`).
