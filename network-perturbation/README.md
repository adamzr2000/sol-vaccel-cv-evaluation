# network-perturbation

Start/stop a fixed-rate iperf3 UDP cross-traffic flow to model uplink
network congestion for the vAccel-vs-ROS2 experiments. Separate from, and
doesn't touch, either benchmark harness.

```bash
# edge host (receives the extra load):
./iperf3_congestion.sh server

# robot host (generates competing uplink traffic toward the edge):
./iperf3_congestion.sh client --target <edge-ip> --rate 50M --duration 120
```

`client` returns immediately once running — run your actual vaccel/ROS
benchmark concurrently. It stops itself after `--duration`; `stop`/`status`
work regardless. `--reverse` models downlink instead of uplink.

**Check for a conflicting system service first:** `systemctl is-active
iperf3` — some hosts already run `iperf3.service` permanently on port
5201. If so, use `--port` to pick a free one (this script never touches
that service either way).

Commands: `server`, `client --target HOST [--rate R] [--duration S] [--port P] [--proto udp|tcp] [--reverse] [--parallel N] [--force]`, `stop [server|client|all]`, `status`.
