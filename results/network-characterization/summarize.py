#!/usr/bin/env python3
"""Summarize raw/ (from measure_network.sh) into summary.csv: mean and std of
RTT, jitter (|difference of consecutive RTTs|) and TCP throughput per direction
(over the 1-s iperf3 intervals of all runs, slow-start seconds omitted)."""
import csv
import glob
import json
import re
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"

rtt = [float(m) for m in re.findall(r"time=([\d.]+) ms", (RAW / "ping.txt").read_text())]
sent = int(re.search(r"(\d+) packets transmitted", (RAW / "ping.txt").read_text()).group(1))
jitter = [abs(b - a) for a, b in zip(rtt, rtt[1:])]

rows = [("RTT latency", "ms", st.mean(rtt), st.stdev(rtt), len(rtt)),
        ("Jitter", "ms", st.mean(jitter), st.stdev(jitter), len(jitter))]
for name, tag in (("Uplink throughput (robot -> edge)", "up"), ("Downlink throughput (edge -> robot)", "down")):
    mbps = []
    for f in sorted(glob.glob(str(RAW / f"iperf_{tag}_*.json"))):
        mbps += [i["sum"]["bits_per_second"] / 1e6 for i in json.load(open(f))["intervals"] if not i["sum"]["omitted"]]
    rows.append((name, "Mbit/s", st.mean(mbps), st.stdev(mbps), len(mbps)))

with open(HERE / "summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric", "unit", "mean", "std", "samples"])
    w.writerows((m, u, f"{a:.2f}", f"{s:.2f}", n) for m, u, a, s, n in rows)
for m, u, a, s, n in rows:
    print(f"{m:38s} {a:8.2f} ± {s:6.2f} {u:7s} (n={n})")
print(f"Packet loss: {100 * (sent - len(rtt)) / sent:.1f}% of {sent} pings")
