#### ML payload bandwidth over TCP (UE ↔ Edge)

[ml_payload_over_tcp_bw.sh](./ml_payload_over_tcp_bw.sh) measures **TCP throughput** and **average transfer time per ML tensor payload** for the above inputs/outputs
  
It prints: total bytes, total time, throughput (MiB/s + Mbit/s), and avg time per payload (ms).

- Uplink (UE → Edge)

On **Edge**:
```bash
nc -lk -p 5001 > /dev/null
```

On **UE**:
```bash
EDGE_IP=10.5.1.20 PORT=5001 ./ml_payload_over_tcp_bw.sh uplink
```

- Downlink (Edge → UE)

On **UE**:
```bash
nc -lk -p 5001 > /dev/null
```

On **Edge**:
```bash
UE_IP=10.3.202.66 PORT=5001 ./ml_payload_over_tcp_bw.sh downlink
```
