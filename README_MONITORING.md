# Monitoring Services

This repository provides two complementary monitoring services that can be used independently or together during experimental runs:

1. **Docker Stats Collector** – per-container resource usage (CPU, memory, disk I/O, network).
2. **System Stats Collector** – host-level CPU and GPU metrics (power, utilization, temperature).

Both services expose REST APIs and optionally export metrics to CSV files for offline analysis.

---

# Docker Stats Monitoring

Per-container resource monitoring based on Docker statistics and host-level fallbacks.

## Metrics Description

Each sample corresponds to a measurement window ending at the reported timestamp and is collected per monitored container.

### CPU Metrics

- **cpu_percent**  
  Container CPU utilization (%) computed from Docker stats CPU time deltas  
  (`cpu_stats` vs `precpu_stats`), normalized by the number of online CPUs.  
  Represents the fraction of host CPU time consumed by the container during the most recent reporting window.

### Memory Metrics

- **mem_mb**  
  Container memory usage (MB) computed as **memory usage minus page cache**  
  (`memory_stats.usage - memory_stats.stats.cache`).  
  Approximates the container working set (actively used memory).

- **mem_limit_mb**  
  Container memory limit (MB) reported by Docker (`memory_stats.limit`).  
  If the reported limit is extremely large (treated as unlimited), this value is recorded as `null`.

- **mem_percent**  
  Container memory usage as a percentage of the configured memory limit:
  ```
  mem_percent = 100 × (mem_mb / mem_limit_mb)
  ```
  If no effective memory limit is configured, this value is recorded as `null`.

### Disk I/O Metrics

- **blk_read_mb**  
  Cumulative block device data read by the container, expressed in MB.  
  Obtained from Docker `blkio_stats` when available, with fallbacks to:
  cgroup v1 blkio files, cgroup v2 `io.stat`, host `/proc/<pid>/io`, or exec-based reads inside the container.  
  This is a **cumulative counter**, not a throughput rate.

- **blk_write_mb**  
  Cumulative block device data written by the container, expressed in MB.  
  Same sourcing and semantics as `blk_read_mb`.  
  This is a **cumulative counter**, not a throughput rate.

### Network I/O Metrics

- **net_rx_mb**  
  Cumulative network data received by the container across all interfaces, expressed in MB,  
  obtained from Docker stats `networks.*.rx_bytes`.  
  This is a **cumulative counter**, not a throughput rate.

- **net_tx_mb**  
  Cumulative network data transmitted by the container across all interfaces, expressed in MB,  
  obtained from Docker stats `networks.*.tx_bytes`.  
  This is a **cumulative counter**, not a throughput rate.

## Timestamp Semantics

- **timestamp**  
  Unix epoch timestamp in milliseconds (UTC).

- **timestamp_iso**  
  ISO-8601 formatted timestamp in UTC.

## Notes on Semantics and Accuracy

- CPU and memory metrics are derived from Docker’s native resource accounting and reflect container usage relative to host resources.  
- Disk and network metrics are cumulative byte counters; per-interval throughput (e.g., MB/s) can be computed offline by differentiating consecutive samples.

## Docker Stats – Usage

Check monitoring status:
```shell
curl localhost:6000/monitor/status | jq
```

Start monitoring (stdout only):
```shell
curl -X POST localhost:6000/monitor/start \
  -H 'Content-Type: application/json' \
  -d '{
    "containers": ["alpine1", "alpine2"],
    "interval": 1.0,
    "csv_dir": null,
    "stdout": true
  }' | jq
```

Start monitoring and export results to CSV:
```shell
curl -X POST localhost:6000/monitor/start \
  -H 'Content-Type: application/json' \
  -d '{
    "containers": ["alpine1", "alpine2"],
    "interval": 1.0,
    "csv_dir": "/results/experiments/docker-stats",
    "csv_names": {
      "alpine1": "experimentA-alpine1.csv",
      "alpine2": "experimentA-alpine2.csv"
    },
    "stdout": false
  }' | jq
```

Stop monitoring:
```shell
curl -X POST http://localhost:6000/monitor/stop | jq
```

---

# System Stats Monitoring

Host-level CPU and GPU monitoring using OS counters, Intel RAPL, and NVIDIA NVML.

## CPU Metrics Description

Each CPU sample corresponds to a measurement window ending at the reported timestamp.

- **cpu_watts**  
  Average CPU package power consumption (W) over the last sampling interval,  
  computed from Intel RAPL energy counters.

- **cpu_util_percent**  
  Average overall CPU utilization (%) computed from operating-system CPU time counters,  
  representing the fraction of time the CPU was busy between consecutive samples.

- **cpu_temp_c**  
  Representative CPU temperature (°C) measured at sampling time.  
  Package temperature is preferred; otherwise, the average of available core temperatures is used.

## GPU Metrics Description

Each GPU sample corresponds to the sampling instant ending at the reported timestamp and is reported per GPU device.

- **power_draw_w**  
  Instantaneous GPU power consumption (W) reported by NVIDIA NVML at sampling time.

- **power_limit_w**  
  Configured GPU power limit (W) enforced by the driver at sampling time.

- **util_gpu_percent**  
  GPU compute utilization (%) reported by NVML, indicating recent GPU engine activity.

- **util_mem_percent**  
  GPU memory controller utilization (%) reported by NVML, indicating recent memory access activity.

- **mem_used_mb**  
  GPU memory usage (MB) measured at sampling time.

- **temp_c**  
  GPU temperature (°C) measured at sampling time.

## Timestamp Semantics

- **timestamp**  
  Unix epoch timestamp in milliseconds (UTC).

- **timestamp_iso**  
  ISO-8601 formatted timestamp in UTC.

GPU metrics reflect instantaneous or short-window measurements provided by NVML,  
whereas CPU power metrics are averaged over the sampling interval.

## System Stats – Usage

Health check:
```shell
curl localhost:6001/health | jq
```

Check monitoring status:
```shell
curl localhost:6001/monitor/status | jq
```

Start monitoring (stdout only):
```shell
curl -X POST localhost:6001/monitor/start \
  -H 'Content-Type: application/json' \
  -d '{
    "interval": 1.0,
    "csv_dir": null,
    "stdout": true,
    "mode": "both"
  }' | jq
```

Start monitoring and export results to CSV:
```shell
curl -X POST localhost:6001/monitor/start \
  -H 'Content-Type: application/json' \
  -d '{
    "interval": 1.0,
    "csv_dir": "/results/experiments/system-stats",
    "mode": "both",
    "csv_names": {
      "cpu": "run1_benchmark_cpu.csv",
      "gpu": "run1_benchmark_gpu.csv"
    },
    "stdout": false
  }' | jq
```

Stop monitoring:
```shell
curl -X POST http://localhost:6001/monitor/stop
```
