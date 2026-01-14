# Docker Stats Monitoring

## Status
```shell
curl localhost:6000/monitor/status | jq
```

## Start monitoring
```shell
curl -X POST localhost:6000/monitor/start \
  -H 'Content-Type: application/json' \
  -d '{"containers":["alpine1","alpine2"],
  "interval":1.0,"csv_dir":null,"stdout":true}' | jq
```

## Start monitoring (and export results to csv)
```shell
curl -X POST localhost:6000/monitor/start \
  -H 'Content-Type: application/json' \
  -d '{"containers":["alpine1","alpine2"],
  "interval":1.0,"csv_dir":"/results/experiments/docker-stats","stdout":false}' | jq
```

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

## Stop monitoring
```shell
curl -X POST "http://localhost:6000/monitor/stop" | jq
```

---

# System (CPU, GPU) Stats Monitoring

## Health
```shell
curl localhost:6001/health | jq
```

## Status
```shell
curl localhost:6001/monitor/status | jq
```

## Start monitoring
```shell
curl -X POST localhost:6001/monitor/start \
  -H 'Content-Type: application/json' \
  -d '{
    "interval":1.0,
    "csv_dir":null,
    "stdout":true,
    "mode": "both"
  }' | jq
```

## Start monitoring (and export results to csv)
```shell
curl -X POST localhost:6001/monitor/start \
  -H 'Content-Type: application/json' \
  -d '{
    "interval":1.0,
    "csv_dir":"/results/experiments/system-stats",
    "tag": "test",
    "stdout":false
  }' | jq
```

## Stop monitoring
```shell
curl -X POST "http://localhost:6001/monitor/stop"
```