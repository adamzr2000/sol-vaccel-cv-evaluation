# Figure caption

`Inference latency measured at the robot for multiple vision models. Local execution uses PyTorch and SOL-optimized libraries on the robot CPU, while remote execution offloads inference to edge CPU/GPU resources via vAccel`

# Key Observations

## Model stats - inference latency
...

## System stats - CPU
...

## System stats - GPU
...

## Docker stats – CPU
...

## Docker stats – RAM
...

## Docker stats – network

- Uplink: ~602 KB per image (all models)
- Downlink: ~4 KB (image classification) and ~4.21 MB (segmentation)

### Robot network traffic (TX) — `torchvision-app` container (remote execution)

* **Segmentation models generate the highest robot TX** (e.g., `deeplabv3_resnet50`, `fcn_resnet50`) because remote inference sustains higher request throughput and keeps the uplink continuously busy during offloading (robot repeatedly sends inputs + protocol overhead).
* **Video classification shows lower robot TX than segmentation** (`mc3_18`, `r3d_18`) since each request carries a larger input tensor (16-frame clip), reducing the achievable request rate; lower requests/sec translates to lower average TX Mbps.

### Edge network traffic (TX) — `vaccel-agent` container (remote execution)

* **Edge TX is very low for classification and video models** because the returned outputs are compact (e.g., 1000-class or 400-class logits), so the downlink response payload is tiny compared to the uplink input stream.
* **Segmentation dominates edge TX** because the agent returns a dense per-pixel output tensor (e.g., `21×224×224` float32 logits), which is orders of magnitude larger than classification/video outputs and therefore drives high edge→robot throughput.
