## Dependencies

Create and activate a Python virtual environment, then install the required packages:

```shell
sudo apt install -y python3-venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install torch torchvision
```

## Download baseline models

Activate the virtual environment and download each baseline model from the PyTorch repository:

```shell
source .venv/bin/activate
python3 download_baseline_model.py --model deeplabv3_resnet50
python3 download_baseline_model.py --model fcn_resnet50
python3 download_baseline_model.py --model resnet50
python3 download_baseline_model.py --model mobilenet_v3_large
python3 download_baseline_model.py --model mc3_18
python3 download_baseline_model.py --model r3d_18
```
