import torch
import sol
import numpy as np
from ultralytics import YOLO
from time import time
# model = YOLO("yolov5s.pt")
model = YOLO("yolov5su.pt")
model.eval()
model.to("cuda")
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True) # alternative download
# ex = torch.export.export(model.model, (input_t, )) # fails. if this would work, the model would be compileable into a binary
 
reps=20
input = np.random.rand(1,3,640,640).astype("float32")
 
sol.config["autotuning"] = True
input_t = torch.Tensor(input).to("cuda")
with torch.no_grad():
    inner_model = model.model
    _ = inner_model(input_t) # warmup
    start=time()
    for _ in range(reps):
        _ = inner_model(input_t)
    print(f"Torch: {(time()-start)/reps}")
 
with torch.no_grad():
    inner_model = sol.optimize(model.model, [input_t], vdims=[False])
    model.model=inner_model
    _ = inner_model(input_t) # warmup
    start=time()
    for _ in range(reps):
        _ = inner_model(input_t)
    print(f"SOL: {(time()-start)/reps}")
 
