import torch, importlib.util, sys, re, glob, numpy as np, os
from PIL import Image
import torchvision.transforms as T
name, lib = sys.argv[1], sys.argv[2]
M = "/models"
tf = T.Compose([T.ToTensor(), T.Resize(256, antialias=True), T.CenterCrop(224), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
spec = importlib.util.spec_from_file_location("m", f"{M}/{name}_sol/{lib}/sol_{name}.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
s = getattr(mod, f"sol_{name}")(f"{M}/{name}_sol/{lib}"); s.init()
files = sorted(glob.glob("/data/imagenet/images/*.jpg"))[:100]
ok, outs = 0, []
for f in files:
    y = int(re.search(r"label_(\d+)", f).group(1))
    x = np.ascontiguousarray(tf(Image.open(f).convert("RGB")).unsqueeze(0).numpy())
    o = np.zeros((1, 1000), np.float32); s.run(x, o); s.sync(); ok += int(o.argmax() == y)
o2 = np.zeros((1, 1000), np.float32); s.run(x, o2); s.sync()
print(f"[container] {name:10s} {lib}: top-1 {ok/len(files):.3f} on {len(files)} imgs; repeat-call max|d| {np.abs(o-o2).max():.2e}")
