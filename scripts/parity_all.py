import torch, importlib.util, sys, numpy as np, torchvision
M = "/home/nextnet/adam/sol-vaccel-cv-evaluation/src/models"
name = sys.argv[1]
seg = name.startswith(("fcn", "deeplab")); vid = name in ("r3d_18", "r2plus1d_18", "swin3d_s", "swin3d_b")
shape = (1, 3, 16, 112, 112) if vid else (1, 3, 224, 224)
sd = torch.load(f"{M}/{name}/{name}_state_dict.pt", map_location="cpu")
kw = {"weights": None}
if seg: kw.update(weights_backbone=None, aux_loss=any(k.startswith("aux_classifier.") for k in sd))
m = torchvision.models.get_model(name, **kw); m.load_state_dict(sd); m = m.cuda().eval()
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
spec = importlib.util.spec_from_file_location("m", f"{M}/{name}_sol/lib_gpu/sol_{name}.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
s = getattr(mod, f"sol_{name}")(f"{M}/{name}_sol/lib_gpu"); s.init()
aoti = torch._inductor.aoti_load_package(f"{M}/{name}/{name}_cuda.pt2")
def cos(a, b): a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64); return a @ b / np.linalg.norm(a) / np.linalg.norm(b)
res = {"sol": [], "aoti": []}; agree = {"sol": 0, "aoti": 0}
rng = np.random.default_rng(0)
for i in range(3):
    x = (rng.random(shape, dtype=np.float32) - 0.45) / 0.226   # ~normalised-image range
    with torch.no_grad():
        r = m(torch.from_numpy(x).cuda()); r = (r["out"] if seg else r).cpu().numpy()
        a = aoti(torch.from_numpy(x).cuda()); a = (a[0] if isinstance(a, (list, tuple)) else a).cpu().numpy()
    o = s.run(np.ascontiguousarray(x)); s.sync(); o = o[0] if isinstance(o, tuple) else o
    for k, v in (("sol", o), ("aoti", a)):
        res[k].append(cos(v, r))
        agree[k] += float((v.argmax(1) == r.argmax(1)).mean())
print(f"{name:20s} SOL cos {min(res['sol']):.6f} argmax-agree {agree['sol']/3:.3f} | AOTI cos {min(res['aoti']):.6f} argmax-agree {agree['aoti']/3:.3f}")
