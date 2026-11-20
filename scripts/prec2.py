import torch, importlib.util, sys, numpy as np, torchvision
name, wts = sys.argv[1], sys.argv[2]; M = "/home/nextnet/adam/sol-vaccel-cv-evaluation/src/models"
torch.hub.set_dir("/tmp/claude-1000/-home-nextnet-adam/6ec546b2-e797-4e75-9032-228e4d2b9126/scratchpad/hub")
seg = name.startswith(("fcn", "deeplab")); vid = name in ("r3d_18", "r2plus1d_18")
shape = (1, 3, 16, 112, 112) if vid else (1, 3, 224, 224)
if wts == "sd":
    sd = torch.load(f"{M}/{name}/{name}_state_dict.pt"); kw = {"weights": None}
    if seg: kw.update(weights_backbone=None, aux_loss=any(k.startswith("aux_classifier.") for k in sd))
    m = torchvision.models.get_model(name, **kw); m.load_state_dict(sd)
else:
    m = torchvision.models.get_model(name, weights=torchvision.models.get_model_weights(name)[wts])
m = m.cuda().eval()
spec = importlib.util.spec_from_file_location("m", f"{M}/{name}_sol/lib_gpu/sol_{name}.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
s = getattr(mod, f"sol_{name}")(f"{M}/{name}_sol/lib_gpu"); s.init()
aoti = torch._inductor.aoti_load_package(f"{M}/{name}/{name}_cuda.pt2") if wts == "sd" else None
x = np.ascontiguousarray((np.random.default_rng(1).random(shape, dtype=np.float32) - 0.45) / 0.226)
def eager(tf32):
    torch.backends.cuda.matmul.allow_tf32 = tf32; torch.backends.cudnn.allow_tf32 = tf32
    with torch.no_grad(): r = m(torch.from_numpy(x).cuda()); return (r["out"] if seg else r).double().cpu().numpy()
ref, t32 = eager(False), eager(True)
o = s.run(x); s.sync(); o = (o[0] if isinstance(o, tuple) else o).astype(np.float64)
rel = lambda v: np.abs(v - ref).max() / np.abs(ref).max()
line = f"{name:20s} rel.err vs FP32: eager-TF32 {rel(t32):.1e} | SOL {rel(o):.1e}"
if aoti is not None:
    torch.backends.cudnn.allow_tf32 = True; torch.backends.cuda.matmul.allow_tf32 = False
    with torch.no_grad(): a = aoti(torch.from_numpy(x).cuda())
    line += f" | AOTI {rel((a[0] if isinstance(a,(list,tuple)) else a).double().cpu().numpy()):.1e}"
print(line)
