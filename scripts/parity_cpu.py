import torch, importlib.util, sys, numpy as np, torchvision
name = sys.argv[1]; M = "/models"
vid = name.startswith("swin3d") or name in ("r3d_18", "r2plus1d_18")
shape = (1, 3, 16, 112, 112) if vid else (1, 3, 224, 224)
m = torchvision.models.get_model(name, weights=None); m.load_state_dict(torch.load(f"{M}/{name}/{name}_state_dict.pt")); m.eval()
spec = importlib.util.spec_from_file_location("m", f"{M}/{name}_sol/lib_cpu/sol_{name}.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
s = getattr(mod, f"sol_{name}")(f"{M}/{name}_sol/lib_cpu"); s.init()
rng = np.random.default_rng(0); agree, cs = 0, []
for i in range(3):
    x = np.ascontiguousarray((rng.random(shape, dtype=np.float32) - 0.45) / 0.226)
    with torch.no_grad(): r = m(torch.from_numpy(x)).numpy().ravel().astype(np.float64)
    o = s.run(x); o = (o[0] if isinstance(o, tuple) else o).ravel().astype(np.float64)
    cs.append(o @ r / np.linalg.norm(o) / np.linalg.norm(r)); agree += int(o.argmax() == r.argmax())
print(f"[container] {name:10s} lib_cpu vs eager: min cos {min(cs):.4f}  argmax agree {agree}/3")
