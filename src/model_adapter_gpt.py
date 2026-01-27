import torch
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import os
import sys
import importlib
import numpy as np

# =========================================================
# SOL execution mode:
#   SOL_RUN_MODE=2 -> Option 2 default (run(*args) each call)
#   SOL_RUN_MODE=3 -> Option 3 default (set_IO + optimize; run() + get_output)
#
# Extra knobs:
#   SOL_OPTIMIZE=0/1 -> enable/disable optimize(2) when GPU
#   SOL_SEG_HEAD=auto|out|aux -> which head to use for segmentation when aux exists
# =========================================================
SOL_RUN_MODE = os.environ.get("SOL_RUN_MODE", "2").strip()
if SOL_RUN_MODE not in ("2", "3"):
    print(f"⚠️  Unknown SOL_RUN_MODE='{SOL_RUN_MODE}', defaulting to 2")
    SOL_RUN_MODE = "2"

SOL_OPTIMIZE = os.environ.get("SOL_OPTIMIZE", "1").strip().lower() in ("1", "true", "yes", "y", "on")

SOL_SEG_HEAD = os.environ.get("SOL_SEG_HEAD", "auto").strip().lower()  # auto|out|aux
if SOL_SEG_HEAD not in ("auto", "out", "aux"):
    print(f"⚠️  Unknown SOL_SEG_HEAD='{SOL_SEG_HEAD}', defaulting to auto")
    SOL_SEG_HEAD = "auto"


class BaseModelAdapter:
    def __init__(self, device):
        self.device = device
        self.model = None
        self.categories = None

    def load_model(self, model_path):
        raise NotImplementedError

    def preprocess(self, image_path):
        raise NotImplementedError

    def infer(self, input_tensor):
        raise NotImplementedError

    def postprocess(self, output):
        return output


# =========================================================
# HELPER: OPENCV VIDEO LOADER
# =========================================================
def load_video_with_cv2(video_path, num_frames=16, transform=None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Video {video_path} has 0 frames.")

    indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    frames_list = []
    current_idx = 0

    for target_frame_idx in indices:
        if target_frame_idx != current_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
            current_idx = target_frame_idx

        ret, frame = cap.read()
        current_idx += 1

        if not ret:
            if frames_list:
                frame = frames_list[-1]
            else:
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if transform:
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1)
            frames_list.append(transform(frame_tensor))
        else:
            frames_list.append(torch.from_numpy(frame))

    cap.release()
    return torch.stack(frames_list, dim=1)


# =========================================================
# 1. ADAPTER FOR PYTORCH BASELINES
# =========================================================
class PyTorchBaselineAdapter(BaseModelAdapter):
    def __init__(self, device, builder_func, weights_filename, model_type, weights_enum=None):
        super().__init__(device)
        self.torch_device = torch.device(device)
        self.builder_func = builder_func
        self.weights_filename = weights_filename
        self.model_type = model_type

        if model_type == "video_classification":
            if hasattr(weights_enum, "meta"):
                self.categories = weights_enum.meta["categories"]

            # Match SOL resolution (112x112) for fair comparison
            self.transform = transforms.Compose([
                transforms.Resize(128, antialias=True),
                transforms.CenterCrop(112),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=[0.432, 0.394, 0.377], std=[0.228, 0.221, 0.217]),
            ])

        elif model_type == "classification":
            self.categories = weights_enum.meta["categories"]
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(256, antialias=True),
                transforms.CenterCrop(224),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        elif model_type == "segmentation":
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(256, antialias=True),
                transforms.CenterCrop(224),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def load_model(self, model_dir):
        print(f"   [Adapter] Loading PyTorch {self.model_type} model from {model_dir}...")

        if self.model_type == "segmentation":
            self.model = self.builder_func(weights=None, weights_backbone=None, aux_loss=True)
        else:
            self.model = self.builder_func(weights=None)

        weights_path = os.path.join(model_dir, self.weights_filename)
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights not found at {weights_path}")

        state_dict = torch.load(weights_path, map_location=self.torch_device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.torch_device)
        self.model.eval()

    def preprocess(self, input_path):
        if self.model_type == "video_classification":
            ext = os.path.splitext(input_path)[1].lower()
            if ext in [".mp4", ".avi", ".mov"]:
                return load_video_with_cv2(input_path, num_frames=16, transform=self.transform).unsqueeze(0)
            else:
                img = cv2.imread(str(input_path))
                if img is None:
                    raise FileNotFoundError(f"Failed: {input_path}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                dummy_tensor = torch.from_numpy(img).permute(2, 0, 1)
                frame_tensor = self.transform(dummy_tensor)
                return torch.stack([frame_tensor] * 16, dim=1).unsqueeze(0)  # CPU tensor
        else:
            img = cv2.imread(str(input_path))
            if img is None:
                raise FileNotFoundError(f"Failed: {input_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return self.transform(img).unsqueeze(0)  # CPU tensor

    def infer(self, input_tensor):
        # Fair timing vs SOL: SOL takes CPU (NumPy) buffers and returns CPU buffers,
        # so its "inference time" includes CPU↔GPU transfers. We do the same here by
        # moving input to GPU and bringing output back to CPU inside this function.
        input_tensor = input_tensor.to(self.torch_device)

        output = self.model(input_tensor)
        if self.model_type == "segmentation":
            output = output["out"]

        # Bring results to CPU inside the timed region (matches SOL output buffers).
        output = output.detach().cpu()
        return output

    def postprocess(self, output_tensor):
        if self.model_type == "segmentation":
            return torch.argmax(output_tensor.squeeze(0), dim=0).byte()
        else:
            probs = torch.nn.functional.softmax(output_tensor, dim=1)
            top_prob, top_class = torch.max(probs, dim=1)
            return top_class, top_prob


# =========================================================
# 2. ADAPTER FOR SOL COMPILER
# =========================================================
class SolAdapter(BaseModelAdapter):
    def __init__(self, device, model_name, model_type="classification"):
        super().__init__(device)
        self.model_name = model_name
        self.model_type = model_type

        # Whether we've bound IO buffers via set_IO() (required for run() with no args)
        self.io_bound = False

        # --- 1. CONFIGURATION BASED ON MODEL TYPE ---
        if self.model_type == "video_classification":
            # Categories (Kinetics-400) where available
            if "mc3_18" in model_name:
                self.categories = models.video.MC3_18_Weights.DEFAULT.meta.get("categories", None)
            elif "r3d_18" in model_name:
                self.categories = models.video.R3D_18_Weights.DEFAULT.meta.get("categories", None)
            elif "r2plus1d_18" in model_name:
                self.categories = models.video.R2Plus1D_18_Weights.DEFAULT.meta.get("categories", None)
            elif "swin3d_t" in model_name:
                self.categories = models.video.Swin3D_T_Weights.DEFAULT.meta.get("categories", None)
            elif "swin3d_s" in model_name:
                self.categories = models.video.Swin3D_S_Weights.DEFAULT.meta.get("categories", None)
            elif "swin3d_b" in model_name:
                self.categories = models.video.Swin3D_B_Weights.DEFAULT.meta.get("categories", None)

            # SOL Video: 112x112
            self.transform = transforms.Compose([
                transforms.Resize(128, antialias=True),
                transforms.CenterCrop(112),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=[0.432, 0.394, 0.377], std=[0.228, 0.221, 0.217]),
            ])

        elif self.model_type == "classification":
            if "resnet50" in model_name:
                self.categories = models.ResNet50_Weights.DEFAULT.meta.get("categories", None)
            elif "mobilenet" in model_name:
                self.categories = models.MobileNet_V3_Large_Weights.DEFAULT.meta.get("categories", None)
            elif "swin_t" in model_name:
                self.categories = models.Swin_T_Weights.DEFAULT.meta.get("categories", None)
            elif "swin_s" in model_name:
                self.categories = models.Swin_S_Weights.DEFAULT.meta.get("categories", None)
            elif "swin_v2_b" in model_name:
                self.categories = models.Swin_V2_B_Weights.DEFAULT.meta.get("categories", None)

            # SOL Image Classification: 224x224
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(256, antialias=True),
                transforms.CenterCrop(224),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        elif self.model_type == "segmentation":
            # SOL Segmentation: 224x224
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(256, antialias=True),
                transforms.CenterCrop(224),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        self.execution_args = []
        self.input_buffer = None
        self.output_buffer = None
        self.vdims = None

    def load_model(self, model_dir):
        lib_path = self._get_model_lib_path(model_dir)
        self._load_model_from_path(lib_path)

    def _get_model_lib_path(self, model_dir):
        # Determine library path based on device
        target_lib = "lib_gpu" if self.device == "cuda" else "lib_cpu"
        lib_path = os.path.join(model_dir, target_lib)

        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"SOL library path not found: {lib_path}")

        return lib_path

    def _load_model_from_path(self, lib_path, use_remote=False):
        print(f"   [Adapter] Loading SOL model from {lib_path}...")

        module_name = f"sol_{self.model_name}"
        module_path = os.path.join(lib_path, f"{module_name}.py")
        try:
            spec = importlib.util.spec_from_file_location(lib_path, module_path)
            sol_module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = sol_module
            spec.loader.exec_module(sol_module)
        except ImportError as e:
            raise ImportError(f"Could not import SOL wrapper for '{module_name}'. Error: {e}")

        model_class = getattr(sol_module, module_name)

        kwargs = {}
        if hasattr(model_class, "use_remote"):
            kwargs["use_remote"] = use_remote

        self.model = model_class(path=lib_path, **kwargs)
        self.model.init()

        # --- 2. BUFFER INITIALIZATION ---
        self.vdims = np.array([1], dtype=np.int64)

        if self.model_type == "video_classification":
            self.input_buffer = np.zeros((1, 3, 16, 112, 112), dtype=np.float32)
            self.output_buffer = np.zeros((1, 400), dtype=np.float32)
            self.execution_args = [self.input_buffer, self.output_buffer, self.vdims]

        elif self.model_type == "classification":
            self.input_buffer = np.zeros((1, 3, 224, 224), dtype=np.float32)
            self.output_buffer = np.zeros((1, 1000), dtype=np.float32)
            self.execution_args = [self.input_buffer, self.output_buffer, self.vdims]

        else:  # Segmentation
            self.input_buffer = np.zeros((1, 3, 224, 224), dtype=np.float32)
            self.output_buffer = np.zeros((1, 21, 224, 224), dtype=np.float32)
            self.aux_buffer = np.zeros((1, 21, 224, 224), dtype=np.float32)
            self.execution_args = [self.input_buffer, self.output_buffer, self.aux_buffer, self.vdims]

        # --- 3. PURE SOL MODE SELECTION ---
        print(f"   [SOL] {'GPU' if self.device == 'cuda' else 'CPU'} mode | SOL_RUN_MODE={SOL_RUN_MODE} | SOL_OPTIMIZE={int(SOL_OPTIMIZE)}")

        # IMPORTANT FIX:
        # If we call set_IO (and especially optimize), the correct execution path is:
        #   run() + get_output()  (Option 3 style)
        #
        # In Mode 2, we can still set_IO/optimize once on GPU to get speed-ups,
        # but then infer() MUST use run() + get_output().
        self.io_bound = False

        if SOL_RUN_MODE == "3":
            # Option 3: bind buffers once; optionally optimize on GPU
            try:
                if hasattr(self.model, "set_IO"):
                    self.model.set_IO(self.execution_args)
                    self.io_bound = True

                if SOL_OPTIMIZE and self.device == "cuda" and hasattr(self.model, "optimize"):
                    print("   [SOL] Running GPU Optimization (Level 2)...")
                    self.model.optimize(2)

            except Exception as e:
                print(f"   ⚠️ SOL set_IO/optimize warning: {e}")

        else:
            # Option 2 default (run(*args)), but for GPU we can still bind IO and optimize once.
            print("   [SOL] Option 2 selected")

            if self.device == "cuda":
                try:
                    if hasattr(self.model, "set_IO"):
                        self.model.set_IO(self.execution_args)
                        self.io_bound = True

                    if SOL_OPTIMIZE and hasattr(self.model, "optimize"):
                        print("   [SOL] (Mode 2) Running GPU Optimization (Level 2)...")
                        self.model.optimize(2)

                except Exception as e:
                    print(f"   ⚠️ SOL (Mode 2) set_IO/optimize warning: {e}")

            # CPU Mode 2: keep pure run(*args) with no binding
            # (io_bound stays False)

    def preprocess(self, input_path):
        if self.model_type == "video_classification":
            ext = os.path.splitext(input_path)[1].lower()
            if ext in [".mp4", ".avi", ".mov"]:
                tensor = load_video_with_cv2(input_path, num_frames=16, transform=self.transform)
            else:
                img = cv2.imread(str(input_path))
                if img is None:
                    raise FileNotFoundError(f"Failed: {input_path}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                dummy_tensor = torch.from_numpy(img).permute(2, 0, 1)
                frame_tensor = self.transform(dummy_tensor)
                tensor = torch.stack([frame_tensor] * 16, dim=1)
            return tensor.unsqueeze(0).numpy()
        else:
            img = cv2.imread(str(input_path))
            if img is None:
                raise FileNotFoundError(f"Failed: {input_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = self.transform(img)
            return tensor.unsqueeze(0).numpy()

    def infer(self, input_numpy):
        # 1. Prepare Input
        input_numpy = np.ascontiguousarray(input_numpy, dtype=np.float32)
        np.copyto(self.input_buffer, input_numpy)

        # 2. Execute
        if self.io_bound:
            # IO bound => run() uses already-bound buffers; get_output() syncs & copies
            self.model.run()
            if hasattr(self.model, "get_output"):
                self.model.get_output()
            elif hasattr(self.model, "sync"):
                self.model.sync()
        else:
            # No IO binding => pass buffers each call
            self.model.run(*self.execution_args)

            # Force synchronization for vAccel remote timing results
            if hasattr(self.model, "sync"):
                self.model.sync()

        # 3. Return outputs
        if self.model_type == "segmentation":
            return (self.output_buffer, getattr(self, "aux_buffer", None))
        return self.output_buffer

    def postprocess(self, output_numpy):
        if self.model_type in ["classification", "video_classification"]:
            output_tensor = torch.from_numpy(output_numpy)
            probs = torch.nn.functional.softmax(output_tensor, dim=1)
            top_prob, top_class = torch.max(probs, dim=1)
            return top_class.cpu(), top_prob.cpu()

        elif self.model_type == "segmentation":
            # output_numpy may be (out, aux) or a single ndarray
            out_np, aux_np = output_numpy if isinstance(output_numpy, tuple) else (output_numpy, None)

            chosen = out_np
            if aux_np is not None:
                if SOL_SEG_HEAD == "aux":
                    chosen = aux_np
                elif SOL_SEG_HEAD == "out":
                    chosen = out_np
                else:
                    # auto: pick less-degenerate logits (higher std + magnitude)
                    out_score = float(np.std(out_np)) + float(np.max(np.abs(out_np)))
                    aux_score = float(np.std(aux_np)) + float(np.max(np.abs(aux_np)))
                    chosen = aux_np if aux_score > out_score else out_np

            output_tensor = torch.from_numpy(chosen)
            return torch.argmax(output_tensor.squeeze(0), dim=0).byte().cpu()

        return torch.from_numpy(output_numpy)


class VaccelSolAdapter(SolAdapter):
    def __init__(self, device, model_name, model_type="classification", use_remote=False):
        self.use_remote = use_remote
        super().__init__(device, model_name, model_type)

    def load_model(self, model_dir):
        lib_path = self._get_model_lib_path(model_dir)
        self._load_model_from_path(os.path.join(lib_path, "vaccel"), self.use_remote)


# =========================================================
# 3. ADAPTER FACTORY
# =========================================================
def get_model_adapter(model_name, backend, device):
    """
    Decides adapter based on model_name string (e.g. ends with '_sol').
    'backend' argument corresponds to execution environment (stock/vaccel)
    """

    # 1. Check if SOL Model (Based on Naming Convention)
    if model_name.endswith("_sol"):
        # Strip suffix to get core name for config (e.g. resnet50_sol -> resnet50)
        # We pass core_name to SolAdapter so it imports 'sol_resnet50', not 'sol_resnet50_sol'
        core_name = model_name.replace("_sol", "")

        VIDEO_CORES = {
            "mc3_18",
            "r3d_18",
            "r2plus1d_18",
            "swin3d_t",
            "swin3d_s",
            "swin3d_b",
        }
        CLS_CORES = {
            "resnet50",
            "mobilenet_v3_large",
            "swin_t",
            "swin_s",
            "swin_v2_b",
        }

        if core_name in VIDEO_CORES:
            m_type = "video_classification"
        elif core_name in CLS_CORES:
            m_type = "classification"
        else:
            m_type = "segmentation"

        if "vaccel" in backend:
            return VaccelSolAdapter(
                device, core_name, model_type=m_type, use_remote="remote" in backend
            )
        else:
            return SolAdapter(device, core_name, model_type=m_type)

    # 2. Else: PyTorch Baseline
    else:
        if "vaccel" in backend:
            raise ValueError("vAccel backends are not supported for Torch models")

        BASELINE_REGISTRY = {
            # Segmentation
            "deeplabv3_resnet50": (models.segmentation.deeplabv3_resnet50, "segmentation", None),
            "deeplabv3_resnet101": (models.segmentation.deeplabv3_resnet101, "segmentation", None),
            "deeplabv3_mobilenet_v3_large": (models.segmentation.deeplabv3_mobilenet_v3_large, "segmentation", None),
            "fcn_resnet50": (models.segmentation.fcn_resnet50, "segmentation", None),
            "fcn_resnet101": (models.segmentation.fcn_resnet101, "segmentation", None),
            "lraspp_mobilenet_v3_large": (models.segmentation.lraspp_mobilenet_v3_large, "segmentation", None),

            # Classification
            "resnet50": (models.resnet50, "classification", models.ResNet50_Weights.DEFAULT),
            "mobilenet_v3_large": (models.mobilenet_v3_large, "classification", models.MobileNet_V3_Large_Weights.DEFAULT),
            "swin_t": (models.swin_t, "classification", models.Swin_T_Weights.DEFAULT),
            "swin_s": (models.swin_s, "classification", models.Swin_S_Weights.DEFAULT),
            "swin_v2_b": (models.swin_v2_b, "classification", models.Swin_V2_B_Weights.DEFAULT),

            # Video
            "mc3_18": (models.video.mc3_18, "video_classification", models.video.MC3_18_Weights.DEFAULT),
            "r3d_18": (models.video.r3d_18, "video_classification", models.video.R3D_18_Weights.DEFAULT),
            "r2plus1d_18": (models.video.r2plus1d_18, "video_classification", models.video.R2Plus1D_18_Weights.DEFAULT),
            "swin3d_t": (models.video.swin3d_t, "video_classification", models.video.Swin3D_T_Weights.DEFAULT),
            "swin3d_s": (models.video.swin3d_s, "video_classification", models.video.Swin3D_S_Weights.DEFAULT),
            "swin3d_b": (models.video.swin3d_b, "video_classification", models.video.Swin3D_B_Weights.DEFAULT),
        }

        if model_name not in BASELINE_REGISTRY:
            raise ValueError(f"Model {model_name} not found in Baseline Registry.")

        builder, m_type, w_enum = BASELINE_REGISTRY[model_name]
        return PyTorchBaselineAdapter(
            device,
            builder,
            f"{model_name}_state_dict.pt",
            model_type=m_type,
            weights_enum=w_enum,
        )
