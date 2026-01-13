# Generated with SOL v0.8.0rc4
import numpy as np
import ctypes
from numpy.ctypeslib import ndpointer, as_ctypes_type

class sol_mc3_18:
	def open(self):
		self.lib = ctypes.CDLL(self.path + "/" + "libsol_mc3_18.so")

		self.call = self.lib.__getattr__("sol_predict") 
		self.call.restype = None

		self.init = self.lib.__getattr__("sol_mc3_18_init")
		self.init.restype = None
		self.init.argtypes = None

		self.free = self.lib.__getattr__("sol_mc3_18_free")
		self.free.restype = None
		self.free.argtypes = None

		self.seed_ = self.lib.__getattr__("sol_mc3_18_set_seed")
		self.seed_.argtypes = [ctypes.c_uint64]
		self.seed_.restype = None

		self.set_IO_ = self.lib.__getattr__("sol_mc3_18_set_IO")
		self.set_IO_.restype = None

		self.call_no_args = self.lib.__getattr__("sol_mc3_18_run")
		self.call_no_args.argtypes = None
		self.call_no_args.restype = None

		self.get_output = self.lib.__getattr__("sol_mc3_18_get_output")
		self.get_output.argtypes = None
		self.get_output.restype = None

		self.sync = self.lib.__getattr__("sol_mc3_18_sync")
		self.sync.argtypes = None
		self.sync.restype = None

		self.opt_ = self.lib.__getattr__("sol_mc3_18_optimize")
		self.opt_.argtypes = [ctypes.c_int]
		self.opt_.restype = None

	def __init__(self, path="."):
		self.path = path
		self.open()

	def __call__(self, *args, **kwargs):
		# TODO keyword args: either ignore or use properly
		return self.run(*args)


	def set_seed(self, s):
		arg = ctypes.c_uint64(s)
		self.seed_(arg)

	def optimize(self, level):
		arg = ctypes.c_int(level)
		self.opt_(arg)

	def set_IO(self, args):
		self.set_IO_.argtypes = [ndpointer(as_ctypes_type(x.dtype), flags="C_CONTIGUOUS") for x in args]
		self.set_IO_(*args)

	def run(self, in__input_0 = None, out__0 = None, vdims = None):
		call_args = [in__input_0, ]
		if any(elem is None for elem in call_args):
			self.call_no_args()
			return
		if vdims is None:
			vdims_0 = call_args[0].shape[0]
			vdims = np.array([vdims_0, ], dtype=np.int64)
		else:
			vdims_0 = vdims[0]
		if out__0 is None:
			out__0 = np.zeros((vdims_0, 400), dtype=np.float32)
		call_args.append(out__0)
		call_args.append(vdims)

		self.call.argtypes = [ndpointer(as_ctypes_type(x.dtype), flags="C_CONTIGUOUS") for x in call_args]
		self.call(*call_args)

		return out__0

	def close(self):
		dlclose_func = ctypes.CDLL(self.path + "/" + "libsol_mc3_18.so").dlclose
		dlclose_func.argtypes = (ctypes.c_void_p,)
		dlclose_func.restype = ctypes.c_int
		return dlclose_func(self.lib._handle)


def run_example():
	####### Option 1: Call the deployed lib directly #######
	vdims_0 = 1
	in__input_0 = np.random.rand(vdims_0, 3, 16, 112, 112).astype(np.float32)
	lib = sol_mc3_18()
	out__0 = lib(in__input_0)
	print(f"Max_V: {np.max(out__0, axis=1)}\nMax_I: {np.argmax(out__0, axis=1)}")

	vdims = np.array([vdims_0, ], dtype=np.int64)

	in__input_0 = np.random.rand(vdims_0, 3, 16, 112, 112).astype(np.float32)
	out__0 = np.zeros((vdims_0, 400), dtype=np.float32)
	dp_args = [in__input_0, out__0, vdims] # Inputs, Outputs, VDims must be in this exact order!

	# Call Function and evaluate output---------------------------------------------
	lib = sol_mc3_18()
	lib.init() # optional, loads parameters on host
	lib.set_seed(271828) # optional

	####### Option 2: Run the underlying lib with predefined buffers #######
	lib.run(*dp_args)
	print(f"Max_V: {np.max(out__0, axis=1)}\nMax_I: {np.argmax(out__0, axis=1)}")

	####### Option 3: Run after setting in- and outputs #######
	lib.set_IO(dp_args)
	lib.optimize(level=2)
	lib.run() # (async)
	lib.get_output() # syncs
	print(f"Max_V: {np.max(out__0, axis=1)}\nMax_I: {np.argmax(out__0, axis=1)}")
	# Free used data and close lib---------------------------------------------
	lib.free()
	lib.close()


if __name__ == "__main__":
	run_example()
