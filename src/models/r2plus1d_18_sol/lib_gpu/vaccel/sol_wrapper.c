/*
 * Generic vAccel exec wrapper for a SOL 0.9 deployment (one per model and device, built by install.sh):
 *   g++ -shared -fPIC -DSOL_NAME='"<model>"' [-DSOL_GPU] wrapper.c $(pkg-config --cflags --libs vaccel) -ldl
 *
 * vAccel resources (src/sol09.py):
 *   LIB  = [this wrapper]                                    (dlopen'ed by the vAccel exec plugin)
 *   DATA = [<model>.so, <model>.sol_weights, SOL runtime libs] (one folder: the library folder locally, the
 *                                                              resource folder on a remote agent)
 * The model library is opened here with RTLD_LOCAL and its functions are looked up on that handle: every SOL 0.9
 * library exports the same C API names (sol_predict, ...), so a globally loaded model would capture the calls of
 * the next model served by the same process. It finds its runtime libs (RPATH $ORIGIN) and its weights
 * (<dir of the library>/<model>.sol_weights) in the same folder.
 *
 * SOL 0.9 C API: sol_predict_create -> handle; sol_predict(handle, input [, offload, stream]) -> output, freed with
 * sol_predict_free. CPU: host input -> host output. GPU: the wrapper does the copies itself (persistent device input
 * buffer, H2D, sol_predict on device pointers (offload=0), D2H straight into the vAccel output buffer) instead of
 * SOL's offload=1 path, which allocates device memory and a host output buffer on every call and needs one more
 * host copy: same work as the PyTorch plugin (H2D, run, D2H), 0.05-0.3 ms less per call, identical outputs.
 * GPU + CUDA graph (default; SOL_CUDA_GRAPH=0 disables): SOL runs on the wrapper's own stream; after GRAPH_WARMUP
 * plain calls, one sol_predict is captured into a CUDA graph (a single SOL inference launches up to ~4000 kernels,
 * e.g. Swin-T), and each later call is H2D, graph launch, D2H on that stream (identical outputs, 5-26% faster). It
 * needs GPU libraries deployed with Conv_NoTextureMemory (deploy.py), the option SOL itself uses for CUDA graphs.
 * If the capture fails, the wrapper stays on the plain path.
 *
 * Entry points (exec_with_resource):
 *   sol_init_unpack   (read: [DATA resource id (int64)])           -> load the model, create the SOL handle
 *   sol_predict_unpack(read: [input f32 buffer], write: [output f32]) -> one inference
 */
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vaccel.h>
#ifndef SOL_GPU
#include <condition_variable>
#include <mutex>
#include <thread>
#endif

extern "C" {

#ifdef SOL_GPU
typedef void *(*create_fn)(int, void *);
typedef void *(*predict_fn)(void *, void *, int, void *);
typedef void (*free_fn)(void *, int, void *);
static void *g_lib = NULL, *g_h = NULL;
static predict_fn g_predict = NULL;
static free_fn g_free = NULL;
/* CUDA runtime, loaded at init (no link-time dependency) */
typedef int (*cuda_malloc_fn)(void **, size_t);
typedef int (*cuda_memcpy_async_fn)(void *, const void *, size_t, int, void *);
typedef int (*cuda_stream_create_fn)(void **);
typedef int (*cuda_stream_fn)(void *);
typedef int (*cuda_begin_capture_fn)(void *, int);
typedef int (*cuda_end_capture_fn)(void *, void **);
typedef int (*cuda_graph_instantiate_fn)(void **, void *, unsigned long long);
typedef int (*cuda_graph_launch_fn)(void *, void *);
static cuda_malloc_fn g_cuda_malloc = NULL;
static cuda_memcpy_async_fn g_cuda_memcpy_async = NULL;
static cuda_stream_fn g_cuda_sync = NULL;
static cuda_begin_capture_fn g_cuda_begin = NULL;
static cuda_end_capture_fn g_cuda_end = NULL;
static cuda_graph_instantiate_fn g_cuda_instantiate = NULL;
static cuda_graph_launch_fn g_cuda_launch = NULL;
static void *g_stream = NULL;  /* the wrapper's CUDA stream (SOL handle created on it) */
static void *g_d_in = NULL;    /* device input buffer, allocated once */
static size_t g_d_in_size = 0;
static int g_graph_state = 0;  /* -1 off, 0..GRAPH_WARMUP-1 warming up, GRAPH_WARMUP ready to capture, 99 captured */
static void *g_graph_exec = NULL, *g_graph_out = NULL;
enum { H2D = 1, D2H = 2, GRAPH_WARMUP = 3, CAPTURE_THREAD_LOCAL = 1, AUTO_FREE_ON_LAUNCH = 1 };

/* one plain inference on the wrapper's stream: H2D, sol_predict (device pointers), D2H, sync; -> 0 on success */
static int gpu_plain(const void *in, size_t in_size, void *out, size_t out_size)
{
	if (g_cuda_memcpy_async(g_d_in, in, in_size, H2D, g_stream) != 0)
		return -1;
	void *p = g_predict(g_h, g_d_in, 0, g_stream);
	if (p == NULL)
		return -1;
	int ret = g_cuda_memcpy_async(out, p, out_size, D2H, g_stream) | g_cuda_sync(g_stream);
	g_free(p, 0, g_stream);
	return ret;
}

/* capture one sol_predict on the wrapper's stream into an executable graph; -> 0 on success */
static int gpu_capture(void)
{
	void *graph = NULL;
	if (g_cuda_begin(g_stream, CAPTURE_THREAD_LOCAL) != 0)
		return -1;
	g_graph_out = g_predict(g_h, g_d_in, 0, g_stream);
	if (g_cuda_end(g_stream, &graph) != 0 || graph == NULL || g_graph_out == NULL)
		return -1;
	return g_cuda_instantiate(&g_graph_exec, graph, AUTO_FREE_ON_LAUNCH) != 0 ? -1 : 0;
}

static int gpu_graph(const void *in, size_t in_size, void *out, size_t out_size)
{
	return g_cuda_memcpy_async(g_d_in, in, in_size, H2D, g_stream) | g_cuda_launch(g_graph_exec, g_stream) |
	       g_cuda_memcpy_async(out, g_graph_out, out_size, D2H, g_stream) | g_cuda_sync(g_stream);
}
#else
typedef void *(*create_fn)(void);
typedef void *(*predict_fn)(void *, void *);
typedef void (*free_fn)(void *);
static void *g_lib = NULL, *g_h = NULL;
static predict_fn g_predict = NULL;
static free_fn g_free = NULL;
#endif


#ifndef SOL_GPU
/* CPU, with SOL_WORKER_THREAD=1 (set by run_agent_sol.sh): run every inference on one persistent worker thread. The
 * agent serves RPCs on several threads and OpenMP keeps one thread team per calling thread, so without this each
 * agent thread gets its own team of OMP_NUM_THREADS threads on the same cores (cold caches, oversubscription). The
 * worker blocks on a condition variable (no spinning) while idle. Off by default: a local client calls from one
 * thread anyway (and the handoff deadlocks inside the Python client process). */
static bool g_use_worker = false;
static std::mutex g_call_mx, g_mx;
static std::condition_variable g_cv;
static void *g_job_in = NULL, *g_job_out = NULL;
static bool g_job_ready = false, g_job_done = false;

static void worker_loop(void)
{
	std::unique_lock<std::mutex> lk(g_mx);
	for (;;) {
		g_cv.wait(lk, [] { return g_job_ready; });
		g_job_ready = false;
		void *in = g_job_in;
		lk.unlock();
		void *out = g_predict(g_h, in);
		lk.lock();
		g_job_out = out;
		g_job_done = true;
		g_cv.notify_all();
	}
}

static void *predict_on_worker(void *in)
{
	std::lock_guard<std::mutex> call(g_call_mx); /* one inference at a time */
	std::unique_lock<std::mutex> lk(g_mx);
	g_job_in = in;
	g_job_done = false;
	g_job_ready = true;
	g_cv.notify_all();
	g_cv.wait(lk, [] { return g_job_done; });
	return g_job_out;
}
#endif

/* folder of the DATA resource: the original files locally, the agent's copies remotely */
static int data_dir(vaccel_id_t id, char *dir, size_t len)
{
	struct vaccel_resource *res;
	if (vaccel_resource_get_by_id(&res, id) != VACCEL_OK)
		return VACCEL_ENOENT;
	const char *f = (res->nr_blobs && res->blobs[0]->path) ? res->blobs[0]->path
		      : (res->nr_paths ? res->paths[0] : NULL);
	if (f == NULL || strrchr(f, '/') == NULL)
		return VACCEL_EINVAL;
	snprintf(dir, len, "%.*s", (int)(strrchr(f, '/') - f), f);
	return VACCEL_OK;
}

static struct vaccel_prof_region init_stats = VACCEL_PROF_REGION_INIT("sol_init");

int sol_init_unpack(struct vaccel_arg *read, size_t nr_read, struct vaccel_arg *write, size_t nr_write)
{
	if (g_h != NULL)
		return VACCEL_OK;
	struct vaccel_arg_array ra;
	int64_t id;
	char dir[4096], path[4400];
	int ret;
	if (nr_read != 1 || (ret = vaccel_arg_array_wrap(&ra, read, nr_read)) ||
	    (ret = vaccel_arg_array_get_int64(&ra, &id)) || (ret = data_dir((vaccel_id_t)id, dir, sizeof(dir)))) {
		vaccel_error("[sol-wrapper] init expects the id of the model's DATA resource");
		return VACCEL_EINVAL;
	}

	vaccel_prof_region_start(&init_stats);
	snprintf(path, sizeof(path), "%s/%s.so", dir, SOL_NAME);
	g_lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
	create_fn create = g_lib ? (create_fn)dlsym(g_lib, "sol_predict_create") : NULL;
	g_predict = g_lib ? (predict_fn)dlsym(g_lib, "sol_predict") : NULL;
	g_free = g_lib ? (free_fn)dlsym(g_lib, "sol_predict_free") : NULL;
#ifdef SOL_GPU
	void *cudart = dlopen("libcudart.so.12", RTLD_NOW | RTLD_LOCAL);
	cuda_stream_create_fn stream_create = cudart ? (cuda_stream_create_fn)dlsym(cudart, "cudaStreamCreate") : NULL;
	g_cuda_malloc = cudart ? (cuda_malloc_fn)dlsym(cudart, "cudaMalloc") : NULL;
	g_cuda_memcpy_async = cudart ? (cuda_memcpy_async_fn)dlsym(cudart, "cudaMemcpyAsync") : NULL;
	g_cuda_sync = cudart ? (cuda_stream_fn)dlsym(cudart, "cudaStreamSynchronize") : NULL;
	g_cuda_begin = cudart ? (cuda_begin_capture_fn)dlsym(cudart, "cudaStreamBeginCapture") : NULL;
	g_cuda_end = cudart ? (cuda_end_capture_fn)dlsym(cudart, "cudaStreamEndCapture") : NULL;
	g_cuda_instantiate = cudart ? (cuda_graph_instantiate_fn)dlsym(cudart, "cudaGraphInstantiateWithFlags") : NULL;
	g_cuda_launch = cudart ? (cuda_graph_launch_fn)dlsym(cudart, "cudaGraphLaunch") : NULL;
	if (stream_create == NULL || g_cuda_malloc == NULL || g_cuda_memcpy_async == NULL || g_cuda_sync == NULL ||
	    g_cuda_begin == NULL || g_cuda_end == NULL || g_cuda_instantiate == NULL || g_cuda_launch == NULL ||
	    stream_create(&g_stream) != 0) {
		vaccel_prof_region_stop(&init_stats);
		vaccel_error("[sol-wrapper] cannot load libcudart.so.12: %s", dlerror());
		return VACCEL_ENOENT;
	}
#endif
	if (create == NULL || g_predict == NULL || g_free == NULL) {
		vaccel_prof_region_stop(&init_stats);
		vaccel_error("[sol-wrapper] cannot load %s: %s", path, dlerror());
		return VACCEL_ENOENT;
	}
#ifdef SOL_GPU
	g_h = create(0, g_stream);
	const char *cg = getenv("SOL_CUDA_GRAPH");
	g_graph_state = (cg != NULL && strcmp(cg, "0") == 0) ? -1 : 0;
#else
	g_h = create();
	const char *w = getenv("SOL_WORKER_THREAD");
	g_use_worker = g_h != NULL && w != NULL && strcmp(w, "1") == 0;
	if (g_use_worker)
		std::thread(worker_loop).detach();
#endif
	vaccel_prof_region_stop(&init_stats);
	vaccel_prof_region_print(&init_stats);
	return g_h == NULL ? VACCEL_EINVAL : VACCEL_OK;
}

static struct vaccel_prof_region predict_unpack_stats = VACCEL_PROF_REGION_INIT("sol_predict_unpack");
static struct vaccel_prof_region predict_stats = VACCEL_PROF_REGION_INIT("sol_predict");

int sol_predict_unpack(struct vaccel_arg *read, size_t nr_read, struct vaccel_arg *write, size_t nr_write)
{
	if (g_h == NULL) {
		vaccel_error("[sol-wrapper] sol_init_unpack has not been called");
		return VACCEL_EINVAL;
	}
	if (nr_read != 1 || nr_write != 1) {
		vaccel_error("[sol-wrapper] predict expects 1 read and 1 write arg, got %zu/%zu", nr_read, nr_write);
		return VACCEL_EINVAL;
	}

	vaccel_prof_region_start(&predict_unpack_stats);
	struct vaccel_arg_array ra, wa;
	void *in, *out;
	size_t in_size, out_size;
	int ret;
	if ((ret = vaccel_arg_array_wrap(&ra, read, nr_read)) || (ret = vaccel_arg_array_wrap(&wa, write, nr_write)) ||
	    (ret = vaccel_arg_array_get_buffer(&ra, &in, &in_size)) ||
	    (ret = vaccel_arg_array_get_buffer(&wa, &out, &out_size))) {
		vaccel_prof_region_stop(&predict_unpack_stats);
		vaccel_error("[sol-wrapper] cannot unpack predict args");
		return ret;
	}
	vaccel_prof_region_stop(&predict_unpack_stats);

	/* region "sol_predict" = host input buffer -> host output buffer (GPU incl. both copies; the D2H copy on the
	 * default stream waits for SOL's kernels), as in the rc5 wrappers; read by agent_server.py /
	 * check_agent_profile.py */
	vaccel_prof_region_start(&predict_stats);
#ifdef SOL_GPU
	if (g_d_in_size != in_size) {
		if (g_cuda_malloc(&g_d_in, in_size) != 0) {
			vaccel_prof_region_stop(&predict_stats);
			vaccel_error("[sol-wrapper] cudaMalloc failed");
			return VACCEL_ENOMEM;
		}
		g_d_in_size = in_size;
	}
	if (g_graph_state == GRAPH_WARMUP) {
		if (gpu_capture() != 0) {
			g_graph_state = -1;
			vaccel_warn("[sol-wrapper] CUDA graph capture failed, plain path");
		} else {
			g_graph_state = 99;
		}
	}
	ret = g_graph_state == 99 ? gpu_graph(in, in_size, out, out_size) : gpu_plain(in, in_size, out, out_size);
	if (g_graph_state >= 0 && g_graph_state < GRAPH_WARMUP)
		g_graph_state++;
	void *p = ret == 0 ? out : NULL;
#else
	void *p = g_use_worker ? predict_on_worker(in) : g_predict(g_h, in);
	if (p != NULL) {
		memcpy(out, p, out_size);
		g_free(p);
	}
#endif
	vaccel_prof_region_stop(&predict_stats);
	vaccel_prof_region_print(&predict_stats);
	return p == NULL ? VACCEL_EINVAL : VACCEL_OK;
}

}
