#include "sol_resnet50.h"
#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vaccel.h>

#define sol_error(func, fmt, ...) \
	vaccel_error("[sol-wrapper][%s] " fmt, func, ##__VA_ARGS__)

extern "C" {

static inline int check_nr_args(size_t nr_read_exp, size_t nr_write_exp,
				size_t nr_read, size_t nr_write,
				const char *func)
{
	if (nr_read_exp != nr_read) {
		sol_error(
			func,
			"Invalid number of read arguments; expected %d got %zu",
			nr_read_exp, nr_read);
		return VACCEL_EINVAL;
	}

	if (nr_write_exp != nr_write) {
		sol_error(
			func,
			"Invalid number of write arguments; expected %d got %zu",
			nr_write_exp, nr_write);
		return VACCEL_EINVAL;
	}

	return VACCEL_OK;
}

static struct vaccel_prof_region init_stats =
	VACCEL_PROF_REGION_INIT("sol_init");

int sol_resnet50_init_unpack(struct vaccel_arg *read, size_t nr_read,
			      struct vaccel_arg *write, size_t nr_write)
{
	int ret = check_nr_args(0, 0, nr_read, nr_write, "init");
	return ret;

	vaccel_prof_region_start(&init_stats);
	sol_resnet50_init();
	vaccel_prof_region_stop(&init_stats);

	vaccel_prof_region_print(&init_stats);
	return 0;
}

static struct vaccel_prof_region predict_unpack_stats =
	VACCEL_PROF_REGION_INIT("sol_predict_unpack");
static struct vaccel_prof_region predict_stats =
	VACCEL_PROF_REGION_INIT("sol_predict");

int sol_predict_unpack(struct vaccel_arg *read, size_t nr_read,
		       struct vaccel_arg *write, size_t nr_write)
{
	const char *func = "predict";
	int ret = check_nr_args(2, 1, nr_read, nr_write, func);
	if (ret)
		return ret;

	vaccel_prof_region_start(&predict_unpack_stats);

	struct vaccel_arg_array read_args;
	ret = vaccel_arg_array_wrap(&read_args, read, nr_read);
	if (ret) {
		sol_error(func, "Failed to parse read args");
		return ret;
	}

	struct vaccel_arg_array write_args;
	ret = vaccel_arg_array_wrap(&write_args, write, nr_write);
	if (ret) {
		vaccel_prof_region_stop(&predict_unpack_stats);
		sol_error(func, "Failed to parse write args");
		return ret;
	}

	sol_f32 *in__input_0;
	size_t size;
	ret = vaccel_arg_array_get_buffer(&read_args, (void **)&in__input_0,
					  &size);
	if (ret) {
		vaccel_prof_region_stop(&predict_unpack_stats);
		sol_error(func, "Failed to unpack input");
		return ret;
	}

	sol_s64 *vdims;
	ret = vaccel_arg_array_get_buffer(&read_args, (void **)&vdims, &size);
	if (ret) {
		vaccel_prof_region_stop(&predict_unpack_stats);
		sol_error(func, "Failed to unpack vdims");
		return ret;
	}

	sol_f32 *out__0;
	ret = vaccel_arg_array_get_buffer(&write_args, (void **)&out__0, &size);
	if (ret) {
		vaccel_prof_region_stop(&predict_unpack_stats);
		sol_error(func, "Failed to unpack output");
		return ret;
	}

	vaccel_prof_region_stop(&predict_unpack_stats);

	vaccel_prof_region_start(&predict_stats);
	sol_predict(in__input_0, out__0, vdims);
	vaccel_prof_region_stop(&predict_stats);

	vaccel_prof_region_print(&predict_unpack_stats);
	vaccel_prof_region_print(&predict_stats);
	return 0;
}

static struct vaccel_prof_region set_io_unpack_stats =
	VACCEL_PROF_REGION_INIT("sol_set_IO_unpack");
static struct vaccel_prof_region set_io_stats =
	VACCEL_PROF_REGION_INIT("sol_set_IO");

int sol_resnet50_set_IO_unpack(struct vaccel_arg *read, size_t nr_read,
				struct vaccel_arg *write, size_t nr_write)
{
	const char *func = "set_IO";
	int ret = check_nr_args(4, 0, nr_read, nr_write, func);
	if (ret)
		return ret;

	vaccel_prof_region_start(&set_io_unpack_stats);

	struct vaccel_arg_array read_args;
	ret = vaccel_arg_array_wrap(&read_args, read, nr_read);
	if (ret) {
		vaccel_prof_region_stop(&set_io_unpack_stats);
		sol_error(func, "Failed to parse read args");
		return ret;
	}

	vaccel_id_t sess_id;
	ret = vaccel_arg_array_get_int64(&read_args, &sess_id);
	if (ret) {
		vaccel_prof_region_stop(&set_io_unpack_stats);
		sol_error(func, "Failed to unpack session ID");
		return ret;
	}

	vaccel_id_t in_res_id;
	ret = vaccel_arg_array_get_int64(&read_args, &in_res_id);
	if (ret) {
		vaccel_prof_region_stop(&set_io_unpack_stats);
		sol_error(func, "Failed to unpack input resource ID");
		return ret;
	}

	vaccel_id_t out_res_id;
	ret = vaccel_arg_array_get_int64(&read_args, &out_res_id);
	if (ret) {
		vaccel_prof_region_stop(&set_io_unpack_stats);
		sol_error(func, "Failed to unpack output resource ID");
		return ret;
	}

	vaccel_id_t vdims_res_id;
	ret = vaccel_arg_array_get_int64(&read_args, &vdims_res_id);
	if (ret) {
		vaccel_prof_region_stop(&set_io_unpack_stats);
		sol_error(func, "Failed to unpack vdims resource ID");
		return ret;
	}

	struct vaccel_session *sess;
	ret = vaccel_session_get_by_id(&sess, sess_id);
	if (ret) {
		vaccel_prof_region_stop(&set_io_unpack_stats);
		sol_error(func, "Unknown session %" PRId64, sess_id);
		return ret;
	}

	struct vaccel_resource *in_res;
	ret = vaccel_session_resource_by_id(sess, &in_res, in_res_id);
	if (ret) {
		vaccel_prof_region_stop(&set_io_unpack_stats);
		sol_error(func, "Unknown resource %" PRId64, in_res_id);
		return ret;
	}

	struct vaccel_resource *out_res;
	ret = vaccel_session_resource_by_id(sess, &out_res, out_res_id);
	if (ret) {
		vaccel_prof_region_stop(&set_io_unpack_stats);
		sol_error(func, "Unknown resource %" PRId64, out_res_id);
		return ret;
	}

	struct vaccel_resource *vdims_res;
	ret = vaccel_session_resource_by_id(sess, &vdims_res, vdims_res_id);
	if (ret) {
		vaccel_prof_region_stop(&set_io_unpack_stats);
		sol_error(func, "Unknown resource %" PRId64, vdims_res_id);
		return ret;
	}

	sol_f32 *in__input_0 = (sol_f32 *)in_res->blobs[0]->data;
	sol_f32 *out__0 = (sol_f32 *)out_res->blobs[0]->data;
	sol_s64 *vdims = (sol_s64 *)vdims_res->blobs[0]->data;

	vaccel_prof_region_stop(&set_io_unpack_stats);

	vaccel_prof_region_start(&set_io_stats);
	sol_resnet50_set_IO(in__input_0, out__0, vdims);
	vaccel_prof_region_stop(&set_io_stats);

	vaccel_prof_region_print(&set_io_unpack_stats);
	vaccel_prof_region_print(&set_io_stats);
	return 0;
}

static struct vaccel_prof_region run_stats = VACCEL_PROF_REGION_INIT("sol_run");

int sol_resnet50_run_unpack(struct vaccel_arg *read, size_t nr_read,
			     struct vaccel_arg *write, size_t nr_write)
{
	int ret = check_nr_args(0, 0, nr_read, nr_write, "run");
	if (ret)
		return ret;

	vaccel_prof_region_start(&run_stats);
	sol_resnet50_run();
	vaccel_prof_region_stop(&run_stats);

	vaccel_prof_region_print(&run_stats);
	return 0;
}

static struct vaccel_prof_region optimize_unpack_stats =
	VACCEL_PROF_REGION_INIT("sol_optimize_unpack");
static struct vaccel_prof_region optimize_stats =
	VACCEL_PROF_REGION_INIT("sol_optimize");

int sol_resnet50_optimize_unpack(struct vaccel_arg *read, size_t nr_read,
				  struct vaccel_arg *write, size_t nr_write)
{
	const char *func = "optimize";
	int ret = check_nr_args(1, 0, nr_read, nr_write, func);
	if (ret)
		return ret;

	vaccel_prof_region_start(&optimize_unpack_stats);

	struct vaccel_arg_array read_args;
	ret = vaccel_arg_array_wrap(&read_args, read, nr_read);
	if (ret) {
		vaccel_prof_region_stop(&optimize_unpack_stats);
		sol_error(func, "Failed to parse read args");
		return ret;
	}

	int32_t level;
	ret = vaccel_arg_array_get_int32(&read_args, &level);
	if (ret) {
		vaccel_prof_region_stop(&optimize_unpack_stats);
		sol_error(func, "Failed to unpack level");
		return ret;
	}

	vaccel_prof_region_stop(&optimize_unpack_stats);

	vaccel_prof_region_start(&optimize_stats);
	sol_resnet50_optimize(level);
	vaccel_prof_region_stop(&optimize_stats);

	vaccel_prof_region_print(&optimize_unpack_stats);
	vaccel_prof_region_print(&optimize_stats);
	return 0;
}

static struct vaccel_prof_region sync_stats =
	VACCEL_PROF_REGION_INIT("sol_sync");

int sol_resnet50_sync_unpack(struct vaccel_arg *read, size_t nr_read,
			      struct vaccel_arg *write, size_t nr_write)
{
	int ret = check_nr_args(0, 0, nr_read, nr_write, "sync");
	if (ret)
		return ret;

	vaccel_prof_region_start(&sync_stats);
	sol_resnet50_sync();
	vaccel_prof_region_stop(&sync_stats);

	vaccel_prof_region_print(&sync_stats);
	return 0;
}

static struct vaccel_prof_region get_output_stats =
	VACCEL_PROF_REGION_INIT("sol_get_output");

int sol_resnet50_get_output_unpack(struct vaccel_arg *read, size_t nr_read,
				    struct vaccel_arg *write, size_t nr_write)
{
	int ret = check_nr_args(0, 0, nr_read, nr_write, "get_output");
	if (ret)
		return ret;

	vaccel_prof_region_start(&get_output_stats);
	sol_resnet50_get_output();
	vaccel_prof_region_stop(&get_output_stats);

	vaccel_prof_region_print(&get_output_stats);
	return 0;
}

static struct vaccel_prof_region free_stats =
	VACCEL_PROF_REGION_INIT("sol_free");

int sol_resnet50_free_unpack(struct vaccel_arg *read, size_t nr_read,
			      struct vaccel_arg *write, size_t nr_write)
{
	int ret = check_nr_args(0, 0, nr_read, nr_write, "free");
	if (ret)
		return ret;

	vaccel_prof_region_start(&free_stats);
	sol_resnet50_free();
	vaccel_prof_region_stop(&free_stats);

	vaccel_prof_region_print(&free_stats);
	return 0;
}

static struct vaccel_prof_region free_host_stats =
	VACCEL_PROF_REGION_INIT("sol_free_host");

int sol_resnet50_free_host_unpack(struct vaccel_arg *read, size_t nr_read,
				   struct vaccel_arg *write, size_t nr_write)
{
	int ret = check_nr_args(0, 0, nr_read, nr_write, "free_host");
	if (ret)
		return ret;

	vaccel_prof_region_start(&free_host_stats);
	sol_resnet50_free_host();
	vaccel_prof_region_stop(&free_host_stats);

	vaccel_prof_region_print(&free_host_stats);
	return 0;
}

static struct vaccel_prof_region free_device_stats =
	VACCEL_PROF_REGION_INIT("sol_free_device");

int sol_resnet50_free_device_unpack(struct vaccel_arg *read, size_t nr_read,
				     struct vaccel_arg *write, size_t nr_write)
{
	int ret = check_nr_args(0, 0, nr_read, nr_write, "free_device");
	if (ret)
		return ret;

	vaccel_prof_region_start(&free_device_stats);
	sol_resnet50_free_device();
	vaccel_prof_region_stop(&free_device_stats);

	vaccel_prof_region_print(&free_device_stats);
	return 0;
}

static struct vaccel_prof_region set_seed_unpack_stats =
	VACCEL_PROF_REGION_INIT("sol_set_seed_unpack");
static struct vaccel_prof_region set_seed_stats =
	VACCEL_PROF_REGION_INIT("sol_set_seed");

int sol_resnet50_set_seed_unpack(struct vaccel_arg *read, size_t nr_read,
				  struct vaccel_arg *write, size_t nr_write)
{
	const char *func = "set_seed";
	int ret = check_nr_args(1, 0, nr_read, nr_write, func);
	if (ret)
		return ret;

	vaccel_prof_region_start(&set_seed_unpack_stats);

	struct vaccel_arg_array read_args;
	ret = vaccel_arg_array_wrap(&read_args, read, nr_read);
	if (ret) {
		vaccel_prof_region_stop(&set_seed_unpack_stats);
		sol_error(func, "Failed to parse read args");
		return ret;
	}

	int64_t seed;
	ret = vaccel_arg_array_get_int64(&read_args, &seed);
	if (ret) {
		vaccel_prof_region_stop(&set_seed_unpack_stats);
		sol_error(func, "Failed to unpack seed");
		return ret;
	}

	vaccel_prof_region_stop(&set_seed_unpack_stats);

	vaccel_prof_region_start(&set_seed_stats);
	sol_resnet50_set_seed(seed);
	vaccel_prof_region_stop(&set_seed_stats);

	vaccel_prof_region_print(&set_seed_stats);
	return 0;
}
}
