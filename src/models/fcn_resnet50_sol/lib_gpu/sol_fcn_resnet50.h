#pragma once
#ifdef __cplusplus
	#include<cinttypes>
#else
	#include<inttypes.h>
#endif

// Generated with SOL v0.8.0rc4
typedef int8_t sol_s8;
typedef int16_t sol_s16;
typedef int32_t sol_s32;
typedef int64_t sol_s64;
typedef uint8_t sol_u8;
typedef uint16_t sol_u16;
typedef uint32_t sol_u32;
typedef uint64_t sol_u64;
typedef float sol_f32;
typedef double sol_f64;

#ifdef __cplusplus
extern "C" {
#endif

	void sol_fcn_resnet50_init(void);
	
	void sol_predict(const sol_f32* in__x, sol_f32* out__0_out, sol_f32* out__0_aux, sol_s64 rvdims[1]);
	void sol_fcn_resnet50_set_IO(const sol_f32* in__x, sol_f32* out__0_out, sol_f32* out__0_aux, sol_s64 rvdims[1]);
	void sol_fcn_resnet50_run(void);
	void sol_fcn_resnet50_optimize(int);

	void sol_fcn_resnet50_sync(void);

	void sol_fcn_resnet50_get_output(void);

	void sol_fcn_resnet50_free(void);
	void sol_fcn_resnet50_free_host(void);
	void sol_fcn_resnet50_free_device(void);
	void sol_fcn_resnet50_free_IO(void);

	void sol_fcn_resnet50_set_seed(int64_t);

#ifdef __cplusplus
}
#endif
