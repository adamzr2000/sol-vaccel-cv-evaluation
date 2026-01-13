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

	void sol_mc3_18_init(void);
	
	void sol_predict(const sol_f32* in__input_0, sol_f32* out__0, sol_s64 rvdims[1]);
	void sol_mc3_18_set_IO(const sol_f32* in__input_0, sol_f32* out__0, sol_s64 rvdims[1]);
	void sol_mc3_18_run(void);
	void sol_mc3_18_optimize(int);

	void sol_mc3_18_sync(void);

	void sol_mc3_18_get_output(void);

	void sol_mc3_18_free(void);
	void sol_mc3_18_free_host(void);
	void sol_mc3_18_free_device(void);
	void sol_mc3_18_free_IO(void);

	void sol_mc3_18_set_seed(int64_t);

#ifdef __cplusplus
}
#endif
