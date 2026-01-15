// Generated with SOL v0.8.0rc5
#include "sol_r3d_18.h"
#include <stdlib.h>
#ifdef __cplusplus
	#include <iostream>
#else
	#include <stdio.h>
#endif

#ifdef __cplusplus
	using std::rand;
	using std::srand;
#endif

int main(int argc, char** argv) {
	
	const int64_t vdims_0 = 1; // this value can be changed
	int64_t vdims[1]; // do not change! (must be int64_t array of size 1)
	vdims[0] = vdims_0;
	
	
	// Define and Initialize Inputs---------------------------------------------
	sol_f32 *_input_0 = (sol_f32*) malloc(vdims_0 * 3ll * 16ll * 112ll * 112ll *  sizeof(sol_f32));
	
	// Define and Initialize Outputs---------------------------------------------
	sol_f32 *_0 = (sol_f32*) malloc(vdims_0 * 400ll *  sizeof(sol_f32));
	
	// Generate Random Input----------------------------------------------------
	srand(0);
	for(size_t n = 0; n < vdims_0 * 3ll * 16ll * 112ll * 112ll * 1; n++)
		_input_0[n] = rand()/(sol_f32)RAND_MAX;
	
	for(size_t n = 0; n < vdims_0 * 400ll * 1; n++)
		_0[n] = 0;
	
	// Call generated library---------------------------------------------
	sol_r3d_18_init(); // optional, reads parameters and moves them to device if necessary
	sol_r3d_18_set_seed(314159); // optional
	sol_predict(_input_0, _0, vdims);
	
	
	// Checking Results Example - Print Max, assuming first dimension is batch
	int max_index = -1;
	sol_f32 max_value = -9999999999;
	size_t data_per_batch = 0;
	
	// Output #0
	data_per_batch = (vdims_0 * 400ll * 1)/vdims_0;
	for(int i = 0; i < vdims_0; i++){
		max_index = -1;
		max_value = -9999999999;
		for(size_t n = i * data_per_batch; n < (i+1) * data_per_batch; n++){
			if(_0[n] > max_value){
				max_value  = _0[n];
				max_index = n - i * data_per_batch;
			}
		}
		#ifdef __cplusplus
			std::cout << "Max_V: " << max_value << std::endl;
			std::cout << "Max_I: " << max_index << std::endl;
		#else
			printf("Max_V: %f\n", max_value);
			printf("Max_I: %d\n", max_index);
		#endif
	}

	sol_r3d_18_free(); // frees allocated parameters by lib
	
	// free all memory allocated by this example
	free(_input_0);
	free(_0);
	
	return 0;
}
