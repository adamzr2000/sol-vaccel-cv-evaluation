// Generated with SOL v0.8.0rc5
#include "sol_swin_t.h"
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
	
	int64_t vdims[0]; // do not change! (must be int64_t array of size 0)
	
	
	// Define and Initialize Inputs---------------------------------------------
	sol_f32 *_input_0 = (sol_f32*) malloc(1ll * 3ll * 224ll * 224ll *  sizeof(sol_f32));
	
	// Define and Initialize Outputs---------------------------------------------
	sol_f32 *_0 = (sol_f32*) malloc(1ll * 1000ll *  sizeof(sol_f32));
	
	// Generate Random Input----------------------------------------------------
	srand(0);
	for(size_t n = 0; n < 1ll * 3ll * 224ll * 224ll * 1; n++)
		_input_0[n] = rand()/(sol_f32)RAND_MAX;
	
	for(size_t n = 0; n < 1ll * 1000ll * 1; n++)
		_0[n] = 0;
	
	// Call generated library---------------------------------------------
	sol_swin_t_init(); // optional, reads parameters and moves them to device if necessary
	sol_swin_t_set_seed(314159); // optional
	sol_predict(_input_0, _0, vdims);
	
	
	// Checking Results Example - Print Max, assuming first dimension is batch
	int max_index = -1;
	sol_f32 max_value = -9999999999;
	size_t data_per_batch = 0;
	
	// Output #0
	data_per_batch = (1ll * 1000ll * 1)/1ll;
	for(int i = 0; i < 1ll; i++){
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

	sol_swin_t_free(); // frees allocated parameters by lib
	
	// free all memory allocated by this example
	free(_input_0);
	free(_0);
	
	return 0;
}
