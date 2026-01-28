.PHONY: all
all: libsol_swin3d_t_wrapper.so

libsol_swin3d_t_wrapper.so: sol_swin3d_t_wrapper.c
	LD_RUN_PATH=.:..:"${LD_RUN_PATH}" \
		$(CXX) -std=c++2b -o $@ $< -fPIC -shared -Wl,--as-needed \
			$(shell pkg-config --cflags --libs vaccel) \
			-L.. -I.. -lsol_swin3d_t

.PHONY: clean
clean:
	rm -f libsol_swin3d_t_wrapper.so
