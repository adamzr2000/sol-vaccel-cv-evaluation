.PHONY: all
all: libsol_swin3d_s_wrapper.so

libsol_swin3d_s_wrapper.so: sol_swin3d_s_wrapper.c
	LD_RUN_PATH=.:..:"${LD_RUN_PATH}" \
		$(CXX) -std=c++2b -o $@ $< -fPIC -shared -Wl,--as-needed \
			$(shell pkg-config --cflags --libs vaccel) \
			-L.. -I.. -lsol_swin3d_s

.PHONY: clean
clean:
	rm -f libsol_swin3d_s_wrapper.so
