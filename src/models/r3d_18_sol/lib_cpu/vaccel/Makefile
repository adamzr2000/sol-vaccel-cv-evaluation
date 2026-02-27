.PHONY: all
all: libsol_r3d_18_wrapper.so

libsol_r3d_18_wrapper.so: sol_r3d_18_wrapper.c
	LD_RUN_PATH=.:..:"${LD_RUN_PATH}" \
		$(CXX) -std=c++2b -o $@ $< -fPIC -shared -Wl,--as-needed \
			$(shell pkg-config --cflags --libs vaccel) \
			-L.. -I.. -lsol_r3d_18

.PHONY: clean
clean:
	rm -f libsol_r3d_18_wrapper.so
