.PHONY: all
all: libsol_mobilenet_v3_large_wrapper.so

libsol_mobilenet_v3_large_wrapper.so: sol_mobilenet_v3_large_wrapper.c
	LD_RUN_PATH=.:..:"${LD_RUN_PATH}" \
		$(CXX) -std=c++2b -o $@ $< -fPIC -shared -Wl,--as-needed \
			$(shell pkg-config --cflags --libs vaccel) \
			-L.. -I.. -lsol_mobilenet_v3_large

.PHONY: clean
clean:
	rm -f libsol_mobilenet_v3_large_wrapper.so
