.PHONY: all
all: libsol_swin_v2_b_wrapper.so

libsol_swin_v2_b_wrapper.so: sol_swin_v2_b_wrapper.c
	LD_RUN_PATH=.:..:"${LD_RUN_PATH}" \
		$(CXX) -std=c++2b -o $@ $< -fPIC -shared -Wl,--as-needed \
			$(shell pkg-config --cflags --libs vaccel) \
			-L.. -I.. -lsol_swin_v2_b

.PHONY: clean
clean:
	rm -f libsol_swin_v2_b_wrapper.so
