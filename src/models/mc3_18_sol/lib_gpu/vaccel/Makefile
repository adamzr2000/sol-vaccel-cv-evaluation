.PHONY: all
all: libsol_mc3_18_wrapper.so

libsol_mc3_18_wrapper.so: sol_mc3_18_wrapper.c
	LD_RUN_PATH=.:..:"${LD_RUN_PATH}" \
		$(CXX) -std=c++2b -o $@ $< -fPIC -shared -Wl,--as-needed \
			$(shell pkg-config --cflags --libs vaccel) \
			-L.. -I.. -lsol_mc3_18

.PHONY: clean
clean:
	rm -f libsol_mc3_18_wrapper.so
