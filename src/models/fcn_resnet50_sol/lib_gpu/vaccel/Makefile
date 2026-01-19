.PHONY: all
all: libsol_fcn_resnet50_wrapper.so

libsol_fcn_resnet50_wrapper.so: sol_fcn_resnet50_wrapper.c
	LD_RUN_PATH=.:..:"${LD_RUN_PATH}" \
		$(CXX) -std=c++2b -o $@ $< -fPIC -shared -Wl,--as-needed \
			$(shell pkg-config --cflags --libs vaccel) \
			-L.. -I.. -lsol_fcn_resnet50

.PHONY: clean
clean:
	rm -f libsol_fcn_resnet50_wrapper.so
