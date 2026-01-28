.PHONY: all
all: libsol_deeplabv3_resnet101_wrapper.so

libsol_deeplabv3_resnet101_wrapper.so: sol_deeplabv3_resnet101_wrapper.c
	LD_RUN_PATH=.:..:"${LD_RUN_PATH}" \
		$(CXX) -std=c++2b -o $@ $< -fPIC -shared -Wl,--as-needed \
			$(shell pkg-config --cflags --libs vaccel) \
			-L.. -I.. -lsol_deeplabv3_resnet101

.PHONY: clean
clean:
	rm -f libsol_deeplabv3_resnet101_wrapper.so
