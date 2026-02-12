# Makefile for cryo-EM denoising (12-layer residual CNN)

FC = nvfortran
FFLAGS = -cuda -O2 -Minfo=accel
LIBS = -lcudnn

# Targets
TARGET_DEEP = cryo_train_deep
TARGET_N2N = cryo_train_n2n

# Source files
SRCS_DEEP = streaming_cryo_loader.cuf conv2d_cudnn.cuf cryo_train_deep.cuf
SRCS_N2N = streaming_n2n_loader.cuf conv2d_cudnn.cuf cryo_train_n2n.cuf

all: $(TARGET_DEEP) $(TARGET_N2N)

$(TARGET_DEEP): $(SRCS_DEEP)
	$(FC) $(FFLAGS) -o $@ $^ $(LIBS)

$(TARGET_N2N): $(SRCS_N2N)
	$(FC) $(FFLAGS) -o $@ $^ $(LIBS)

clean:
	rm -f $(TARGET_DEEP) $(TARGET_N2N) *.mod *.o

.PHONY: all clean
