# Makefile for cryo_train_deep (12-layer residual CNN)

FC = nvfortran
FFLAGS = -cuda -O2 -Minfo=accel
LIBS = -lcudnn

TARGET = cryo_train_deep

SRCS = streaming_cryo_loader.cuf conv2d_cudnn.cuf cryo_train_deep.cuf

all: $(TARGET)

$(TARGET): $(SRCS)
	$(FC) $(FFLAGS) -o $@ $^ $(LIBS)

clean:
	rm -f $(TARGET) *.mod *.o

.PHONY: all clean
