# Makefile for cryo_train_perceptual

FC = nvfortran
FFLAGS = -cuda -O2 -Minfo=accel
LIBS = -lcudnn -lcufft

TARGET = cryo_train_perceptual

SRCS = streaming_cryo_loader.cuf conv2d_cudnn.cuf fourier_loss.cuf \
       ssim_loss.cuf gradient_loss.cuf cryo_train_perceptual.cuf

all: $(TARGET)

$(TARGET): $(SRCS)
	$(FC) $(FFLAGS) -o $@ $^ $(LIBS)

clean:
	rm -f $(TARGET) *.mod *.o

.PHONY: all clean
