#
# Makefile for Patus stencil benchmark
#
# Note: $(PATUS_*) variables will be automatically replaced by the
# required runtime files by Patus.
#

CC = nvcc
NVCCFLAGS = -O3 -arch=sm_13 -I$(HOME)/Software/NVIDIA_GPU_Computing_SDK/C/common/inc
#NVCCFLAGS = -O0 -g -arch=sm_13 -I/home/christen/NVIDIA_GPU_Computing_SDK/C/common/inc

bench: kernel.cu driver.cu $(PATUS_RUNTIME_FILES)
	$(CC) $(NVCCFLAGS) -o $@ $+
	
clean:
	rm -rf *.o bench