#
# Makefile for Patus stencil benchmark
#
# Note: $(PATUS_*) variables will be automatically replaced by the
# required runtime files by Patus.
#

CC = gcc
CFLAGS = -O3 -fopenmp -I.

#$(PATUS_INIT_NONAUTOTUNE_EXE_PARAMS)


bench: kernel.o driver.o ssemathlib.o $(PATUS_RUNTIME_OBJECT_FILES)
	$(CC) $(CFLAGS) -o $@ $+

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<
	
#tune:
#$(PATUS_TEST_NONAUTOTUNE_EXE_PARAMS)
#	$(PATUS_AUTOTUNER) bench $(PATUS_EXE_PARAMS)
	
clean:
	rm -rf *.o bench
