#
# Makefile for Patus stencil benchmark
#
# Note: $(PATUS_*) variables will be automatically replaced by the
# required runtime files by Patus.
#

CC = cc
CFLAGS = -O3 -fopenmp -I.

bench: kernel.o driver.o $(PATUS_RUNTIME_OBJECT_FILES)
	$(CC) $(CFLAGS) -o $@ $+ -lm

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<
	
tune:
$(PATUS_TEST_NONAUTOTUNE_EXE_PARAMS)
	$(PATUS_AUTOTUNER) bench $(PATUS_EXE_PARAMS) | tee `date +tune_%F_%H.%M.%S.txt`

clean:
	rm -rf *.o bench