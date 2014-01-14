#
# Makefile for Patus stencil benchmark
#
# Note: $(PATUS_*) variables will be automatically replaced by the
# required runtime files by Patus.
#

CC = icc

CFLAGS = -O3 -fno-alias -fno-fnalias -g -openmp -I.

bench: kernel.o driver.o util.o timer.o
	$(CC) $(CFLAGS) -o $@ $+

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

	
tune:
$(PATUS_TEST_NONAUTOTUNE_EXE_PARAMS)
	$(PATUS_AUTOTUNER) bench $(PATUS_EXE_PARAMS) | tee `date +tune_%F_%H.%M.%S.txt`
	

clean:
	rm -rf *.o bench
