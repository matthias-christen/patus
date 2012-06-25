#
# Makefile for Patus stencil benchmark
#
# Note: $(PATUS_*) variables will be automatically replaced by the
# required runtime files by Patus.
#

CC = icc
#CFLAGS = -O3 -fno-alias -fno-fnalias -g -openmp -offload-build -offload-copts="-openmp -g -O3" -I.

# only build native MIC for now... 
CFLAGS = -mmic -O3 -fno-alias -fno-fnalias -g -openmp -I.

bench: kernel.o driver.o $(PATUS_RUNTIME_OBJECT_FILES)
	$(CC) $(CFLAGS) -o $@ $+

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<
	
clean:
	rm -rf *.o bench
