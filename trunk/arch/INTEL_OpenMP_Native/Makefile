#
# Makefile for Patus stencil benchmark
#
# Note: $(PATUS_*) variables will be automatically replaced by the
# required runtime files by Patus.
#

CC = icc
#CFLAGS = -O3 -fno-alias -fno-fnalias -g -openmp -offload-build -offload-copts="-openmp -g -O3" -I.

# only build native MIC for now... 
CFLAGS = -O3 -mmic -fno-alias -fno-fnalias -g -openmp -I.

bench: kernel.o driver.o util.o timer.o
        $(CC) $(CFLAGS) -o $@ $+

%.o: %.c
        $(CC) $(CFLAGS) -c -o $@ $<
        
clean:
        rm -rf *.o bench
