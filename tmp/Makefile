#
# Makefile for Patus stencil benchmark
#
# Note: $(PATUS_*) variables will be automatically replaced by the
# required runtime files by Patus.
#

CC = icc
CFLAGS = -O3 -openmp

bench: kernel.o driver.o timer.o
	$(CC) $(CFLAGS) -o $@ $+

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<
	
clean:
	rm -rf *.o bench
