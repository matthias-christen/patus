TARGETS = blur divergence edge game-of-life gradient laplacian tricubic wave-1 wave-2

ALT_ARCH = "x86_64 AVX"
# or, if your CPU does not support AVX, use
#ALT_ARCH = "x86_64 SSE"


all: $(TARGETS)

blur: blur.stc
	patus --outdir=$@ $<
	$(MAKE) -C $@

divergence: divergence.stc
	patus --outdir=$@ $<
	$(MAKE) -C $@

edge: edge.stc
	patus --outdir=$@ --architecture=$(ALT_ARCH) $<
	$(MAKE) -C $@

game-of-life: game-of-life.stc
	patus --outdir=$@ $<
	$(MAKE) -C $@

gradient: gradient.stc
	patus --outdir=$@ $<
	$(MAKE) -C $@

laplacian: laplacian.stc
	patus --outdir=$@ $<
	$(MAKE) -C $@

tricubic: tricubic.stc
	patus --outdir=$@ --architecture=$(ALT_ARCH) $<
	$(MAKE) -C $@

wave-1: wave-1.stc
	patus --outdir=$@ --validation-tolerance=1e-3 $<
	$(MAKE) -C $@

wave-2: wave-2.stc
	patus --outdir=$@ --validation-tolerance=1e-3 $<
	$(MAKE) -C $@


clean:
	rm -rf $(TARGETS)
	rm -rf out
	rm -f *~

