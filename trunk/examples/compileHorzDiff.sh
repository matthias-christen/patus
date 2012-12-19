#!/bin/bash
patus --architecture="x86_64 AVX asm" --create-prefetching=yes --outdir=HorzDiffLaplacian HorzDiffLaplacian.stc
patus --architecture="x86_64 AVX" --outdir=HorzDiffFluxes HorzDiffFluxes.stc
patus --architecture="x86_64 AVX asm" --create-prefetching=no --outdir=HorzDiffOut HorzDiffOut.stc


#cd HorzDiffLaplacian
#make
#cd ../HorzDiffFluxes
#make
#cd ../HorzDiffOut
#make
#cd ..

