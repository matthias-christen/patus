# Paths to the JDK and to the Cbc solver
JDK_DIR=/usr/local/java/jdk1.7.0_03
CBC_DIR=/home/christen/Software/Cbc

#
# CBC was built with the following configure command line:
# ./configure CXX=icpc CC=icc ADD_CXXFLAGS="-DCLP_FAST_CODE -DCOIN_FAST_CODE -DCOIN_USE_RESTRICT" --with-blas="-L$MKLROOT/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -openmp -lpthread -lm" --enable-static=yes --enable-cbc-parallel
#
# - static MKL:
# ./configure CXX=icpc CC=icc ADD_CXXFLAGS="-DCLP_FAST_CODE -DCOIN_FAST_CODE -DCOIN_USE_RESTRICT -fPIC" LDFLAGS=-fPIC --with-blas="-Wl,--start-group,$MKLROOT/lib/intel64/libmkl_intel_lp64.a,$MKLROOT/lib/intel64/libmkl_intel_thread.a,$MKLROOT/lib/intel64/libmkl_core.a,--end-group -openmp -lpthread -lm" --enable-static=yes  --enable-cbc-parallel
#
# At some point the Cbc build will fail due to undefined references in CoinUtil to some Lapack functions.
# Go to CoinUtil/src, make clean, make, then add the MKL linking stuff to the compiler (linker) invocation
# and re-link, then re-do the ar and ranlib commands which followed in make.
# Go up to the Cbc base directory again, and make. This time the build should complete.
#

# --------------------------------------------------------------------------------------------------

# Set the compiler
CXX=icpc
CFLAGS=-O2 -g
OMP=-openmp

MKL=-Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_intel_thread.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -openmp -lpthread -lm
CBC_LIBDIR=$(CBC_DIR)/lib/

# Name of the JNI library to create
LIB_NAME=libilpsolver.so

ilpsolver: CoinInterface.cpp ch_unibas_cs_hpwc_patus_ilp_ILPSolver.h ch_unibas_cs_hpwc_patus_ilp_ILPModel.h ch_unibas_cs_hpwc_patus_ilp_ILPSolution.h
	$(CXX) $(CFLAGS) -c -fPIC $(OMP) -o $(<:cpp=o) -I$(JDK_DIR)/include -I$(JDK_DIR)/include/linux -I$(CBC_DIR)/include/coin $<
	$(CXX) -o $(LIB_NAME) -fPIC -shared -Wl,-soname,$(LIB_NAME) $(<:cpp=o) -lz -lrt $(OMP) -Wl,--start-group $(CBC_LIBDIR)libCoinUtils.a $(CBC_LIBDIR)libOsiClp.a $(CBC_LIBDIR)libOsi.a $(CBC_LIBDIR)libClp.a $(CBC_LIBDIR)libCgl.a $(CBC_LIBDIR)libCbc.a $(CBC_LIBDIR)libCbcSolver.a -Wl,--end-group $(MKL)


ch_unibas_cs_hpwc_patus_ilp_ILPSolver.h: ../src/ch/unibas/cs/hpwc/patus/ilp/ILPSolver.java
	$(JDK_DIR)/bin/javah -classpath ../src ch.unibas.cs.hpwc.patus.ilp.ILPSolver
	
ch_unibas_cs_hpwc_patus_ilp_ILPModel.h: ../src/ch/unibas/cs/hpwc/patus/ilp/ILPModel.java
	$(JDK_DIR)/bin/javah -classpath ../src ch.unibas.cs.hpwc.patus.ilp.ILPModel
	
ch_unibas_cs_hpwc_patus_ilp_ILPSolution.h: ../src/ch/unibas/cs/hpwc/patus/ilp/ILPSolution.java
	$(JDK_DIR)/bin/javah -classpath ../src ch.unibas.cs.hpwc.patus.ilp.ILPSolution


clean:
	rm -rf *.o *.so
