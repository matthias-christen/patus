# Paths to the JDK and to the Cbc solver
JDK_DIR=/usr/local/java/jdk1.7.0_03
CBC_DIR=/home/christen/Software/Cbc

#
# CBC was built with the following configure command line:
# ./configure CXX=icpc CC=icc --with-blas="-L$MKLROOT/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -openmp -lpthread -lm" --enable-static=yes
#
# - static MKL:
# ./configure CXX=icpc CC=icc --with-blas="-Wl,--start-group $MKLROOT/lib/intel64/libmkl_intel_lp64.a $MKLROOT/lib/intel64/libmkl_intel_thread.a $MKLROOT/lib/intel64/libmkl_core.a -Wl,--end-group -openmp -lpthread -lm" --enable-static=yes
#

# --------------------------------------------------------------------------------------------------

# Set the compiler
CXX=icpc
CFLAGS=-O2 -g

# Name of the JNI library to create
LIB_NAME=libilpsolver.so

ilpsolver: CoinInterface.cpp ch_unibas_cs_hpwc_patus_ilp_ILPSolver.h ch_unibas_cs_hpwc_patus_ilp_ILPModel.h ch_unibas_cs_hpwc_patus_ilp_ILPSolution.h
	$(CXX) $(CFLAGS) -c -fPIC -o $(<:cpp=o) -I$(JDK_DIR)/include -I$(JDK_DIR)/include/linux -I$(CBC_DIR)/include/coin $<
	$(CXX) -o $(LIB_NAME) -fPIC -shared -Wl,-soname,$(LIB_NAME) $(<:cpp=o) -mkl -lz -L$(CBC_DIR)/lib -lCoinUtils -lOsiClp -lOsi -lClp -lCgl -lCbc -lCbcSolver


ch_unibas_cs_hpwc_patus_ilp_ILPSolver.h: ../ch/unibas/cs/hpwc/patus/ilp/ILPSolver.java
	$(JDK_DIR)/bin/javah -classpath .. ch.unibas.cs.hpwc.patus.ilp.ILPSolver
	
ch_unibas_cs_hpwc_patus_ilp_ILPModel.h: ../ch/unibas/cs/hpwc/patus/ilp/ILPModel.java
	$(JDK_DIR)/bin/javah -classpath .. ch.unibas.cs.hpwc.patus.ilp.ILPModel
	
ch_unibas_cs_hpwc_patus_ilp_ILPSolution.h: ../ch/unibas/cs/hpwc/patus/ilp/ILPSolution.java
	$(JDK_DIR)/bin/javah -classpath .. ch.unibas.cs.hpwc.patus.ilp.ILPSolution


clean:
	rm -rf *.o *.so
