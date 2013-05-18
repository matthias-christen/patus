# Define the target architecture
ARCHITECTURE = "x86_64 AVX"
# ... or if your CPU doesn't support AVX:
#ARCHITECTURE = "x86_64 SSE"

# Define the optimization flags for gcc
GCC_FLAGS = -O3 -fgcse-sm -funroll-loops -funroll-all-loops -g -fopenmp -mavx
# ... or if your CPU doesn't support AVX:
#GCC_FLAGS = -O3 -fgcse-sm -funroll-loops -funroll-all-loops -g -fopenmp -msse2


# Find the grid size in the source file and use it for auto-tuning
GRIDSIZE = $(shell grep "\#define GRID_SIZE" wave.c | sed 's/[^0-9]\+//')


# Runs Patus (code generation and tuning) and creates the header files for wave.c
OUT_DIR = $(<:.c=_kernel)
define run-patus
	@echo Generating the code in a subfolder...
	patus $< --outdir=$(OUT_DIR) --embedded --create-prefetching=no --architecture=$(ARCHITECTURE)
	
	@echo Copying the generated kernel.c file to the main directory and rename appropriately...
	cp $(OUT_DIR)/kernel.c $(@:_pp.c=_patus.c)
	cp $(OUT_DIR)/kernel.h $(@:_pp.c=_patus.h)
	
	@echo Copying the header files...
	cp $(OUT_DIR)/*.h .
	
	@echo Running the auto-tuner...
	make --directory=$(OUT_DIR)
	make --directory=$(OUT_DIR) tune size=$(GRIDSIZE)
	
	@echo Copying tuned parameters...
	cat /dev/null > tuned_params.h
	cat */tuned_params.h >> tuned_params.h
endef


# Build and run the program
all: wave wave_basic run

# Create the executable
wave: wave_pp.c
	gcc $(GCC_FLAGS) $+ wave_patus.c -o $@ -lm
	
# Pre-process and generate code
# In the "--embedded" mode, the Patus preprocessor will always create a "*_pp.c" file
wave_pp.c: wave.c
	$(run-patus)

# Build the basic version (without Patus)
wave_basic: wave_basic.c
	gcc $(GCC_FLAGS) $+ -o $@ -lm

# Run the executable and plot the result
run:
	./wave
	./visualize.sh

# Remove any generated files
clean:
	rm -f cycle.h ssemathlib.h avxmathlib.h avxreductions.h ssereductions.h patusrt.h tuned_params.h wave.h
	rm -rf *_patus.* *_pp.c wave_kernel
	rm -f wave wave_basic
	rm -f [0-9][0-9][0-9][0-9].txt
	rm -f *~

# Remove any files created by the programs
cleanoutput:
	rm -f ????.txt
	rm -f waves.gp

# Clean for distribution
distclean: clean cleanoutput
