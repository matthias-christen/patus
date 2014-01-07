# Find the grid size in the source file and use it for auto-tuning
GRIDSIZE = $(shell grep "\#define GRID_SIZE" WhisperingGallery.c | sed 's/[^0-9]\+//')
PLOTSIZE = $(shell echo $(GRIDSIZE)-0.5 | bc)


# Runs Patus (code generation and tuning) and creates the header files for WhisperingGallery.c
define run-patus
	@echo Generating the code in a subfolder...
	patus $< --outdir=$(<:.stc=) --create-prefetching=no
	
	@echo Copying the generated kernel.c file to the main directory and rename appropriately...
	cp $(<:.stc=)/kernel.c $@
	cp $(<:.stc=)/kernel.h $(@:c=h)
	
	@echo Copying the header files...
	cp $(<:.stc=)/*.h .
	
	@echo Running the auto-tuner...
	make --directory=$(<:.stc=)
	make --directory=$(<:.stc=) tune x_max=$(GRIDSIZE) y_max=$(GRIDSIZE)
	
	@echo Copying tuned parameters...
	cat /dev/null > tuned_params.h
	cat */tuned_params.h >> tuned_params.h
endef


# Build and run the program
all: WhisperingGallery run

# Create the executable
WhisperingGallery: WhisperingGallery.c fdtdE2D.c fdtdH2D.c integrate.c
	gcc -O3 -fgcse-sm -funroll-loops -funroll-all-loops -g -fopenmp -mavx $+ -o $@ -lm

# Generate code from stencil specifications
%.c: %.stc
	$(run-patus)

# Run the executable and plot the result
run:
	./WhisperingGallery
	sed s/@PLOTSIZE@/$(PLOTSIZE)/g plot.gp > plot.tmp.gp
	gnuplot plot.tmp.gp
	rm plot.tmp.gp
	evince output.pdf &

# Remove any generated files
clean:
	rm -f fdtd*.[ch] integrate.[ch] WhisperingGallery
	rm -f cycle.h ssemathlib.h avxmathlib.h avxreductions.h ssereductions.h patusrt.h tuned_params.h
	rm -rf fdtdE2D fdtdH2D integrate
	rm -f *~

distclean: clean
	rm -f output.*
