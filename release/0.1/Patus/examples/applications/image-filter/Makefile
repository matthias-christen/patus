# define the names of the input and the output images
IMAGE_INPUT_FILE = hpc.png
IMAGE_OUTPUT_FILE = hpc-filtered.png

# the image viewer program used to display the result image
IMAGE_VIEWER = eog


# compiler optimization flags
ifeq ($(DEBUG), 1)
    # default optimizations
    OPT_FLAGS = -O0 -g
else
    # for debugging
    OPT_FLAGS = -O3
endif


# get the image dimensions
# NOTE that the image width must be divisible by 8
IMAGE_WIDTH = $(shell file $(IMAGE_INPUT_FILE) | egrep -o '[0-9]+' | sed -n '1,1p')
IMAGE_HEIGHT = $(shell file $(IMAGE_INPUT_FILE) | egrep -o '[0-9]+' | sed -n '2,1p')


# do everything
all: show

# run the program
run: filter
	./filter $(IMAGE_INPUT_FILE) $(IMAGE_OUTPUT_FILE)

# display the image
show: $(IMAGE_OUTPUT_FILE)
	$(IMAGE_VIEWER) $< &


# ---- build ----

filter: pngfilter.c gen/kernel.c
	gcc -mavx $(OPT_FLAGS) -fopenmp $+ -lpng -o $@

gen/kernel.c: filter.stc
	patus --outdir=gen $<
	make -C gen
	make -C gen tune width=$(IMAGE_WIDTH) height=$(IMAGE_HEIGHT)

$(IMAGE_OUTPUT_FILE): run


# ---- clean up ----

clean:
	rm -f filter
	rm -rf gen
	rm -f $(IMAGE_OUTPUT_FILE)
	rm -f *~

