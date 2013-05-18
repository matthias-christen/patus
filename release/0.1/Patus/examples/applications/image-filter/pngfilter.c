/**
 * Original file from http://zarb.org/~gc/html/libpng.html
 * Copyright 2002-2010 Guillaume Cottenceau.
 *
 * Modified by Matthias Christen for applying a Patus-generated
 * image filter.
 */

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <sys/time.h>

#define PNG_DEBUG 3
#include <png.h>


/* forward-declare the Patus-generated kernel */
#include "gen/kernel.h"


int width, height;
png_byte color_type;
png_byte bit_depth;

png_structp png_ptr;
png_infop info_ptr;
int number_of_passes;
png_bytep * row_pointers;


void abort_(const char * s, ...)
{
	va_list args;
	va_start(args, s);
	vfprintf(stderr, s, args);
	fprintf(stderr, "\n");
	va_end(args);
	abort();
}

double gettime()
{
	struct timeval tp;
	struct timezone tzp;
	int i;

	i = gettimeofday (&tp, &tzp);
	return ((double) tp.tv_sec * 1.e6 + (double) tp.tv_usec);
}

void read_png_file(char* file_name)
{
	int y;
	char header[8];    // 8 is the maximum size that can be checked

	/* open file and test for it being a png */
	FILE *fp = fopen(file_name, "rb");
	if (!fp)
		abort_("[read_png_file] File %s could not be opened for reading", file_name);
	fread(header, 1, 8, fp);
	if (png_sig_cmp(header, 0, 8))
		abort_("[read_png_file] File %s is not recognized as a PNG file", file_name);

	/* initialize stuff */
	png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

	if (!png_ptr)
		abort_("[read_png_file] png_create_read_struct failed");

	info_ptr = png_create_info_struct(png_ptr);
	if (!info_ptr)
		abort_("[read_png_file] png_create_info_struct failed");

	if (setjmp(png_jmpbuf(png_ptr)))
		abort_("[read_png_file] Error during init_io");

	png_init_io(png_ptr, fp);
	png_set_sig_bytes(png_ptr, 8);

	png_read_info(png_ptr, info_ptr);

	width = png_get_image_width(png_ptr, info_ptr);
	height = png_get_image_height(png_ptr, info_ptr);
	color_type = png_get_color_type(png_ptr, info_ptr);
	bit_depth = png_get_bit_depth(png_ptr, info_ptr);

	number_of_passes = png_set_interlace_handling(png_ptr);
	png_read_update_info(png_ptr, info_ptr);

	/* read file */
	if (setjmp(png_jmpbuf(png_ptr)))
		abort_("[read_png_file] Error during read_image");

	row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * height);
	for (y=0; y<height; y++)
		row_pointers[y] = (png_byte*) malloc(png_get_rowbytes(png_ptr,info_ptr));
	png_read_image(png_ptr, row_pointers);

	fclose(fp);
}

void write_png_file(char* file_name)
{
	int y;

	/* create file */
	FILE *fp = fopen(file_name, "wb");
	if (!fp)
		abort_("[write_png_file] File %s could not be opened for writing", file_name);

	/* initialize stuff */
	png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

	if (!png_ptr)
		abort_("[write_png_file] png_create_write_struct failed");

	info_ptr = png_create_info_struct(png_ptr);
	if (!info_ptr)
		abort_("[write_png_file] png_create_info_struct failed");

	if (setjmp(png_jmpbuf(png_ptr)))
		abort_("[write_png_file] Error during init_io");

	png_init_io(png_ptr, fp);

	/* write header */
	if (setjmp(png_jmpbuf(png_ptr)))
		abort_("[write_png_file] Error during writing header");

	png_set_IHDR(png_ptr, info_ptr, width, height,
		bit_depth, color_type, PNG_INTERLACE_NONE,
		PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

	png_write_info(png_ptr, info_ptr);

	/* write bytes */
	if (setjmp(png_jmpbuf(png_ptr)))
		abort_("[write_png_file] Error during writing bytes");

	png_write_image(png_ptr, row_pointers);

	/* end write */
	if (setjmp(png_jmpbuf(png_ptr)))
		abort_("[write_png_file] Error during end of write");
	png_write_end(png_ptr, NULL);

	/* cleanup heap allocation */
	for (y=0; y<height; y++)
	free(row_pointers[y]);
	free(row_pointers);

	fclose(fp);
}

inline int clamp(float a)
{
	if (a < 0)
		return 0;
	if (a > 255)
		return 255;
	return (int) a;
}

void process_file(void)
{
	int x, y;
	double t0, t1;
	float *red_in, *green_in, *blue_in;
	float *red_out, *green_out, *blue_out;
	float *dummy1, *dummy2, *dummy3;

	if (png_get_color_type(png_ptr, info_ptr) == PNG_COLOR_TYPE_RGB)
		abort_("[process_file] input file is PNG_COLOR_TYPE_RGB but must be PNG_COLOR_TYPE_RGBA (lacks the alpha channel)");

	if (png_get_color_type(png_ptr, info_ptr) != PNG_COLOR_TYPE_RGBA)
		abort_("[process_file] color_type of input file must be PNG_COLOR_TYPE_RGBA (%d) (is %d)",
			PNG_COLOR_TYPE_RGBA, png_get_color_type(png_ptr, info_ptr));

	red_in = (float*) malloc(width * height * sizeof(float));
	green_in = (float*) malloc(width * height * sizeof(float));
	blue_in = (float*) malloc(width * height * sizeof(float));

	red_out = (float*) malloc(width * height * sizeof(float));
	green_out = (float*) malloc(width * height * sizeof(float));
	blue_out = (float*) malloc(width * height * sizeof(float));

	if (!red_in || !green_in || !blue_in || !red_out || !green_out || !blue_out)
		abort_("[process_file] could not allocate memory for processing the image");

	#pragma omp parallel
	initialize_filter(red_in, red_out, green_in, green_out, blue_in, blue_out, width, height);

	for (y = 0; y < height; y++)
	{
		png_byte* row = row_pointers[y];
		for (x=0; x<width; x++)
		{
			png_byte* ptr = &(row[x*4]);
			int idx = y * width + x;
			red_in[idx] = ptr[0];
			green_in[idx] = ptr[1];
			blue_in[idx] = ptr[2]; 
		}
	}

	/* apply the Patus-generated filter kernel */
	t0 = gettime();
	#pragma omp parallel
	filter(&dummy1, &dummy2, &dummy3, red_in, red_out, green_in, green_out, blue_in, blue_out, width, height);
	t1 = gettime();

	printf("filter took %f µs to complete.\n", t1 - t0);
	
	for (y=0; y<height; y++)
	{
		png_byte* row = row_pointers[y];
		for (x=0; x<width; x++)
		{
			png_byte* ptr = &(row[x*4]);
			int idx = y * width + x;
			ptr[0] = clamp(red_out[idx]);
			ptr[1] = clamp(green_out[idx]);
			ptr[2] = clamp(blue_out[idx]); 
		}
	}

	free(red_in);
	free(green_in);
	free(blue_in);
	free(red_out);
	free(green_out);
	free(blue_out);
}

int main(int argc, char **argv)
{
	if (argc != 3)
		abort_("Usage: program_name <file_in> <file_out>");

	read_png_file(argv[1]);
	process_file();
	write_png_file(argv[2]);

	return 0;
}

