#include <stdio.h>
#include <string.h>
#include <stdarg.h>

#include "patusrt.h"


int has_arg (char* szArg, int argc, char** argv)
{
	int i;
	for (i = 0; i < argc; i++)
		if (strcmp (argv[i], szArg) == 0)
			return 1;
	return 0;
}

void write_data_f (char* szFilename, int nDim, float* pData, ...)
{
	int i, j;
	int nSizeX, nSizeY, nOffset;
	
	// nothing to print if the dimension i 0
	if (nDim == 0)
		return;
	
	va_list vl;
	va_start (vl, pData);
	
	nSizeX = va_arg (vl, int);
	if (nDim > 1)
		nSizeY = va_arg (vl, int);
	else
		nSizeY = 1;
		
	nOffset = 0;
	if (nDim > 2)
	{
		// to write the 2D slice of the grid, projecting to grid(x0, x1, x2/2, x3/2,..., xn/2) (all x0, x1),
		// calculate the offset in the linear array
		
		nOffset = 0;
		int nStride = nSizeX * nSizeY;
		for (i = 2; i < nDim; i++)
		{
			int nSize = va_arg (vl, int);
			nOffset += (nSize / 2) * nStride;
			nStride *= nSize;
		}
	}
	
	// write the data to file
	FILE* file = fopen (szFilename, "w");
	for (j = 0; j < nSizeY; j++)
	{
		for (i = 0; i < nSizeX; i++)
			fprintf (file, "%e ", pData[i + j * nSizeX + nOffset]);
		fprintf (file, "\n");
	}		
	fclose (file);
}

void write_data_d (char* szFilename, int nDim, double* pData, ...)
{
	int i, j;
	int nSizeX, nSizeY, nOffset;
	
	// nothing to print if the dimension i 0
	if (nDim == 0)
		return;
	
	va_list vl;
	va_start (vl, pData);
	
	nSizeX = va_arg (vl, int);
	if (nDim > 1)
		nSizeY = va_arg (vl, int);
	else
		nSizeY = 1;
		
	nOffset = 0;
	if (nDim > 2)
	{
		// to write the 2D slice of the grid, projecting to grid(x0, x1, x2/2, x3/2,..., xn/2) (all x0, x1),
		// calculate the offset in the linear array
		
		nOffset = 0;
		int nStride = nSizeX * nSizeY;
		for (i = 2; i < nDim; i++)
		{
			int nSize = va_arg (vl, int);
			nOffset += (nSize / 2) * nStride;
			nStride *= nSize;
		}
	}
	
	// write the data to file
	FILE* file = fopen (szFilename, "w");
	for (j = 0; j < nSizeY; j++)
	{
		for (i = 0; i < nSizeX; i++)
			fprintf (file, "%e ", pData[i + j * nSizeX + nOffset]);
		fprintf (file, "\n");
	}		
	fclose (file);
}
