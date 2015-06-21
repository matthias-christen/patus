#include <iostream>
#include <fstream>

#include "PerfStats.h"

#include "Algorithms/Trap.h"

#pragma patus forward_decls


class PATUS_STENCIL_NAME : public Stencil<PATUS_STENCIL_DIMENSIONALITY>
{
private:
	Grid<PATUS_STENCIL_DIMENSIONALITY, PATUS_STENCIL_FPTYPE>* m_pGrid[PATUS_NUM_GRIDS];

public:
	PATUS_STENCIL_NAME (Coords<PATUS_STENCIL_DIMENSIONALITY> size)
		: Stencil<PATUS_STENCIL_DIMENSIONALITY> (Coords<PATUS_STENCIL_DIMENSIONALITY> (PATUS_BOUNDINGBOX_MIN), Coords<PATUS_STENCIL_DIMENSIONALITY> (PATUS_BOUNDINGBOX_MAX))
	{
		for (int i = 0; i < PATUS_NUM_GRIDS; i++)
			m_pGrid[i] = new Grid<PATUS_STENCIL_DIMENSIONALITY, PATUS_STENCIL_FPTYPE> (size);
		
		#pragma patus initialize_grids
	}

	~PATUS_STENCIL_NAME ()
	{
		for (int i = 0; i < PATUS_NUM_GRIDS; i++)
			delete m_pGrid[i];
	}

	virtual void compute(int nTmin, int nTmax, Coords<PATUS_STENCIL_DIMENSIONALITY> xmin, Coords<PATUS_STENCIL_DIMENSIONALITY> xmax, Slopes<PATUS_STENCIL_DIMENSIONALITY>& slopes)
	{
		#pragma patus compute_stencil(m_pGrid, nTmin, nTmax, xmin, xmax, slopes)
	}

	virtual void computePoint (int t, Coords<PATUS_STENCIL_DIMENSIONALITY>& pt)
	{
		#pragma patus compute_point
	}

	Grid<PATUS_STENCIL_DIMENSIONALITY, PATUS_STENCIL_FPTYPE>& getGrid (int t)
	{
		return *(m_pGrid[t % PATUS_NUM_GRIDS]);
	}

	virtual long getFlopsCount ()
	{
		return m_pGrid[0]->getElementsCount () * PATUS_FLOPS_PER_STENCIL;
	}
};


int main (int argc, char** argv)
{
#ifdef PARALLEL
	tbb::task_scheduler_init init (-1);
#endif

	const int nTMax = PATUS_NUM_TIMESTEPS;
	Coords<PATUS_STENCIL_DIMENSIONALITY> size (PATUS_GRID_SIZE);
	Coords<PATUS_STENCIL_DIMENSIONALITY> xmin (PATUS_DOMAIN_MIN);
	Coords<PATUS_STENCIL_DIMENSIONALITY> xmax (PATUS_DOMAIN_MAX);

	PATUS_STENCIL_NAME stencil (size);
	
	PerfStats ps (nTMax * stencil.getFlopsCount ());
	StencilAlgorithmTrap<PATUS_STENCIL_DIMENSIONALITY> algo;

	ps.tic ();	
	algo.run (stencil, nTMax, xmin, xmax);
	ps.toc ();

	std::ofstream out ("result.txt");
	stencil.getGrid (nTMax).printSlice (out);
	out.close ();

	// print performance statistics
	std::cout << ps << std::endl;

	return 0;
}

