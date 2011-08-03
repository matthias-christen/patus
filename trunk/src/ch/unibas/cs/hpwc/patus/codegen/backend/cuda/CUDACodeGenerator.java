package ch.unibas.cs.hpwc.patus.codegen.backend.cuda;

import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;

/**
 *
 * @author Matthias-M. Christen
 */
public class CUDACodeGenerator extends AbstractCUDACodeGenerator
{
	///////////////////////////////////////////////////////////////////
	// Constants

	private final static int[] INDEXING_LEVEL_DIMENSIONALITIES = new int[] { 3, 2 };


	///////////////////////////////////////////////////////////////////
	// Implementation

	public CUDACodeGenerator (CodeGeneratorSharedObjects data)
	{
		super (data);
	}

	@Override
	protected int getIndexingLevelDimensionality (int nIndexingLevel)
	{
		return INDEXING_LEVEL_DIMENSIONALITIES[nIndexingLevel];
	}
}
