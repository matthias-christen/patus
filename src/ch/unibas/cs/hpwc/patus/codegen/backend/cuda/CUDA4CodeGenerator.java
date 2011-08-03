package ch.unibas.cs.hpwc.patus.codegen.backend.cuda;

import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;

public class CUDA4CodeGenerator extends AbstractCUDACodeGenerator
{
	///////////////////////////////////////////////////////////////////
	// Constants

	/**
	 * CUDA 4.0 and above allow (3, 3)-indexing on Fermi cards
	 */
	private final static int[] INDEXING_LEVEL_DIMENSIONALITIES = new int[] { 3, 3 };


	///////////////////////////////////////////////////////////////////
	// Implementation

	public CUDA4CodeGenerator (CodeGeneratorSharedObjects data)
	{
		super (data);
	}

	@Override
	protected int getIndexingLevelDimensionality (int nIndexingLevel)
	{
		return INDEXING_LEVEL_DIMENSIONALITIES[nIndexingLevel];
	}
}
