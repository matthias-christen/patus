package ch.unibas.cs.hpwc.patus.codegen.backend.cuda;

import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;

/**
 *
 * @author Matthias-M. Christen
 */
public class CUDA1DCodeGenerator extends AbstractCUDACodeGenerator
{
	///////////////////////////////////////////////////////////////////
	// Implementation

	public CUDA1DCodeGenerator (CodeGeneratorSharedObjects data)
	{
		super (data);
	}

	@Override
	protected int getIndexingLevelDimensionality (int nIndexingLevel)
	{
		return 1;
	}
}
