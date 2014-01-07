/*******************************************************************************
 * Copyright (c) 2011 Matthias-M. Christen, University of Basel, Switzerland.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the GNU Lesser Public License v2.1
 * which accompanies this distribution, and is available at
 * http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
 * 
 * Contributors:
 *     Matthias-M. Christen, University of Basel, Switzerland - initial API and implementation
 ******************************************************************************/
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
