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
package ch.unibas.cs.hpwc.patus.codegen.backend;

import cetus.hir.Expression;
import ch.unibas.cs.hpwc.patus.ast.StatementList;

/**
 *
 * @author Matthias-M. Christen
 */
public interface INonKernelFunctions
{
	/**
	 * Initialize code generator data structures
	 */
	public abstract void initializeNonKernelFunctionCG ();
	
	
	///////////////////////////////////////////////////////////////////
	// Patus Pragmas

	/**
	 * Creates a {@link StatementList} with forward declarations of the
	 * initialization and stencil kernel functions.
	 */
	public abstract StatementList forwardDecls ();

	/**
	 * Creates the code declaring the grids to be passed to the
	 * initialization/stencil kernel functions.
	 */
	public abstract StatementList declareGrids ();

	/**
	 * Creates the code allocating memory for the stencil grids.
	 */
	public abstract StatementList allocateGrids ();

	/**
	 * Creates the code to call the generated initialization function.
	 */
	public abstract StatementList initializeGrids ();

	/**
	 * Creates the code to transfer the data to the compute units.
	 */
	public abstract StatementList sendData ();

	/**
	 * Creates the code to receive computed data from the compute units.
	 */
	public abstract StatementList receiveData ();

	/**
	 * Creates the code calling the generated stencil kernel. 
	 */
	public abstract StatementList computeStencil ();

	/**
	 * Creates the code to validate the computation done by the generated
	 * stencil kernel.
	 */
	public abstract StatementList validateComputation ();

	public abstract StatementList writeGrids (String strFilenameFormat, String strType);

	/**
	 * Creates the code to free the data.
	 */
	public abstract StatementList deallocateGrids ();


	///////////////////////////////////////////////////////////////////
	// Patus Variables

	public abstract Expression getFlopsPerStencil ();

	public abstract Expression getGridPointsCount ();

	public abstract Expression getBytesTransferred ();

	public abstract Expression getDoValidation ();

	public abstract Expression getValidates ();


	///////////////////////////////////////////////////////////////////
	// Makefile Variables
		
	/**
	 * Creates the code to test whether the size parameters are still 0 (i.e.,
	 * haven't been changed on the command line to <code>make</code>) and in
	 * this case emits an error message.
	 * 
	 * @return Makefile code as a {@link String}
	 */
	public abstract String getTestNonautotuneExeParams ();

	/**
	 * Returns the command to call the Patus auto-tuner.
	 */
	public abstract String getAutotuner ();
	
	/**
	 * Returns the list of command line parameters to the auto-tuner (excluding
	 * the name of the executable).
	 */
	public abstract String getExeParams ();
}
