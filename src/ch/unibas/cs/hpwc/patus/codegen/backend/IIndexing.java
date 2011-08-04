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

/**
 *
 * @author Matthias-M. Christen
 */
public interface IIndexing
{
	///////////////////////////////////////////////////////////////////
	// Inner Types

	public interface IIndexingLevel
	{
		/**
		 * Returns the number of dimensions in this particular indexing level.
		 * @return The number of dimensions in this indexing level
		 */
		public abstract int getDimensionality ();

		/**
		 * Specifies whether the index is a programming model-specific built-in
		 * variable (e.g. in CUDA or UPC) or a function call.
		 * @return Returns <code>true</code> if the index is a built-in variable
		 * 	or <code>false</code> if the index is retrieved from a function call
		 */
		public abstract boolean isVariable ();

		/**
		 * Returns the index for the dimension <code>nDimension</code>.
		 * @param nDimension
		 * @return
		 */
		public abstract Expression getIndexForDimension (int nDimension);

		public abstract Expression getSizeForDimension (int nDimension);

		public abstract int getDefaultBlockSize (int nDimension);
	}

	/**
	 * Threading type of the architecture (roughly reflects the core count),
	 * i.e. specifies whether the architecture describes a multicore or manycore
	 * machine.
	 */
	public enum EThreading
	{
		/**
		 * Traditional multi-threading
		 */
		MULTI,

		/**
		 * Manycore architecture with virtually infinitely many cores
		 */
		MANY
	}


	///////////////////////////////////////////////////////////////////
	// Implementation

	/**
	 * Returns the number of indexing levels, e.g. 1 on CPUs,
	 * 2 on GPUs for grids and thread blocks, etc.
	 * @return The number of indexing levels
	 */
	public abstract int getIndexingLevelsCount ();

	/**
	 * Returns the indexing level description for a particular indexing level, <code>nIndexingLevel</code>.
	 * The deepest indexing level is returned for <code>nIndexingLevel == 0</code>.
	 * @param nIndexingLevel The indexing level
	 * @return The indexing level description
	 */
	public abstract IIndexingLevel getIndexingLevel (int nIndexingLevel);

	/**
	 * Returns the indexing level description for the parallelism level <code>nParallismLevel</code>.
	 * The deepest indexing level is returned for the highest parallelism level.
	 * @param nParallelismLevel The parallelism level
	 * @return
	 */
	public abstract IIndexingLevel getIndexingLevelFromParallelismLevel (int nParallelismLevel);

	/**
	 * Returns the threading paradigm of the hardware architecture
	 * (i.e. whether the machine implements a multicore or a manycore architecture).
	 * @return The hardware's threading properties
	 */
	public abstract EThreading getThreading ();
}
