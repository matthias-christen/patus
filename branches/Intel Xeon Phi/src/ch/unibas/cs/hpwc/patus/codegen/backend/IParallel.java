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

import cetus.hir.Statement;

/**
 * Hardware-/programming model-specific parallelization intrinsics
 * (that are used for stencil kernel code generation).
 *
 * @author Matthias-M. Christen
 */
public interface IParallel
{
	/**
	 * Implements a barrier.
	 */
	public abstract Statement getBarrier (int nParallelismLevel);
}
