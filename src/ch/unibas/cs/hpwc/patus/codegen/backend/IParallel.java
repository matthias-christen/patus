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
	public abstract Statement getBarrier ();
}
