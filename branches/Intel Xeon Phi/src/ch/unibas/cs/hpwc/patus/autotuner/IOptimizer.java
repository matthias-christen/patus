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
package ch.unibas.cs.hpwc.patus.autotuner;

/**
 *
 * @author Matthias-M. Christen
 */
public interface IOptimizer
{
	/**
	 * Runs the optimizer for the executable <code>run</code>.
	 * @param run The executable
	 */
	public abstract void optimize (IRunExecutable run);

	/**
	 * Returns the optimal result parameters (values of the &quot;decision variables&quot;).
	 * @return The result parameter set
	 */
	public abstract int[] getResultParameters ();

	/**
	 * Returns the timing for the optimal parameters determined by the optimizer.
	 * @return The timing for the optimal parameter set
	 */
	public abstract double getResultTiming ();

	/**
	 * Returns the program output for the optimum found.
	 * @return The program output for the optimal run
	 */
	public abstract String getProgramOutput ();

	/**
	 * Returns the optimizer's / method name.
	 * @return The optimizer name
	 */
	public abstract String getName ();
	
	public abstract boolean checkBounds ();
	
	public abstract void setCheckBounds (boolean bCheckBounds);
}
