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

import java.util.List;

/**
 *
 * @author Matthias-M. Christen
 */
public interface IRunExecutable
{

	/**
	 * Runs the executable and returns the time that has been measured.
	 * @param rgParams The command line (autotuning) parameters
	 * @param sbResult A {@link StringBuilder} to which the program output is appended, or
	 * 	<code>null</code> if the output is not desired
	 * @return The execution time of the executable with the parameter set <code>rgParams</code>
	 * 	or {@link Double#MAX_VALUE} if an error occurs or the parameter set <code>rgParams</code>
	 * 	is not within the predefined bounds
	 */
	public abstract double execute (int[] rgParams, StringBuilder sbResult, boolean bCheckBounds);

	/**
	 * Returns the number of autotuning parameters.
	 * @return
	 */
	public abstract int getParametersCount ();

	/**
	 * Returns an array of lower bounds for the parameter values.
	 * @return Array of lower bound values
	 */
	public abstract int[] getParameterValueLowerBounds ();

	/**
	 * Returns an array of upper bounds for the parameter values.
	 * @return Array of upper bound values
	 */
	public abstract int[] getParameterValueUpperBounds ();

	/**
	 * Returns the lower bounds for the optimizer.
	 * These are indices into the parameter set defined in this class.
	 * @return The lower indices
	 */
	public abstract int[] getParameterLowerBounds ();

	/**
	 * Returns the upper bounds for the optimizer.
	 * These are indices into the parameter set defined in this class.
	 * @return The upper indices
	 */
	public abstract int[] getParameterUpperBounds ();

	/**
	 * Sets the parameter set. <code>rgParamSet</code> is an array of arrays,
	 * each array contains all the possible values for a parameter.
	 * @param rgParamSets The parameter sets
	 */
	public abstract void setParameterSets (List<int[]> listParamSets);

	/**
	 * Returns the actual values from the optimizer params.
	 * @param rgParamsFromOptimizer
	 * @return
	 */
	public abstract int[] getParameters (int[] rgParamsFromOptimizer);

	/**
	 * Get the number of program runs that have been done.
	 * @return The number of program runs
	 */
	public abstract int getRunsCount ();

	/**
	 * Checks whether the constraints are satisfied for the actual parameters.
	 * @param rgParams
	 * @return
	 */
	boolean areConstraintsSatisfied (int[] rgParams);
}
