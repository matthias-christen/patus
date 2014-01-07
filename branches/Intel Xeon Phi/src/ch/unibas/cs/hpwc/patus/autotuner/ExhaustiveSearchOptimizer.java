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

import ch.unibas.cs.hpwc.patus.util.DomainPointEnumerator;


/**
 *
 * @author Matthias-M. Christen
 */
public class ExhaustiveSearchOptimizer extends AbstractOptimizer
{
	///////////////////////////////////////////////////////////////////
	// Implementation
	
	@Override
	public void optimize (IRunExecutable run)
	{
		int nParamsCount = run.getParametersCount ();
		StringBuilder sbResult = new StringBuilder ();

		DomainPointEnumerator dpe = new DomainPointEnumerator ();
		for (int j = 0; j < nParamsCount; j++)
			dpe.addDimension (run.getParameterLowerBounds ()[j], run.getParameterUpperBounds ()[j]);

		for (int[] rgParams : dpe)
		{
			// execute
			sbResult.setLength (0);
			double fRuntime = run.execute (rgParams, sbResult, checkBounds ());

			// check whether the runtime is better
			if (fRuntime < getResultTiming ())
			{
				setResultTiming (fRuntime);
				setResultParameters (rgParams);
				setProgramOutput (sbResult);
			}
		}
	}

	@Override
	public String getName ()
	{
		return "Exhaustive search";
	}
}
