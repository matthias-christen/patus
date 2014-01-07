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

import org.apache.log4j.Logger;

import ch.unibas.cs.hpwc.patus.util.StringUtil;



/**
 * Optimizer that uses the Powell method (orthogonal optimization,
 * <a href="http://math.fullerton.edu/mathews/n2003/PowellMethodMod.html">
 * http://math.fullerton.edu/mathews/n2003/PowellMethodMod.html</a>)
 * to find a minimum.
 *
 * @author Matthias-M. Christen
 */
public class GreedyOptimizer extends AbstractOptimizer
{
	///////////////////////////////////////////////////////////////////
	// Constants

	private static final int MAX_RUNS = 5;

	private static final Logger LOGGER = Logger.getLogger (GreedyOptimizer.class);


	///////////////////////////////////////////////////////////////////
	// Implementation

	@Override
	public void optimize (IRunExecutable run)
	{
		int nParamsCount = run.getParametersCount ();
		int[] rgResult = new int[nParamsCount];
		StringBuilder sbResult = new StringBuilder ();

		System.arraycopy (run.getParameterLowerBounds (), 0, rgResult, 0, nParamsCount);

		for (int i = 0; i < MAX_RUNS; i++)
		{
			LOGGER.info (StringUtil.concat ("Run ", String.valueOf (i)));

			for (int nDir = 0; nDir < nParamsCount; nDir++)
			{
				int nMinParam = rgResult[nDir];
				for (int nVal = run.getParameterLowerBounds ()[nDir]; nVal <= run.getParameterUpperBounds ()[nDir]; nVal++)
				{
					// execute
					rgResult[nDir] = nVal;
					sbResult.setLength (0);
					double fRuntime = run.execute (rgResult, sbResult, checkBounds ());

					// check whether the runtime is better
					if (fRuntime < getResultTiming ())
					{
						setResultTiming (fRuntime);
						setResultParameters (rgResult);
						setProgramOutput (sbResult);
						nMinParam = nVal;
					}
				}

				// fix the param
				rgResult[nDir] = nMinParam;
			}
		}
	}

	@Override
	public String getName ()
	{
		return "Powell search method";
	}
}
