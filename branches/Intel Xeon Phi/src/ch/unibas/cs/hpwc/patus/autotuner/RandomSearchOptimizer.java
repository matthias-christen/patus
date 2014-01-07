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


public class RandomSearchOptimizer extends AbstractOptimizer
{
	private final static int MAX_SEARCHES = 500;
	
	private int[] m_rgCurrentParams = null;
		
	
	@Override
	public void optimize (IRunExecutable run)
	{
		StringBuilder sbResult = new StringBuilder ();
		for (int i = 0; i < MAX_SEARCHES; i++)
		{
			createNewParams (run);
						
			sbResult.setLength (0);
			double fObjValue = run.execute (m_rgCurrentParams, sbResult, checkBounds ());
			
			OptimizerLog.step (i, m_rgCurrentParams, fObjValue);

			if (fObjValue < getResultTiming ())
			{
				setResultTiming (fObjValue);
				setResultParameters (m_rgCurrentParams);				
				setProgramOutput (sbResult);
			}
		}
	}
	
	private void createNewParams (IRunExecutable run)
	{
		if (m_rgCurrentParams == null)
			m_rgCurrentParams = new int[run.getParametersCount ()];

		OptimizerUtil.getRandomPointWithinBounds (m_rgCurrentParams, run.getParameterLowerBounds (), run.getParameterUpperBounds ());
	}

	@Override
	public String getName ()
	{
		return "RandomSearch";
	}
}
