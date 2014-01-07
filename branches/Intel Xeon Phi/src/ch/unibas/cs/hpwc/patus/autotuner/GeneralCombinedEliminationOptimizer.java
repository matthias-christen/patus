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

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class GeneralCombinedEliminationOptimizer extends AbstractOptimizer
{
	private static class ParamConfig implements Comparable<ParamConfig>
	{
		private int m_nIndex;
		private int m_nParamValue;
		private double m_fRIP;
		
		public ParamConfig (int nIndex, int nParamValue, double fRIP)
		{
			m_nIndex = nIndex;
			m_nParamValue = nParamValue;
			m_fRIP = fRIP;
		}
		
		public int getIndex ()
		{
			return m_nIndex;
		}

		public int getParamValue ()
		{
			return m_nParamValue;
		}

		public double getRIP ()
		{
			return m_fRIP;
		}

		@Override
		public int compareTo (ParamConfig config)
		{
			if (m_fRIP < config.getRIP ())
				return -1;
			if (m_fRIP > config.getRIP ())
				return 1;
			return 0;
		}
	}

	private double m_fCurrentMinimum = Double.MAX_VALUE;
	
	
	private double execute (IRunExecutable run, int[] rgParams)
	{
		StringBuilder sbOutput = new StringBuilder ();
		double fResult = run.execute (rgParams, sbOutput, checkBounds ());
		
		if (fResult < m_fCurrentMinimum)
		{
			m_fCurrentMinimum = fResult;

			setResultTiming (fResult);
			setResultParameters (rgParams);
			setProgramOutput (sbOutput);
		}
		
		return fResult;
	}
	
	private static void copyParams (int[] rgParamsDest, int[] rgParamsSrc)
	{
		System.arraycopy (rgParamsSrc, 0, rgParamsDest, 0, rgParamsSrc.length);
	}
	
	private static void replaceParamValue (int[] rgParamsResult, int[] rgParams, ParamConfig config)
	{
		GeneralCombinedEliminationOptimizer.copyParams (rgParamsResult, rgParams);
		if (config.getIndex () < rgParams.length)
			rgParamsResult[config.getIndex ()] = config.getParamValue ();
	}
	
	@Override
	public void optimize (IRunExecutable run)
	{
		// initialize the index set
		Set<Integer> setIndexSet = new HashSet<> ();
		for (int i = 0; i < run.getParametersCount (); i++)
			setIndexSet.add (i);
		
		// represent the set X
		List<ParamConfig> listBest = new ArrayList<> ();
		
		int[] rgParams = new int[run.getParametersCount ()];
		int[] rgParamsNew = new int[run.getParametersCount ()];
		
		// initialize the base
		OptimizerUtil.getRandomPointWithinBounds (rgParams, run.getParameterLowerBounds (), run.getParameterUpperBounds ());
		
		do
		{
			listBest.clear ();
			
			// measure the RIPs relative to the start value for all parameters and parameter values
			double fBaseObjVal = execute (run, rgParams);
			copyParams (rgParamsNew, rgParams);
			for (int i : setIndexSet)
			{
				double fMinRIP = Double.MAX_VALUE;
				int nMinXi = Integer.MAX_VALUE;
				
				for (int nXi = run.getParameterLowerBounds ()[i]; nXi <= run.getParameterUpperBounds ()[i]; nXi++)
				{
					rgParamsNew[i] = nXi;
					double fRIP = (execute (run, rgParamsNew) - fBaseObjVal) / fBaseObjVal;
					if (fRIP < 0 && fRIP < fMinRIP)
					{
						fMinRIP = fRIP;
						nMinXi = nXi;
					}					
				}

				if (nMinXi != Integer.MAX_VALUE)
					listBest.add (new ParamConfig (i, nMinXi, fMinRIP));

				// restore the old parameter configuration
				rgParamsNew[i] = rgParams[i];
			}
			
			Collections.sort (listBest);
			for (ParamConfig config : listBest)
			{
				GeneralCombinedEliminationOptimizer.replaceParamValue (rgParamsNew, rgParams, config);
				double fNewVal = execute (run, rgParamsNew);
	
				if (fNewVal < fBaseObjVal)
				{
					setIndexSet.remove (config.getIndex ());
					copyParams (rgParams, rgParamsNew);
					fBaseObjVal = fNewVal;
				}
			}
		} while (listBest.size () > 0);
	}
	
	@Override
	public String getName ()
	{
		return "General Combined Elimination";
	}
}
