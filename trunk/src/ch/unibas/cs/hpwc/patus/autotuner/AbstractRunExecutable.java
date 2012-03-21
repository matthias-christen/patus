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

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import cetus.hir.Expression;
import ch.unibas.cs.hpwc.patus.util.IntArray;

/**
 * <p>Abstract implementation of {@link IRunExecutable}.</p>
 * <p>Does not define the actual program execution, but provides constraint checking and
 * caching of program results, and maps parameter ranges between the ones used in the
 * auto-tuner and the actual program command line arguments.</p> 
 * 
 * @author Matthias-M. Christen
 */
public abstract class AbstractRunExecutable implements IRunExecutable
{
	///////////////////////////////////////////////////////////////////
	// Constants

	//private final static Logger LOGGER = Logger.getLogger (AbstractRunExecutable.class);


	///////////////////////////////////////////////////////////////////
	// Inner Types
	
	private class ProgramExecutionResult
	{
		private double m_fExecutionTime;
		private String m_strProgramOutput;
		
		public ProgramExecutionResult (double fExecutionTime, String strProgramOutput)
		{
			m_fExecutionTime = fExecutionTime;
			m_strProgramOutput = strProgramOutput;
		}

		public double getExecutionTime ()
		{
			return m_fExecutionTime;
		}

		public String getProgramOutput ()
		{
			return m_strProgramOutput;
		}
	}

	
	///////////////////////////////////////////////////////////////////
	// Member Variables

	protected int[] m_rgParamValueLowerBounds;
	protected int[] m_rgParamValueUpperBounds;

	protected int[] m_rgParamsLowerBounds;
	protected int[] m_rgParamsUpperBounds;

	/**
	 * The sets of parameters; the first dimension are the distinct parameters,
	 * the second dimension describe all possible values for one parameter
	 */
	protected int[][] m_rgParamsSets;

	protected List<Expression> m_listConstraints;

	protected Map<IntArray, ProgramExecutionResult> m_mapCachedExecutions;
	

	///////////////////////////////////////////////////////////////////
	// Implementation

	public AbstractRunExecutable (List<int[]> listParamSets, List<Expression> listConstraints)
	{
		setParameterSets (listParamSets);
		m_listConstraints = listConstraints;

		m_mapCachedExecutions = new HashMap<> ();
	}

	/* (non-Javadoc)
	 * @see ch.unibas.cs.hpwc.patus.autotuner.IRunExecutable#getParametersCount()
	 */
	@Override
	public int getParametersCount ()
	{
		return m_rgParamsLowerBounds.length;
	}

	/* (non-Javadoc)
	 * @see ch.unibas.cs.hpwc.patus.autotuner.IRunExecutable#getParameterValueLowerBounds()
	 */
	@Override
	public int[] getParameterValueLowerBounds ()
	{
		return m_rgParamValueLowerBounds;
	}

	/* (non-Javadoc)
	 * @see ch.unibas.cs.hpwc.patus.autotuner.IRunExecutable#getParameterValueUpperBounds()
	 */
	@Override
	public int[] getParameterValueUpperBounds ()
	{
		return m_rgParamValueUpperBounds;
	}

	/* (non-Javadoc)
	 * @see ch.unibas.cs.hpwc.patus.autotuner.IRunExecutable#getParameterLowerBounds()
	 */
	@Override
	public int[] getParameterLowerBounds ()
	{
		return m_rgParamsLowerBounds;
	}

	/* (non-Javadoc)
	 * @see ch.unibas.cs.hpwc.patus.autotuner.IRunExecutable#getParameterUpperBounds()
	 */
	@Override
	public int[] getParameterUpperBounds ()
	{
		return m_rgParamsUpperBounds;
	}

	/* (non-Javadoc)
	 * @see ch.unibas.cs.hpwc.patus.autotuner.IRunExecutable#setParameterSets(java.util.List)
	 */
	@Override
	public void setParameterSets (List<int[]> listParamSets)
	{
		m_rgParamsSets = new int[listParamSets.size ()][];
		int i = 0;
		for (int[] rgParams : listParamSets)
			m_rgParamsSets[i++] = rgParams;

		// set the upper and lower bounds
		m_rgParamsLowerBounds = new int[m_rgParamsSets.length];
		m_rgParamsUpperBounds = new int[m_rgParamsSets.length];
		m_rgParamValueLowerBounds = new int[m_rgParamsSets.length];
		m_rgParamValueUpperBounds = new int[m_rgParamsSets.length];

		for (i = 0; i < m_rgParamsSets.length; i++)
		{
			m_rgParamValueLowerBounds[i] = m_rgParamsSets[i][0];
			m_rgParamValueUpperBounds[i] = m_rgParamsSets[i][m_rgParamsSets[i].length - 1];
			m_rgParamsLowerBounds[i] = 0;
			m_rgParamsUpperBounds[i] = m_rgParamsSets[i].length - 1;
		}
	}

	/* (non-Javadoc)
	 * @see ch.unibas.cs.hpwc.patus.autotuner.IRunExecutable#getParameters(int[])
	 */
	@Override
	public int[] getParameters (int[] rgParamsFromOptimizer)
	{
		int[] rgValues = new int[rgParamsFromOptimizer.length];
		for (int i = 0; i < rgParamsFromOptimizer.length; i++)
			rgValues[i] = m_rgParamsSets[i][rgParamsFromOptimizer[i]];
		return rgValues;
	}

	public List<Expression> getConstraints ()
	{
		return m_listConstraints;
	}

	protected int[][] getParameterSets ()
	{
		return m_rgParamsSets;
	}
	
	@Override
	public boolean areConstraintsSatisfied (int[] rgParams)
	{
		int[] rgActualParams = new int[rgParams.length];
		for (int i = 0; i < rgParams.length; i++)
			rgActualParams[i] = getParameterSets ()[i][rgParams[i]];
		return OptimizerUtil.areConstraintsSatisfied (rgActualParams, getConstraints ());
	}

	@Override
	public double execute (int[] rgParams, StringBuilder sbResult, boolean bCheckBounds)
	{
		ProgramExecutionResult cached = m_mapCachedExecutions.get (new IntArray (rgParams));
		if (cached != null)
		{
			if (sbResult != null)
				sbResult.append (cached.getProgramOutput ());
			return cached.getExecutionTime ();
		}
		
		if (!OptimizerUtil.isWithinBounds (rgParams, getParameterLowerBounds (), getParameterUpperBounds ()))
			return Double.MAX_VALUE;

		int[] rgActualParams = new int[rgParams.length];
		for (int i = 0; i < rgParams.length; i++)
			rgActualParams[i] = getParameterSets ()[i][rgParams[i]];
		if (bCheckBounds && !OptimizerUtil.areConstraintsSatisfied (rgActualParams, getConstraints ()))
			return Double.MAX_VALUE;

		double fResult = runPrograms (rgActualParams, sbResult);
		m_mapCachedExecutions.put (new IntArray (rgParams, true), new ProgramExecutionResult (fResult, sbResult == null ? "" : sbResult.toString ()));

		return fResult;
	}

	/**
	 * Returns an execution time histogram as a map data structure.
	 * @return
	 */
	public Histogram<Double, int[]> createHistogram ()
	{
		// create the histogram map
		Histogram<Double, int[]> histogram = new Histogram<> ();
		histogram.setAcceptableRange (0.0, 1.0e50);

		// fill the histogram map
		for (IntArray arrParams : m_mapCachedExecutions.keySet ())
		{
			// convert to actual parameters
			int[] rgActualParams = new int[arrParams.length ()];
			for (int i = 0; i < arrParams.length (); i++)
				rgActualParams[i] = getParameterSets ()[i][arrParams.get (i)];

			double fVal = m_mapCachedExecutions.get (arrParams).getExecutionTime ();
			if (fVal != Double.MAX_VALUE)
				histogram.addSample (fVal, rgActualParams);
		}

		return histogram;
	}

	protected abstract double runPrograms (int[] rgParams, StringBuilder sbResult);
	
	@Override
	public int getRunsCount ()
	{
		return m_mapCachedExecutions.size ();
	}
}
