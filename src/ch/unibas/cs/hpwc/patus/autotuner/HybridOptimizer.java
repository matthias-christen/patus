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
import java.util.LinkedList;
import java.util.List;

import cetus.hir.Expression;

/**
 *
 * @author Matthias-M. Christen
 */
public class HybridOptimizer implements IOptimizer
{
	///////////////////////////////////////////////////////////////////
	// Inner Types

	private static List<int[]> getIntArrayList (List<ParamSet> listParamSet)
	{
		List<int[]> list = new ArrayList<> (listParamSet.size ());
		for (ParamSet set : listParamSet)
			list.add (set.getParams ());
		return list;
	}

	private static class ExecutionResult
	{
		private int[] m_rgParams;
		private double m_fResult;

		public ExecutionResult (int[] rgParams, double fResult)
		{
			m_rgParams = new int[rgParams.length];
			System.arraycopy (rgParams, 0, m_rgParams, 0, rgParams.length);
			m_fResult = fResult;
		}

		public int[] getParams ()
		{
			return m_rgParams;
		}

		public double getResult ()
		{
			return m_fResult;
		}
	}

	/**
	 * Wrapper class to pass arguments to the {@link HybridOptimizer#optimize(IRunExecutable)} method.
	 */
	public static class HybridRunExecutable extends AbstractRunExecutable
	{
		private String m_strExecutableFilename;
		private List<ParamSet> m_listParamSets;

		private List<int[]> m_listParamMain;
		private List<int[]> m_listParamExhaustive;

		private List<ExecutionResult> m_listExecutionResults;


		public HybridRunExecutable (String strExecutableFilename, List<ParamSet> listParamSets, List<Expression> listConstraints)
		{
			super (getIntArrayList (listParamSets), listConstraints);

			m_strExecutableFilename = strExecutableFilename;
			m_listParamSets = listParamSets;
			m_listConstraints = listConstraints;

			m_listParamMain = new ArrayList<> (listParamSets.size ());
			m_listParamExhaustive = new ArrayList<> (listParamSets.size ());

			m_listExecutionResults = new LinkedList<> ();

			for (ParamSet set : listParamSets)
			{
				if (set.useExhaustive ())
					m_listParamExhaustive.add (set.getParams ());
				else
					m_listParamMain.add (set.getParams ());
			}
		}

		/**
		 * Assembles the total parameter array from the splitted arrays <code>rgParamsMain</code>
		 * (used for the main optimizer, <code>m_optMain</code>) and <code>rgParamsExhaustive</code>
		 * (used for the exhaustive search optimizer).
		 * @param rgParamsMain
		 * @param rgParamsExhaustive
		 * @return
		 */
		public int[] assembleParameters (int[] rgParamsMain, int[] rgParamsExhaustive)
		{
			int[] rgParams = new int[rgParamsMain.length + rgParamsExhaustive.length];

			int i = 0;
			int nIdxMain = 0;
			int nIdxExhaustive = 0;
			for (ParamSet set : m_listParamSets)
			{
				if (set.useExhaustive ())
					rgParams[i] = rgParamsExhaustive[nIdxExhaustive++];
				else
					rgParams[i] = rgParamsMain[nIdxMain++];

				i++;
			}

			return rgParams;
		}

		public String getExecutable ()
		{
			return m_strExecutableFilename;
		}

		public List<int[]> getMainParamSet ()
		{
			return m_listParamMain;
		}

		public List<int[]> getExhaustiveParamSet ()
		{
			return m_listParamExhaustive;
		}

		@Override
		public List<Expression> getConstraints ()
		{
			return m_listConstraints;
		}

		public boolean hasExhaustiveParams ()
		{
			return m_listParamExhaustive.size () > 0;
		}

		@Override
		protected double runPrograms (int[] rgParams, StringBuilder sbResult)
		{
			return 0;
		}

		public void addExecution (int[] rgParams, double fResult)
		{
			m_listExecutionResults.add (new ExecutionResult (rgParams, fResult));
		}

		@Override
		public Histogram<Double, int[]> createHistogram ()
		{
			// create the histogram map
			Histogram<Double, int[]> histogram = new Histogram<> ();
			histogram.setAcceptableRange (0.0, 1.0e50);

			// fill the histogram map
			for (ExecutionResult er : m_listExecutionResults)
				if (er.getResult () != Double.MAX_VALUE)
					histogram.addSample (er.getResult (), er.getParams ());

			return histogram;
		}
	}


	///////////////////////////////////////////////////////////////////
	// Member Variables

	private IOptimizer m_optMain;
	private IOptimizer m_optExhaustive;

	private int[] m_rgResult;
	private double m_fMinRuntime;
	private String m_strProgramOutput;
	private boolean m_bCheckBounds;


	///////////////////////////////////////////////////////////////////
	// Implementation

	public HybridOptimizer (IOptimizer optMain)
	{
		m_optMain = optMain;
		m_optExhaustive = OptimizerFactory.getOptimizer (ExhaustiveSearchOptimizer.class.getName ());
		m_bCheckBounds = true;
		
		m_optMain.setCheckBounds (false);
		m_optExhaustive.setCheckBounds (false);
	}

	@Override
	public void optimize (IRunExecutable run0)
	{
		if (!(run0 instanceof HybridRunExecutable))
			throw new RuntimeException ("The argument to HybridOptimizer.optimize must be of type HybridOptimizer.HybridRunExecutable.");
		final HybridRunExecutable run = (HybridRunExecutable) run0;

		if (run.hasExhaustiveParams ())
		{
			// with exhaustive parameters

			m_fMinRuntime = Double.MAX_VALUE;

			// outer, exhaustive loop
			IRunExecutable runExhaustive = new AbstractRunExecutable (run.getExhaustiveParamSet (), run.getConstraints ())
			{
				@Override
				protected double runPrograms (final int[] rgActualParamsExhaustive, StringBuilder sbResultExhaustive)
				{
					// inner loop with m_optMain
					m_optMain.optimize (new RunExecutable (run.getExecutable (), run.getMainParamSet (), run.getConstraints ())
					{
						@Override
						protected double runPrograms (int[] rgActualParamsInner, StringBuilder sbResult)
						{
							// assemble the parameter list and run the program
							int[] rgActualTotalParams = run.assembleParameters (rgActualParamsInner, rgActualParamsExhaustive);
							if (!OptimizerUtil.areConstraintsSatisfied (rgActualTotalParams, getConstraints ()))
								return Double.MAX_VALUE;
							
							double fResult = super.runPrograms (rgActualTotalParams, sbResult);
							run.addExecution (rgActualTotalParams, fResult);
							return fResult;
						}
					});

					// has a new optimum been found?
					if (m_optMain.getResultTiming () < m_fMinRuntime)
					{
						// find the parameter index for the values in rgActualParamsExhaustive
						int[] rgParamsExhaustive = new int[rgActualParamsExhaustive.length];
						for (int i = 0; i < rgActualParamsExhaustive.length; i++)
						{
							for (int j = 0; j < getParameterSets ()[i].length; j++)
							{
								if (getParameterSets ()[i][j] == rgActualParamsExhaustive[i])
								{
									rgParamsExhaustive[i] = j;
									break;
								}
							}
						}

						m_fMinRuntime = m_optMain.getResultTiming ();
						m_rgResult = run.assembleParameters (m_optMain.getResultParameters (), rgParamsExhaustive);
						m_strProgramOutput = m_optMain.getProgramOutput ();
					}

					return m_fMinRuntime;
				}
			};

			m_optExhaustive.optimize (runExhaustive);
		}
		else
		{
			// without exhaustive parameters: just proxy to the main optimizer
			m_optMain.setCheckBounds (true);
			m_optMain.optimize (new RunExecutable (run.getExecutable (), run.getMainParamSet (), run.getConstraints ())
			{
				@Override
				protected double runPrograms (int[] rgActualParams, StringBuilder sbResult)
				{
					double fResult = super.runPrograms (rgActualParams, sbResult);
					run.addExecution (rgActualParams, fResult);
					return fResult;
				}
			});

			m_fMinRuntime = m_optMain.getResultTiming ();
			m_rgResult = m_optMain.getResultParameters ();
			m_strProgramOutput = m_optMain.getProgramOutput ();
		}
	}

	@Override
	public int[] getResultParameters ()
	{
		return m_rgResult;
	}

	@Override
	public double getResultTiming ()
	{
		return m_fMinRuntime;
	}

	@Override
	public String getProgramOutput ()
	{
		return m_strProgramOutput;
	}

	@Override
	public String getName ()
	{
		return "HybridOptimizer";
	}
	
	@Override
	public boolean checkBounds ()
	{
		return m_bCheckBounds;
	}
	
	@Override
	public void setCheckBounds (boolean bCheckBounds)
	{
		m_bCheckBounds = bCheckBounds;
	}
}
