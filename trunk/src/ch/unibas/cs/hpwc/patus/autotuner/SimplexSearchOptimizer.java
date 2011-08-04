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

import java.util.Arrays;

import org.apache.log4j.Logger;

import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 * This class represents the routine to find a minimum of multidimensional
 * function f with the simplex method (aka polytope method, or Nelder-Mead
 * algorithm) in mutidimensional. The main references used for this
 * implementation are: <BR>
 * <ul>
 * 	<li>P.E. Gill, W. Murray and M.H. Wright, Practical Optimization, Academic Press</li>
 * 	<li>J.C. Lagarias, J.A. Reeds, M.H. Wright and P.E. Wright, Convergence Properites of the Nelder-Mead Simplex Method
 * 		in Low Dimension, SIAM Journal of Optimization, Vol 9, Number 1, pp 112-147, 1998</li>
 * </ul>
 *
 * @author Pierre-Yves Mignotte, Markus Schmies, Vitali Lieder: JTEM - Java Tools for Experimental Mathematics
 * @author Matthias Christen (Adaptation for Patus)
 */
public class SimplexSearchOptimizer extends AbstractOptimizer
{
	///////////////////////////////////////////////////////////////////
	// Inner Types

	private enum ETransformationType
	{
		EXPAND,
		REFLECT,
		CONTRACT_OUTSIDE,
		CONTRACT_INSIDE
	}


	///////////////////////////////////////////////////////////////////
	// Constants

	private final static Logger LOGGER = Logger.getLogger (SimplexSearchOptimizer.class);


	/**
	 * Number of optimizer runs (each with a different starting point)
	 */
	private final static int NUM_RUNS = 5;

	/**
	 * maximum number of function evaluation in search method.
	 */
	private final static int MAX_ITER = 1000;

	private final static double TOLERANCE = 0.001;

	/**
	 * Reflection parameter (Default = 1.0)
	 */
	private final static double RHO = 1.0D;

	/**
	 * Expansion parameter (Default = 2.0)
	 */
	private final static double CHI = 2.0D;

	/**
	 * Contraction parameter (Default = 0.5)
	 */
	private final static double GAMMA = 0.5D;

	/**
	 * Shrink parameter (Default = 0.5)
	 */
	private final static double SIGMA = 0.5D;


	private static final boolean KEEP_IN_BOUNDS = true;


	///////////////////////////////////////////////////////////////////
	// Inner Classes
	
	private class Vertex implements Comparable<Vertex>
	{
		private int[] m_rgCoords;
		private double m_fObjValue;
		private StringBuilder m_sbProgramOutput;
		
		public Vertex ()
		{
			m_rgCoords = new int[m_nParametersCount];
			m_fObjValue = Double.MAX_VALUE;
			m_sbProgramOutput = new StringBuilder ();
		}
		
		public Vertex (int[] rgStartConfiguration)
		{
			m_rgCoords = new int[m_nParametersCount];
			System.arraycopy (rgStartConfiguration, 0, m_rgCoords, 0, rgStartConfiguration.length);
			m_fObjValue = Double.MAX_VALUE;
			m_sbProgramOutput = new StringBuilder ();
		}

		public int[] getCoords ()
		{
			return m_rgCoords;
		}
		
		public double getObjValue ()
		{
			return m_fObjValue;
		}
		
		@Override
		public int compareTo (Vertex vOther)
		{
			if (m_fObjValue < vOther.getObjValue ())
				return -1;
			if (m_fObjValue > vOther.getObjValue ())
				return 1;
			return 0;
		}

		public void setAsCurrentBest ()
		{
			setResultTiming (m_fObjValue);
			setResultParameters (m_rgCoords);
			setProgramOutput (m_sbProgramOutput);
		}

		public int getCoord (int nCoord)
		{
			return m_rgCoords[nCoord];
		}

		public void setCoord (int nCoord, int nValue)
		{
			m_rgCoords[nCoord] = nValue;
		}

		public void offsetCoord (int nCoord, int nOffset)
		{
			m_rgCoords[nCoord] += nOffset;
		}

		public boolean hasEqualCoords (Vertex vOther)
		{
			return Arrays.equals (m_rgCoords, vOther.getCoords ());
		}

		public void execute ()
		{
			m_sbProgramOutput.setLength (0);
			m_fObjValue = m_run.execute (m_rgCoords, m_sbProgramOutput, checkBounds ());
		}

		public int getDim ()
		{
			return m_rgCoords.length;
		}
		
		@Override
		public String toString ()
		{
			return StringUtil.concat ("(", Arrays.toString (m_rgCoords), ") [", m_fObjValue, "]");
		}
		
		public Vertex clone ()
		{
			Vertex vx = new Vertex (m_rgCoords);
			vx.m_fObjValue = m_fObjValue;
			vx.m_sbProgramOutput.append (m_sbProgramOutput);
			
			return vx;
		}
	}

	
	///////////////////////////////////////////////////////////////////
	// Member Variables

	private IRunExecutable m_run;
	private int m_nParametersCount;

	private int m_nIterationNum;
	private Vertex[] m_rgVertices;
	private Vertex[] m_rgVerticesOld;

	/**
	 * The centroid vertex
	 */
	private double[] m_rgVertexCentroid;

	/**
	 * The type of the last transform
	 */
	ETransformationType m_typeLastTransform;


	///////////////////////////////////////////////////////////////////
	// Implementation

	@Override
	public void optimize (IRunExecutable run)
	{
		// initialize
		m_run = run;

		for (int i = 0; i < NUM_RUNS; i++)
		{
			m_nIterationNum = 0;

			// initialize the optimizer with a new starting point
			initialize (OptimizerUtil.getRandomPointWithinBounds (m_run.getParameterLowerBounds (), m_run.getParameterUpperBounds ()));

			// start the optimizer
			search ();

			// has a new minimum been found?
			if (m_rgVertices[0].getObjValue () < getResultTiming ())
				m_rgVertices[0].setAsCurrentBest ();
		}
	}

	protected void initialize (int[] rgStartConfiguration)
	{
		m_nParametersCount = m_run.getParametersCount ();

		// create objects
		m_rgVertices = new Vertex[m_nParametersCount + 1];
		m_rgVerticesOld = null;

		// initialize the vertices
		for (int j = 0; j < m_nParametersCount; j++)
			m_rgVertices[0] = new Vertex (rgStartConfiguration);

		// MATLAB version for vertices initialization;
		// original suggestion of L. Pfeffer (Stanford)
		for (int i = 0; i < m_nParametersCount; i++)
		{
			m_rgVertices[i + 1] = new Vertex (rgStartConfiguration);

			if (m_rgVertices[i + 1].getCoord (0) == 0)
				m_rgVertices[i + 1].setCoord (i, m_run.getParameterUpperBounds ()[i] / 2);
			else
			{
				int nOffset = Math.max (m_rgVertices[i + 1].getCoord (i) / 2, 1);
				
				if (m_rgVertices[i + 1].getCoord (i) <= m_run.getParameterUpperBounds ()[i])
					m_rgVertices[i + 1].offsetCoord (i, nOffset);
				else
				{
					if (m_rgVertices[i + 1].getCoord (i) - nOffset >= m_run.getParameterLowerBounds ()[i])
						m_rgVertices[i + 1].offsetCoord (i, -nOffset);
					else
					{
						if (Math.random () < 0.5)
							m_rgVertices[i + 1].setCoord (i, m_run.getParameterUpperBounds ()[i]);
						else
							m_rgVertices[i + 1].setCoord (i, m_run.getParameterLowerBounds ()[i]);
					}
				}
			}
		}
		
		// make sure vertices are distinct
		int nConfigsCount = 1;
		for (int i = 0; i < m_nParametersCount; i++)
			nConfigsCount *= m_run.getParameterUpperBounds ()[i] - m_run.getParameterLowerBounds ()[i] + 1;
		if (nConfigsCount >= m_nParametersCount + 1)
		{
			for (int i = 1; i < m_nParametersCount + 1; i++)
			{
				for (int j = 0; j < i; j++)
				{
					if (m_rgVertices[i].hasEqualCoords (m_rgVertices[j]))
					{
						// two vertices are equal; try to modify vertex i such that it is distinct from the other ones
						OptimizerUtil.getNextConfigInPlace (m_rgVertices[i].getCoords (), m_run.getParameterLowerBounds (), m_run.getParameterUpperBounds ());
						
						// repeat the check
						i--;
						break;
					}
				}
			}
		}

		m_rgVertexCentroid = new double[m_nParametersCount];
	}

	/**
	 * Search the minimum of function f with Simplex method in initial basis xi
	 * and return the value of the minimum. <BR>
	 * xi is used to create the first vertices. The set of vertex is p0 and the
	 * n vertices where one variable of p0 is modified. If p0[i] != 0, then
	 * pi[i] = p0[i]*(1+0.05*xi[i][i]), ie +5% of itself. If p0[i] == 0 then we
	 * prefer to add 0.00025*xi[i][i]. Sometimes, this value doesn't allow to
	 * reach the optimum if it's too far.
	 */
	protected Vertex search ()
	{
		Vertex vxReflected = null;
		Vertex vxExpanded = null;
		Vertex vxContracted = null;

		// Evaluation for each node of the vertex
		evaluate ();

		// sort the nodes according to the evaluation
		Arrays.sort (m_rgVertices);

		while ((Math.abs (m_rgVertices[m_nParametersCount].getObjValue () - m_rgVertices[0].getObjValue ()) > TOLERANCE) && (m_nIterationNum++ < MAX_ITER) && !isConverged ())
		{
			// Compute of the centroid of the best vertices
			computeCentroid ();

			vxReflected = reflect ();
			if (vxReflected.getObjValue () < m_rgVertices[0].getObjValue ())
			{
				// first case: expansion
				vxExpanded = expand ();
				if (vxExpanded.getObjValue () < vxReflected.getObjValue ())
				{
					// accept expanded
					accept (vxExpanded, ETransformationType.EXPAND);
				}
				else
				{
					// accept reflected
					accept (vxReflected, ETransformationType.REFLECT);
				}
			}
			else
			{
				// second case: no expansion
				if (vxReflected.getObjValue () >= m_rgVertices[m_nParametersCount - 1].getObjValue ())
				{
					// contraction
					if (vxReflected.getObjValue () < m_rgVertices[m_nParametersCount].getObjValue ())
					{
						// contract outside
						vxContracted = contractOutside ();
						if (vxContracted.getObjValue () <= vxReflected.getObjValue ())
						{
							// accept contracted
							accept (vxContracted, ETransformationType.CONTRACT_OUTSIDE);
						}
						else
							shrink ();
					}
					else
					{
						// contract inside
						vxContracted = contractInside ();
						if (vxContracted.getObjValue () < vxReflected.getObjValue ())
						{
							// accept contracted
							accept (vxContracted, ETransformationType.CONTRACT_INSIDE);
						}
						else
							shrink ();
					}
				}
				else
				{
					// accept reflected (not reflected)
					accept (vxReflected, ETransformationType.REFLECT);
				}
			}

			// output progress information
			OptimizerLog.step (m_nIterationNum, m_rgVertices[0].getCoords (), m_rgVertices[0].getObjValue (),
				m_typeLastTransform == null ? "" : m_typeLastTransform.toString ());

			// sort for next iteration
			Arrays.sort (m_rgVertices);
		}

		OptimizerLog.terminate (m_nIterationNum, MAX_ITER);

		// by construction, the solution is in m_rgVertices[0]
		return m_rgVertices[0];
	}

	/**
	 * Run the executable for each configuration.
	 */
	private void evaluate ()
	{
		for (int i = 0; i < m_nParametersCount + 1; i++)
			m_rgVertices[i].execute ();
	}

	/**
	 * Accepts the result of a transformation.
	 * @param rgVertex
	 * @param fObjValue
	 * @param type
	 */
	private void accept (Vertex vx, ETransformationType type)
	{
		m_rgVertices[m_nParametersCount] = vx;
		m_typeLastTransform = type;
	}

	/**
	 * Compute the centroid of the best n vertices.
	 * The method will write the centroid vertex to {@link SimplexSearch#m_rgVertexCentroid}.
	 */
	private void computeCentroid ()
	{
		for (int i = 0; i < m_nParametersCount; i++)
		{
			m_rgVertexCentroid[i] = 0;
			for (int j = 0; j < m_nParametersCount; j++)
				m_rgVertexCentroid[i] += m_rgVertices[j].getCoord (i);
			m_rgVertexCentroid[i] /= m_nParametersCount;
		}
	}

	/**
	 * Reflects a vertex.
	 * @param rgVertexReflected
	 */
	private Vertex reflect ()
	{
		Vertex vx = new Vertex ();
		for (int i = 0; i < m_nParametersCount; i++)
			vx.setCoord (i, (int) Math.round ((1 + RHO) * m_rgVertexCentroid[i] - RHO * m_rgVertices[m_nParametersCount].getCoord (i)));
		
		if (KEEP_IN_BOUNDS)
			OptimizerUtil.adjustToBounds (vx.getCoords (), m_run.getParameterLowerBounds (), m_run.getParameterUpperBounds ());
		
		vx.execute ();
		return vx;
	}

	/**
	 * Expands a vertex.
	 * @param rgVertexExpanded
	 */
	private Vertex expand ()
	{
		Vertex vx = new Vertex ();
		for (int i = 0; i < m_nParametersCount; i++)
			vx.setCoord (i, (int) Math.round ((1 + RHO * CHI) * m_rgVertexCentroid[i] - RHO * CHI * m_rgVertices[m_nParametersCount].getCoord (i)));

		if (KEEP_IN_BOUNDS)
			OptimizerUtil.adjustToBounds (vx.getCoords (), m_run.getParameterLowerBounds (), m_run.getParameterUpperBounds ());
		
		vx.execute ();
		return vx;
	}

	/**
	 *
	 * @param rgVertexContracted
	 * @return
	 */
	private Vertex contractOutside ()
	{
		Vertex vx = new Vertex ();
		for (int i = 0; i < m_nParametersCount; i++)
			vx.setCoord (i, (int) Math.round ((1 + RHO * GAMMA) * m_rgVertexCentroid[i] - RHO * GAMMA * m_rgVertices[m_nParametersCount].getCoord (i)));
		
		if (KEEP_IN_BOUNDS)
			OptimizerUtil.adjustToBounds (vx.getCoords (), m_run.getParameterLowerBounds (), m_run.getParameterUpperBounds ());

		vx.execute ();
		return vx;
	}

	/**
	 *
	 * @param rgVertexContracted
	 * @return
	 */
	private Vertex contractInside ()
	{
		Vertex vx = new Vertex ();
		for (int i = 0; i < m_nParametersCount; i++)
			vx.setCoord (i, (int) Math.round ((1 - GAMMA) * m_rgVertexCentroid[i] + GAMMA * m_rgVertices[m_nParametersCount].getCoord (i)));

		if (KEEP_IN_BOUNDS)
			OptimizerUtil.adjustToBounds (vx.getCoords (), m_run.getParameterLowerBounds (), m_run.getParameterUpperBounds ());

		vx.execute ();
		return vx;
	}

	/**
	 * Perform the shrink step.
	 */
	private void shrink ()
	{
		Arrays.sort (m_rgVertices);

		for (int i = 1; i < m_rgVertices.length; i++)
		{
			for (int j = 0; j < m_rgVertices[0].getDim (); j++)
				m_rgVertices[i].setCoord (j, (int) Math.round ((1 - SIGMA) * m_rgVertices[0].getCoord (j) + SIGMA * m_rgVertices[i].getCoord (j)));
			
			if (KEEP_IN_BOUNDS)
				OptimizerUtil.adjustToBounds (m_rgVertices[i].getCoords (), m_run.getParameterLowerBounds (), m_run.getParameterUpperBounds ());

			m_rgVertices[i].execute ();
		}

		Arrays.sort (m_rgVertices);
	}

	private void copyToOld ()
	{
		for (int i = 0; i < m_rgVertices.length; i++)
			m_rgVerticesOld[i] = m_rgVertices[i].clone ();
	}

	/**
	 * Determines whether the vertices haven't changed since the last iteration.
	 * @return <code>true</code> iff the vertices haven't changed since the last iteration
	 */
	private boolean isConverged ()
	{
		if (m_rgVerticesOld == null)
		{
			m_rgVerticesOld = new Vertex[m_nParametersCount + 1];
			copyToOld ();
			return false;
		}

		for (int i = 0; i < m_rgVertices.length; i++)
			if (!m_rgVertices[i].hasEqualCoords (m_rgVerticesOld[i]))
			{
				copyToOld ();
				return false;
			}

		// all entries of old and new vertices match: converged
		LOGGER.info (StringUtil.concat ("Converged at config ", m_rgVertices[0]));
		return true;
	}

	@Override
	public String getName ()
	{
		return "Simplex search";
	}	
}
