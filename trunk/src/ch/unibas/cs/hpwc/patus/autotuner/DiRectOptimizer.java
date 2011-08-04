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
 * <p>Implements the DIRECT algorithm.</p>
 * 
 * <b>Slides:</b>
 * <ul>
 * <li><a href="http://www.inf.ethz.ch/personal/ybrise/data/talks/msem20080401.pdf">http://www.inf.ethz.ch/personal/ybrise/data/talks/msem20080401.pdf</a></li>
 * <li><a href="http://www.scai.fraunhofer.de/fileadmin/ArbeitsgruppeTrottenberg/WS0809/seminar/Lammert.pdf">http://www.scai.fraunhofer.de/fileadmin/ArbeitsgruppeTrottenberg/WS0809/seminar/Lammert.pdf</a></li>
 * </ul>
 * 
 * <b>Literature:</b>
 * <ul>
 * <li><a href="http://www4.ncsu.edu/~ctk/Finkel_Direct/DirectUserGuide_pdf.pdf">http://www4.ncsu.edu/~ctk/Finkel_Direct/DirectUserGuide_pdf.pdf</a></li>
 * <li><a href="http://www.springerlink.com/content/kn467t1876721411/">http://www.springerlink.com/content/kn467t1876721411/</a></li>
 * </ul>
 * 
 * @author Matthias-M. Christen
 */
public class DiRectOptimizer extends AbstractOptimizer
{
	///////////////////////////////////////////////////////////////////
	// Constants

	private final static int MAX_SEGMENTS = 3;
	
	private final static Logger LOGGER = Logger.getLogger (DiRectOptimizer.class);
	
	
	///////////////////////////////////////////////////////////////////
	// Inner Types

	private class Rectangle
	{
		private int[] m_rgMin;
		private int[] m_rgMax;
		private double m_fMinVal;
		
		
		public Rectangle (int[] rgMin, int[] rgMax)
		{
			m_rgMin = rgMin;
			m_rgMax = rgMax;
			m_fMinVal = Double.MAX_VALUE;
		}
		
		private void evaluateRecursively (IRunExecutable run, int[] rgBase)
		{
			if (rgBase.length == m_rgMin.length)
			{
				// rgBase has been fully constructed: execute
				
				StringBuilder sbResult = new StringBuilder ();
				double fVal = run.execute (rgBase, sbResult, checkBounds ());
				
				if (fVal < getResultTiming ())
				{
					setResultTiming (fVal);
					setResultParameters (rgBase);
					setProgramOutput (sbResult);
				}
				
				m_fMinVal = Math.min (m_fMinVal, fVal);
			}
			else
			{
				// not yet fully constructed: add next coordinate
				
				int[] rgCoords0 = new int[rgBase.length + 1];
				System.arraycopy (rgBase, 0, rgCoords0, 0, rgBase.length);
				rgCoords0[rgBase.length] = m_rgMin[rgBase.length];
				evaluateRecursively (run, rgCoords0);
	
				if (m_rgMin[rgBase.length] != m_rgMax[rgBase.length])
				{
					int[] rgCoords1 = new int[rgBase.length + 1];
					System.arraycopy (rgBase, 0, rgCoords1, 0, rgBase.length);
					rgCoords1[rgBase.length] = m_rgMax[rgBase.length];			
					evaluateRecursively (run, rgCoords1);
				}
			}
		}
		
		public void evaluate (IRunExecutable run, int nParts, boolean bDoSubdivision)
		{
			if (LOGGER.isDebugEnabled ())
				LOGGER.debug (StringUtil.concat ("Evaluating ", toString ()));

			m_fMinVal = Double.MAX_VALUE;
			evaluateRecursively (run, new int[] { });
			
			if (LOGGER.isDebugEnabled ())
				LOGGER.debug (StringUtil.concat ("Evaluated ", toString ()));

			// do new subdivision
			if (bDoSubdivision && m_fMinVal < getResultTiming () * 1.5)
				subdivide (run, nParts);
		}
		
		private void subdivideRecursively (IRunExecutable run, int nParts, int[] rgMinBase, int[] rgMaxBase)
		{
			if (rgMinBase.length == m_rgMin.length)
			{
				// fully constructed
				new Rectangle (rgMinBase, rgMaxBase).evaluate (run, 2, true);
			}
			else
			{
				final int n = rgMinBase.length;
				final int nStep = m_rgMin[n] == m_rgMax[n] ? 0 : Math.max (1, (m_rgMax[n] - m_rgMin[n]) / nParts);
				
				int nMin = m_rgMin[n];
				int nMax = Math.min (nMin + nStep, m_rgMax[n]);
				do
				{
					int[] rgMin = new int[n + 1];
					int[] rgMax = new int[n + 1];
					
					System.arraycopy (rgMinBase, 0, rgMin, 0, n);
					rgMin[n] = nMin;
					
					System.arraycopy (rgMaxBase, 0, rgMax, 0, n);
					rgMax[n] = nMax;
					
					subdivideRecursively (run, nParts, rgMin, rgMax);
					
					nMin += nStep;
					nMax = Math.min (nMin + nStep, m_rgMax[n]);
				} while (nMin + nStep < m_rgMax[n]);				
			}
		}
		
		public void subdivide (IRunExecutable run, int nParts)
		{
			if (LOGGER.isDebugEnabled ())
				LOGGER.debug (StringUtil.concat ("Subdividing ", toString ()));
			
			// check if the rectangle can be subdivided
			boolean bCanSubdivide = false;
			for (int i = 0; i < m_rgMin.length; i++)
			{
				if (m_rgMax[i] - m_rgMin[i] > 1)
				{
					bCanSubdivide = true;
					break;
				}
			}
			
			// if it can, do the subdivision, otherwise just evaluate
			if (bCanSubdivide)
				subdivideRecursively (run, nParts, new int[] { }, new int[] { });
			else
				evaluate (run, nParts, false);
		}
		
		@Override
		public String toString ()
		{
			return StringUtil.concat ("[", StringUtil.join (m_rgMin, ", "), "] x [", StringUtil.join (m_rgMax, ", "), "]  { ", m_fMinVal, " }");
		}
	}


	///////////////////////////////////////////////////////////////////
	// Implementation
	
	@Override
	public void optimize (IRunExecutable run)
	{
		new Rectangle (run.getParameterLowerBounds (), run.getParameterUpperBounds ()).subdivide (run, MAX_SEGMENTS);
		
		if (LOGGER.isDebugEnabled ())
			LOGGER.debug (StringUtil.concat ("# Program runs: ", run.getRunsCount ()));
	}

	@Override
	public String getName ()
	{
		return "DIRECT";
	}
}
