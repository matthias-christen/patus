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
 * <p>Implements the Hooke-Jeeves Algorithm (Pattern Search).</p>
 * 
 * <b>Example codes:</b>
 * <ul>
 * <li><a href="http://www.serc.iisc.ernet.in/~amohanty/SE288/hja.html">http://www.serc.iisc.ernet.in/~amohanty/SE288/hja.html</a></li>
 * <li><a href="http://www.netlib.org/opt/hooke.c">http://www.netlib.org/opt/hooke.c</a></li>
 * </ul>
 * 
 * <b>Slides:</b>
 * <ul>
 * <li><a href="http://www.scai.fraunhofer.de/fileadmin/ArbeitsgruppeTrottenberg/WS0809/seminar/Lammert.pdf">http://www.scai.fraunhofer.de/fileadmin/ArbeitsgruppeTrottenberg/WS0809/seminar/Lammert.pdf</a></li>
 * </ul>
 * 
 * <b>More Literature:</b>
 * <ul>
 * <li><a href="http://arantxa.ii.uam.es/~fdiez/docencia/04-05/OSI/Hooke-Jeeves.pdf">http://arantxa.ii.uam.es/~fdiez/docencia/04-05/OSI/Hooke-Jeeves.pdf</a></li>
 * <li><a href="http://camo.ici.ro/journal/vol11/v11a4.pdf">http://camo.ici.ro/journal/vol11/v11a4.pdf</a></li>
 * </ul>
 * 
 * @author Matthias-M. Christen
 */
public class HookeJeevesOptimizer extends AbstractOptimizer
{
	private final static int MAX_RUNS = 5;
	private final static int MAX_ITER = 25;
	private final static int MAX_SEGMENTS = 8;
	
	/**
	 * Number of times to &quot;roll the dice&quot; for a new starting point
	 */
	private final static int MAX_STARTPOINT_TRIES = 50;
	
	
	/**
	 * Performs exploratory moves along each coordinate axis with step sizes +/-<code>rgStep</code>.
	 * @param run
	 * @param rgStart Start coordinates
	 * @param rgCurrent The current coordinates on function exit
	 * @param rgStep The steps in each direction
	 * @return <code>true</code> iff the method was able to improve the objective value
	 */
	private boolean explore (IRunExecutable run, int[] rgStart, int[] rgCurrent, int[] rgStep)
	{
		final int nParamsCount = run.getParametersCount ();
		StringBuilder sbResult = new StringBuilder ();
		boolean bImproved = false;
		
		HookeJeevesOptimizer.copy (rgStart, rgCurrent);

		for (int i = 0; i < nParamsCount; i++)
		{
			if (rgStep[i] == 0)
				continue;
			
			rgCurrent[i] += rgStep[i];
			sbResult.setLength (0);
			double fObjVal = run.execute (rgCurrent, sbResult, checkBounds ());
			if (fObjVal < getResultTiming ())
			{
				// improvement achieved: accept point
				bImproved = true;
				setResultTiming (fObjVal);
				setResultParameters (rgCurrent);
				setProgramOutput (sbResult);
			}
			else
			{
				// no improvement: search in other direction
				rgCurrent[i] -= 2 * rgStep[i];
				
				sbResult.setLength (0);
				fObjVal = run.execute (rgCurrent, sbResult, checkBounds ());
				if (fObjVal < getResultTiming ())
				{
					bImproved = true;
					setResultTiming (fObjVal);
					setResultParameters (rgCurrent);
					setProgramOutput (sbResult);
				}
				else
				{
					// no improvement: revert
					rgCurrent[i] = rgStart[i];
				}
			}
		}
		
		return bImproved;
	}
	
	@Override
	public void optimize (IRunExecutable run)
	{
		int[] rgLower = run.getParameterLowerBounds ();
		int[] rgUpper = run.getParameterUpperBounds ();
				
		int[] rgStart = new int[rgLower.length];
		int[] rgCurrent = new int[rgStart.length];

		for (int j = 0; j < MAX_RUNS; j++)
		{
			for (int k = 0; k < MAX_STARTPOINT_TRIES; k++)
			{
				OptimizerUtil.getRandomPointWithinBounds (rgStart, rgLower, rgUpper);
				if (run.areConstraintsSatisfied (rgStart))
					break;
			}
			
			// initialize steps
			int[] rgStep = new int[rgStart.length];
			boolean[] rgParamFixed = new boolean[rgStart.length];
			for (int i = 0; i < rgStep.length; i++)
			{
				rgParamFixed[i] = rgUpper[i] == rgLower[i];
				rgStep[i] = (rgUpper[i] - rgLower[i]) / MAX_SEGMENTS;
				if (rgStep[i] == 0 && !rgParamFixed[i])
					rgStep[i] = 1;
			}
			
			for (int k = 0; k < MAX_ITER; k++)
			{
				// explore as long as there is improvement			
				for (boolean bLastImproved = true; ; )
				{
					boolean bImproved = explore (run, rgStart, rgCurrent, rgStep);
					if (!bImproved && !bLastImproved)
						break;
					
					if (bImproved)
					{
						for (int i = 0; i < rgStart.length; i++)
							rgStart[i] = 2 * rgCurrent[i] - rgStart[i];
					}
					else
					{
						for (int i = 0; i < rgStart.length; i++)
							rgStart[i] = rgCurrent[i];
					}
					OptimizerUtil.adjustToBounds (rgStart, rgLower, rgUpper);
					
					bLastImproved = bImproved;
				}
				
				// tighten the mesh
				for (int i = 0; i < rgStep.length; i++)
				{
					if (!rgParamFixed[i])
					{
						rgStep[i] /= 2;
						if (rgStep[i] == 0)
							rgStep[i] = 1;
					}
				}
			}
		}
	}
	
	private static void copy (int[] rgSrc, int[] rgDest)
	{
		System.arraycopy (rgSrc, 0, rgDest, 0, rgSrc.length);
	}

	@Override
	public String getName ()
	{
		return "Hooke Jeeves Algorithm";
	}
}
