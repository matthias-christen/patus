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

public abstract class AbstractOptimizer implements IOptimizer
{
	private int[] m_rgResult = null;
	private double m_fObjValue = Double.MAX_VALUE;
	private String m_strProgramOutput = "";
	private boolean m_bCheckBounds = true;
	

	protected void setResultParameters (int[] rgResult)
	{
		if (m_rgResult == null)
			m_rgResult = new int[rgResult.length];
		System.arraycopy (rgResult, 0, m_rgResult, 0, rgResult.length);
	}

	@Override
	public int[] getResultParameters ()
	{
		return m_rgResult;
	}
	
	protected void setResultTiming (double fObjValue)
	{
		m_fObjValue = fObjValue;
	}

	@Override
	public double getResultTiming ()
	{
		return m_fObjValue;
	}
	
	protected void setProgramOutput (StringBuilder sbProgramOutput)
	{
		if (sbProgramOutput != null && sbProgramOutput.length () > 0)
			m_strProgramOutput = sbProgramOutput.toString ();
	}
	
	protected void setProgramOutput (String strProgramOutput)
	{
		if (strProgramOutput != null && !"".equals (strProgramOutput))
			m_strProgramOutput = strProgramOutput;
	}

	@Override
	public String getProgramOutput ()
	{
		return m_strProgramOutput;
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
