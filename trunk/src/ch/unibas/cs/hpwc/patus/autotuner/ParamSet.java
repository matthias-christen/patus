package ch.unibas.cs.hpwc.patus.autotuner;

import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 * Encapsulates a parameter set passed to the autotuner.
 * Lists all acceptable values for one parameter axis.
 *
 * @author Matthias-M. Christen
 */
public class ParamSet
{
	///////////////////////////////////////////////////////////////////
	// Member Variables

	private int[] m_rgParamSet;
	private boolean m_bUseExhaustive;


	///////////////////////////////////////////////////////////////////
	// Implementation

	public ParamSet (int[] rgParamSet, boolean bUseExhaustive)
	{
		m_rgParamSet = rgParamSet;
		m_bUseExhaustive = bUseExhaustive;
	}

	public int[] getParams ()
	{
		return m_rgParamSet;
	}

	public boolean useExhaustive ()
	{
		return m_bUseExhaustive;
	}
	
	@Override
	public String toString ()
	{
		StringBuilder sb = new StringBuilder ("Parameter set ");
		if (m_bUseExhaustive)
			sb.append (", Exhaustive ");
		sb.append ("{ ");
		
		boolean bIsFirst = true;
		for (int nParamValue : m_rgParamSet)
		{
			if (!bIsFirst)
				sb.append (", ");
			sb.append (nParamValue);
			bIsFirst = false;
		}
		
		sb.append (" }");
		
		return sb.toString ();
	}
}
