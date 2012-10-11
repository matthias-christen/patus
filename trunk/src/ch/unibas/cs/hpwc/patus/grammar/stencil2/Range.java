package ch.unibas.cs.hpwc.patus.grammar.stencil2;

import ch.unibas.cs.hpwc.patus.util.StringUtil;

public class Range
{
	private int m_nStart;
	private int m_nEnd;
	
	public Range (int nStart, int nEnd)
	{
		if (nStart > nEnd)
			throw new RuntimeException ("'start' must not be larger than 'end'.");
		m_nStart = nStart;
		m_nEnd = nEnd;
	}
	
	public int getStart ()
	{
		return m_nStart;
	}
	
	public int getEnd ()
	{
		return m_nEnd;
	}
	
	public int getSize ()
	{
		return m_nEnd - m_nStart + 1;
	}
	
	@Override
	public String toString ()
	{
		return StringUtil.concat ("[", m_nStart, "..", m_nEnd, "]");
	}
}