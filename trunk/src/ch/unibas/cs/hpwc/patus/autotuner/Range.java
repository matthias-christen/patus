package ch.unibas.cs.hpwc.patus.autotuner;

/**
 *
 * @author Matthias-M. Christen
 */
public class Range<T extends Number> implements Comparable<Range<T>>
{
	///////////////////////////////////////////////////////////////////
	// Member Variables

	private T m_min;
	private T m_max;


	///////////////////////////////////////////////////////////////////
	// Implementation

	public Range ()
	{
		setRange (null, null);
	}

	public Range (T min, T max)
	{
		setRange (min, max);
	}

	public void setRange (T min, T max)
	{
		m_min = min;
		m_max = max;
	}

	public void setMin (T min)
	{
		m_min = min;
	}

	public void setMax (T max)
	{
		m_max = max;
	}

	public T getMin ()
	{
		return m_min;
	}

	public T getMax ()
	{
		return m_max;
	}

	public boolean inRange (T val)
	{
		if (m_min == null || m_max == null || val == null)
			return false;
		return m_min.doubleValue () <= val.doubleValue () && val.doubleValue () <= m_max.doubleValue ();
	}

	@Override
	public int compareTo (Range<T> rangeOther)
	{
		double fValue1 = m_min.doubleValue ();
		double fValue2 = rangeOther.getMin ().doubleValue ();
		return fValue1 < fValue2 ? -1 : (fValue1 == fValue2 ? 0 : 1);
	}
}
