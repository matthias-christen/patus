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
