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
package ch.unibas.cs.hpwc.patus.util;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

/**
 *
 * @author Matthias-M. Christen
 */
public class DomainPointEnumerator implements Iterable<int[]>
{
	///////////////////////////////////////////////////////////////////
	// Inner Types

	public static class MinMax
	{
		private int m_nMinimum;
		private int m_nMaximum;

		public MinMax (int nMinimum, int nMaximum)
		{
			m_nMinimum = nMinimum;
			m_nMaximum = nMaximum;
		}

		public int getMinimum ()
		{
			return m_nMinimum;
		}

		public int getMaximum ()
		{
			return m_nMaximum;
		}

		public int size ()
		{
			return m_nMaximum - m_nMinimum + 1;
		}
	}

	class DomainIterator implements Iterator<int[]>
	{
		private int[] m_rgCurrentPoint;
		private int m_nCurrent;

		public DomainIterator ()
		{
			m_rgCurrentPoint = new int[m_listDimensions.size ()];
			for (int i = 0; i < m_listDimensions.size (); i++)
				m_rgCurrentPoint[i] = m_listDimensions.get (i).getMinimum ();
			m_nCurrent = 0;
		}

		@Override
		public boolean hasNext ()
		{
			return m_nCurrent < m_nPointsCount;
		}

		@Override
		public int[] next ()
		{
			// return a copy of the current point
			int[] rgConfig = new int[m_rgCurrentPoint.length];
			System.arraycopy (m_rgCurrentPoint, 0, rgConfig, 0, m_rgCurrentPoint.length);

			// advance the point
			for (int i = 0; i < m_rgCurrentPoint.length; i++)
			{
				m_rgCurrentPoint[i]++;
				if (m_rgCurrentPoint[i] > m_listDimensions.get (i).getMaximum ())
					m_rgCurrentPoint[i] = m_listDimensions.get (i).getMinimum ();
				else
					break;
			}
			m_nCurrent++;

			return rgConfig;
		}

		@Override
		public void remove ()
		{
			// can't remove elements
		}
	}


	///////////////////////////////////////////////////////////////////
	// Member Variables

	private List<MinMax> m_listDimensions;

	private int m_nPointsCount;


	///////////////////////////////////////////////////////////////////
	// Implementation

	public DomainPointEnumerator ()
	{
		m_listDimensions = new ArrayList<> ();
		m_nPointsCount = 0;
	}

	public DomainPointEnumerator (MinMax... rgDimensions)
	{
		this ();
		for (MinMax dim : rgDimensions)
			addDimension (dim);
	}

	public void addDimension (int nMinimum, int nMaximum)
	{
		addDimension (new MinMax (nMinimum, nMaximum));
	}

	public void addDimension (MinMax dimension)
	{
		m_listDimensions.add (dimension);
		if (m_nPointsCount == 0)
			m_nPointsCount = 1;
		m_nPointsCount *= dimension.size ();
	}

	public int size ()
	{
		return m_nPointsCount;
	}

	@Override
	public Iterator<int[]> iterator ()
	{
		return new DomainIterator ();
	}

	public static void main (String[] args)
	{
		DomainPointEnumerator dpe = new DomainPointEnumerator ();
		dpe.addDimension (0, 2);
		dpe.addDimension (0, 0);
		dpe.addDimension (1, 4);

		System.out.println (dpe.size ());
		System.out.println ("--------->");
		for (int[] x : dpe)
			System.out.println (Arrays.toString (x));
	}
}
