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

import java.util.Arrays;
import java.util.Iterator;

/**
 *
 * @author Matthias-M. Christen
 */
public class IntArray implements Iterable<Integer>, Comparable<IntArray>
{
	///////////////////////////////////////////////////////////////////
	// Member Variables

	private int[] m_rgValues;


	///////////////////////////////////////////////////////////////////
	// Implementation

	public IntArray (int[] rgValues)
	{
		this (rgValues, false);
	}

	public IntArray (int[] rgValues, boolean bCreateCopy)
	{
		if (bCreateCopy)
		{
			if (rgValues == null)
				m_rgValues = null;
			else
			{
				m_rgValues = new int[rgValues.length];
				System.arraycopy (rgValues, 0, m_rgValues, 0, rgValues.length);
			}
		}
		else
			m_rgValues = rgValues;
	}

	public int[] get ()
	{
		return m_rgValues;
	}

	public int get (int nIdx)
	{
		return m_rgValues[nIdx];
	}

	public int length ()
	{
		return m_rgValues.length;
	}

	/**
	 * Adds the offset <code>rgOffset</code>.
	 * @param rgOffset
	 */
	public void add (int[] rgOffset)
	{
		for (int i = 0; i < Math.min (m_rgValues.length, rgOffset.length); i++)
			m_rgValues[i] += rgOffset[i];
	}

	public void append (int... rgValues)
	{
		int[] rgValuesOld = m_rgValues;
		m_rgValues = new int[rgValuesOld.length + rgValues.length];
		System.arraycopy (rgValuesOld, 0, m_rgValues, 0, rgValuesOld.length);
		System.arraycopy (rgValues, 0, m_rgValues, rgValuesOld.length, rgValues.length);
	}

	@Override
	public boolean equals (Object obj)
	{
		if (obj instanceof int[])
			return Arrays.equals (m_rgValues, (int[]) obj);
		if (obj instanceof IntArray)
			return Arrays.equals (m_rgValues, ((IntArray) obj).get ());
		return false;
	}

	@Override
	public int hashCode ()
	{
		return Arrays.hashCode (m_rgValues);
	}

	@Override
	public int compareTo (IntArray arr)
	{
		if (m_rgValues.length != arr.length ())
		{
			//throw new RuntimeException ("Only arrays of same lengths can be compared");
			return m_rgValues.length - arr.length ();
		}

		int nIdx = 0;
		while (m_rgValues[nIdx] == arr.get ()[nIdx])
		{
			nIdx++;

			// if the end is reached and all entries have been equal, the arrays are equal
			if (nIdx == m_rgValues.length)
				return 0;
		}

		return m_rgValues[nIdx] - arr.get ()[nIdx];
	}

	@Override
	public Iterator<Integer> iterator ()
	{
		return new Iterator<Integer> ()
		{
			private int m_nIdx = 0;

			@Override
			public boolean hasNext ()
			{
				return m_nIdx < m_rgValues.length;
			}

			@Override
			public Integer next ()
			{
				return m_rgValues[m_nIdx++];
			}

			@Override
			public void remove ()
			{
				throw new RuntimeException ("Method not supported");
			}
		};
	}

	@Override
	public String toString ()
	{
		return Arrays.toString (m_rgValues);
	}

	/**
	 * Returns an <code>int</code> array of length <code>nLength</code>
	 * filled with values <code>nFill</code>.
	 * @param nLength The array length
	 * @param nFill The fill value
	 * @return An array of length <code>nLength</code> filled with values <code>nFill</code>
	 */
	public static int[] getArray (int nLength, int nFill)
	{
		int[] rgArr = new int[nLength];
		Arrays.fill (rgArr, nFill);
		return rgArr;
	}
}
