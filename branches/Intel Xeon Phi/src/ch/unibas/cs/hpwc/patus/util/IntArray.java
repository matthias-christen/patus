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
 * An array of integers that provides some operations on the array (adding offsets, appending new data)
 * and can be used as key in maps.
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

	/**
	 * Constructs a new {@link IntArray} and initializes it with the values in <code>rgValues</code>.
	 * The value array is used by reference, i.e., if values in the original <code>rgValues</code>
	 * array are changed, the changes will be reflected in the values of this {@link IntArray} instance.
	 * 
	 * @param rgValues The values to assign to the array
	 */
	public IntArray (int[] rgValues)
	{
		this (rgValues, false);
	}
	
	/**
	 * Constructs a new {@link IntArray} and initializes it with the values in <code>rgValues</code>.
	 * The value array is either used by reference (if <code>bCreateCopy == false</code>) or the
	 * values are copied to the internal array and don't change if entries of the original <code>rgValues</code>
	 * are modified (if <code>bCreateCopy == true</code>).
	 * 
	 * @param rgValues The values to assign to the array
	 * @param bCreateCopy Determines whether a copy of the original array is created or the internal array
	 * 	is assigned by reference
	 */
	public IntArray (int[] rgValues, boolean bCreateCopy)
	{
		if (bCreateCopy)
			initialize (rgValues, 0, rgValues.length - 1);
		else
			m_rgValues = rgValues;
	}
	
	public IntArray (int[] rgValues, int nStart, int nEnd)
	{
		initialize (rgValues, nStart, nEnd);
	}
	
	private void initialize (int[] rgValues, int nStart, int nEnd)
	{
		if (rgValues == null)
			m_rgValues = null;
		else
		{
			m_rgValues = new int[nEnd - nStart + 1];
			System.arraycopy (rgValues, nStart, m_rgValues, 0, nEnd - nStart + 1);
		}
	}

	/**
	 * Returns the values as <code>int[]</code>.
	 * @return The array values
	 */
	public int[] get ()
	{
		return m_rgValues;
	}

	/**
	 * Returns one entry of the array.
	 * @param nIdx The index of the entry to retrieve
	 * @return The value of the entry at index <code>nIdx</code>
	 */
	public int get (int nIdx)
	{
		return m_rgValues[nIdx];
	}
	
	/**
	 * Sets the entry at index <code>nIdx</code> to <code>nValue</code>.
	 * @param nIdx The index of the entry to modify
	 * @param nValue The new value
	 */
	public void set (int nIdx, int nValue)
	{
		m_rgValues[nIdx] = nValue;
	}

	/**
	 * Returns the length of the array.
	 * @return
	 */
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

	/**
	 * Appends new values to the array. The values are by-value, i.e., if an entry in the
	 * original array <code>rgValue</code> is modified, the changes won't be reflected
	 * in this {@link IntArray instance.
	 * 
	 * @param rgValues The values to append
	 */
	public void append (int... rgValues)
	{
		int[] rgValuesOld = m_rgValues;
		m_rgValues = new int[rgValuesOld.length + rgValues.length];
		System.arraycopy (rgValuesOld, 0, m_rgValues, 0, rgValuesOld.length);
		System.arraycopy (rgValues, 0, m_rgValues, rgValuesOld.length, rgValues.length);
	}
	
	/**
	 * Determines whether all the array values are 0.
	 * @return <code>true</code> iff all the array values are 0
	 */
	public boolean isZero ()
	{
		for (int nVal : m_rgValues)
			if (nVal != 0)
				return false;
		return true;
	}
	
	public IntArray abs ()
	{
		IntArray arr = new IntArray (m_rgValues, true);
		for (int i = 0; i < arr.m_rgValues.length; i++)
			arr.m_rgValues[i] = Math.abs (arr.m_rgValues[i]);
		return arr;
	}
	
	public IntArray neg ()
	{
		IntArray arr = new IntArray (m_rgValues, true);
		for (int i = 0; i < arr.m_rgValues.length; i++)
			arr.m_rgValues[i] = -arr.m_rgValues[i];
		return arr;
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
	
	/**
	 * Performs a linear search to check whether the array contains the value <code>nValue</code>.
	 * @param nValue The value to look for
	 * @return <code>true</code> iff the array contains <code>nValue</code>
	 */
	public boolean contains (int nValue)
	{
		for (int v : m_rgValues)
			if (v == nValue)
				return true;
		return false;
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
