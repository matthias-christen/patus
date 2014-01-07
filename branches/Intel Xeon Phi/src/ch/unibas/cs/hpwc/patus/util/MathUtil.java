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

/**
 * Integer math utility functions.
 * 
 * @author Matthias-M. Christen
 */
public class MathUtil
{
	/**
	 * Computes the GCD (greatest common divisor) of <code>a</code> and
	 * <code>b</code>.
	 * 
	 * @param a
	 *            The first number
	 * @param b
	 *            The second number
	 * @return The GCD of <code>a</code> and <code>b</code>
	 */
	public static int getGCD (int a, int b)
	{
		int a0 = a;
		int b0 = b;
		
		if (b0 > a0)
		{
			// swap the arguments
			int nTmp = a0;
			a0 = b0;
			b0 = nTmp;
		}

		// ensure that the arguments are non-negative
		if (a0 < 0)
			a0 = -a0;
		if (b0 < 0)
			b0 = -b0;

		// Euclidean algorithm
		for (int p; b0 != 0; )
		{
			p = a0;
			a0 = b0;
			b0 = p % b0;
		}

		return a0;
	}

	/**
	 * Computes the GCD (greatest common divisor) of the numbers
	 * <code>rgNumbers</code>.
	 * 
	 * @param rgNumbers
	 *            The numbers for which to compute the least common multiple
	 * @return The GCD of <code>rgNumbers</code>
	 */
	public static int getGCD (int... rgNumbers)
	{
		if (rgNumbers.length == 0)
			return 0;
		if (rgNumbers.length == 1)
			return rgNumbers[0];

		int nGCD = rgNumbers[0];
		for (int i = 1; i < rgNumbers.length; i++)
			nGCD = MathUtil.getGCD (nGCD, rgNumbers[i]);

		return nGCD;
	}

	/**
	 * Computes the LCM (least common multiple) of <code>a</code> and
	 * <code>b</code>.
	 * 
	 * @param a
	 *            The first number
	 * @param b
	 *            The second number
	 * @return The LCM of <code>a</code> and <code>b</code>
	 */
	public static int getLCM (int a, int b)
	{
		return (a * b) / MathUtil.getGCD (a, b);
	}

	/**
	 * Computes the LCM (least common multiple) of the numbers
	 * <code>rgNumbers</code>.
	 * 
	 * @param rgNumbers
	 *            The numbers for which to compute the least common multiple
	 * @return The LCM of <code>rgNumbers</code>
	 */
	public static int getLCM (int... rgNumbers)
	{
		if (rgNumbers.length == 0)
			return 1;
		if (rgNumbers.length == 1)
			return rgNumbers[0];

		int nLCM = rgNumbers[0];
		for (int i = 1; i < rgNumbers.length; i++)
			nLCM = MathUtil.getLCM (nLCM, rgNumbers[i]);

		return nLCM;
	}

	/**
	 * Determines if <var>nNum</var> is a power of 2.
	 * 
	 * @param nNum
	 *            The integer to test
	 * @return <code>true</code> iff <var>nNum</var> is a power of 2.
	 */
	public static boolean isPowerOfTwo (int nNum)
	{
		/*
    	for (int i = 1; i <= nNum; i <<= 1)
    		if (nNum == i)
    			return true;
    	return false;
    	*/
		
		return nNum > 0 && ((nNum & (nNum - 1)) == 0); 
	}
	
	/**
	 * Returns the sign of <var>a</var>.
	 * 
	 * @param a
	 *            The integer for which to find the sign
	 * @return The sign of <var>a</var>
	 */
	public static int sgn (final int a)
	{
		if (a < 0)
			return -1;
		if (a > 0)
			return 1;
		return 0;
	}

	/**
	 * Computes log2 (ceil (<code>a</code>)).
	 * @param a
	 * @return
	 */
	public static long log2 (final long a)
	{
		if (a <= 0)
			throw new RuntimeException ("The argument must be strictly positive");
		
		/*
		int j = -1;
		for (int i = a; i > 0; i >>= 1)
			j++;
		return j;
		*/
		
		long v = a;
		final long b[] = new long[] { 0x00000002, 0x0000000c, 0x000000f0L, 0x0000ff00L, 0xffff0000L, 0xffffffff00000000L };
		long r = 0;
		int s = 32;
		
		for (int i = 5; i >= 0; i--)
		{
			if ((v & b[i]) != 0)
			{
				v >>= s;
				r |= s; 
			}
			s >>= 1;
		}
		
		return r;
	}
	
	/**
	 * Finds the integer <var>x</var> such that
	 * <var>x</var>=2<sup><var>q</var></sup> for some integer <var>q</var>, and
	 * <var>x</var> &le; <var>a</var>.
	 * 
	 * @param a
	 *            The value for which to find the power of 2 less or equal to
	 *            <var>a</var>
	 * @return A power of 2-integer less or equal to <var>a</var>
	 */
	public static long getPrevPower2 (final long a)
	{		
		long v = a;
		v |= v >> 1;
		v |= v >> 2;
		v |= v >> 4;
		v |= v >> 8;
		v |= v >> 16;
		v |= v >> 32;
		
		return (v >> 1) + 1;
	}

	/**
	 * Returns
	 * <code>ceil (<var>nNumerator</var> / <var>nDenominator</var>)</code>.
	 * 
	 * @param nNumerator
	 *            The numerator
	 * @param nDenominator
	 *            The denomiator
	 * @return <code>ceil (<var>nNumerator</var> / <var>nDenominator</var>)</code>
	 */
	public static int divCeil (int nNumerator, int nDenominator)
	{
		return (nNumerator + nDenominator - 1) / nDenominator;
	}
	
	/**
	 * Returns the maximum of the values in the array <var>a</var>.
	 * 
	 * @param a
	 *            The array in which to find the maximum value
	 * @return The maximum value in <var>a</var>
	 */
	public static int max (int... a)
	{
		int nMax = Integer.MIN_VALUE;
		for (int n : a)
			nMax = (nMax < n) ? n : nMax;
		return nMax;
	}
	
	/**
	 * Returns the minimum of the values in the array <var>a</var>.
	 * 
	 * @param a
	 *            The array in which to find the minimum value
	 * @return The minimum value in <var>a</var>
	 */
	public static int min (int... a)
	{
		int nMin = Integer.MAX_VALUE;
		for (int n : a)
			nMin = (nMin > n) ? n : nMin;
		return nMin;		
	}
}
