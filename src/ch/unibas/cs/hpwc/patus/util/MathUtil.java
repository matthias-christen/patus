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
 *
 * @author Matthias-M. Christen
 */
public class MathUtil
{
	///////////////////////////////////////////////////////////////////
	// Member Variables


	///////////////////////////////////////////////////////////////////
	// Implementation

	/**
	 * Computes the GCD (greatest common divisor) of <code>a</code> and <code>b</code>.
	 * @param a The first number
	 * @param b The second number
	 * @return The GCD of <code>a</code> and <code>b</code>
	 */
	public static int getGCD (int a, int b)
	{
		if (b > a)
		{
			// swap the arguments
			int nTmp = a;
			a = b;
			b = nTmp;
		}

		// ensure that the arguments are non-negative
		if (a < 0)
			a = -a;
		if (b < 0)
			b = -b;

		// Euclidean algorithm
		for (int p; b != 0; )
		{
			p = a;
			a = b;
			b = p % b;
		}

		return a;
	}

	/**
	 * Computes the GCD (greatest common divisor) of the numbers <code>rgNumbers</code>.
	 * @param rgNumbers The numbers for which to compute the least common multiple
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
	 * Computes the LCM (least common multiple) of <code>a</code> and <code>b</code>.
	 * @param a The first number
	 * @param b The second number
	 * @return The LCM of <code>a</code> and <code>b</code>
	 */
	public static int getLCM (int a, int b)
	{
		return (a * b) / MathUtil.getGCD (a, b);
	}

	/**
	 * Computes the LCM (least common multiple) of the numbers <code>rgNumbers</code>.
	 * @param rgNumbers The numbers for which to compute the least common multiple
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

	public static boolean isPowerOfTwo (int nNum)
	{
    	for (int i = 1; i <= nNum; i <<= 1)
    		if (nNum == i)
    			return true;
    	return false;
	}
	
	public static int sgn (int a)
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
	public static long log2 (int a)
	{
		if (a <= 0)
			throw new RuntimeException ("The argument must be strictly positive");
		
		int j = -1;
		for (int i = a; i > 0; i >>= 1)
			j++;
		return j;
	}
}
