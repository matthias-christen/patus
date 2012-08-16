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

import cetus.hir.Expression;
import cetus.hir.IntegerLiteral;
import cetus.hir.UnaryExpression;
import cetus.hir.UnaryOperator;


/**
 * This class implements some elementwise vector operations.
 * @author Matthias-M. Christen
 */
public class VectorUtil
{
	///////////////////////////////////////////////////////////////////
	// Implementation

	/**
	 * Determines the minimum dimension of the vectors <code>rgVectors</code>.
	 * @return The minimum length of the vectors in the array <code>rgVectors</code>
	 */
	public static int getDimension (int[]... rgVectors)
	{
		int nDimension = Integer.MAX_VALUE;
		for (int[] rgVector : rgVectors)
		{
			if (rgVector == null)
				continue;
			nDimension = Math.min (nDimension, rgVector.length);
		}

		if (nDimension == Integer.MAX_VALUE)
			return 0;
		return nDimension;
	}

	public static <T> int getDimension (T[]... rgVectors)
	{
		int nDimension = Integer.MAX_VALUE;
		for (T[] rgVector : rgVectors)
		{
			if (rgVector == null)
				continue;
			nDimension = Math.min (nDimension, rgVector.length);
		}

		if (nDimension == Integer.MAX_VALUE)
			return 0;
		return nDimension;
	}

	/**
	 * Returns the minimum values in the array <code>rgVectors</code>
	 * in the sense of an upper bound, i.e. the method computes an elementwise
	 * minimum and returns a vector containing these minimum values.
	 * @return The elementwise minimum
	 */
	public static int[] getMinimum (int[]... rgVectors)
	{
		// get the dimension of the vectors
		int nDimension = VectorUtil.getDimension (rgVectors);

		// create the index that will contain the maximum values
		int[] rgMinimum = new int[nDimension];
		for (int i = 0; i < nDimension; i++)
			rgMinimum[i] = Integer.MAX_VALUE;

		// find minimum values
		for (int[] rgVector : rgVectors)
		{
			if (rgVector == null)
				continue;
			for (int i = 0; i < nDimension; i++)
				if (rgVector[i] < rgMinimum[i])
					rgMinimum[i] = rgVector[i];
		}

		// set non-set values to 0
		for (int i = 0; i < nDimension; i++)
			if (rgMinimum[i] == Integer.MAX_VALUE)
				rgMinimum[i] = 0;

		return rgMinimum;
	}
	
	public static Expression[] getMinimum (Expression[]... rgVectors)
	{
		// get the dimension of the vectors
		int nDimension = VectorUtil.getDimension (rgVectors);

		// create the index that will contain the maximum values
		Expression[] rgMinimum = new Expression[nDimension];
		for (int i = 0; i < nDimension; i++)
		{
			Expression[] rgArgs = new Expression[rgVectors.length];
			for (int j = 0; j < rgVectors.length; j++)
				rgArgs[j] = rgVectors[j] == null ? null : rgVectors[j][i];
			rgMinimum[i] = ExpressionUtil.min (rgArgs);
		}

		return rgMinimum;
	}

	/**
	 * Returns the maximum values in the array <code>rgVectors</code>
	 * in the sense of an upper bound, i.e. the method computes an elementwise
	 * maximum and returns a vector containing these maximum values.
	 * @return The elementwise maximum
	 */
	public static int[] getMaximum (int[]... rgVectors)
	{
		// get the dimension of the vectors
		int nDimension = VectorUtil.getDimension (rgVectors);

		// create the index that will contain the maximum values
		int[] rgMaximum = new int[nDimension];
		for (int i = 0; i < nDimension; i++)
			rgMaximum[i] = Integer.MIN_VALUE;

		// find maximum values
		for (int[] rgVector : rgVectors)
		{
			if (rgVector == null)
				continue;
			for (int i = 0; i < nDimension; i++)
				if (rgVector[i] > rgMaximum[i])
					rgMaximum[i] = rgVector[i];
		}

		// set non-set values to 0
		for (int i = 0; i < nDimension; i++)
			if (rgMaximum[i] == Integer.MAX_VALUE)
				rgMaximum[i] = 0;

		return rgMaximum;
	}

	/**
	 * Adds the two vectors <code>rgVector1</code> and <code>rgVector2</code>
	 * and returns a new vector containing the sum.
	 * @param rgVector1 The first summand
	 * @param rgVector2 The second summand
	 * @return A new vector being the sum of <code>rgVector1</code> and <code>rgVector2</code>
	 */
	public static int[] add (int[] rgVector1, int[] rgVector2)
	{
		// get the dimension of the vectors
		int nDimension = VectorUtil.getDimension (rgVector1, rgVector2);

		int[] rgSum = new int[nDimension];
		for (int i = 0; i < nDimension; i++)
			rgSum[i] = (rgVector1 == null ? 0 : rgVector1[i]) + (rgVector2 == null ? 0 : rgVector2[i]);

		return rgSum;
	}

	/**
	 * Subtracts the vector <code>rgSubtrahend</code> from <code>rgMinuend</code>
	 * and returns a new vector containing the difference.
	 * @param rgMinuend The vector from which to subtract
	 * @param rgSubtrahend The vector that is subtracted
	 * @return A new vector containing the difference <code>rgMinuend</code> - <code>rgSubtrahend</code>
	 */
	public static int[] subtract (int[] rgMinuend, int[] rgSubtrahend)
	{
		// get the dimension of the vectors
		int nDimension = VectorUtil.getDimension (rgMinuend, rgSubtrahend);

		int[] rgDifference = new int[nDimension];
		for (int i = 0; i < nDimension; i++)
			rgDifference[i] = (rgMinuend == null ? 0 : rgMinuend[i]) - (rgSubtrahend == null ? 0 : rgSubtrahend[i]);

		return rgDifference;
	}

	/**
	 * Returns a new vector containing -<code>rgVector</code>.
	 * @param rgVector The vector to negate
	 * @return -<code>rgVector</code>
	 */
	public static int[] negate (int[] rgVector)
	{
		return VectorUtil.subtract (null, rgVector);
	}
	
	public static Expression[] negate (Expression[] rgVector)
	{
		if (rgVector == null)
			return null;
		
		Expression[] rgResult = new Expression[rgVector.length];
		for (int i = 0; i < rgVector.length; i++)
			rgResult[i] = new UnaryExpression (UnaryOperator.MINUS, rgVector[i].clone ());
		return rgResult;
	}

	/**
	 * Determines whether the vector <code>rgVector</code> is the zero vector.
	 * @param rgVector The vector to test
	 * @return <code>true</code> iff <code>rgVector</code> is the zero vector
	 */
	public static boolean isZero (int[] rgVector)
	{
		for (int nVal : rgVector)
			if (nVal != 0)
				return false;
		return true;
	}
}
