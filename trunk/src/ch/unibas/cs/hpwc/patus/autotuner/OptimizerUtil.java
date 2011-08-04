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

import java.util.Random;

import cetus.hir.BinaryExpression;
import cetus.hir.DepthFirstIterator;
import cetus.hir.Expression;
import cetus.hir.IDExpression;
import cetus.hir.IntegerLiteral;
import ch.unibas.cs.hpwc.patus.util.ExpressionUtil;

public class OptimizerUtil
{
	///////////////////////////////////////////////////////////////////
	// Constants

	private final static Random RANDOM = new Random ();


	///////////////////////////////////////////////////////////////////
	// Implementation

	/**
	 * Checks whether the arrays <code>rgLowerBound</code> and <code>rgUpperBound</code>
	 * meet the specifications (i.e. are not <code>null</code> and have the same length.
	 * @param rgLowerBound The lower bounds array
	 * @param rgUpperBound The upper bounds array
	 */
	public static void checkBoundVars (int[] rgLowerBound, int[] rgUpperBound)
	{
		if (rgLowerBound == null || rgUpperBound == null)
			throw new RuntimeException ("Lower and upper bounds must not be null.");
		if (rgLowerBound.length != rgUpperBound.length)
			throw new RuntimeException ("Lower and upper bounds arrays must have the same length");
	}

	/**
	 * Determines whether <code>rgParams</code> lies within the bounds <code>rgLowerBound</code>, <code>rgUpperBound</code>.
	 * @param rgParams The parameter to check
	 * @param rgLowerBound The coordinate-wise lower bounds
	 * @param rgUpperBound The coordinate-wise upper bounds
	 * @return
	 */
	public static boolean isWithinBounds (int[] rgParams, int[] rgLowerBound, int[] rgUpperBound)
	{
		// sanity checks...
		OptimizerUtil.checkBoundVars (rgLowerBound, rgUpperBound);

		if (rgParams == null)
			throw new NullPointerException ("The argument rgParams must not be null.");
		if (rgParams.length != rgLowerBound.length)
			throw new RuntimeException ("The parameter array must be of the same length (" + rgLowerBound.length + ") as the upper and lower bounds.");

		// do the check
		for (int i = 0; i < rgParams.length; i++)
			if (rgParams[i] < rgLowerBound[i] || rgParams[i] > rgUpperBound[i])
				return false;

		// check succeeded
		return true;
	}

	/**
	 * Determines whether <code>rgParams</code> lies within the bounds <code>rgLowerBound</code>, <code>rgUpperBound</code>
	 * and whether the constraints in <code>itConstraints</code> are satisfied.
	 * @param rgParams The parameter to check
	 * @param rgLowerBound The coordinate-wise lower bounds
	 * @param rgUpperBound The coordinate-wise upper bounds
	 * @param itConstraints An iterable over constraints to check
	 * @return
	 */
	public static boolean isWithinBounds (int[] rgParams, int[] rgLowerBound, int[] rgUpperBound, Iterable<Expression> itConstraints)
	{
		// check whether within bounds
		if (!OptimizerUtil.isWithinBounds (rgParams, rgLowerBound, rgUpperBound))
			return false;

		// check whether all the constraints are satisfied
		if (!OptimizerUtil.areConstraintsSatisfied (rgParams, itConstraints))
			return false;

		return true;
	}

	/**
	 * Determines whether the constraints <code>itConstraints</code> are satisfied.
	 * @param rgParams
	 * @param itConstraints
	 * @return
	 */
	public static boolean areConstraintsSatisfied (int[] rgParams, Iterable<Expression> itConstraints)
	{
		for (Expression expr : itConstraints)
		{
			if (!(expr instanceof BinaryExpression))
				throw new RuntimeException ("Constraints must be comparison expressions");

			BinaryExpression bexprSubst = (BinaryExpression) OptimizerUtil.substituteValues (expr, rgParams);

			Expression exprLHS = cetus.hir.Symbolic.simplify (bexprSubst.getLHS ());
			Expression exprRHS = cetus.hir.Symbolic.simplify (bexprSubst.getRHS ());
			if (!(exprLHS instanceof IntegerLiteral) && !(exprRHS instanceof IntegerLiteral))
				return false;
			if (!ExpressionUtil.compare ((IntegerLiteral) exprLHS, ((BinaryExpression) expr).getOperator (), (IntegerLiteral) exprRHS))
				return false;
		}

		return true;
	}

	/**
	 * Substitutes the values into variables &quot;$n&quot; in the expression <code>expr</code>.
	 * @param expr
	 * @param rgValues
	 * @return
	 */
	public static Expression substituteValues (Expression expr, int[] rgValues)
	{
		Expression exprNew = expr.clone ();
		for (DepthFirstIterator it = new DepthFirstIterator (exprNew); it.hasNext (); )
		{
			Object obj = it.next ();
			if (obj instanceof IDExpression)
			{
				String strName = ((IDExpression) obj).getName ();
				if (strName.charAt (0) == '$')
					((IDExpression) obj).swapWith (new IntegerLiteral (rgValues[Integer.valueOf (strName.substring (1)) - 1]));
			}
		}

		return exprNew;
	}
	
	public static void adjustToBounds (int[] rgParams, int[] rgLower, int[] rgUpper)
	{
		if (rgLower == null || rgUpper == null)
			throw new RuntimeException ("Lower and upper bounds must not be null.");

		for (int i = 0; i < rgParams.length; i++)
		{
			if (i < rgLower.length)
			{
				if (rgParams[i] < rgLower[i])
					rgParams[i] = rgLower[i];
			}
			
			if (i < rgUpper.length)
			{
				if (rgParams[i] > rgUpper[i])
					rgParams[i] = rgUpper[i];
			}
		}
	}

	/**
	 * Checks whether the parameter <code>rgParams</code> is in the parameter set <code>rgParamSet</code>.
	 * @param rgParams The parameter to check
	 * @param rgParamSet The set of possible parameters
	 * @return <code>true</code> iff <code>rgParams</code> is in <code>rgParamSet</code>
	 */
	public static boolean isInParamSet (int[] rgParams, int[][] rgParamSet)
	{
		// sanity checks...
		if (rgParams == null)
			throw new NullPointerException ("The argument rgParams must not be null.");
		if (rgParamSet == null)
			throw new NullPointerException ("The parameter set must not be null.");
		if (rgParams.length != rgParamSet.length)
			throw new RuntimeException ("The number of parameters in rgParams and the parameter set rgParamSet must have the same length.");

		// do the check
		for (int i = 0; i < rgParams.length; i++)
		{
			boolean bIsFound = false;
			for (int j = 0; j < rgParamSet[i].length; j++)
			{
				if (rgParams[i] == rgParamSet[i][j])
				{
					bIsFound = true;
					break;
				}
			}

			if (!bIsFound)
				return false;
		}

		return true;
	}
	
	/**
	 * Returns a random point that lies within the bounds <code>rgLowerBound</code> and <code>rgUpperBound</code>.
	 * @param rgLowerBound The coordinate-wise lower bounds
	 * @param rgUpperBound The coordinate-wise upper bounds
	 * @return A random point between <code>rgLowerBound</code> and <code>rgUpperBound</code>
	 */
	public static void getRandomPointWithinBounds (int[] rgPoint, int[] rgLowerBound, int[] rgUpperBound)
	{
		OptimizerUtil.checkBoundVars (rgLowerBound, rgUpperBound);
		for (int i = 0; i < Math.min (rgPoint.length, rgLowerBound.length); i++)
			rgPoint[i] = OptimizerUtil.RANDOM.nextInt (rgUpperBound[i] - rgLowerBound[i] + 1) + rgLowerBound[i];
	}	

	/**
	 * Returns a random point that lies within the bounds <code>rgLowerBound</code> and <code>rgUpperBound</code>.
	 * @param rgLowerBound The coordinate-wise lower bounds
	 * @param rgUpperBound The coordinate-wise upper bounds
	 * @return A random point between <code>rgLowerBound</code> and <code>rgUpperBound</code>
	 */
	public static int[] getRandomPointWithinBounds (int[] rgLowerBound, int[] rgUpperBound)
	{
		OptimizerUtil.checkBoundVars (rgLowerBound, rgUpperBound);

		int[] rgResult = new int[rgLowerBound.length];
		OptimizerUtil.getRandomPointWithinBounds (rgResult, rgLowerBound, rgUpperBound);

		return rgResult;
	}

	/**
	 * Returns a random point from the parameter set <code>rgParamSet</code>.
	 * @param rgParamSet The parameter set (an array of arrays of all possible values for a parameter)
	 * @return A random point from the parameter set
	 */
	public static int[] getRandomPoint (int[][] rgParamSet)
	{
		int[] rgResult = new int[rgParamSet.length];
		for (int i = 0; i < rgParamSet.length; i++)
			rgResult[i] = rgParamSet[i][OptimizerUtil.RANDOM.nextInt (rgParamSet[i].length)];

		return rgResult;
	}

	public static void getNextConfigInPlace (int[] rgConfig, int[] rgParameterLowerBounds, int[] rgParameterUpperBounds)
	{
		// advance the point
		for (int i = 0; i < rgConfig.length; i++)
		{
			rgConfig[i]++;
			if (rgConfig[i] > rgParameterUpperBounds[i])
				rgConfig[i] = rgParameterLowerBounds[i];
			else
				break;
		}		
	}
	
	public static int[] getNextConfig (int[] rgCurrent, int[] rgParameterLowerBounds, int[] rgParameterUpperBounds)
	{
		// return a copy of the current point
		int[] rgConfig = new int[rgCurrent.length];
		System.arraycopy (rgCurrent, 0, rgConfig, 0, rgCurrent.length);
		OptimizerUtil.getNextConfigInPlace (rgConfig, rgParameterLowerBounds, rgParameterUpperBounds);
		return rgConfig;
	}
}
