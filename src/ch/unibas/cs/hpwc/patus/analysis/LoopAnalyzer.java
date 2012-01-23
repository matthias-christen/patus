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
package ch.unibas.cs.hpwc.patus.analysis;

import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.Expression;
import cetus.hir.IntegerLiteral;
import cetus.hir.Typecast;
import ch.unibas.cs.hpwc.patus.ast.RangeIterator;
import ch.unibas.cs.hpwc.patus.codegen.Globals;
import ch.unibas.cs.hpwc.patus.symbolic.MaximaTimeoutException;
import ch.unibas.cs.hpwc.patus.symbolic.Symbolic;
import ch.unibas.cs.hpwc.patus.util.CodeGeneratorUtil;
import ch.unibas.cs.hpwc.patus.util.ExpressionUtil;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 * Obtain information about loops.
 * @author Matthias-M. Christen
 */
public class LoopAnalyzer
{
	///////////////////////////////////////////////////////////////////
	// Implementation

	/**
	 * Determines whether the <code>loop</code> has a constant trip count.
	 * @param loop
	 * @return
	 */
	public static boolean isConstantTripCount (RangeIterator loop)
	{
		return LoopAnalyzer.isConstantTripCount (loop.getStart (), loop.getEnd (), loop.getStep ());
	}

	/**
	 * Determines whether the loop defined by the start, step, and end expressions has a constant trip count.
	 * @param exprStart
	 * @param exprEnd
	 * @param exprStep
	 * @return
	 */
	public static boolean isConstantTripCount (Expression exprStart, Expression exprEnd, Expression exprStep)
	{
		return LoopAnalyzer.getConstantTripCount (exprStart, exprEnd, exprStep) != null;
	}

	/**
	 * Returns the trip count of the <code>loop</code>.
	 * @param loop
	 * @return
	 */
	public static Expression getTripCount (RangeIterator loop)
	{
		return LoopAnalyzer.getTripCount (loop.getStart (), loop.getEnd (), loop.getStep ());
	}

	/**
	 * Calculates the loop trip count and returns the expression (an {@link IntegerLiteral}
	 * if the trip count is constant, or <code>null</code> otherwise.
	 * @param loop
	 * @return The loop trip count if the loop trip count is constant, or <code>null</code>
	 * 	if it is not
	 */
	public static Expression getConstantTripCount (RangeIterator loop)
	{
		return LoopAnalyzer.getConstantTripCount (loop.getStart (), loop.getEnd (), loop.getStep ());
	}

	/**
	 * Calculates the loop trip count and returns the expression.
	 * @param exprStart The start loop index
	 * @param exprEnd The end loop index
	 * @param exprStep The loop step
	 * @return The loop trip count expression (possibly a symbolic value)
	 */
	public static Expression getTripCount (Expression exprStart, Expression exprEnd, Expression exprStep)
	{
		// check whether the trip count is constant
		// loop trip count is
		//     |  end - start  |
		//     |  -----------  |  +  1
		//     +-   stride    -+
		//
		// ( ceil((end - start + 1) / stride) = floor((end - start + 1 + (stride - 1)) / stride) = floor((end - start) / stride + 1) = floor ((end - start) / stride) + 1 )

		try
		{
			return Symbolic.evaluateExpression (
				StringUtil.concat ("ratsimp(floor(((", exprEnd, ")-(", exprStart, "))/(", exprStep, ")) + 1)"),
				exprStart, exprEnd, exprStep);
		}
		catch (MaximaTimeoutException e)
		{
			return new BinaryExpression (
				new Typecast (CodeGeneratorUtil.specifiers (Globals.SPECIFIER_INDEX),
					new BinaryExpression (
						new BinaryExpression (exprEnd, BinaryOperator.SUBTRACT, exprStart),
						BinaryOperator.DIVIDE,
						exprStep)),
				BinaryOperator.ADD,
				new IntegerLiteral (1));
		}
	}

	/**
	 * Calculates the loop trip count and returns the expression (an {@link IntegerLiteral}
	 * if the trip count is constant, or <code>null</code> otherwise.
	 * @param exprStart
	 * @param exprEnd
	 * @param exprStep
	 * @return The loop trip count if the loop trip count is constant, or <code>null</code>
	 * 	if it is not
	 */
	public static Expression getConstantTripCount (Expression exprStart, Expression exprEnd, Expression exprStep)
	{
		Expression exprTripCount = LoopAnalyzer.getTripCount (exprStart, exprEnd, exprStep);

		//if (Symbolic.LogicalValue.TRUE.equals (Symbolic.isNumber (exprTripCount)))
		return ExpressionUtil.isNumberLiteral (exprTripCount) ? exprTripCount : null;
	}
}
