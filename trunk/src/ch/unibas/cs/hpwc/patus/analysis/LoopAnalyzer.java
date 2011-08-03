package ch.unibas.cs.hpwc.patus.analysis;

import java.util.ArrayList;
import java.util.List;

import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.Expression;
import cetus.hir.ForLoop;
import cetus.hir.IntegerLiteral;
import cetus.hir.Typecast;
import ch.unibas.cs.hpwc.patus.ast.RangeIterator;
import ch.unibas.cs.hpwc.patus.codegen.Globals;
import ch.unibas.cs.hpwc.patus.codegen.unrollloop.UniformlyIncrementingLoopNestPart;
import ch.unibas.cs.hpwc.patus.codegen.unrollloop.UnrollLoopSharedObjects;
import ch.unibas.cs.hpwc.patus.symbolic.MaximaTimeoutException;
import ch.unibas.cs.hpwc.patus.symbolic.Symbolic;
import ch.unibas.cs.hpwc.patus.util.CodeGeneratorUtil;
import ch.unibas.cs.hpwc.patus.util.ExpressionUtil;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 * TODO: Unify with architecture of loop unroller...
 * @author Matthias-M. Christen
 */
public class LoopAnalyzer
{
	///////////////////////////////////////////////////////////////////
	// Member Variables

	///////////////////////////////////////////////////////////////////
	// Implementation

	/**
	 *
	 * @param loop
	 * @return
	 */
	public static boolean isConstantTripCount (ForLoop loop)
	{
		UniformlyIncrementingLoopNestPart loopInfo = new UniformlyIncrementingLoopNestPart ();
		List<int[]> l = new ArrayList<int[]> ();
		l.add (new int[] { 1 });
		UnrollLoopSharedObjects data = new UnrollLoopSharedObjects (l);
		loopInfo.init (data, loop, 0);
		return loopInfo.isConstantTripCount ();
	}

	/**
	 *
	 * @param loop
	 * @return
	 */
	public static boolean isConstantTripCount (RangeIterator loop)
	{
		return LoopAnalyzer.isConstantTripCount (loop.getStart (), loop.getEnd (), loop.getStep ());
	}

	/**
	 *
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
	 *
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
