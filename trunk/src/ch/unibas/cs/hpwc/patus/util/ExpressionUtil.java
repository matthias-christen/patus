package ch.unibas.cs.hpwc.patus.util;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import org.apache.log4j.Logger;

import cetus.hir.AssignmentExpression;
import cetus.hir.AssignmentOperator;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.DepthFirstIterator;
import cetus.hir.Expression;
import cetus.hir.FloatLiteral;
import cetus.hir.FunctionCall;
import cetus.hir.IntegerLiteral;
import cetus.hir.Literal;
import cetus.hir.NameID;
import cetus.hir.Specifier;
import cetus.hir.Traversable;
import cetus.hir.Typecast;
import ch.unibas.cs.hpwc.patus.analysis.HIRAnalyzer;
import ch.unibas.cs.hpwc.patus.codegen.Globals;
import ch.unibas.cs.hpwc.patus.symbolic.ExpressionData;
import ch.unibas.cs.hpwc.patus.symbolic.MaximaTimeoutException;
import ch.unibas.cs.hpwc.patus.symbolic.Symbolic;

public class ExpressionUtil
{
	///////////////////////////////////////////////////////////////////
	// Member Variables

	private final static Logger LOGGER = Logger.getLogger (ExpressionUtil.class);

	private static Map<String, Map<String, Map<Boolean, Expression>>> m_mapCeil = new HashMap<String, Map<String,Map<Boolean,Expression>>> ();


	///////////////////////////////////////////////////////////////////
	// Implementation


	/**
	 * Deterines whether <code>expr</code> is a number literal.
	 * @param expr The expression to test
	 * @return <code>true</code> iff <code>expr</code> is a number literal
	 */
	public static boolean isNumberLiteral (Expression expr)
	{
		return (expr instanceof IntegerLiteral) || (expr instanceof FloatLiteral);
	}

	/**
	 * Determines whether the expression <code>expr</code> is zero.
	 * @param expr The expression to test
	 * @return <code>true</code> iff <code>expr</code> is zeor
	 */
	public static boolean isZero (Expression expr)
	{
		if (expr instanceof Literal)
			return ExpressionUtil.isZeroLiteral ((Literal) expr);

		Expression exprSimplified = Symbolic.simplify (expr);
		if (exprSimplified instanceof Literal)
			return ExpressionUtil.isZeroLiteral ((Literal) exprSimplified);

		return false;
	}

	/**
	 *
	 * @param literal
	 * @return
	 */
	public static boolean isZeroLiteral (Literal literal)
	{
		if (literal instanceof IntegerLiteral)
			return ((IntegerLiteral) literal).getValue () == 0;
		if (literal instanceof FloatLiteral)
			return ((FloatLiteral) literal).getValue () == 0.0;
		return false;
	}

	/**
	 * Determines whether the expression <code>expr</code> has the value <code>nValue</code>.
	 * @param expr The expression to test
	 * @param nValue The expected value
	 * @return <code>true</code> if <code>expr</code> evaluates to <code>nValue</code>
	 */
	public static boolean isValue (Expression expr, int nValue)
	{
		if (expr instanceof Literal)
			return ExpressionUtil.isValueLiteral ((Literal) expr, nValue);

		Expression exprSimplified = Symbolic.simplify (expr);
		if (exprSimplified instanceof Literal)
			return ExpressionUtil.isValue (exprSimplified, nValue);

		return false;
	}

	/**
	 * Determines whether the literal <code>literal</code> is an integer literal and has the value <code>nValue</code>.
	 * @param literal The literal to examine
	 * @param nValue The value
	 * @return <code>true</code> iff literal is an {@link IntegerLiteral} and has the value <code>nValue</code>
	 */
	public static boolean isValueLiteral (Literal literal, int nValue)
	{
		if (literal instanceof IntegerLiteral)
			return ((IntegerLiteral) literal).getValue () == nValue;
		return false;
	}

	/**
	 * Tries to extract an integer constant from <code>expr</code>.
	 * @param expr The expression from which to extract an integer constant
	 * @return The integer represented by <code>expr</code>
	 */
	public static int getIntegerValue (Expression expr)
	{
		if (expr instanceof IntegerLiteral)
			return (int) ((IntegerLiteral) expr).getValue ();
		if (expr instanceof FloatLiteral)
			return (int) ((FloatLiteral) expr).getValue ();

		// try to simplify the expression
		Expression exprSimple = Symbolic.simplify (expr);
		if (exprSimple instanceof IntegerLiteral)
			return (int) ((IntegerLiteral) exprSimple).getValue ();
		if (exprSimple instanceof FloatLiteral)
			return (int) ((FloatLiteral) exprSimple).getValue ();

        throw new RuntimeException ("Cannot extract integer value");
	}

	/**
	 *
	 * @param expr
	 * @return
	 */
	public static double getFloatValue (Expression expr)
	{
		if (expr instanceof IntegerLiteral)
			return ((IntegerLiteral) expr).getValue ();
		if (expr instanceof FloatLiteral)
			return ((FloatLiteral) expr).getValue ();

		// try to simplify the expression
		Expression exprSimple = Symbolic.simplify (expr);
		if (exprSimple instanceof IntegerLiteral)
			return ((IntegerLiteral) exprSimple).getValue ();
		if (exprSimple instanceof FloatLiteral)
			return ((FloatLiteral) exprSimple).getValue ();

        throw new RuntimeException ("Cannot extract float value");
	}

	/**
	 * Creates an expression that calculates the exponential expression <code>exprBase</code>^<code>exprExponent</code>.
	 * @param exprBase The base
	 * @param exprExponent The exponent
	 * @return An expression that calculates <code>exprBase</code>^<code>exprExponent</code>
	 */
	public static Expression createExponentExpression (Expression exprBase, Expression exprExponent)
	{
		return createExponentExpression (
			new ExpressionData (exprBase, 0, Symbolic.EExpressionType.EXPRESSION),
			new ExpressionData (exprExponent, 0, Symbolic.EExpressionType.EXPRESSION)).getExpression ();
	}

	/**
	 *
	 * @param exprBase
	 * @param nExponent
	 * @return
	 */
	private static ExpressionData createExponentExpression (ExpressionData exprBase, int nExponent)
	{
		switch (nExponent)
		{
		case 0:
			return new ExpressionData (new IntegerLiteral (1), 0, Symbolic.EExpressionType.EXPRESSION);
		case 1:
			return exprBase;
		case -1:
			return new ExpressionData (
				new BinaryExpression (new FloatLiteral (1.0), BinaryOperator.DIVIDE, exprBase.getExpression ().clone ()),
				exprBase.getFlopsCount () + 1,
				Symbolic.EExpressionType.EXPRESSION);
		default:
			if (nExponent <= 5)
			{
				int nFlopsCount = exprBase.getFlopsCount ();
				Expression expr = exprBase.getExpression ();
				Expression exprNew = exprBase.getExpression ().clone ();

				for (int i = 0; i < Math.abs (nExponent) - 1; i++)
				{
					exprNew = new BinaryExpression (exprNew.clone (), BinaryOperator.MULTIPLY, expr.clone ());
					nFlopsCount += exprBase.getFlopsCount () + 1;
				}
				expr = exprNew;

				if (nExponent < 0)
				{
					return new ExpressionData (
						new BinaryExpression (new FloatLiteral (1.0), BinaryOperator.DIVIDE, expr.clone ()),
						nFlopsCount + 1,
						Symbolic.EExpressionType.EXPRESSION);
				}

				return new ExpressionData (expr, nFlopsCount, Symbolic.EExpressionType.EXPRESSION);
			}
		}

		return new ExpressionData (
			new FunctionCall (new NameID ("pow"), CodeGeneratorUtil.expressions (exprBase.getExpression (), new IntegerLiteral (nExponent))),
			exprBase.getFlopsCount () + 1,
			Symbolic.EExpressionType.EXPRESSION);
	}

	/**
	 *
	 * @param exprBase
	 * @param exprExponent
	 * @return
	 */
	public static ExpressionData createExponentExpression (ExpressionData exprBase, ExpressionData exprExponent)
	{
		if (ExpressionUtil.isZero (exprBase.getExpression ()))
			return new ExpressionData (new IntegerLiteral (0), 0, Symbolic.EExpressionType.EXPRESSION);
		if (ExpressionUtil.isValue (exprExponent.getExpression (), 1))
			return new ExpressionData (new IntegerLiteral (1), 0, Symbolic.EExpressionType.EXPRESSION);

		if (exprExponent.getExpression () instanceof IntegerLiteral)
			return ExpressionUtil.createExponentExpression (exprBase, (int) ((IntegerLiteral) exprExponent.getExpression ()).getValue ());

		if (exprExponent.getExpression () instanceof FloatLiteral)
		{
			double fExponent = ((FloatLiteral) exprExponent.getExpression ()).getValue ();

			if (Math.floor (fExponent) == fExponent)
				return ExpressionUtil.createExponentExpression (exprBase, (int) fExponent);

			if (fExponent == 0.5)
			{
				return new ExpressionData (
					new FunctionCall (new NameID ("sqrt"), CodeGeneratorUtil.expressions (exprBase.getExpression ())),
					exprBase.getFlopsCount () + 1,
					Symbolic.EExpressionType.EXPRESSION);
			}
		}

		// the generic case
		return new ExpressionData (
			new FunctionCall (new NameID ("pow"), CodeGeneratorUtil.expressions (exprBase.getExpression (), exprExponent.getExpression ())),
			exprBase.getFlopsCount () + exprExponent.getFlopsCount () + 1,
			Symbolic.EExpressionType.EXPRESSION);
	}

	/**
	 * Calculates the sum of the expressions <code>rgExpressions</code>.
	 * @param rgExpressions The expressions to add
	 * @return The sum of all the expressions <code>rgExpressions</code>
	 */
	public static Expression sum (Expression... rgExpressions)
	{
		Expression exprSum = null;
		for (Expression expr : rgExpressions)
		{
			if (exprSum == null)
				exprSum = expr.clone ();
			else
				exprSum = new BinaryExpression (exprSum, BinaryOperator.ADD, expr.clone ());
		}

		return exprSum == null ? new IntegerLiteral (0) : Symbolic.optimizeExpression (exprSum);
	}

	/**
	 * Calculates the product of the expressions <code>rgExpressions</code>.
	 * @param rgExpressions The expressions to multiply
	 * @return The product of all the expressions <code>rgExpressions</code>
	 */
	public static Expression product (Expression... rgExpressions)
	{
		Expression exprProduct = null;
		for (Expression expr : rgExpressions)
		{
			if (exprProduct == null)
				exprProduct = expr.clone ();
			else
				exprProduct = new BinaryExpression (exprProduct, BinaryOperator.MULTIPLY, expr.clone ());
		}

		return exprProduct == null ? new IntegerLiteral (1) : Symbolic.optimizeExpression (exprProduct);
	}

	/**
	 * Compares the literals <code>listLHS</code> and <code>litRHS</code> by means of the comparison
	 * operator <code>op</code>.
	 * @param litLHS The left hand side of the comparison
	 * @param op The comparison operator
	 * @param litRHS The right hand side of the comparison
	 * @return <code>true</code> iff the relation <code>litLHS</code> &lt;op&gt; <code>litRHS</code> holds
	 */
	public static boolean compare (IntegerLiteral litLHS, BinaryOperator op, IntegerLiteral litRHS)
	{
		long nLHS = litLHS.getValue ();
		long nRHS = litRHS.getValue ();

		if (op == BinaryOperator.COMPARE_EQ)
			return nLHS == nRHS;
		if (op == BinaryOperator.COMPARE_GE)
			return nLHS >= nRHS;
		if (op == BinaryOperator.COMPARE_GT)
			return nLHS > nRHS;
		if (op == BinaryOperator.COMPARE_LE)
			return nLHS <= nRHS;
		if (op == BinaryOperator.COMPARE_LT)
			return nLHS < nRHS;
		if (op == BinaryOperator.COMPARE_NE)
			return nLHS != nRHS;

		throw new RuntimeException (StringUtil.concat ("The operator ", op.toString (), " is no comparison operator"));
	}

	/**
	 * Returns the minimum of the expressions <code>rgExprs</code>
	 * or an expression that computes the minimum.
	 * @param rgExprs An array of expressions from which to find the minimum
	 * @return min(<code>expr1</code>, <code>expr2</code>, ...)
	 */
	public static Expression min (Expression... rgExprs)
	{
		List<Expression> listArgs = new LinkedList<Expression> ();

		long nMin = Long.MAX_VALUE;
		double fMin = Double.MAX_VALUE;
		for (Expression expr : rgExprs)
		{
			if (expr instanceof IntegerLiteral)
				nMin = Math.min (nMin, ((IntegerLiteral) expr).getValue ());
			else if (expr instanceof FloatLiteral)
				fMin = Math.min (fMin, ((FloatLiteral) expr).getValue ());
			else
				listArgs.add (expr.clone ());
		}

		if (nMin != Long.MAX_VALUE)
		{
			if (fMin != Double.MAX_VALUE)
			{
				// both nMax and fMax are used
				if (nMin < fMin)
					listArgs.add (new IntegerLiteral (nMin));
				else
					listArgs.add (new FloatLiteral (fMin));
			}
			else
			{
				// only nMax is used
				listArgs.add (new IntegerLiteral (nMin));
			}
		}
		else
		{
			if (fMin != Double.MAX_VALUE)
			{
				// only fMax is used
				listArgs.add (new FloatLiteral (fMin));
			}
		}

		if (listArgs.size () == 1)
			return listArgs.get (0);
		return new FunctionCall (Globals.FNX_MIN.clone (), listArgs);
	}

	/**
	 * Returns the minimum of the expressions <code>listExpressions</code>
	 * or an expression that computes the minimum.
	 * @param rgExprs An array of expressions from which to find the minimum
	 * @return min(<code>expr1</code>, <code>expr2</code>, ...)
	 */
	public static Expression min (List<Expression> listExpressions)
	{
		Expression[] rgExprs = new Expression[listExpressions.size ()];
		listExpressions.toArray (rgExprs);
		return ExpressionUtil.min (rgExprs);
	}

	/**
	 * Returns the maximum of the expressions <code>rgExpr</code>
	 * or an expression that computes the maximum.
	 * @param rgExprs An array of expressions from which to find the maximum
	 * @return max(<code>expr1</code>, <code>expr2</code>, ...)
	 */
	public static Expression max (Expression... rgExprs)
	{
		List<Expression> listArgs = new LinkedList<Expression> ();

		long nMax = Long.MIN_VALUE;
		double fMax = Double.MIN_VALUE;
		for (Expression expr : rgExprs)
		{
			if (expr instanceof IntegerLiteral)
				nMax = Math.max (nMax, ((IntegerLiteral) expr).getValue ());
			else if (expr instanceof FloatLiteral)
				fMax = Math.max (fMax, ((FloatLiteral) expr).getValue ());
			else
				listArgs.add (expr.clone ());
		}

		if (nMax != Long.MIN_VALUE)
		{
			if (fMax != Double.MIN_VALUE)
			{
				// both nMax and fMax are used
				if (nMax > fMax)
					listArgs.add (new IntegerLiteral (nMax));
				else
					listArgs.add (new FloatLiteral (fMax));
			}
			else
			{
				// only nMax is used
				listArgs.add (new IntegerLiteral (nMax));
			}
		}
		else
		{
			if (fMax != Double.MIN_VALUE)
			{
				// only fMax is used
				listArgs.add (new FloatLiteral (fMax));
			}
		}

		if (listArgs.size () == 1)
			return listArgs.get (0);
		return new FunctionCall (Globals.FNX_MAX.clone (), listArgs);
	}

	/**
	 * Returns the maximum of the expressions <code>listExpressions</code>
	 * or an expression that computes the maximum.
	 * @param rgExprs An array of expressions from which to find the maximum
	 * @return max(<code>expr1</code>, <code>expr2</code>, ...)
	 */
	public static Expression max (List<Expression> listExpressions)
	{
		Expression[] rgExprs = new Expression[listExpressions.size ()];
		listExpressions.toArray (rgExprs);
		return ExpressionUtil.max (rgExprs);
	}

	/**
	 * Returns an expression computing the floor of <code>expr</code>.
	 * @param expr The expression for which to compute the floor
	 * @return The floor of <code>expr</code>
	 */
	public static Expression floor (Expression expr)
	{
		Expression exprSimplified = Symbolic.simplify (expr);
		if (ExpressionUtil.isNumberLiteral (exprSimplified))
			return new IntegerLiteral (ExpressionUtil.getIntegerValue (exprSimplified));
		else if (exprSimplified instanceof BinaryExpression && ((BinaryExpression) exprSimplified).getOperator () == BinaryOperator.DIVIDE)
		{
			Expression exprNumerator = ((BinaryExpression) exprSimplified).getLHS ();
			Expression exprDenominator = ((BinaryExpression) exprSimplified).getRHS ();
			if (ExpressionUtil.isNumberLiteral (exprNumerator) && ExpressionUtil.isNumberLiteral (exprDenominator))
				return new IntegerLiteral (ExpressionUtil.getIntegerValue (exprNumerator) / ExpressionUtil.getIntegerValue (exprDenominator));

			return new Typecast (CodeGeneratorUtil.specifiers (ExpressionUtil.getSpecifier (new Traversable[] { expr }, Specifier.INT)), exprSimplified);
		}

		return new Typecast (CodeGeneratorUtil.specifiers (ExpressionUtil.getSpecifier (new Traversable[] { expr }, Specifier.INT)), exprSimplified);
	}

	/**
	 * Calculates the ceiling of the expression <code>expr</code>.
	 * If the expression is a number, it is simplified. If it is a fraction it is treated specially:
	 * we make use of the fact that <code>ceil(a/b) = floor((a+b-1)/b)</code>.
	 * @param expr The expression of which to calculate the ceiling.
	 * @return ceil(expr)
	 */
	public static Expression ceil (Expression expr)
	{
		return ExpressionUtil.ceil (expr, true);
	}

	/**
	 * Calculates the ceiling of the expression <code>expr</code>.
	 * If the expression is a number, it is simplified. If it is a fraction it is treated specially:
	 * we make use of the fact that <code>ceil(a/b) = floor((a+b-1)/b)</code>.
	 * @param expr The expression of which to calculate the ceiling.
	 * @return ceil(expr)
	 */
	public static Expression ceil (Expression expr, boolean bSimplify)
	{
		if (expr instanceof IntegerLiteral)
			return expr;

		if (expr instanceof FloatLiteral)
			return new IntegerLiteral ((int) Math.ceil (((FloatLiteral) expr).getValue ()));

		if (expr instanceof BinaryExpression && ((BinaryExpression) expr).getOperator ().equals (BinaryOperator.DIVIDE))
		{
			// if the expression is of the form ceil(a/b) convert it to (int)((a+b-1)/b)

			Expression exprNumerator = ((BinaryExpression) expr).getLHS ();
			Expression exprDenominator = ((BinaryExpression) expr).getRHS ();
			if (bSimplify)
			{
				exprNumerator = Symbolic.simplify (exprNumerator);
				exprDenominator = Symbolic.simplify (exprDenominator);
			}

			// return the typecast expression
			return new Typecast (
				// cast to int
				CodeGeneratorUtil.specifiers (ExpressionUtil.getSpecifier (new Traversable[] { expr }, Specifier.INT)),

				// compute (a+b-1) / b
				new BinaryExpression (
					new BinaryExpression (
						exprNumerator.clone (),
						BinaryOperator.ADD,
						new BinaryExpression (exprDenominator.clone (), BinaryOperator.SUBTRACT, new IntegerLiteral (1))),
					BinaryOperator.DIVIDE,
					exprDenominator.clone ()));
		}

		// try to simplify
		if (bSimplify)
		{
			try
			{
				return Symbolic.evaluateExpression (StringUtil.concat ("ceiling(", expr, ")"), expr);
			}
			catch (MaximaTimeoutException e)
			{
				// something went wrong; resort to the default...
			}
		}

		return new FunctionCall (new NameID ("ceil"), CodeGeneratorUtil.expressions (expr));
	}

	/**
	 * Calculates the ceiling of the expression <code>exprNumerator / exprDenominator</code>.
	 * If the expression is a number, it is simplified. Otherwise
	 * we make use of the fact that <code>ceil(a/b) = floor((a+b-1)/b)</code>.
	 * @param expr The expression of which to calculate the ceiling.
	 * @return ceil(expr)
	 */
	public static Expression ceil (Expression exprNumerator, Expression exprDenominator)
	{
		return ExpressionUtil.ceil (exprNumerator, exprDenominator, true);
	}

	/**
	 * Calculates the ceiling of the expression <code>exprNumerator / exprDenominator</code>.
	 * If the expression is a number, it is simplified. Otherwise
	 * we make use of the fact that <code>ceil(a/b) = floor((a+b-1)/b)</code>.
	 * @param expr The expression of which to calculate the ceiling.
	 * @return ceil(expr)
	 */
	public static Expression ceil (Expression exprNumerator, Expression exprDenominator, boolean bSimplify)
	{
		// determine whether the expression has been calculated already
		String strNumerator = exprNumerator.toString ();
		String strDenominator = exprDenominator.toString ();
		Map<String, Map<Boolean, Expression>> mapCeil2 = m_mapCeil.get (strNumerator);
		if (mapCeil2 == null)
			m_mapCeil.put (strNumerator, mapCeil2 = new HashMap<String, Map<Boolean, Expression>> ());
		Map<Boolean, Expression> map = mapCeil2.get (strDenominator);
		if (map == null)
			mapCeil2.put (strDenominator, map = new HashMap<Boolean, Expression> ());
		Expression exprResult = map.get (bSimplify);
		if (exprResult != null)
		{
			if (LOGGER.isDebugEnabled ())
				LOGGER.debug (StringUtil.concat ("Computing ceil(", strNumerator, ", ", strDenominator, ") = ", exprResult.toString (), " [cached]"));
			return exprResult.clone ();
		}

		// calculate explicitly if both numerator and denomiator are literals
		if (exprNumerator instanceof Literal && exprDenominator instanceof Literal)
		{
			double fNumerator = 0;
			double fDenominator = 1;

			if (exprNumerator instanceof IntegerLiteral)
				fNumerator = ((IntegerLiteral) exprNumerator).getValue ();
			else if (exprNumerator instanceof FloatLiteral)
				fNumerator = ((FloatLiteral) exprNumerator).getValue ();

			if (exprDenominator instanceof IntegerLiteral)
				fDenominator = ((IntegerLiteral) exprDenominator).getValue ();
			else if (exprDenominator  instanceof FloatLiteral)
				fDenominator = ((FloatLiteral) exprDenominator).getValue ();

			exprResult = new IntegerLiteral ((long) Math.ceil (fNumerator / fDenominator));
		}

		if (exprResult == null)
		{
			// simplify the term added to the numerator
			Expression exprDenomAddToNumerator = null;
			if (exprDenominator instanceof IntegerLiteral)
				exprDenomAddToNumerator = new IntegerLiteral (((IntegerLiteral) exprDenominator).getValue () - 1);
			else if (exprDenominator instanceof FloatLiteral)
			{
				// use ceil for floating point numbers
				exprResult = new FunctionCall (
					new NameID ("ceil"),
					CodeGeneratorUtil.expressions (new BinaryExpression (exprNumerator.clone (), BinaryOperator.DIVIDE, exprDenominator.clone ())));
			}
			else
				exprDenomAddToNumerator = new BinaryExpression (exprDenominator.clone (), BinaryOperator.SUBTRACT, new IntegerLiteral (1));

			if (exprResult == null)
			{
				// create the new numerator and simplify if desired
				Expression exprDenominatorNew = new BinaryExpression (exprNumerator.clone (), BinaryOperator.ADD, exprDenomAddToNumerator);
				if (bSimplify)
					exprDenominatorNew = Symbolic.simplify (exprDenominatorNew, Symbolic.ALL_VARIABLES_INTEGER);

				// divide and type cast
				// (note that the type cast prevents further simplification of the integer division by Maxima)
				exprResult = new Typecast (
					CodeGeneratorUtil.specifiers (ExpressionUtil.getSpecifier (new Traversable[] { exprNumerator, exprDenominator }, Specifier.INT)),
					new BinaryExpression (exprDenominatorNew, BinaryOperator.DIVIDE, exprDenominator.clone ()));
			}
		}

		map.put (bSimplify, exprResult);
		if (LOGGER.isDebugEnabled ())
			LOGGER.debug (StringUtil.concat ("Computing ceil(", strNumerator, ", ", strDenominator, ") = ", exprResult.toString ()));
		return exprResult;
	}

	/**
	 * Retrieves the first specifier that is encountered within <code>trv</code>.
	 * @param trv The traversable in which to search for a {@link Specifier}
	 * @return The first specifier encountered in <code>trv</code> or <code>specDefault</code>
	 * 	if no {@link Specifier} could be found
	 */
	public static Specifier getSpecifier (Traversable[] rgTraversables, Specifier specDefault)
	{
		for (Traversable trv : rgTraversables)
		{
			for (DepthFirstIterator it = new DepthFirstIterator (trv); it.hasNext (); )
			{
				Object obj = it.next ();
				if (obj instanceof Specifier && HIRAnalyzer.isIntegerSpecifier ((Specifier) obj))
					return (Specifier) obj;
				if (obj instanceof Typecast)
				{
					for (Object objSpec : ((Typecast) obj).getSpecifiers ())
						if ((objSpec instanceof Specifier) && HIRAnalyzer.isIntegerSpecifier ((Specifier) objSpec))
							return (Specifier) objSpec;
				}
			}
		}

		return specDefault;
	}

	/**
	 * Counts the number of Flops in the expression <code>expr</code>.
	 * @param expr The expression in which to count the number of Flops
	 * @return The number of Flops in the expression <code>expr</code>
	 */
	public static int getNumberOfFlops (Expression expr)
	{
		int nFlops = 0;
		for (DepthFirstIterator it = new DepthFirstIterator (expr); it.hasNext (); )
		{
			Object o = it.next ();
			if (o instanceof BinaryExpression)
			{
				// assume floating point expression: only +, -, *, / are valid operators
				BinaryOperator op = ((BinaryExpression) o).getOperator ();
				if (op.equals (BinaryOperator.ADD) || op.equals (BinaryOperator.SUBTRACT) || op.equals (BinaryOperator.MULTIPLY) || op.equals (BinaryOperator.DIVIDE))
					nFlops++;
			}
			else if (o instanceof AssignmentExpression)
			{
				// assume floating point expression: only +, -, *, / are valid operators
				AssignmentOperator op = ((AssignmentExpression) o).getOperator ();
				if (op.equals (AssignmentOperator.ADD) || op.equals (AssignmentOperator.SUBTRACT) || op.equals (AssignmentOperator.MULTIPLY) || op.equals (AssignmentOperator.DIVIDE))
					nFlops++;
			}
		}

		return nFlops;
	}

	/**
	 * Returns a binary expression for <code>expr+1</code>.
	 * @param expr
	 * @return
	 */
	public static Expression increment (Expression expr)
	{
		return new BinaryExpression (expr, BinaryOperator.ADD, Globals.ONE.clone ());
	}

	/**
	 * Returns a binary expression for <code>expr-1</code>.
	 * @param expr
	 * @return
	 */
	public static Expression decrement (Expression expr)
	{
		return new BinaryExpression (expr, BinaryOperator.SUBTRACT, Globals.ONE.clone ());
	}
}
