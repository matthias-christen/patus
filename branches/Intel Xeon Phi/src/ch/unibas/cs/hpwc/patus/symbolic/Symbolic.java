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
package ch.unibas.cs.hpwc.patus.symbolic;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import cetus.hir.ArrayAccess;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.DepthFirstIterator;
import cetus.hir.Expression;
import cetus.hir.FunctionCall;
import cetus.hir.IDExpression;
import cetus.hir.IntegerLiteral;
import cetus.hir.Literal;
import cetus.hir.NameID;
import cetus.hir.Specifier;
import cetus.hir.Typecast;
import ch.unibas.cs.hpwc.patus.representation.StencilNode;
import ch.unibas.cs.hpwc.patus.util.CodeGeneratorUtil;
import ch.unibas.cs.hpwc.patus.util.ExpressionUtil;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 * Class for doing symbolic calculations and symbolic reasoning.
 * Uses the Maxima computer algebra system.
 *
 * @author Matthias-M. Christen
 */
public class Symbolic
{
	///////////////////////////////////////////////////////////////////
	// Constants

	public static final List<Expression> ALL_VARIABLES_POSITIVE = new ArrayList<> ();
	public static final List<Expression> ALL_VARIABLES_INTEGER = new ArrayList<> ();


	// Static initialization
	static
	{
		Symbolic.ALL_VARIABLES_POSITIVE.add (new BinaryExpression (new NameID ("ALL_VARIABLES___"), BinaryOperator.COMPARE_GT, new IntegerLiteral (0)));
		Symbolic.ALL_VARIABLES_INTEGER.add (new FunctionCall (new NameID ("declare"), CodeGeneratorUtil.expressions (new NameID ("ALL_VARIABLES___"), new NameID ("integer"))));
	}


	///////////////////////////////////////////////////////////////////
	// Inner Types

	public enum ELogicalValue
	{
		TRUE,
		FALSE,
		UNKNOWN;

		public static ELogicalValue fromString (String s)
		{
			if ("true".equals (s))
				return TRUE;
			if ("false".equals (s))
				return FALSE;
			return UNKNOWN;
		}
	}

	/**
	 * Type of an expression: simple expression, equation, or inequality.
	 */
	public enum EExpressionType
	{
		/**
		 * The expression is a simple expression
		 */
		EXPRESSION,

		/**
		 * The expression is an equality expression
		 */
		EQUATION,

		/**
		 * The expression is an inequality expression
		 */
		INEQUALITY
	}

	private static Map<Expression, ExpressionData> m_mapCache1 = new HashMap<> ();
	private static Map<String, ExpressionData> m_mapCache2 = new HashMap<> ();


	///////////////////////////////////////////////////////////////////
	// Implementation

	/**
	 * Resets the Maxima variables.
	 */
	public static void reset ()
	{
		try
		{
			Maxima maxima = Maxima.getInstance ();
			if (maxima != null)
				maxima.executeExpectingSingleOutput ("reset(); kill(all);");
		}
		catch (MaximaTimeoutException e)
		{
		}
	}

	/**
	 * Evaluates the expression <code>expr</code>. If it can't be evaluated,
	 * the original expression is returned.
	 * @param expr The expression to evaluate
	 * @return The result of the evaluation of <code>expr</code>
	 */
	public static Expression evaluateExpression (Expression expr)
	{
		try
		{
			return Symbolic.evaluateExpression (Symbolic.toMaximaString (expr), expr);
		}
		catch (Exception e)
		{
			return expr;
		}
	}

	/**
	 *
	 * @param strExpression
	 * @return
	 * @throws MaximaTimeoutException
	 */
	public static Expression evaluateExpression (String strExpression, Expression... rgExprOrigs) throws MaximaTimeoutException
	{
		return Symbolic.evaluateExpressionEx (strExpression, rgExprOrigs).getExpression ();
	}

	/**
	 *
	 * @param strExpression
	 * @return The evaluated expression, wrapped in an {@link ExpressionData} object. The result can be <code>null</code> if there was a Maxima error.
	 * @throws MaximaTimeoutException
	 */
	public static ExpressionData evaluateExpressionEx (String strExpression) throws MaximaTimeoutException
	{
		return Symbolic.evaluateExpressionEx (strExpression);
	}

	/**
	 * Evaluates the expression <code>strExpression</code> and returns the simplified expression along with some expression metrics.
	 * @param strExpression The expression to simplify
	 * @param rgExprOrigs Original expressions which contain identifiers occurring in <code>strExpression</code>.
	 * 	The identifiers found in <code>rgExprOrigs</code> will be used when parsing the result expression
	 * @return The evaluated expression, wrapped in an {@link ExpressionData} object. The result can be <code>null</code> if there was a Maxima error.
	 * @throws MaximaTimeoutException
	 */
	public static ExpressionData evaluateExpressionEx (String strExpression, Expression... rgExprOrigs) throws MaximaTimeoutException
	{
		ExpressionData edResult = Symbolic.m_mapCache2.get (strExpression);
		if (edResult != null)
			return new ExpressionData (edResult);

		Maxima maxima = Maxima.getInstance ();
		if (maxima != null)
		{
			// build the Maxima expression string that can be parsed and feed the expression to Maxima
			Symbolic.m_mapCache2.put (
				strExpression,
				edResult = ExpressionParser.parseEx (
					maxima.executeExpectingSingleOutput (StringUtil.concat ("string(", strExpression, ");")),
					rgExprOrigs
				)
			);

			return edResult;
		}

		// Maxima not found...
		// use the Cetus simplifier
		Symbolic.m_mapCache2.put (
			strExpression,
			edResult = ExpressionParser.parseEx (
				cetus.hir.Symbolic.simplify (ExpressionParser.parse (strExpression)).toString (),
				rgExprOrigs
			)
		);

		return edResult;
	}

	/**
	 * Evaluates an expression that is expected to return a logical value (<code>true</code>, <code>false</code>,
	 * <code>unknown</code>). The expression is executed under the assumption that the expressions in
	 * <code>listAssumptions</code> hold.
	 * If the expression can't be evaulated, {@link ELogicalValue#UNKNOWN} is returned.
	 * @param expr The expression to evaluate
	 * @param listAssumptions A list of assumptions (inequalities that hold). Can be <code>null</code> if there are no assumptions
	 * @return The logical value the expression <code>expr</code> (under assumption of the expression in
	 * <code>listAssumptions</code>) evaluates to
	 * @throws MaximaTimeoutException
	 */
	public static ELogicalValue evaluateLogical (Expression expr, List<Expression> listAssumptions) throws MaximaTimeoutException
	{
		try
		{
			return Symbolic.evaluateLogical (Symbolic.toMaximaString (expr), listAssumptions);
		}
		catch (Exception e)
		{
			return ELogicalValue.UNKNOWN;
		}
	}

	/**
	 * Evaluates the Maxima expression <code>strExpression</code> that is expected to return a logical value
	 * (<code>true</code>, <code>false</code>, <code>unknown</code>).
	 * The expression is executed under the assumption that the expressions in <code>listAssumptions</code> hold.
	 * @param strExpression The Maxima expression to evaluate
	 * @param listAssumptions A list of assumptions (inequalities that hold). Can be <code>null</code> if there are no assumptions
	 * @return The logical value the expression <code>expr</code> (under assumption of the expression in
	 * <code>listAssumptions</code>) evaluates to
	 * @return The logical value the expression <code>expr</code> (under assumption of the expression in
	 * <code>listAssumptions</code>) evaluates to
	 * @throws MaximaTimeoutException
	 */
	public static ELogicalValue evaluateLogical (String strExpression, List<Expression> listAssumptions) throws MaximaTimeoutException
	{
		Maxima maxima = Maxima.getInstance ();
		if (maxima != null)
		{
			// make assumptions about variables
			Symbolic.makeAssumptions (strExpression, listAssumptions);

			// evaluate the expression
			ELogicalValue lvResult = ELogicalValue.fromString (maxima.executeExpectingSingleOutput (strExpression));

			// reset the system
			if (listAssumptions != null)
				maxima.executeExpectingSingleOutput ("reset(); kill(all);");

			return lvResult;
		}

		// Maxima not found...
		return ELogicalValue.UNKNOWN;
	}

	/**
	 * Make assumptions about variables in Maxima.
	 * @param strExpression
	 * @param listAssumptions
	 * @throws MaximaTimeoutException
	 */
	public static void makeAssumptions (String strExpression, List<Expression> listAssumptions) throws MaximaTimeoutException
	{
		Maxima maxima = Maxima.getInstance ();
		if (maxima != null)
		{
			// build a list of predicates and feed them to Maxima
			if (listAssumptions != null)
			{
				// assume that all variable are positive
				if (listAssumptions.equals (Symbolic.ALL_VARIABLES_POSITIVE))
					maxima.executeExpectingSingleOutput (StringUtil.concat ("for var in listofvars(", strExpression, ") do assume(var > 0);"));
				else if (listAssumptions.equals (Symbolic.ALL_VARIABLES_INTEGER))
				{
					String strMaximaResult = maxima.executeExpectingSingleOutput (StringUtil.concat ("string (listofvars(", strExpression, "));"));
					for (String strVar : strMaximaResult.substring (1, strMaximaResult.length () - 1).split (","))
					{
						strVar = strVar.trim ().replaceAll ("\\[.*\\]", "");
						if (!"".equals (strVar))
							maxima.executeExpectingSingleOutput (StringUtil.concat ("declare(", strVar, ", integer);"));
					}
				}
				else
				{
					// build the assume statement
					StringBuilder sb = new StringBuilder ("assume(");
					StringUtil.joinAsBuilder (listAssumptions, ",", sb);
					sb.append (");");

					maxima.executeExpectingSingleOutput (sb.toString ());
				}
			}
		}
	}

	/**
	 * Determines whether the expression <code>expr</code> evaluates to TRUE.
	 * @param expr The expression to test
	 * @param listAssumptions A list of assumptions (inequalities that hold).
	 * 	Can be <code>null</code> if there are no assumptions.
	 * @return
	 */
	public static ELogicalValue isTrue (BinaryExpression expr, List<Expression> listAssumptions)
	{
		ELogicalValue lvResult = ELogicalValue.UNKNOWN;
		try
		{
			String strExpression = Symbolic.toMaximaString (expr);

			// make assumptions before building the new expression
			Symbolic.makeAssumptions (strExpression, listAssumptions);

			StringBuilder sb = new StringBuilder ("is(");
			sb.append (strExpression);
			sb.append (");");

			// evaluate; the assumptions have already been made
			lvResult = Symbolic.evaluateLogical (sb.toString (), null);
		}
		catch (NotConvertableException e)
		{
		}
		catch (MaximaTimeoutException e)
		{
		}

		if (listAssumptions != null)
			Symbolic.reset ();
		return lvResult;
	}

	/**
	 * Determines whether the expression <code>expr</code> is positive.
	 * @param expr The expression to test
	 * @param listAssumptions A list of assumptions (inequalities that hold). Can be <code>null</code> if there are no assumptions.
	 * @return
	 */
	public static ELogicalValue isPositive (Expression expr, List<Expression> listAssumptions)
	{
		return Symbolic.isTrue (
			new BinaryExpression (expr.clone (), BinaryOperator.COMPARE_GT, new IntegerLiteral (0)),
			listAssumptions);
	}

	/**
	 * Determines whether the expression <code>expr</code> evaluates to a number
	 * (as opposed to a symbolic expression).
	 * If it can't be determined whether the expression evalutates to a numeric value, {@link ELogicalValue#UNKNOWN}
	 * is returned.
	 * @param expr The expression to test
	 * @return {@link ELogicalValue#TRUE} if it can be safely said that <code>expr</code> is a numeric value
	 */
	public static ELogicalValue isNumber (Expression expr)
	{
		try
		{
			StringBuilder sb = new StringBuilder ("numberp(ratsimp(");
			sb.append (Symbolic.toMaximaString (expr));
			sb.append ("))");

			return Symbolic.evaluateLogical (sb.toString (), null);
		}
		catch (NotConvertableException e)
		{
			return ELogicalValue.UNKNOWN;
		}
		catch (MaximaTimeoutException e)
		{
			return ELogicalValue.UNKNOWN;
		}
	}

	/**
	 * Determines whether the expression <code>expr</code> is dependent of the identifier <code>id</code>.
	 * Note that the expression isn't simply scanned for occurrences of <code>id</code>, but instead
	 * the expression is simplified symbolically and the analysis is done on the simplified expression.
	 * @param expr The expression to examine
	 * @param id The identifier to look for in the expression <code>expr</code>
	 * @return
	 */
	public static ELogicalValue isIndependentOf (Expression expr, IDExpression id)
	{
		try
		{
			StringBuilder sb = new StringBuilder ("freeof(");
			sb.append (id.toString ());
			sb.append (", ratsimp(");
			sb.append (Symbolic.toMaximaString (expr));
			sb.append ("));");

			return Symbolic.evaluateLogical (sb.toString (), null);
		}
		catch (NotConvertableException e)
		{
			return ELogicalValue.UNKNOWN;
		}
		catch (MaximaTimeoutException e)
		{
			return ELogicalValue.UNKNOWN;
		}
	}

	public static Expression simplify (Expression expr)
	{
		return Symbolic.simplify (expr, null);
	}

	/**
	 * Tries to simplify the expression <code>expr</code>.
	 * @param expr The expression to simplify
	 * @return The simplified expression
	 */
	public static Expression simplify (Expression expr, List<Expression> listAssumptions)
	{
		ExpressionData ed = Symbolic.simplifyEx (expr, listAssumptions);
		return ed == null ? null : ed.getExpression ();
	}
	
	public static ExpressionData simplifyEx (Expression expr)
	{
		return Symbolic.simplifyEx (expr, null);
	}
	
	public static ExpressionData simplifyEx (Expression expr, List<Expression> listAssumptions)
	{
		// literals can't be simplified any further
		if ((expr instanceof Literal) || (expr instanceof IDExpression))
			return new ExpressionData (expr, 0, Symbolic.EExpressionType.EXPRESSION);

		// TODO: factor the assumptions into the expression cache
		ExpressionData edResult = Symbolic.m_mapCache1.get (expr);
		if (edResult != null)
			return edResult.clone ();

		try
		{
			// convert the expression
			String strExpression = Symbolic.toMaximaString (expr);

			// make assumptions about variables
			Symbolic.makeAssumptions (strExpression, listAssumptions);

			// evaluate the expression
			edResult = Symbolic.evaluateExpressionEx (StringUtil.concat ("ratsimp(", strExpression, ")"), expr);
			if (isAcceptable (edResult.getExpression (), listAssumptions))
				Symbolic.m_mapCache1.put (expr, edResult);
			else
				edResult = new ExpressionData (expr, ExpressionUtil.getNumberOfFlops (expr), Symbolic.EExpressionType.EXPRESSION);

			// reset the system
			if (listAssumptions != null)
				Symbolic.reset ();

			return edResult;
		}
		catch (NotConvertableException e)
		{
			// an error has occurred; just return the original expression
			Symbolic.m_mapCache1.put (expr, edResult = new ExpressionData (expr, ExpressionUtil.getNumberOfFlops (expr), Symbolic.EExpressionType.EXPRESSION));
			return edResult;
		}
		catch (MaximaTimeoutException e)
		{
			// an error has occurred; just return the original expression
			Symbolic.m_mapCache1.put (expr, edResult = new ExpressionData (expr, ExpressionUtil.getNumberOfFlops (expr), Symbolic.EExpressionType.EXPRESSION));
			return edResult;
		}
	}

	public static boolean isAcceptable (Expression expr, List<Expression> listAssumptions)
	{
		if (listAssumptions == ALL_VARIABLES_INTEGER)
		{
			// we don't want any "ceil"s in integer-only expressions
			boolean bContainsCeil = false;
			for (DepthFirstIterator it = new DepthFirstIterator (expr); it.hasNext (); )
			{
				Object o = it.next ();
				if (o instanceof FunctionCall)
				{
					Expression exprName = ((FunctionCall) o).getName ();
					if (exprName instanceof IDExpression)
					{
						if (((IDExpression) exprName).getName ().equals ("ceil"))
						{
							bContainsCeil = true;
							break;
						}
					}
				}
			}

			if (bContainsCeil)
				return false;
		}

		return true;
	}

	/**
	 * Optimizes the expression <code>expr</code>, i.e. tries to simplify it so that
	 * the number of operations becomes minimal.
	 * @param expr The expression to optimize
	 * @return The optimized expression
	 */
	public static Expression optimizeExpression (Expression expr)
	{
		return Symbolic.optimizeExpression (expr, null);
	}

	public static Expression optimizeExpression (Expression expr, List<Expression> listAssumptions)
	{
		try
		{
			return ExpressionOptimizer.optimize (expr, listAssumptions);
		}
		catch (NotConvertableException e)
		{
			return Symbolic.simplify (expr, listAssumptions);
		}
	}

	private static Expression replaceOperators (Expression expr, boolean bModified)
	{
		boolean bModifiedLocal = bModified;
		Expression exprNew = expr.clone ();

		// replace % operator
		for (DepthFirstIterator it = new DepthFirstIterator (exprNew); it.hasNext (); )
		{
			Object obj = it.next ();
			if (obj instanceof BinaryExpression)
			{
				BinaryExpression bexpr = (BinaryExpression) obj;
				if (bexpr.getOperator () == BinaryOperator.MODULUS)
				{
					bModifiedLocal = true;

					Expression exprLHS = bexpr.getLHS ();
					Expression exprRHS = bexpr.getRHS ();
					exprLHS.setParent (null);
					exprRHS.setParent (null);

					if (bexpr == exprNew)
						return Symbolic.replaceOperators (new FunctionCall (new NameID ("mod"), CodeGeneratorUtil.expressions (exprLHS, exprRHS)), true);
					bexpr.swapWith (new FunctionCall (new NameID ("mod"), CodeGeneratorUtil.expressions (exprLHS, exprRHS)));
				}
			}
			else if (obj instanceof Typecast)
			{
				Typecast cast = (Typecast) obj;
				if (cast.getSpecifiers ().size () == 1 && cast.getSpecifiers ().get (0).equals (Specifier.INT))
				{
					// cast to int: replace by "floor" or the cast expression
					bModifiedLocal = true;

					Expression exprCast = (Expression) cast.getChildren ().get (0);
					exprCast.setParent (null);

					if (cast == exprNew)
						return Symbolic.replaceOperators (new FunctionCall (new NameID ("floor"), CodeGeneratorUtil.expressions (exprCast)), true);
					cast.swapWith (new FunctionCall (new NameID ("floor"), CodeGeneratorUtil.expressions (exprCast)));
				}
			}
		}

		return bModifiedLocal ? exprNew : expr;
	}

	/**
	 * Converts the expression <code>expr</code> to a Maxima-compatible string.
	 * @param expr The expression to convert
	 * @return The expression <code>expr</code> as a Maxima-compatible string
	 */
	public static String toMaximaString (Expression expr) throws NotConvertableException
	{
		// replace operators if necessary
		Expression expr1 = Symbolic.replaceOperators (expr, false);

		// Hack
		// Maxima uses ":" as assignment, "=" as comparison

		List<StencilNode> listNodes = new ArrayList<> ();
		for (DepthFirstIterator it = new DepthFirstIterator (expr1); it.hasNext (); )
		{
			Object o = it.next ();
			if (o instanceof StencilNode)
			{
				StencilNode n = (StencilNode) o;
				n.setExpandedPrintMethod ();
				listNodes.add (n);
			}
			else if (o instanceof FunctionCall)
			{
				// replace "pow(a,b)" by "a^b" 
				FunctionCall f = (FunctionCall) o;
				if (f.getName () instanceof IDExpression && ("pow".equals (((IDExpression) f.getName ()).getName ()) || "powf".equals (((IDExpression) f.getName ()).getName ())) && f.getNumArguments () == 2)
				{
					Expression exprReplace = new BinaryExpression (f.getArgument (0).clone (), BinaryOperator.BITWISE_EXCLUSIVE_OR, f.getArgument (1).clone ());
					if (o == expr1)
					{
						expr1 = exprReplace;
						break;
					}
					f.swapWith (exprReplace);
				}
			}
		}
		
		String strExpr = expr1.toString ();
		
		for (StencilNode node : listNodes)
			node.setDefaultPrintMethod ();
		
		if (strExpr.indexOf (">>") >= 0)
			throw new NotConvertableException (expr1);
		if (strExpr.indexOf ("<<") >= 0)
			throw new NotConvertableException (expr1);

		return strExpr.replaceAll ("==", "=");
	}


	///////////////////////////////////////////////////////////////////
	// Testing...

	public static void main (String[] args) throws Exception
	{
//		System.out.println (Symbolic.isPositive (new NameID ("i"), Symbolic.ALL_VARIABLES_POSITIVE));
//		System.out.println (Symbolic.isPositive (new NameID ("i"), null));
//		System.out.println (Symbolic.isTrue (new BinaryExpression (new NameID ("i"), BinaryOperator.COMPARE_EQ, new IntegerLiteral (0)), null));
//
//		System.out.println (Symbolic.simplify (new BinaryExpression (new NameID ("a"), BinaryOperator.MODULUS, new IntegerLiteral (1)), ALL_VARIABLES_INTEGER));

		Expression exprSubscripted1 = new ArrayAccess (new NameID ("u"), CodeGeneratorUtil.expressions (new IntegerLiteral (1), new NameID ("idx0")));
		System.out.println (exprSubscripted1.toString ());
		System.out.println (Symbolic.optimizeExpression (exprSubscripted1));

		Expression exprSubscripted2 = new ArrayAccess (new ArrayAccess (new NameID ("u"), new IntegerLiteral (1)), new NameID ("idx0"));
		System.out.println (exprSubscripted2.toString ());
		System.out.println (Symbolic.optimizeExpression (exprSubscripted2));

		Maxima maxima = Maxima.getInstance ();
		if (maxima != null)
			maxima.close ();

		System.out.println (cetus.hir.Symbolic.simplify (new BinaryExpression (new IntegerLiteral (3), BinaryOperator.ADD, new IntegerLiteral (5))));
	}
}
