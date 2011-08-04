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

import java.util.List;

import cetus.hir.Expression;
import cetus.hir.IDExpression;
import cetus.hir.Literal;
import ch.unibas.cs.hpwc.patus.util.ExpressionUtil;

/**
 *
 * @author Matthias-M. Christen
 */
public class ExpressionOptimizer
{
	///////////////////////////////////////////////////////////////////
	// Constants

	/**
	 * Maxima simplification functions.
	 */
	protected enum ESimplifactionFunctions
	{
		RATSIMP ("", "ratsimp"),
		FULLRATSIMP ("", "fullratsimp"),
		FACTOR ("", "factor"),
		FACTORSUM ("", "factorsum"),
		GCFAC ("scifac", "gcfac"),
		SIMPLIFYSUM ("simplify_sum", "simplify_sum");


		private String m_strPackage;
		private String m_strFunction;

		private ESimplifactionFunctions (String strPackage, String strFunction)
		{
			m_strPackage = strPackage;
			m_strFunction = strFunction;
		}

		/**
		 * Evaluates the expression <code>strExpression</code> using the simplification function.
		 * @param strExpression The expression to simplify
		 * @param rgExprOrigs Original expressions which contain identifiers occurring in <code>strExpression</code>.
		 * 	The identifiers found in <code>rgExprOrigs</code> will be used when parsing the result expression
		 * @return The simplified expression
		 * @throws MaximaTimeoutException
		 */
		public ExpressionData evaluate (String strExpression, Expression... rgExprOrigs) throws MaximaTimeoutException
		{
			StringBuilder sb = new StringBuilder ();

			// load the package
			if (!"".equals (m_strPackage))
			{
				sb.append ("if properties(");
				sb.append (m_strFunction);
				sb.append (") = [] then load(");
				sb.append (m_strPackage);
				sb.append (");");
				Maxima.getInstance ().executeExpectingSingleOutput (sb.toString ());
			}

			sb.setLength (0);
			sb.append (m_strFunction);
			sb.append ("(");
			sb.append (strExpression);
			sb.append (")");
			return Symbolic.evaluateExpressionEx (sb.toString (), rgExprOrigs);
		}
	}


	///////////////////////////////////////////////////////////////////
	// Implementation

	/**
	 * Tries to optimize the expression <code>expr</code> such that the
	 * Flop count is minimized.
	 * @param expr The expression to simplify
	 * @return
	 * @throws NotConvertableException
	 */
	public static Expression optimize (Expression expr) throws NotConvertableException
	{
		// literals and ID expressions can't be simplified any further
		if ((expr instanceof Literal) || (expr instanceof IDExpression))
			return expr;

		return ExpressionOptimizer.optimizeEx (expr).getExpression ();
	}

	public static Expression optimize (Expression expr, List<Expression> listAssumptions) throws NotConvertableException
	{
		// literals and ID expressions can't be simplified any further
		if ((expr instanceof Literal) || (expr instanceof IDExpression))
			return expr;

		return ExpressionOptimizer.optimizeEx (expr, listAssumptions).getExpression ();
	}

//	/**
//	 * Tries to optimize the expression <code>strExpression</code> such that the
//	 * Flop count is minimized.
//	 * @param strExpression The expression to simplify
//	 * @return
//	 */
//	public static Expression optimize (String strExpression, Expression... rgExprOrigs)
//	{
//		return ExpressionOptimizer.optimizeEx (strExpression, rgExprOrigs).getExpression ();
//	}

	public static ExpressionData optimizeEx (Expression expr) throws NotConvertableException
	{
		return ExpressionOptimizer.optimizeEx (
			Symbolic.toMaximaString (expr),
			null,
			new ExpressionData (expr, ExpressionUtil.getNumberOfFlops (expr), Symbolic.EExpressionType.EXPRESSION),
			expr
		);
	}

	public static ExpressionData optimizeEx (Expression expr, List<Expression> listAssumptions) throws NotConvertableException
	{
		return ExpressionOptimizer.optimizeEx (
			Symbolic.toMaximaString (expr),
			listAssumptions,
			new ExpressionData (expr, ExpressionUtil.getNumberOfFlops (expr), Symbolic.EExpressionType.EXPRESSION),
			expr
		);
	}

	/**
	 * Simplifies the expression <code>strExpression</code>.
	 * @param expr The expression to optimize
	 * @param rgExprOrigs The
	 * @return
	 */
	public static ExpressionData optimizeEx (String strExpression, List<Expression> listAssumptions, ExpressionData dataOrig, Expression... rgExprOrigs) throws NotConvertableException
	{
		ExpressionData dataOpt = dataOrig;
		int nFlops = dataOpt.getFlopsCount ();

		// make assumptions about variables
		try
		{
			Symbolic.makeAssumptions (strExpression, listAssumptions);
		}
		catch (MaximaTimeoutException e1)
		{
		}

		for (ESimplifactionFunctions fnx : ESimplifactionFunctions.values ())
		{
			try
			{
				// evaluate the expression using the simplification function fnx
				ExpressionData data = fnx.evaluate (strExpression, rgExprOrigs);

				// data can be null if an error occurred
				if (data == null || !Symbolic.isAcceptable (data.getExpression (), listAssumptions))
					continue;

				if (data.getFlopsCount () < nFlops)
				{
					nFlops = data.getFlopsCount ();
					dataOpt = data;

					// we expect we can't get below 1 Flop (if an op is included)
					if (nFlops <= 1)
						break;
				}
			}
			catch (MaximaTimeoutException e)
			{
			}
		}

//		if (dataOpt == null)
//			return ExpressionParser.parseEx (strExpression, expr);
		return dataOpt;
	}
}
