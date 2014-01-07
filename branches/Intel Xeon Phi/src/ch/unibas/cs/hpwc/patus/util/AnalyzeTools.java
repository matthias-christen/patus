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

import cetus.hir.BinaryOperator;
import cetus.hir.DepthFirstIterator;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.IDExpression;
import cetus.hir.Statement;

/**
 *
 * @author Matthias-M. Christen
 */
public class AnalyzeTools
{
	///////////////////////////////////////////////////////////////////
	// Member Variables

	///////////////////////////////////////////////////////////////////
	// Implementation

	/**
	 * Checks whether the expression <code>expr</code> depends on the identifier <code>id</code>
	 * @param expr The expression to check whether it depends on <code>id</code>
	 * @param id The identifier
	 * @return <code>true</code> iff <code>id</code> occurs in <code>expr</code>
	 */
	public static boolean dependsExpressionOnIdentifier (Expression expr, IDExpression id)
	{
		for (DepthFirstIterator it = new DepthFirstIterator (expr); it.hasNext (); )
		{
			Object obj = it.next ();
			if (obj instanceof IDExpression)
				if (obj.equals (id))
					return true;
		}

		return false;
	}

	/**
	 * Determines whether the statement <code>stmt</code> contains a statement
	 * that &quot;does something&quot;.
	 * @param stmt
	 * @return
	 */
	public static boolean containsEffectiveStatement (Statement stmt)
	{
		for (DepthFirstIterator it = new DepthFirstIterator (stmt); it.hasNext (); )
			if (it.next () instanceof ExpressionStatement)
				return true;

		return false;
	}
	
	public static boolean isComparisonOperator (BinaryOperator op)
	{
		return op.equals (BinaryOperator.COMPARE_EQ) || op.equals (BinaryOperator.COMPARE_NE) ||
			op.equals (BinaryOperator.COMPARE_LT) || op.equals (BinaryOperator.COMPARE_LE) ||
			op.equals (BinaryOperator.COMPARE_GT) || op.equals (BinaryOperator.COMPARE_GE);
	}
}
