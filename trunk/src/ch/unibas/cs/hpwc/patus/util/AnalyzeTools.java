package ch.unibas.cs.hpwc.patus.util;

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
}
