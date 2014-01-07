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

import cetus.hir.Expression;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 * A wrapper class around an expression that provides some more information
 * extracted from the Maxima expression when it is being parsed.
 */
public class ExpressionData
{
	private Expression m_expression;
	private int m_nFlops;
	private Symbolic.EExpressionType m_type;

	public ExpressionData (Expression expr, int nFlops, Symbolic.EExpressionType type)
	{
		m_expression = expr;
		m_nFlops = nFlops;
		m_type = type;
	}

	public ExpressionData (ExpressionData ed)
	{
		m_expression = ed.getExpression ().clone ();
		m_nFlops = ed.getFlopsCount ();
		m_type = ed.getType ();
	}

	public Expression getExpression ()
	{
		return m_expression;
	}

	public int getFlopsCount ()
	{
		return m_nFlops;
	}

	public Symbolic.EExpressionType getType ()
	{
		return m_type;
	}

	@Override
	public ExpressionData clone ()
	{
		return new ExpressionData (this);
	}

	@Override
	public String toString ()
	{
		return StringUtil.concat (m_expression.toString (), "  (", m_nFlops, " Flops)");
	}
}
