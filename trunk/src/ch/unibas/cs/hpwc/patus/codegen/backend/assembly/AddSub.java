package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import cetus.hir.BinaryOperator;
import cetus.hir.Expression;
import ch.unibas.cs.hpwc.patus.arch.TypeBaseIntrinsicEnum;
import ch.unibas.cs.hpwc.patus.codegen.Globals;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 * 
 * @author Matthias-M. Christen
 */
class AddSub
{
	///////////////////////////////////////////////////////////////////
	// Member Variables

	private Expression m_expr;
	private BinaryOperator m_op;
	
	
	///////////////////////////////////////////////////////////////////
	// Implementation

	public AddSub (BinaryOperator op, Expression expr)
	{
		m_op = op;
		m_expr = expr;
	}
	
	public Expression getExpression ()
	{
		return m_expr;
	}
	
	public TypeBaseIntrinsicEnum getBaseIntrinsic ()
	{
		return Globals.getIntrinsicBase (m_op);
	}
	
	public BinaryOperator getOperator ()
	{
		return m_op;
	}
	
	@Override
	public String toString ()
	{
		return StringUtil.concat (m_op.toString (), m_expr.toString ());
	}
}