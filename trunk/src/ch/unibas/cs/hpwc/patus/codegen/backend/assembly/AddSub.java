package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import cetus.hir.BinaryOperator;
import cetus.hir.Expression;
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
	
	public String getBaseIntrinsic ()
	{
		return Globals.getIntrinsicBase (m_op).value ();
	}
	
	/*
	public String getInstruction ()
	{
		if (m_op.equals (BinaryOperator.ADD))
			return arch.getIntrinsic (m_op, null);
		if (m_op.equals (BinaryOperator.SUBTRACT))
			return IBackendAssemblyCodeGenerator.INSTR_SUB;
		return null;
	}*/		
	
	@Override
	public String toString ()
	{
		return StringUtil.concat (m_op.toString (), m_expr.toString ());
	}
}