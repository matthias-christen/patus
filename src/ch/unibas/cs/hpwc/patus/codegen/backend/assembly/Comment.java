package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import java.util.Map;

import ch.unibas.cs.hpwc.patus.arch.TypeBaseIntrinsicEnum;
import ch.unibas.cs.hpwc.patus.ast.ParameterAssignment;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

public class Comment extends AbstractInstruction
{
	private String m_strComment;

	
	public Comment (String strComment)
	{
		m_strComment = strComment;
	}
	
	public Comment (String strComment, ParameterAssignment pa)
	{
		this (strComment);
		setParameterAssignment (pa);
	}

	@Override
	public void issue (StringBuilder sbResult)
	{
		sbResult.append ("/* ");
		sbResult.append (m_strComment);
		sbResult.append (" */");
	}
	
	@Override
	public TypeBaseIntrinsicEnum getIntrinsic ()
	{
		// this instruction doesn't correspond to an intrinsic
		return null;
	}
	
	@Override
	public String toString ()
	{
		return StringUtil.concat ("/* ", m_strComment, " */");
	}
	
	@Override
	public String toJavaCode (Map<IOperand, String> mapOperands)
	{
		return StringUtil.concat ("new Comment (\"", m_strComment, "\");");
	}
}
