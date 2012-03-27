package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import ch.unibas.cs.hpwc.patus.util.StringUtil;

public class Comment implements IInstruction
{
	private String m_strComment;
	
	public Comment (String strComment)
	{
		m_strComment = strComment;
	}

	@Override
	public void issue (StringBuilder sbResult)
	{
		sbResult.append ("/* ");
		sbResult.append (m_strComment);
		sbResult.append (" */");
	}
	
	@Override
	public String toString ()
	{
		return StringUtil.concat ("/* ", m_strComment, " */");
	}
}
