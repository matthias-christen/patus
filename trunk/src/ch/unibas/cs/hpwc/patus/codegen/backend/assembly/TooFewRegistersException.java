package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import ch.unibas.cs.hpwc.patus.arch.TypeRegisterType;

public class TooFewRegistersException extends Exception
{
	private static final long serialVersionUID = 1L;

	private TypeRegisterType m_regtype;
	private int m_nExcessRegisterRequirement;
	
	public TooFewRegistersException (TypeRegisterType regtype, int nExcessRegisterRequirement)
	{
		m_regtype = regtype;
		m_nExcessRegisterRequirement = nExcessRegisterRequirement;
	}

	public TypeRegisterType getRegisterType ()
	{
		return m_regtype;
	}
	
	public int getExcessRegisterRequirement ()
	{
		return m_nExcessRegisterRequirement;
	}
}
