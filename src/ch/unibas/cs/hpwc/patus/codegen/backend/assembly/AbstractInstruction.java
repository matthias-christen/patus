package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import ch.unibas.cs.hpwc.patus.ast.ParameterAssignment;

public abstract class AbstractInstruction implements IInstruction
{
	///////////////////////////////////////////////////////////////////
	// Member Variables
	
	protected ParameterAssignment m_pa;

	
	///////////////////////////////////////////////////////////////////
	// Implementation
	
	public AbstractInstruction ()
	{
		m_pa = null;
	}

	@Override
	public void setParameterAssignment (ParameterAssignment pa)
	{
		m_pa = pa;
	}

	@Override
	public ParameterAssignment getParameterAssignment ()
	{
		return m_pa;
	}
}
