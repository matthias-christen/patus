package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import cetus.hir.Specifier;
import ch.unibas.cs.hpwc.patus.arch.IArchitectureDescription;
import ch.unibas.cs.hpwc.patus.arch.TypeArchitectureType.Intrinsics.Intrinsic;

public class Instruction
{
	private Intrinsic m_intrinsic;
	private IOperand[] m_rgOperands;
	
	public Instruction (Intrinsic intrinsic, IOperand... rgOperands)
	{
		m_intrinsic = intrinsic;
		m_rgOperands = rgOperands;
	}
	
	public Instruction (String strInstruction, IOperand... rgOperands)
	{
		this (new Intrinsic (), rgOperands);
		m_intrinsic.setName (strInstruction);
	}
	
	public void issue (Specifier specDatatype, IArchitectureDescription arch, StringBuilder sbResult)
	{
		boolean bIsVectorInstruction = arch.getSIMDVectorLength (specDatatype) > 1;

		sbResult.append (m_intrinsic.getName ());
		sbResult.append (" ");
		
		boolean bFirst = true;
		for (IOperand op : m_rgOperands)
		{
			if (!bFirst)
				sbResult.append (", ");
			sbResult.append (op.toString ());
			bFirst = false;
		}
		
		sbResult.append ("\\n\\t");
	}
}
