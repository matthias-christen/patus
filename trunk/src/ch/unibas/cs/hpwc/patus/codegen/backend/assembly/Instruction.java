package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import cetus.hir.Specifier;
import ch.unibas.cs.hpwc.patus.arch.IArchitectureDescription;
import ch.unibas.cs.hpwc.patus.arch.TypeArchitectureType.Intrinsics.Intrinsic;

/**
 * This class encapsulates a single inline assembly instruction (mnemonic + operands).
 * 
 * @author Matthias-M. Christen
 */
public class Instruction implements IInstruction
{
	private String m_strIntrinsicBaseName;
	private IOperand[] m_rgOperands;
	
	public Instruction (String strIntrinsicBaseName, IOperand... rgOperands)
	{
		m_strIntrinsicBaseName = strIntrinsicBaseName;
		m_rgOperands = rgOperands;
	}
	
	public void issue (Specifier specDatatype, IArchitectureDescription arch, StringBuilder sbResult)
	{
		// try to find the intrinsic corresponding to m_strIntrinsicBaseName
		Intrinsic intrinsic = arch.getIntrinsic (m_strIntrinsicBaseName, specDatatype);
		
		// if the base name doesn't correspond to an intrinsic defined in the architecture description,
		// use m_strIntrinsicBaseName as instruction mnemonic
		String strInstruction = intrinsic == null ? m_strIntrinsicBaseName : intrinsic.getName ();
	
		//boolean bIsVectorInstruction = arch.getSIMDVectorLength (specDatatype) > 1;

		sbResult.append (strInstruction);
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
