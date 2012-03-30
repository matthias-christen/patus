package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import java.util.Map;
import java.util.Set;

import ch.unibas.cs.hpwc.patus.arch.TypeRegisterType;

public class InstructionListAnalyzer
{
	/**
	 * 
	 * @param reg
	 * @param nCurrentInstrIdx
	 * @return
	 */
	public static boolean isLastRead (InstructionList il, IOperand.PseudoRegister reg, int nCurrentInstrIdx,
		Map<IOperand.PseudoRegister, Set<IOperand.PseudoRegister>> mapSubstitutes)
	{
		if (!reg.getRegisterType ().equals (TypeRegisterType.SIMD))
			throw new RuntimeException ("Only implemented for SIMD registers");
		
		Set<IOperand.PseudoRegister> setSubstitutes = mapSubstitutes == null ? null : mapSubstitutes.get (reg);
		
		int nIdx = 0;
		for (IInstruction instruction : il)
		{
			if (nIdx > nCurrentInstrIdx)
			{
				if (instruction instanceof Instruction)
				{
					// check input operands (i.e., all operands except the last)
					IOperand[] rgOps = ((Instruction) instruction).getOperands ();
					for (int j = 0; j < rgOps.length - 1; j++)
					{
						// check whether the operand is equal to reg or if the operand is a substitutee for reg 
						if (rgOps[j] instanceof IOperand.PseudoRegister &&
							(reg.equals (rgOps[j]) || (setSubstitutes != null && setSubstitutes.contains (rgOps[j]))))
						{
							// another, later read was found
							return false;
						}
					}
				}				
			}
				
			nIdx++;
		}
		
		return true;
	}
}
