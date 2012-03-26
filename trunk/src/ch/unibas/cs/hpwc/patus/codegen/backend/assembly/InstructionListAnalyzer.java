package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

public class InstructionListAnalyzer
{
	/**
	 * 
	 * @param reg
	 * @param nCurrentInstrIdx
	 * @return
	 */
	public static boolean isLastRead (InstructionList il, IOperand.PseudoRegister reg, int nCurrentInstrIdx)
	{
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
						if (rgOps[j] instanceof IOperand.PseudoRegister && reg.equals (rgOps[j]))
						{
							// another, later read was found
							return false;
						}
				}				
			}
				
			nIdx++;
		}
		
		return true;
	}
}
