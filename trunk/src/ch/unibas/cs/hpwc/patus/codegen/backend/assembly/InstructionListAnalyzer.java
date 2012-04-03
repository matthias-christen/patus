package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import java.util.Map;
import java.util.Set;

import ch.unibas.cs.hpwc.patus.arch.TypeRegisterType;

public class InstructionListAnalyzer
{
	/**
	 * Determines whether the instruction at index <code>nCurrentInstrIdx</code>
	 * in the instruction list <code>il</code> contains the last read of the
	 * SIMD pseudo register <code>reg</code>. It is assumed that pseudo
	 * registers are written to only once.
	 * 
	 * @param il
	 *            The instruction list to examine
	 * @param reg
	 *            The register to check for reads
	 * @param nCurrentInstrIdx
	 *            The index of the instruction within the instruction list
	 *            <code>il</code> to examine
	 * @return <code>true</code> iff the pseudo register <code>reg</code> is not
	 *         read anymore beyond the instruction with index
	 *         <code>nCurrentInstrIdx</code>
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
	
	public static boolean isDependent (IInstruction instr1, IInstruction instr2)
	{
		// check for flow dependences:
		// instr1: write to {PR_i}
		// instr2: read {PR_i}
		if (InstructionListAnalyzer.isFlowDependent (instr1, instr2))
			return true;
		
		// check for antidependences:
		// instr1: read {PR_i}
		// instr2: write to {PR_i}
		if (InstructionListAnalyzer.isAntiDependent (instr1, instr2))
			return true;
		
		// check for output dependences:
		// instr1: write to {PR_i}
		// instr2: write to {PR_i}
		if (InstructionListAnalyzer.isOutputDependent (instr1, instr2))
			return true;
		
		return false;
	}

	public static boolean isFlowDependent (IInstruction instr1, IInstruction instr2)
	{
		if (!(instr1 instanceof Instruction))
			return false;
		if (!(instr2 instanceof Instruction))
			return false;
		
		IOperand opOut = ((Instruction) instr1).getOperands ()[((Instruction) instr1).getOperands ().length - 1];
		for (int i = 0; i < ((Instruction) instr2).getOperands ().length - 1; i++)
			if (((Instruction) instr2).getOperands ()[i].equals (opOut))
				return true;
		
		return false;
	}

	public static boolean isAntiDependent (IInstruction instr1, IInstruction instr2)
	{
		return InstructionListAnalyzer.isFlowDependent (instr2, instr1);
	}

	public static boolean isOutputDependent (IInstruction instr1, IInstruction instr2)
	{
		if (!(instr1 instanceof Instruction))
			return false;
		if (!(instr2 instanceof Instruction))
			return false;
		
		IOperand opOut1 = ((Instruction) instr1).getOperands ()[((Instruction) instr1).getOperands ().length - 1];
		IOperand opOut2 = ((Instruction) instr2).getOperands ()[((Instruction) instr2).getOperands ().length - 1];
		
		return opOut1.equals (opOut2);
	}
}
