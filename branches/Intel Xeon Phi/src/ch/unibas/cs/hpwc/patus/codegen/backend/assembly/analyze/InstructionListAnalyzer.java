package ch.unibas.cs.hpwc.patus.codegen.backend.assembly.analyze;

import java.util.Map;
import java.util.Set;

import ch.unibas.cs.hpwc.patus.arch.IArchitectureDescription;
import ch.unibas.cs.hpwc.patus.arch.TypeRegisterType;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.IInstruction;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.IOperand;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.Instruction;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.InstructionList;

public class InstructionListAnalyzer
{
	/**
	 * Determines whether the instruction at index <code>nCurrentInstrIdx</code>
	 * in the instruction list <code>il</code> contains the last read of the
	 * SIMD pseudo register <code>reg</code>. It is assumed that pseudo
	 * registers are written to only once.
	 * 
	 * <p><b>Assumes that operations are non-destructive and that the result operand is the last operand</b></p>
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
	public static boolean isLastRead (IArchitectureDescription arch,
		InstructionList il, InstructionList.EInstructionListType iltype,
		IOperand.PseudoRegister reg, int nCurrentInstrIdx,
		Map<IOperand.PseudoRegister, Set<IOperand.PseudoRegister>> mapSubstitutes)
	{
		if (!reg.getRegisterType ().equals (TypeRegisterType.SIMD))
			throw new RuntimeException ("Only implemented for SIMD registers");
		if (!arch.hasNonDestructiveOperations () && iltype.equals (InstructionList.EInstructionListType.SPECIFIC))
			throw new RuntimeException ("Not implemented for operations with destructive operands");
		
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
	
	/**
	 * Determines whether <code>instr2</code> depends on <code>instr1</code>.
	 * @param instr1
	 * @param instr2
	 * @return
	 */
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

	/**
	 * Determines whether there is a flow dependence from <code>instr1</code> to
	 * <code>instr2</code>, i.e. <code>instr2</code> is flow dependent on
	 * <code>instr1</code> (<code>instr1</code> &delta;<sup>f</sup>
	 * <code>instr2</code>)
	 * 
	 * @param instr1
	 * @param instr2
	 * @return
	 */
	public static boolean isFlowDependent (IInstruction instr1, IInstruction instr2)
	{
		if (!(instr1 instanceof Instruction))
			return false;
		if (!(instr2 instanceof Instruction))
			return false;
		
		int nOperandsCount = ((Instruction) instr1).getOperands ().length;
		if (nOperandsCount == 0)
			return false;
		
		IOperand opOut = ((Instruction) instr1).getOperands ()[nOperandsCount - 1];
		
		// check input operands
		int nOperandsCount2 = ((Instruction) instr2).getOperands ().length;
		for (int i = 0; i < nOperandsCount2 - 1; i++)
			if (((Instruction) instr2).getOperands ()[i].equals (opOut))
				return true;
		
		// check output operand if it is an address
		IOperand opOut2 = ((Instruction) instr2).getOperands ()[nOperandsCount2 - 1];
		if (opOut2 instanceof IOperand.Address)
			if (((IOperand.Address) opOut2).getRegBase ().equals (opOut) || ((((IOperand.Address) opOut2).getRegIndex () != null) && ((IOperand.Address) opOut2).getRegIndex ().equals (opOut)))
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
		
		int nOpsCount1 = ((Instruction) instr1).getOperands ().length;
		if (nOpsCount1 == 0)
			return false;
		
		int nOpsCount2 = ((Instruction) instr2).getOperands ().length;
		if (nOpsCount2 == 0)
			return false;

		IOperand opOut1 = ((Instruction) instr1).getOperands ()[nOpsCount1 - 1];
		IOperand opOut2 = ((Instruction) instr2).getOperands ()[nOpsCount2 - 1];
		
		return opOut1.equals (opOut2);
	}

	/**
	 * Determines whether the instruction <code>instr</code> moves data from/to
	 * memory.
	 * 
	 * @param instr
	 *            The instruction to examine
	 * @return <code>true</code> iff <code>instr</code> moves data from/to
	 *         memory
	 */
	public static boolean movesDataBetweenMemory (IInstruction instr)
	{
		if (!(instr instanceof Instruction))
			return false;
		
		for (IOperand op : ((Instruction) instr).getOperands ())
			if (op instanceof IOperand.Address)
				return true;
		return false;
	}
}
