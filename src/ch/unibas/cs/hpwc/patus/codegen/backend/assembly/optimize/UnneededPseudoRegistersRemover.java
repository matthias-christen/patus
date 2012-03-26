package ch.unibas.cs.hpwc.patus.codegen.backend.assembly.optimize;

import java.util.HashMap;
import java.util.Map;

import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.IInstruction;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.IOperand;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.IOperand.PseudoRegister;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.Instruction;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.InstructionList;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.InstructionListAnalyzer;

/**
 * Replaces instructions of the form
 * <pre>
 * 	op arg<sub>1</sub>, arg<sub>2</sub>, ..., arg<sub>n</sub>, {pseudoreg-j}
 * </pre>
 * where one of the arguments <code>arg<sub>i</sub></code> is a pseudo register,
 * say, <code>{pseudoreg-k}</code>, by
 * <pre>
 * 	op arg<sub>1</sub>, ..., arg<sub>i-1</sub>, {pseudoreg-k}, arg<sub>i+1</sub>, ..., arg<sub>n</sub>, {pseudoreg-k}
 * </pre>
 * if this is possible (i.e., if <code>{pseudoreg-j}</code> is never read anymore
 * in the instruction list.
 * This means, that, by replaceing <code>{pseudoreg-j}</code> by <code>{pseudoreg-k}</code>
 * one pseudo register is saved.
 * 
 * @author Matthias-M. Christen
 */
public class UnneededPseudoRegistersRemover implements IInstructionListOptimizer
{
	private Map<IOperand.PseudoRegister, IOperand.PseudoRegister> m_mapSubstitute;
	
	public UnneededPseudoRegistersRemover ()
	{
		m_mapSubstitute = new HashMap<> ();
	}
	
	/**
	 * Substitutes any erased pseudo registers in the operands array by the pseudo register
	 * which was substituted for the erased one.
	 * 
	 * @param rgOps
	 *            The array of instruction arguments
	 * @return The array of instruction arguments with the erased pseudo registers substituted appropriately
	 */
	private IOperand[] substitutePseudoRegisters (IOperand[] rgOps)
	{
		if (m_mapSubstitute.isEmpty ())
			return rgOps;
		
		IOperand[] rgOpsNew = new IOperand[rgOps.length];
		
		for (int i = 0; i < rgOps.length; i++)
		{
			if (rgOps[i] instanceof IOperand.PseudoRegister)
			{
				rgOpsNew[i] = m_mapSubstitute.get (rgOps[i]);
				if (rgOpsNew[i] == null)
					rgOpsNew[i] = rgOps[i];
			}
			else
				rgOpsNew[i] = rgOps[i];
		}
		
		return rgOpsNew;
	}
	
	/**
	 * Determines whether the instruction has a pseudo register as an input.
	 * 
	 * @param rgOps
	 *            The array of instruction arguments
	 * @return <code>true</code> iff at least one of the input arguments in the
	 *         array <code>rgOps</code> is a {@link PseudoRegister}
	 */
	private static boolean hasInputPseudoRegisters (IOperand[] rgOps)
	{
		// omit the last (=result) operand
		for (int i = 0; i < rgOps.length - 1; i++)
			if (rgOps[i] instanceof IOperand.PseudoRegister)
				return true;
		return false;
	}
	
	@Override
	public InstructionList optimize (InstructionList il)
	{
		InstructionList ilResult = new InstructionList ();
		int nCurrentInstructionIdx = 0;
		
		for (IInstruction instruction : il)
		{
			if (instruction instanceof Instruction)
			{
				Instruction instr = (Instruction) instruction;
				IOperand[] rgOps = instr.getOperands ();
				
				boolean bHasPseudoRegisters = UnneededPseudoRegistersRemover.hasInputPseudoRegisters (rgOps);
				boolean bIsResultPseudoRegister = rgOps[rgOps.length - 1] instanceof IOperand.PseudoRegister;
				
				// substitute any pseudo register with pseudo registers previously marked as substitutees
				IOperand[] rgOpsNew = bHasPseudoRegisters || bIsResultPseudoRegister ? substitutePseudoRegisters (rgOps) : rgOps;
				
				// if the result operand is a pseudo register, check if there are other
				// pseudo register operands, and if there are, and the result register
				// differs from all the other pseudo register operands, try to find
				// a pseudo register operand, which is never read anymore, and substitute
				// that for the result pseudo register
				
				if (bIsResultPseudoRegister && bHasPseudoRegisters)
				{
					IOperand.PseudoRegister regNewResult = null;
					for (int i = 0; i < rgOpsNew.length - 1; i++)
					{
						if (rgOpsNew[i] instanceof IOperand.PseudoRegister)
						{
							if (InstructionListAnalyzer.isLastRead (il, (IOperand.PseudoRegister) rgOpsNew[i], nCurrentInstructionIdx))
							{
								regNewResult = (IOperand.PseudoRegister) rgOpsNew[i];
								break;
							}
						}
					}
					
					// substitute
					if (regNewResult != null)
					{
						rgOpsNew[rgOpsNew.length - 1] = regNewResult;
						m_mapSubstitute.put ((IOperand.PseudoRegister) rgOps[rgOps.length - 1], regNewResult);
					}
				}
				
				ilResult.addInstruction (new Instruction (instr.getIntrinsicBaseName (), rgOpsNew));
			}
			else
				ilResult.addInstruction (instruction);
			
			nCurrentInstructionIdx++;
		}
		
		return ilResult;
	}
}