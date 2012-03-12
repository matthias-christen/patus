package ch.unibas.cs.hpwc.patus.codegen.backend.assembly.optimize;

import java.util.HashMap;
import java.util.Map;

import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.IInstruction;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.IOperand;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.Instruction;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.InstructionList;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.InstructionListAnalyzer;

public class UnneededPseudoRegistersRemover implements IInstructionListOptimizer
{
	private Map<IOperand.PseudoRegister, IOperand.PseudoRegister> m_mapSubstitute;
	
	public UnneededPseudoRegistersRemover ()
	{
		m_mapSubstitute = new HashMap<IOperand.PseudoRegister, IOperand.PseudoRegister> ();
	}
	
	private IOperand[] substitutePseudoRegisters (IOperand[] rgOps)
	{
		IOperand[] rgOpsNew = new IOperand[rgOps.length];
		
		for (int i = 0; i < rgOps.length; i++)
		{
			if (rgOps[i] instanceof IOperand.PseudoRegister)
			{
				rgOpsNew[i] = m_mapSubstitute.get (rgOps[i]);
				if (rgOpsNew[i] == null)
					rgOpsNew[i] = rgOps[i];
			}
		}
		
		return rgOpsNew;
	}
	
	private boolean hasInputPseudoRegisters (IOperand[] rgOps)
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
				
				boolean bHasPseudoRegisters = hasInputPseudoRegisters (rgOps);
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