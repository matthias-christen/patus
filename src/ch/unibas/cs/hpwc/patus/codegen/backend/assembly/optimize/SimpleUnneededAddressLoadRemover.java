package ch.unibas.cs.hpwc.patus.codegen.backend.assembly.optimize;

import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.IInstruction;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.IOperand;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.Instruction;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.InstructionList;

/**
 * Removes unnecessary &quot;mov&quot;s from an instruction list.
 * @author Matthias-M. Christen
 */
public class SimpleUnneededAddressLoadRemover implements IInstructionListOptimizer
{
	public SimpleUnneededAddressLoadRemover ()
	{
	}

	@Override
	public InstructionList optimize (InstructionList il)
	{
		InstructionList ilOut = new InstructionList ();
		
		Instruction instrLastMove = null;
		IOperand.Register regDest = null;
		
		for (IInstruction instruction : il)
		{
			if (instruction instanceof Instruction)
			{
				Instruction instr = (Instruction) instruction;
				if (instr.getOperands ().length == 0)
					continue;
				
				boolean bIsInstructionNeeded = true;
				
				// check for "mov" instructions
				if (instr.getInstructionName ().equals (UnneededAddressLoadRemover.MOV_INSTRUCTION))
				{
					// if the previous "mov" is the same as the current one and the destination register
					// (rgDest) hasn't been modified (instrLastLoad is set to null), this one is not needed
					if (instrLastMove != null)
					{
						if (instrLastMove.equals (instr))
							bIsInstructionNeeded = false;
					}
					
					// if the "mov" instruction is not the same as the previous one, record the new
					// destination register regDest and set the "last mov" instruction
					if (bIsInstructionNeeded)
					{
						for (IOperand op : instr.getOperands ())
							if (op instanceof IOperand.Register)
							{
								regDest = (IOperand.Register) op;
								break;
							}
						
						instrLastMove = instr;
					}
				}
				else
				{
					// if not a "mov" instruction, check whether regDest is modified
					if (instr.getOperands ()[instr.getOperands ().length - 1].equals (regDest))
						instrLastMove = null;
				}
				
				if (bIsInstructionNeeded)
					ilOut.addInstruction (instr);
			}
			else
				ilOut.addInstruction (instruction);
		}
		
		return ilOut;
	}
}
