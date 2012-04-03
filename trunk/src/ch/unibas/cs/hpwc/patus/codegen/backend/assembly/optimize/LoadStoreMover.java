package ch.unibas.cs.hpwc.patus.codegen.backend.assembly.optimize;

import java.util.Arrays;

import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.IInstruction;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.IOperand;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.Instruction;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.InstructionList;

public class LoadStoreMover implements IInstructionListOptimizer
{
	private final static int INSTRUCTIONS_SKIP = 2;
	
	@Override
	public InstructionList optimize (InstructionList il)
	{
		IInstruction[] rgInstrBuffer = new IInstruction[il.size ()];
		int i = 0;
		for (IInstruction instruction : il)
			rgInstrBuffer[i++] = instruction;
		
		for (i = 0; i < rgInstrBuffer.length; i++)
		{
			if (rgInstrBuffer[i] instanceof Instruction)
			{
				Instruction instr = (Instruction) rgInstrBuffer[i];
				if (LoadStoreMover.isAddressLoad (instr))
					LoadStoreMover.moveArrayEntry (rgInstrBuffer, i, -INSTRUCTIONS_SKIP);
			}
		}
		
		InstructionList ilOut = new InstructionList ();
		ilOut.addInstructions (rgInstrBuffer);
		return ilOut;
	}

	private static boolean isAddressLoad (Instruction instr)
	{
		if (UnneededAddressLoadRemover.MOV_INSTRUCTION.equals (instr.getIntrinsicBaseName ()))
		{
			return instr.getOperands ().length == 2 && 
				(instr.getOperands ()[0] instanceof IOperand.Address) &&
				(instr.getOperands ()[1] instanceof IOperand.PseudoRegister); 
		}
		
		return false;
	}
	
	private static <T> void moveArrayEntry (T[] rgArray, int nIndex, int nOffset)
	{
		int nEffOffset = nOffset;
		if (nIndex + nEffOffset < 0)
			nEffOffset = -nIndex;
		if (nIndex + nEffOffset >= rgArray.length)
			nEffOffset = rgArray.length - nIndex - 1;
		
		int nDir = nEffOffset < 0 ? -1 : 1;
		
		T tmp = rgArray[nIndex];
		for (int i = nIndex; i != nIndex + nEffOffset; i += nDir)
			rgArray[i] = rgArray[i + nDir];
		rgArray[nIndex + nEffOffset] = tmp;
	}
	
	public static void main (String[] args)
	{
		Integer[] a = { 1, 2, 3, 4, 5, 6, 7, 8 };
		System.out.println (Arrays.toString (a));
		moveArrayEntry (a, 0, -2);
		System.out.println (Arrays.toString (a));
	}
}
