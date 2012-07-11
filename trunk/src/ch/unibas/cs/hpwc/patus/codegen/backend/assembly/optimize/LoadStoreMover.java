package ch.unibas.cs.hpwc.patus.codegen.backend.assembly.optimize;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import cetus.hir.Specifier;
import ch.unibas.cs.hpwc.patus.arch.IArchitectureDescription;
import ch.unibas.cs.hpwc.patus.arch.TypeArchitectureType.Intrinsics.Intrinsic;
import ch.unibas.cs.hpwc.patus.arch.TypeBaseIntrinsicEnum;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.IInstruction;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.IOperand;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.Instruction;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.InstructionList;

/**
 * Moves loads up and stores down to remove latencies between value loads and use of the values.
 * Superseded by instruction scheduling.
 * @author Matthias-M. Christen
 */
@Deprecated
public class LoadStoreMover implements IInstructionListOptimizer
{
	private final static int INSTRUCTIONS_SKIP = 2;
	private Intrinsic m_iLoadFpr;
	private Intrinsic m_iLoadFprUa;
	private Intrinsic m_iStoreFpr;
	private Intrinsic m_iStoreFprUa;
	
	
	public LoadStoreMover (IArchitectureDescription arch)
	{
		m_iLoadFpr = arch.getIntrinsic (TypeBaseIntrinsicEnum.LOAD_FPR_ALIGNED.value (), Specifier.FLOAT);
		m_iLoadFprUa = arch.getIntrinsic (TypeBaseIntrinsicEnum.LOAD_FPR_UNALIGNED.value (), Specifier.FLOAT);

		m_iStoreFpr = arch.getIntrinsic (TypeBaseIntrinsicEnum.STORE_FPR_ALIGNED.value (), Specifier.FLOAT);
		m_iStoreFprUa = arch.getIntrinsic (TypeBaseIntrinsicEnum.STORE_FPR_UNALIGNED.value (), Specifier.FLOAT);
	}
	
	@Override
	public InstructionList optimize (InstructionList il)
	{
		// TODO: this doesn't work reliably yet...
		if (true)
			return il;
		
		IInstruction[] rgInstrBuffer = new IInstruction[il.size ()];
		int i = 0;
		for (IInstruction instruction : il)
			rgInstrBuffer[i++] = instruction;
		
//		for (i = 0; i < rgInstrBuffer.length; i++)
//		{
//			if (rgInstrBuffer[i] instanceof Instruction)
//			{
//				Instruction instr = (Instruction) rgInstrBuffer[i];
//				if (isAddressLoad (instr))
//				{
//					LoadStoreMover.moveArrayEntry (rgInstrBuffer, i, -INSTRUCTIONS_SKIP);
//				}
//			}
//		}
//		
//		InstructionList ilOut = new InstructionList ();
//		ilOut.addInstructions (rgInstrBuffer);
//		return ilOut;

		InstructionList ilOut = new InstructionList ();
		Set<Integer> setAdded = new HashSet<> ();
		for (i = 0; i < rgInstrBuffer.length; i++)
		{
			if (!setAdded.contains (i + INSTRUCTIONS_SKIP))
			{
				int nSize = i + INSTRUCTIONS_SKIP < rgInstrBuffer.length ? getAddressLoadBlockSize (rgInstrBuffer, i + INSTRUCTIONS_SKIP) : 0;
				for (int j = 0; j < nSize; j++)
				{
					int nIdx = i + INSTRUCTIONS_SKIP + j;
					ilOut.addInstruction (rgInstrBuffer[nIdx]);
					setAdded.add (nIdx);
				}
			}
			
			if (!setAdded.contains (i))
				ilOut.addInstruction (rgInstrBuffer[i]);
		}
		
		return ilOut;
	}

	private boolean isAddressLoad (IInstruction instruction)
	{
		if (!(instruction instanceof Instruction))
			return false;
		
		Instruction instr = (Instruction) instruction;
		
		//if (UnneededAddressLoadRemover.MOV_INSTRUCTION.equals (instr.getIntrinsicBaseName ()))
		if (instr.getInstructionName ().equals (m_iLoadFpr.getName ()) || instr.getInstructionName ().equals (m_iLoadFprUa.getName ()) ||
			instr.getInstructionName ().equals (m_iStoreFpr.getName ()) || instr.getInstructionName ().equals (m_iStoreFprUa.getName ()))
		{
			return instr.getOperands ().length == 2 && 
				(instr.getOperands ()[0] instanceof IOperand.Address) &&
				(instr.getOperands ()[1] instanceof IOperand.PseudoRegister); 
		}
		
		return false;
	}
	
	/**
	 * Quick and dirty; would need detailed dependence analysis to determine which MOVs can be moved and which additional instructions are needed!!!
	 * @param instruction
	 * @return
	 */
	private static boolean isNeg (IInstruction instruction)
	{
		if (!(instruction instanceof Instruction))
			return false;
		
		Instruction instr = (Instruction) instruction;		
		return instr.getInstructionName ().equals ("neg");
	}
	
	private int getAddressLoadBlockSize (IInstruction[] rgInstrs, int nIdx)
	{
		int i = nIdx;
		boolean bIsNegStart = LoadStoreMover.isNeg (rgInstrs[i]);
		if (bIsNegStart)
			i++;
		
		boolean bAddrLoadFound = false;
		while (isAddressLoad (rgInstrs[i]))
		{
			bAddrLoadFound = true;
			i++;
		}
		
		if (!bAddrLoadFound)
			return 0;
		
		if (bIsNegStart && LoadStoreMover.isNeg (rgInstrs[i]))
			i++;
		
		return i - nIdx;
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
