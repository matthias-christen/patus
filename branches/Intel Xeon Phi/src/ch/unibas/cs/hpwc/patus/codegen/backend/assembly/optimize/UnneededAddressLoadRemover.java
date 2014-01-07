package ch.unibas.cs.hpwc.patus.codegen.backend.assembly.optimize;

import java.util.HashMap;
import java.util.Map;

import cetus.hir.Specifier;
import ch.unibas.cs.hpwc.patus.arch.IArchitectureDescription;
import ch.unibas.cs.hpwc.patus.arch.TypeArchitectureType.Intrinsics.Intrinsic;
import ch.unibas.cs.hpwc.patus.arch.TypeBaseIntrinsicEnum;
import ch.unibas.cs.hpwc.patus.codegen.Globals;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.IInstruction;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.IOperand;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.Instruction;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.InstructionList;

/**
 * Removes unnecessary &quot;mov&quot;s from an instruction list.
 * @author Matthias-M. Christen
 */
public class UnneededAddressLoadRemover implements IInstructionListOptimizer
{
	public static final String MOV_INSTRUCTION = "mov";

	
	private IArchitectureDescription m_arch;
	
	
	public UnneededAddressLoadRemover (IArchitectureDescription arch)
	{
		m_arch = arch;
	}

	@Override
	public InstructionList optimize (InstructionList il)
	{
		if (true)
			throw new RuntimeException ("Not implemented");
		
		// -------------------------------------------------------------------------------
		// the implementation is not complete and will lead to bugs in the generated code
		// -------------------------------------------------------------------------------
		
		InstructionList ilOut = new InstructionList ();
		
		Map<IOperand.Register, Instruction> mapLastMoveInstructionPerRegister = new HashMap<> ();
		
		for (IInstruction instruction : il)
		{
			if (instruction instanceof Instruction)
			{
				Instruction instr = (Instruction) instruction;
				if (instr.getOperands ().length == 0)
					continue;
				
				boolean bIsInstructionNeeded = true;
				
				// check for "mov" instructions
				if (UnneededAddressLoadRemover.isMoveInstruction (instr, m_arch))
				{
					IOperand.Register regDest = UnneededAddressLoadRemover.getRegister (instr);
					if (regDest != null)
					{
						Instruction instrLastMove = mapLastMoveInstructionPerRegister.get (regDest);
					
						// if the previous "mov" is the same as the current one and the destination register
						// (rgDest) hasn't been modified (instrLastLoad is set to null), this one is not needed
						if (instrLastMove != null)
						{
							
							// CHECK if contains address whether addr regs have changed!
							// =~~> dependence analysis...
							
							if (instrLastMove.equals (instr))
								bIsInstructionNeeded = false;
						}
						
						// if the "mov" instruction is not the same as the previous one, record the new
						// destination register regDest and set the "last mov" instruction
						if (bIsInstructionNeeded)
							mapLastMoveInstructionPerRegister.put (regDest, instr);
					}
				}
				else
				{
					// if not a "mov" instruction, check whether regDest is modified
					IOperand opResult = instr.getOperands ()[instr.getOperands ().length - 1];
					if (opResult instanceof IOperand.Register)
					{
						if (mapLastMoveInstructionPerRegister.containsKey (opResult))
						{
							mapLastMoveInstructionPerRegister.remove (opResult);
						
//							// find and delete instructions that contain opResult in an address calculation
//							List<IOperand.Register> listRemove = new ArrayList<> ();
//							for (IOperand.Register reg : mapLastMoveInstructionPerRegister.keySet ())
//								if (UnneededAddressLoadRemover.containsIndirect (mapLastMoveInstructionPerRegister.get (reg), (IOperand.Register) opResult))
//									listRemove.add (reg);
//							for (IOperand.Register reg : listRemove)
//								mapLastMoveInstructionPerRegister.remove (reg);
						}
					}
				}
				
				if (bIsInstructionNeeded)
					ilOut.addInstruction (instr);
			}
			else
				ilOut.addInstruction (instruction);
		}
		
		return ilOut;
	}
	
	private static boolean containsIndirect (Instruction instr, IOperand.Register reg)
	{
		for (IOperand op : instr.getOperands ())
			if (op instanceof IOperand.Address)
				if (reg.equals (((IOperand.Address) op).getRegBase ()) || reg.equals (((IOperand.Address) op).getRegIndex ()))
					return true;
		
		return false;
	}

	private static IOperand.Register getRegister (Instruction instr)
	{
		for (IOperand op : instr.getOperands ())
			if (op instanceof IOperand.Register)
				return (IOperand.Register) op;
		
		return null;
	}

	private static boolean isMoveInstruction (Instruction instr, IArchitectureDescription arch)
	{
		if (instr == null)
			return false;
		
		String strName = instr.getInstructionName ();
		if (strName == null)
			return false;
		
		if (strName.equals (MOV_INSTRUCTION))
			return true;
		
		String[] rgMoveInstructions = {
			TypeBaseIntrinsicEnum.LOAD_GPR.value (),
			TypeBaseIntrinsicEnum.LOAD_FPR_ALIGNED.value (),
			TypeBaseIntrinsicEnum.LOAD_FPR_UNALIGNED.value (),
			
			TypeBaseIntrinsicEnum.STORE_GPR.value (),
			TypeBaseIntrinsicEnum.STORE_FPR_ALIGNED.value (),
			TypeBaseIntrinsicEnum.STORE_FPR_UNALIGNED.value ()
		};
		
		for (String strMoveInstruction : rgMoveInstructions)
			if (strName.equals (strMoveInstruction))
				return true;

		for (String strMoveInstruction : rgMoveInstructions)
		{
			for (Specifier specType : Globals.BASE_DATATYPES)
			{
				Intrinsic intrinsic = arch.getIntrinsic (strMoveInstruction, specType);
				if (intrinsic != null && intrinsic.getName ().equals (strName))
					return true;
			}
		}
		
		return false;
	}
}
