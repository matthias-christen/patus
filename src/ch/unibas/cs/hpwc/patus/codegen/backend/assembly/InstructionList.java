package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.apache.log4j.Logger;

import ch.unibas.cs.hpwc.patus.arch.TypeRegisterType;

/**
 * 
 * @author Matthias-M. Christen
 */
public class InstructionList implements Iterable<IInstruction>
{
	///////////////////////////////////////////////////////////////////
	// Constants
	
	private final static Logger LOGGER = Logger.getLogger (InstructionList.class);

	
	///////////////////////////////////////////////////////////////////
	// Member Variables

	/**
	 * The list of instructions within this portion of the inline assembly section
	 */
	private List<IInstruction> m_listInstructions;
		
	
	///////////////////////////////////////////////////////////////////
	// Implementation

	public InstructionList ()
	{
		m_listInstructions = new ArrayList<> ();
	}

	public void addInstruction (IInstruction instruction)
	{
		if (instruction != null)
		{
			///
//			if (instruction instanceof Instruction && !m_listInstructions.isEmpty ())
//			{
//				if ("mov".equals (((Instruction) instruction).getIntrinsicBaseName ()))
//				{
//					if (m_listInstructions.get (m_listInstructions.size () - 1).toString ().equals (instruction.toString ()))
//						return;
//				}
//			}
			///
				
			m_listInstructions.add (instruction);
		}
	}
	
	public void addInstruction (IInstruction instruction, StencilAssemblySection.OperandWithInstructions op)
	{
		addInstructions (op.getInstrPre ());
		addInstruction (instruction);
		addInstructions (op.getInstrPost ());
	}
	
	public void addInstructions (InstructionList il)
	{
		if (il != null)
		{
			for (IInstruction instr : il)
				addInstruction (instr);
		}
	}
	
	public void addInstructions (IInstruction[] rgInstructions)
	{
		if (rgInstructions != null)
		{
			for (IInstruction instr : rgInstructions)
				addInstruction (instr);
		}
	}

	@Override
	public Iterator<IInstruction> iterator ()
	{
		return m_listInstructions.iterator ();
	}
	
	/**
	 * 
	 * @param as
	 * @return
	 */
	public InstructionList allocateRegisters (AssemblySection as)
	{
		LOGGER.info ("Performing live analysis and allocating registers...");
		
		// do a live analysis
		LiveAnalysis analysis = new LiveAnalysis (this);
		Map<TypeRegisterType, LAGraph> mapGraphs = analysis.run ();
		
		// allocate registers
		Map<IOperand.PseudoRegister, IOperand.IRegisterOperand> map = RegisterAllocator.mapPseudoRegistersToRegisters (mapGraphs, as);
		
		// replace the pseudo registers by allocated registers
		return replacePseudoRegisters (map);
	}
	
	/**
	 * 
	 * @param graph
	 * @return
	 */
	public InstructionList replacePseudoRegisters (Map<IOperand.PseudoRegister, IOperand.IRegisterOperand> mapPseudoRegsToRegs)
	{
		InstructionList il = new InstructionList ();
		for (IInstruction instr : this)
		{
			IInstruction instrNew = instr;
			if (instr instanceof Instruction)
			{
				IOperand[] rgOps = ((Instruction) instr).getOperands ();
				IOperand[] rgOpsNew = null;

				// search all operands of the instruction and replace pseudo register instances by actual registers
				for (int i = 0; i < rgOps.length; i++)
				{
					IOperand opNew = null;
					
					// translate pseudo registers
					if (rgOps[i] instanceof IOperand.PseudoRegister)
						opNew = mapPseudoRegsToRegs.get (rgOps[i]);
					else if (rgOps[i] instanceof IOperand.Address)
					{
						IOperand.Address opAddr = (IOperand.Address) rgOps[i];
						
						IOperand.IRegisterOperand regBase = null;
						IOperand.IRegisterOperand regIndex = null;
						if (opAddr.getRegBase () instanceof IOperand.PseudoRegister)
							regBase = mapPseudoRegsToRegs.get (opAddr.getRegBase ());
						if (opAddr.getRegIndex () != null && (opAddr.getRegIndex () instanceof IOperand.PseudoRegister))
							regIndex = mapPseudoRegsToRegs.get (opAddr.getRegIndex ());
						
						if (regBase != null || regIndex != null)
						{
							opNew = new IOperand.Address (
								regBase == null ? opAddr.getRegBase () : regBase,
								regIndex == null ? opAddr.getRegIndex () : regIndex,
								opAddr.getScale (),
								opAddr.getDisplacement ()
							);
						}
					}

					// set the new operand to the operands array
					if (opNew != null)
					{
						// create the new operands array if it hasn't been created yet
						if (rgOpsNew == null)
						{
							rgOpsNew = new IOperand[rgOps.length];
							for (int j = 0; j < rgOps.length; j++)
								rgOpsNew[j] = rgOps[j];
						}

						rgOpsNew[i] = opNew;
					}
				}
				
				// generate a new instruction if one of the operands has been modified
				if (rgOpsNew != null)
					instrNew = new Instruction (((Instruction) instrNew).getIntrinsicBaseName (), rgOpsNew);
			}
			
			il.addInstruction (instrNew);
		}
		
		return il;
	}
	
	/**
	 * Replaces the instructions in the key set of the map
	 * <code>mapInstructionReplacements</code> by the corresponding map values.
	 * 
	 * @param mapInstructionReplacements
	 *            The map defining the mapping between old and new instruction
	 *            names
	 * @return A new instruction list with instructions replaced as defined in
	 *         the map <code>mapInstructionReplacements</code>
	 */
	public InstructionList replaceInstructions (Map<String, String> mapInstructionReplacements)
	{
		InstructionList il = new InstructionList ();
		for (IInstruction instr : this)
		{
			IInstruction instrNew = instr;
			if (instr instanceof Instruction)
			{
				String strInstrRepl = mapInstructionReplacements.get (((Instruction) instr).getIntrinsicBaseName ());
				if (strInstrRepl != null)
					instrNew = new Instruction (strInstrRepl, ((Instruction) instr).getOperands ());
			}

			il.addInstruction (instrNew);
		}
		
		return il;
	}

	/**
	 * Returns number of instructions in this instruction list.
	 * 
	 * @return The number of instructions in the instruction list
	 */
	public int size ()
	{
		return m_listInstructions.size ();
	}
	
	public boolean isEmpty ()
	{
		return m_listInstructions.isEmpty ();
	}

	@Override
	public String toString ()
	{
		StringBuilder sb = new StringBuilder ();
		for (IInstruction i : this)
		{
			sb.append (i.toString ());
			sb.append ("\n");
		}
		
		return sb.toString ();
	}
}
