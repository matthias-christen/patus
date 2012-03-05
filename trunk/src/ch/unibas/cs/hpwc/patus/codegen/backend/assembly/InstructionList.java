package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 * 
 * @author Matthias-M. Christen
 */
public class InstructionList implements Iterable<IInstruction>
{
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
		m_listInstructions = new ArrayList<IInstruction> ();
	}

	public void addInstruction (IInstruction instruction)
	{
		m_listInstructions.add (instruction);
	}
	
	public void addInstructions (InstructionList il)
	{
		for (IInstruction instr : il)
			m_listInstructions.add (instr);
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
		// do a live analysis
		LiveAnalysis analysis = new LiveAnalysis (this);
		LAGraph graph = analysis.run ();
		
		// allocate registers
		Map<IOperand.PseudoRegister, IOperand.IRegisterOperand> map = RegisterAllocator.mapPseudoRegistersToRegisters (graph, as);
		
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
				boolean bConstructedNew = false;

				// search all operands of the instruction and replace pseudo register instances by actual registers
				for (int i = 0; i < rgOps.length; i++)
				{
					if (rgOps[i] instanceof IOperand.PseudoRegister)
					{
						if (!bConstructedNew)
						{
							IOperand[] rgOpsTmp = rgOps;
							rgOps = new IOperand[rgOpsTmp.length];
							for (int j = 0; j < rgOpsTmp.length; j++)
								rgOps[j] = rgOpsTmp[j];
							bConstructedNew = true;
						}
						
						IOperand.IRegisterOperand reg = mapPseudoRegsToRegs.get (rgOps[i]);
						if (reg != null)
							rgOps[i] = reg;
					}
				}
			}
			
			il.addInstruction (instrNew);
		}
		
		return il;
	}
	
	/**
	 * Replaces the instructions in the key set of the map <code>mapInstructionReplacements</code>
	 * by the corresponding map values.
	 * @param mapInstructionReplacements The map defining the mapping between old and new instruction names
	 * @return A new instruction list with instructions replaced as defined in the map
	 * 	<code>mapInstructionReplacements</code>
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
	 * @return The number of instructions in the instruction list
	 */
	public int size ()
	{
		return m_listInstructions.size ();
	}
}
