package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

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

	@Override
	public Iterator<IInstruction> iterator ()
	{
		return m_listInstructions.iterator ();
	}
	
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
}
