package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class InstructionList implements Iterable<Instruction>
{
	///////////////////////////////////////////////////////////////////
	// Member Variables

	/**
	 * The list of instructions within this portion of the inline assembly section
	 */
	private List<Instruction> m_listInstructions;
		
	
	///////////////////////////////////////////////////////////////////
	// Implementation

	public InstructionList ()
	{
		m_listInstructions = new ArrayList<Instruction> ();
	}

	public void addInstruction (Instruction instruction)
	{
		m_listInstructions.add (instruction);
	}

	@Override
	public Iterator<Instruction> iterator ()
	{
		return m_listInstructions.iterator ();
	}
}
