package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

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
}
