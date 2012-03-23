package ch.unibas.cs.hpwc.patus.codegen.backend.assembly.optimize;

import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.IInstruction;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.IOperand;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.InstructionList;

public class RemoveMultipleMemoryLoad implements IInstructionListOptimizer
{
	///////////////////////////////////////////////////////////////////
	// Constants
	
	private final static int NUM_READAHEAD_INSTRUCTIONS = 4;

	
	///////////////////////////////////////////////////////////////////
	// Member Variables
	
	private IInstruction[] m_rgReadAheadInstructions;
	
	private int m_nReadAheadInstructionIdx;

	
	///////////////////////////////////////////////////////////////////
	// Implementation

	public RemoveMultipleMemoryLoad ()
	{
		m_rgReadAheadInstructions = new IInstruction[NUM_READAHEAD_INSTRUCTIONS];
		m_nReadAheadInstructionIdx = 0;
	}
	
	@Override
	public InstructionList optimize (InstructionList il)
	{
		InstructionList ilNew = new InstructionList ();

		for (IInstruction instr : il)
		{
			addInstructionToBuffer (instr);
			
			IOperand.Address[] rgCommonAddrs = getCommonMemoryReferences ();
			if (rgCommonAddrs != null)
				addInstruction (ilNew, rgCommonAddrs);
		}
		
		// empty the buffer
		for (int i = 0; i < NUM_READAHEAD_INSTRUCTIONS; i++)
			addInstruction (ilNew, null);
		
		return ilNew;
	}
	
	private void addInstructionToBuffer (IInstruction instr)
	{		
		m_rgReadAheadInstructions[m_nReadAheadInstructionIdx] = instr;
			
		m_nReadAheadInstructionIdx++;
		if (m_nReadAheadInstructionIdx >= NUM_READAHEAD_INSTRUCTIONS)
			m_nReadAheadInstructionIdx -= NUM_READAHEAD_INSTRUCTIONS;
	}
	
	private void addInstruction (InstructionList il, IOperand.Address[] rgCommonAddrs)
	{
		
	}
	
	private IOperand.Address[] getCommonMemoryReferences ()
	{
		return null;
	}
}
