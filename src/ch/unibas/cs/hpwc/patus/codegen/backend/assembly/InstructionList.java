package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import java.util.ArrayList;
import java.util.HashMap;
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
	
	public InstructionList replacePseudoRegisters (LAGraph graph)
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
						
//						rgOps[i] = mapRegisters.get ();
					}
				}
			}
			
			il.addInstruction (instrNew);
		}
		
		return il;
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

	/**
	 * Returns number of instructions in this instruction list.
	 * @return The number of instructions in the instruction list
	 */
	public int size ()
	{
		return m_listInstructions.size ();
	}
}
