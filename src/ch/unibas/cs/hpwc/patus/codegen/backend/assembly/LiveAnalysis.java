package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import java.util.Arrays;

import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.IOperand.PseudoRegister;

/**
 * 
 * @author Matthias-M. Christen
 */
public class LiveAnalysis
{
	///////////////////////////////////////////////////////////////////
	// Constants
	
	private final static byte STATE_LIVE = 1;
	private final static byte STATE_DEAD = 0;
	private final static byte STATE_UNASSIGNED = -1;

	
	///////////////////////////////////////////////////////////////////
	// Member Variables

	private IInstruction[] m_rgInstructions;
	
	private byte[][] m_rgLivePseudoRegisters;
	
	private IOperand.PseudoRegister[] m_rgPseudoRegisters;
	
	
	///////////////////////////////////////////////////////////////////
	// Implementation

	public LiveAnalysis (InstructionList il)
	{
		m_rgInstructions = new IInstruction[il.size ()];
		int j = 0;
		for (IInstruction instr : il)
		{
			m_rgInstructions[j] = instr;
			j++;
		}

		m_rgLivePseudoRegisters = null;
	}
	
	private int getMaxPseudoRegIndex ()
	{
		int nMaxIdx = 0;
		for (IInstruction instr : m_rgInstructions)
		{
			if (instr instanceof Instruction)
			{
				for (IOperand op : ((Instruction) instr).getOperands ())
					if (op instanceof PseudoRegister)
						nMaxIdx = Math.max (nMaxIdx, ((PseudoRegister) op).getNumber ());
			}
		}
		
		return nMaxIdx;
	}
	
	public LAGraph run ()
	{
		LAGraph graph = new LAGraph ();
		
		// construct the matrix
		createStateMatrix (graph);
		
		// construct the graph from the matrix
		for (int i = 0; i < m_rgLivePseudoRegisters.length; i++)
		{
			for (int j = 0; j < m_rgLivePseudoRegisters[i].length; j++)
			{
				for (int k = 0; k < m_rgLivePseudoRegisters[i].length; k++)
				{
					if (j == k)
						continue;
					
					if (m_rgLivePseudoRegisters[i][j] == STATE_LIVE && m_rgLivePseudoRegisters[i][k] == STATE_LIVE)
						graph.addEdge (m_rgPseudoRegisters[j], m_rgPseudoRegisters[k]);
				}
			}
		}
		
		return graph;
	}
	
	private void createStateMatrix (LAGraph graph)
	{
		// create the matrix of pseudo registers
		int nPseudoRegistersCount = getMaxPseudoRegIndex ();
		m_rgLivePseudoRegisters = new byte[m_rgInstructions.length][nPseudoRegistersCount];
		for (int i = 0; i < m_rgInstructions.length; i++)
			Arrays.fill (m_rgLivePseudoRegisters[i], STATE_UNASSIGNED);
		m_rgPseudoRegisters = new IOperand.PseudoRegister[nPseudoRegistersCount];
				
		for (int i = 0; i < m_rgInstructions.length; i++)
		{
			IInstruction instr = m_rgInstructions[i];
			if (instr instanceof Instruction)
			{
				IOperand[] rgOps = ((Instruction) instr).getOperands ();
				
				// - the last operand is the output operand
				// - a register becomes live if it is written to (it is the output operand)
				// - a register is killed when the instruction contains the last read
				//   (pseudo registers are not reused, i.e., they are written only once)
				
				// check whether a register gets killed
				for (int j = 0; j < rgOps.length - 1; j++)
				{
					if (rgOps[j] instanceof IOperand.PseudoRegister)
					{
						IOperand.PseudoRegister reg = (IOperand.PseudoRegister) rgOps[j];
						graph.addNode (reg);
						m_rgPseudoRegisters[reg.getNumber ()] = reg;
						
						if (isLastRead (reg, i))
							m_rgLivePseudoRegisters[i][reg.getNumber ()] = STATE_DEAD;
					}
				}
				
				// new write register becomes live
				if (rgOps[rgOps.length - 1] instanceof IOperand.PseudoRegister)
				{
					IOperand.PseudoRegister reg = (IOperand.PseudoRegister) rgOps[rgOps.length - 1];
					graph.addNode (reg);
					m_rgPseudoRegisters[reg.getNumber ()] = reg;
					
					m_rgLivePseudoRegisters[i][reg.getNumber ()] = STATE_LIVE;
				}
				
				// promote unassigned flags from previous instruction
				for (int j = 0; j < m_rgLivePseudoRegisters[i].length; j++)
				{
					if (i == 0)
					{
						if (m_rgLivePseudoRegisters[i][j] == STATE_UNASSIGNED)
							m_rgLivePseudoRegisters[i][j] = STATE_DEAD;
					}
					else
					{
						if (m_rgLivePseudoRegisters[i][j] == STATE_UNASSIGNED)
							m_rgLivePseudoRegisters[i][j] = m_rgLivePseudoRegisters[i - 1][j];						
					}
				}
			}
		}
	}
	
	private boolean isLastRead (IOperand.PseudoRegister reg, int nCurrentInstrIdx)
	{
		for (int i = nCurrentInstrIdx + 1; i < m_rgInstructions.length; i++)
		{
			IInstruction instr = m_rgInstructions[i];
			if (instr instanceof Instruction)
			{
				// check input operands (i.e., all operands except the last
				IOperand[] rgOps = ((Instruction) instr).getOperands ();
				for (int j = 0; j < rgOps.length - 1; j++)
					if (reg.equals (rgOps[j]))
					{
						// another, later read was found
						return false;
					}
			}
		}
		
		return true;
	}
}
