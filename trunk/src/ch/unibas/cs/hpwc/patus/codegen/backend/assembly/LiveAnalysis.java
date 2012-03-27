package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import java.text.DecimalFormat;
import java.util.Arrays;

import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.IOperand.PseudoRegister;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

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

	/**
	 * The array of instructions to analyze
	 */
	private IInstruction[] m_rgInstructions;
	
	/**
	 * A matrix showing which pseudo registers (second index) are live at which instruction (first index)
	 */
	private byte[][] m_rgLivePseudoRegisters;
	
	/**
	 * The array of pseudo registers used in the instructions
	 */
	private IOperand.PseudoRegister[] m_rgPseudoRegisters;
	
	
	///////////////////////////////////////////////////////////////////
	// Implementation

	/**
	 * Creates a live analysis analyzer for the instruction list <code>il</code>.
	 * Call the {@link LiveAnalysis#run()} method to start the analysis.
	 * 
	 * @param il
	 *            The instruction list to analyze
	 */
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

	/**
	 * Finds the maximum pseudo register index.
	 * 
	 * @return The maximum pseudo register index
	 */
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
	
	/**
	 * Runs the live analysis.
	 * 
	 * @return The live analysis graph
	 */
	public LAGraph run ()
	{
		LAGraph graph = new LAGraph ();
		
		// construct the matrix and add the vertices to the LAGraph
		createStateMatrix (graph);
		
		// construct the graph from the matrix:
		// add an edge between two vertices if the two corresponding pseudo registers are live at the same time
		// (the vertices were added in createStateMatrix)
		
		for (int i = 0; i < m_rgLivePseudoRegisters.length; i++)
		{
			for (int j = 0; j < m_rgLivePseudoRegisters[i].length; j++)
			{
				for (int k = j + 1; k < m_rgLivePseudoRegisters[i].length; k++)
				{
					if (m_rgLivePseudoRegisters[i][j] == STATE_LIVE && m_rgLivePseudoRegisters[i][k] == STATE_LIVE)
					{
						graph.addEdge (new LAGraph.Vertex (m_rgPseudoRegisters[j]), new LAGraph.Vertex (m_rgPseudoRegisters[k]));
						graph.addEdge (new LAGraph.Vertex (m_rgPseudoRegisters[k]), new LAGraph.Vertex (m_rgPseudoRegisters[j]));
					}
				}
			}
		}
				
		return graph;
	}
	
	/**
	 * Constructs a matrix, <code>m_rgLivePseudoRegisters</code>, which captures which pseudo registers
	 * (columns of the matrix) are live in which instruction (rows of the matrix).
	 * 
	 * @param graph
	 *            An instance of the live analysis graph, to which vertices are added in this method.
	 */
	private void createStateMatrix (LAGraph graph)
	{
		// create the matrix of pseudo registers
		int nPseudoRegistersCount = getMaxPseudoRegIndex () + 1;
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
						graph.addVertex (new LAGraph.Vertex (reg));
						m_rgPseudoRegisters[reg.getNumber ()] = reg;
						
						if (isLastRead (reg, i))
						{
							m_rgLivePseudoRegisters[i][reg.getNumber ()] = STATE_LIVE;
							if (i < m_rgInstructions.length - 1)
								m_rgLivePseudoRegisters[i + 1][reg.getNumber ()] = STATE_DEAD;
						}
					}
				}
				
				// new write register becomes live
				if (rgOps[rgOps.length - 1] instanceof IOperand.PseudoRegister)
				{
					IOperand.PseudoRegister reg = (IOperand.PseudoRegister) rgOps[rgOps.length - 1];
					graph.addVertex (new LAGraph.Vertex (reg));
					m_rgPseudoRegisters[reg.getNumber ()] = reg;
					
					m_rgLivePseudoRegisters[i][reg.getNumber ()] = STATE_LIVE;
				}
				
			}

			// promote unassigned flags from previous instruction
			promoteUnassignedFlags (i);
		}
	}
	
	/**
	 * Promotes unassigned flags from previous instruction.
	 * 
	 * @param nCurIdx
	 *            The current instruction index
	 */
	private void promoteUnassignedFlags (int nCurIdx)
	{
		for (int j = 0; j < m_rgLivePseudoRegisters[nCurIdx].length; j++)
		{
			if (nCurIdx == 0)
			{
				if (m_rgLivePseudoRegisters[nCurIdx][j] == STATE_UNASSIGNED)
					m_rgLivePseudoRegisters[nCurIdx][j] = STATE_DEAD;
			}
			else
			{
				if (m_rgLivePseudoRegisters[nCurIdx][j] == STATE_UNASSIGNED)
					m_rgLivePseudoRegisters[nCurIdx][j] = m_rgLivePseudoRegisters[nCurIdx - 1][j];
			}
		}		
	}
	
	/**
	 * Determines whether the last read of the pseudo register <code>reg</code> occurs in
	 * the instruction with index <code>nCurrentIstrIdx</code>.
	 * 
	 * @param reg
	 *            The pseudo register
	 * @param nCurrentInstrIdx
	 *            The index of the instruction
	 * @return <code>true</code> iff the last read of the register <code>reg</code> occurs
	 *         in the instruction with index <code>nCurrentInstrIdx</code>
	 */
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
	
	@Override
	public String toString ()
	{
		StringBuilder sb = new StringBuilder ();
		
		final int nInstrStrLen = 60;
		DecimalFormat fmt = new DecimalFormat (" 00");
		
		sb.append (StringUtil.padRight ("Instr \\ Reg", nInstrStrLen));
		for (int i = 0; i < m_rgLivePseudoRegisters[0].length; i++)
			sb.append (fmt.format (i));
		sb.append ('\n');
		
		for (int i = 0; i < m_rgLivePseudoRegisters.length; i++)
		{
			sb.append (StringUtil.padRight (m_rgInstructions[i].toString (), nInstrStrLen));

			for (int j = 0; j < m_rgLivePseudoRegisters[i].length; j++)
			{
				sb.append ("  ");
				sb.append (m_rgLivePseudoRegisters[i][j] == STATE_LIVE ? '*' : ' ');
			}
			sb.append ('\n');
		}
		
		return sb.toString ();
	}
}
