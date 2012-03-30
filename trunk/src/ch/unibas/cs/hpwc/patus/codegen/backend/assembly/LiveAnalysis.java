package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import ch.unibas.cs.hpwc.patus.arch.TypeRegisterType;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 * Performs a pseudo register live analysis on the provided instruction list.
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
					if (op instanceof IOperand.PseudoRegister)
						nMaxIdx = Math.max (nMaxIdx, ((IOperand.PseudoRegister) op).getNumber ());
			}
		}
		
		return nMaxIdx;
	}
	
	/**
	 * Runs the live analysis.
	 * 
	 * @return The live analysis graph
	 */
	public Map<TypeRegisterType, LAGraph> run ()
	{
		Map<TypeRegisterType, LAGraph> mapGraphs = new HashMap<> ();
		for (TypeRegisterType type : TypeRegisterType.values ())
			mapGraphs.put (type, new LAGraph ());
		
		// construct the matrix and add the vertices to the LAGraph
		createStateMatrix (mapGraphs);
		
		// construct the graph from the matrix:
		// add an edge between two vertices if the two corresponding pseudo registers are live at the same time
		// (the vertices were added in createStateMatrix)
		
		for (int i = 0; i < m_rgLivePseudoRegisters.length; i++)
		{
			for (int j = 0; j < m_rgLivePseudoRegisters[i].length; j++)
			{
				for (int k = j + 1; k < m_rgLivePseudoRegisters[i].length; k++)
				{
					if (m_rgLivePseudoRegisters[i][j] == STATE_LIVE && m_rgLivePseudoRegisters[i][k] == STATE_LIVE &&
						m_rgPseudoRegisters[j].getRegisterType ().equals (m_rgPseudoRegisters[k].getRegisterType ()))
					{
						LAGraph graph = mapGraphs.get (m_rgPseudoRegisters[j].getRegisterType ());
						graph.addEdge (new LAGraph.Vertex (m_rgPseudoRegisters[j]), new LAGraph.Vertex (m_rgPseudoRegisters[k]));
						graph.addEdge (new LAGraph.Vertex (m_rgPseudoRegisters[k]), new LAGraph.Vertex (m_rgPseudoRegisters[j]));
					}
				}
			}
		}
				
		return mapGraphs;
	}
	
	/**
	 * Constructs a matrix, <code>m_rgLivePseudoRegisters</code>, which captures which pseudo registers
	 * (columns of the matrix) are live in which instruction (rows of the matrix).
	 * 
	 * @param graph
	 *            An instance of the live analysis graph, to which vertices are added in this method.
	 */
	private void createStateMatrix (Map<TypeRegisterType, LAGraph> mapGraphs)
	{
		// create the matrix of pseudo registers
		int nPseudoRegistersCount = getMaxPseudoRegIndex () + 1;
		m_rgLivePseudoRegisters = new byte[m_rgInstructions.length][nPseudoRegistersCount];
		for (int i = 0; i < m_rgInstructions.length; i++)
			Arrays.fill (m_rgLivePseudoRegisters[i], STATE_UNASSIGNED);
		m_rgPseudoRegisters = new IOperand.PseudoRegister[nPseudoRegistersCount];
				
		IOperand.PseudoRegister[] rgRegs = new IOperand.PseudoRegister[2]; 

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
				for (int j = 0; j < rgOps.length; j++)
				{
					// check the last operand (the output) only if it is an address
					// if it is an address, it might contain register reads
					if (j == rgOps.length - 1 && !(rgOps[j] instanceof IOperand.Address))
						break;
					
					// gather registers to check
					rgRegs[0] = null;
					rgRegs[1] = null;
					
					if (rgOps[j] instanceof IOperand.PseudoRegister)
						rgRegs[0] = (IOperand.PseudoRegister) rgOps[j];
					else if (rgOps[j] instanceof IOperand.Address)
					{
						IOperand.Address opAddr = (IOperand.Address) rgOps[j];
						int k = 0;
						if (opAddr.getRegBase () instanceof IOperand.PseudoRegister)
							rgRegs[k++] = (IOperand.PseudoRegister) opAddr.getRegBase ();
						if (opAddr.getRegIndex () instanceof IOperand.PseudoRegister)
							rgRegs[k++] = (IOperand.PseudoRegister) opAddr.getRegIndex ();
					}
					
					// check the registers
					for (IOperand.PseudoRegister reg : rgRegs)
					{
						if (reg == null)
							continue;
						
						mapGraphs.get (reg.getRegisterType ()).addVertex (new LAGraph.Vertex (reg));
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
					mapGraphs.get (reg.getRegisterType ()).addVertex (new LAGraph.Vertex (reg));
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
				IOperand[] rgOps = ((Instruction) instr).getOperands ();
								
				// check input operands (i.e., all operands except the last)
				for (int j = 0; j < rgOps.length - 1; j++)
				{
					if (reg.equals (rgOps[j]))
					{
						// another, later read was found
						return false;
					}
					
					// check whether the register is read within an address
					if (rgOps[j] instanceof IOperand.Address)
						if (reg.equals (((IOperand.Address) rgOps[j]).getRegBase ()) || reg.equals (((IOperand.Address) rgOps[j]).getRegIndex ()))
							return false;					
				}

				// check output operand: last read if no read occurred previously and the register is written to
				IOperand opOut = rgOps[rgOps.length - 1];
				if (reg.equals (opOut))
					return true;
				
				// if the output operand is an address, check whether reg is the base or the index register,
				// in which case there is a read
				if (opOut instanceof IOperand.Address)
					if (reg.equals (((IOperand.Address) opOut).getRegBase ()) || reg.equals (((IOperand.Address) opOut).getRegIndex ()))
						return false;
			}
		}
		
		return true;
	}
	
	public String visualize (Set<TypeRegisterType> setRegTypesToShow)
	{
		StringBuilder sb = new StringBuilder ();
		
		final int nInstrStrLen = 100;
		DecimalFormat fmt = new DecimalFormat (" 00");
		
		sb.append (StringUtil.padRight ("Instr \\ Reg", nInstrStrLen));
		for (int i = 0; i < m_rgPseudoRegisters.length; i++)
			if (setRegTypesToShow == null || (m_rgPseudoRegisters[i] != null && setRegTypesToShow.contains (m_rgPseudoRegisters[i].getRegisterType ())))
				sb.append (fmt.format (i));
		sb.append ('\n');
		
		for (int i = 0; i < m_rgLivePseudoRegisters.length; i++)
		{
			sb.append (StringUtil.padRight (m_rgInstructions[i].toString (), nInstrStrLen));

			for (int j = 0; j < m_rgLivePseudoRegisters[i].length; j++)
			{
				if (setRegTypesToShow != null && (m_rgPseudoRegisters[j] == null || !setRegTypesToShow.contains (m_rgPseudoRegisters[j].getRegisterType ())))
					continue;
				
				sb.append ("  ");
				switch (m_rgLivePseudoRegisters[i][j])
				{
				case STATE_LIVE:
					sb.append ('*');
					break;
				case STATE_DEAD:
					sb.append (' ');
					break;
				case STATE_UNASSIGNED:
					sb.append ('ø');
					break;
				}
			}
			sb.append ('\n');
		}
		
		return sb.toString ();
	}
	
	@Override
	public String toString ()
	{
		return visualize (null);
	}
}
