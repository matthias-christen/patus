package ch.unibas.cs.hpwc.patus.codegen.backend.assembly.analyze;

import cetus.hir.Specifier;
import ch.unibas.cs.hpwc.patus.arch.IArchitectureDescription;
import ch.unibas.cs.hpwc.patus.arch.TypeArchitectureType.Intrinsics.Intrinsic;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.IInstruction;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.IOperand;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.Instruction;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.InstructionList;

public class DependenceAnalysis
{
	///////////////////////////////////////////////////////////////////
	// Constants
	
	private final static int LATENCY_DEFAULT = 1;
	private final static int LATENCY_MEMORY_MOVE = 50;
	
	
	///////////////////////////////////////////////////////////////////
	// Member Variables
	
	private IArchitectureDescription m_arch;

	/**
	 * The array of instructions to analyze
	 */
	private IInstruction[] m_rgInstructions;
	private int m_nInstructionsCount;
		
	
	///////////////////////////////////////////////////////////////////
	// Implementation

	public DependenceAnalysis (InstructionList il, IArchitectureDescription arch)
	{
		m_arch = arch;
		m_rgInstructions = new IInstruction[il.size ()];
		
		int j = 0;
		for (IInstruction instr : il)
		{
			if (instr instanceof Instruction)
			{
				m_rgInstructions[j] = instr;
				j++;
			}
		}
		m_nInstructionsCount = j;
	}
	
	public DAGraph run (Specifier specDatatype)
	{
		DAGraph graph = new DAGraph ();
		DAGraph.Vertex[] rgVertices = new DAGraph.Vertex[m_nInstructionsCount];
		
		for (int i = 0; i < m_nInstructionsCount; i++)
		{
			graph.addVertex (rgVertices[i] = new DAGraph.Vertex (m_rgInstructions[i]));
			
			IOperand[] rgOperands = null;
			if (rgVertices[i].getInstruction () instanceof Instruction)
				rgOperands = ((Instruction) rgVertices[i].getInstruction ()).getOperands ();
			Intrinsic intrinsicI = m_arch.getIntrinsic (m_rgInstructions[i].getIntrinsic (), specDatatype, rgOperands);
			if (intrinsicI != null && intrinsicI.getExecUnitTypeIds () != null && intrinsicI.getExecUnitTypeIds ().size () > 0)
				rgVertices[i].setExecUnitTypes (m_arch.getExecutionUnitTypesByIDs (intrinsicI.getExecUnitTypeIds ()));
			
			for (int j = 0; j < i; j++)
			{
				if (m_rgInstructions[j] instanceof Instruction)
				{
					// does the current instruction (i) depend on a previous one (j)?
					if (InstructionListAnalyzer.isFlowDependent (m_rgInstructions[j], m_rgInstructions[i]))
					{
						DAGraph.Edge edge = graph.addEdge (rgVertices[j], rgVertices[i]);
					
						// set the latency (the edge weight)
						
//						if (InstructionListAnalyzer.movesDataBetweenMemory (m_rgInstructions[j]))
//							edge.setLatency (LATENCY_MEMORY_MOVE);
//						else
						{
							Intrinsic intrinsicJ = m_arch.getIntrinsic (m_rgInstructions[j].getIntrinsic (), specDatatype);
							edge.setLatency (intrinsicJ == null ? LATENCY_DEFAULT : intrinsicJ.getLatency ());
						}
					}
				}
			}
		}
		
		return graph;
	}	
}
