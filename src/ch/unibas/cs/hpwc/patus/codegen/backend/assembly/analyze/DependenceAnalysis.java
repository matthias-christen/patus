package ch.unibas.cs.hpwc.patus.codegen.backend.assembly.analyze;

import ch.unibas.cs.hpwc.patus.arch.IArchitectureDescription;
import ch.unibas.cs.hpwc.patus.arch.TypeArchitectureType.Intrinsics.Intrinsic;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.IInstruction;
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
		
	
	///////////////////////////////////////////////////////////////////
	// Implementation

	public DependenceAnalysis (InstructionList il, IArchitectureDescription arch)
	{
		m_arch = arch;
		m_rgInstructions = new IInstruction[il.size ()];
		
		int j = 0;
		for (IInstruction instr : il)
		{
			m_rgInstructions[j] = instr;
			j++;
		}
	}
	
	public DAGraph run ()
	{
		DAGraph graph = new DAGraph ();
		DAGraph.Vertex[] rgVertices = new DAGraph.Vertex[m_rgInstructions.length];
		
		for (int i = 0; i < m_rgInstructions.length; i++)
		{
			graph.addVertex (rgVertices[i] = new DAGraph.Vertex (m_rgInstructions[i]));
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
							Intrinsic intrinsic = m_arch.getIntrinsicByIntrinsicName (((Instruction) m_rgInstructions[j]).getIntrinsicBaseName ());
							edge.setLatency (intrinsic == null || intrinsic.getLatency () == null ? LATENCY_DEFAULT : intrinsic.getLatency ().intValue ());
						}
					}
				}
			}
		}
		
		return graph;
	}	
}
