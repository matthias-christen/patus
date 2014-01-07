package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import org.apache.log4j.Logger;

import ch.unibas.cs.hpwc.patus.arch.IArchitectureDescription;
import ch.unibas.cs.hpwc.patus.arch.TypeBaseIntrinsicEnum;
import ch.unibas.cs.hpwc.patus.arch.TypeExecUnitType;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.analyze.DAGraph;
import ch.unibas.cs.hpwc.patus.graph.IVertex;
import ch.unibas.cs.hpwc.patus.graph.algorithm.GraphUtil;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 * <p>Reorders the instructions, which are given as a dependence graph,
 * so that latencies between instructions are minimized.</p>
 * 
 * <p>The algorithms are described in the following papers:</p>
 * <ul>
 * 	<li>K. Wilken, J. Liu, M. Heffernan: Optimal Instruction Scheduling Using Integer Programming, PLDI 2000</li>
 * 	<li>D. K&auml;stner, S. Winkel: ILP-based Instruction Scheduling for IA-64</li>
 * </ul>
 * 
 * @author Matthias-M. Christen
 */
public class InstructionScheduler extends AbstractInstructionScheduler
{
	///////////////////////////////////////////////////////////////////
	// Constants
	
	private final static Logger LOGGER = Logger.getLogger (InstructionScheduler.class);
	

	///////////////////////////////////////////////////////////////////
	// Inner Types
	
	private static class NopInstruction extends AbstractInstruction
	{
		@Override
		public void issue (StringBuilder sbResult)
		{
			// empty instruction: nothing to do
		}
		
		@Override
		public TypeBaseIntrinsicEnum getIntrinsic ()
		{
			return null;
		}
		
		@Override
		public String toString ()
		{
			return "<nop>";
		}

		@Override
		public String toJavaCode (Map<IOperand, String> mapIOperands)
		{
			return "new NopInstruction ();";
		}		
	}
			

	///////////////////////////////////////////////////////////////////
	// Member Variables
	
	private List<TypeExecUnitType> m_listAllExecUnitTypes;

	
	///////////////////////////////////////////////////////////////////
	// Implementation

	public InstructionScheduler (DAGraph graph, IArchitectureDescription arch)
	{
		super (graph, arch);
		
		if (arch.getAssemblySpec () != null && arch.getAssemblySpec ().getExecUnitTypes () != null)
			m_listAllExecUnitTypes = arch.getAssemblySpec ().getExecUnitTypes ().getExecUnitType ();
		else
			m_listAllExecUnitTypes = new ArrayList<> (1);
	}
		
	@Override
	protected int doSchedule (InstructionList ilOut)
	{
		LOGGER.info (StringUtil.concat ("Scheduling instructions (DAG has ", getGraph ().getVerticesCount (), " vertices and ", getGraph ().getEdgesCount (), " edges)."));
		
		boolean bIsFirst = true;
		int nScheduleLength = 0;
		
		createStandardForm ();
		for (DAGraph subgraph : partitionGraph ())
		{
			InstructionRegionScheduler sched = new InstructionRegionScheduler (subgraph, getArchitectureDescription ());

			// add the scheduled instructions to the global instruction list
			int i = 0;
			for (IInstruction instr : sched.schedule ())
			{
				// add the first instruction only if this is the first partition
				// (as the last vertex of a partition and the first vertex of the next partition coincide)
				if (((i == 0 && bIsFirst) || (i > 0)) && !(instr instanceof NopInstruction))
					ilOut.addInstruction (instr);
				i++;
			}
			
			nScheduleLength += sched.getScheduleLength ();
			bIsFirst = false;
		}
		
		return nScheduleLength;
	}
	
	/**
	 * Create a DAG with exactly one root node and exactly one leaf node.
	 */
	protected void createStandardForm ()
	{
		DAGraph graph = getGraph ();
		
		// if there is more than one root nodes, create an artificial root nodes
		// which immediately precedes all the original root nodes
		Collection<DAGraph.Vertex> collRoots = GraphUtil.getRootVertices (graph);
		if (collRoots.size () > 1)
		{
			DAGraph.Vertex vertRootNew = new DAGraph.Vertex (new NopInstruction ());
			graph.addVertex (vertRootNew);
			vertRootNew.setExecUnitTypes (m_listAllExecUnitTypes);
			
			for (DAGraph.Vertex vertRootOld : collRoots)
				graph.addEdge (vertRootNew, vertRootOld).setLatency (1);
		}
		
		// if there is more than one root nodes, create an artificial root nodes
		// which immediately precedes all the original root nodes
		Collection<DAGraph.Vertex> collLeaves = GraphUtil.getLeafVertices (graph);
		if (collLeaves.size () > 1)
		{
			DAGraph.Vertex vertLeafNew = new DAGraph.Vertex (new NopInstruction ());
			graph.addVertex (vertLeafNew);
			vertLeafNew.setExecUnitTypes (m_listAllExecUnitTypes);
			
			for (DAGraph.Vertex vertLeafOld : collLeaves)
				graph.addEdge (vertLeafOld, vertLeafNew).setLatency (1);
		}
	}
	
	protected List<DAGraph> partitionGraph ()
	{
		DAGraph graph = getGraph ();
		
		List<DAGraph> listGraphs = new LinkedList<> ();
		IVertex[] rgVerticesInTopologicalOrder = GraphUtil.getTopologicalSort (graph);
				
		// determine partition nodes and build subgraphs
		int nVertLatestIdx = -1;
		DAGraph graphCurrent = new DAGraph ();
		listGraphs.add (graphCurrent);
		
		for (int i = 0; i < rgVerticesInTopologicalOrder.length; i++)
		{
			DAGraph.Vertex v = (DAGraph.Vertex) rgVerticesInTopologicalOrder[i];
			graphCurrent.addVertex (v);
			
			if (i == nVertLatestIdx)
			{
				// this is a partition node; create a new subgraph
				graphCurrent = new DAGraph ();
				listGraphs.add (graphCurrent);				
				graphCurrent.addVertex (v);
			}
			
			for (DAGraph.Edge edge : graph.getOutgoingEdges (v))
			{
				// check whether w is later in the topological order than "latest"
				for (int j = i; j < rgVerticesInTopologicalOrder.length; j++)
					if (rgVerticesInTopologicalOrder[j] == edge.getHeadVertex () && j > nVertLatestIdx)
						nVertLatestIdx = j;
			}
		}
		
		// add edges to the subgraphs
		List<DAGraph> listGraphsToRemove = new ArrayList<> (listGraphs.size ());
		for (DAGraph subgraph : listGraphs)
		{
			if (subgraph.getVerticesCount () <= 1)
				listGraphsToRemove.add (subgraph);
			else
			{
				for (DAGraph.Vertex v : subgraph.getVertices ())
					for (DAGraph.Edge edge : graph.getOutgoingEdges (v))
					{
						if (subgraph.containsVertex (edge.getHeadVertex ()))
						{
							DAGraph.Edge edgeNew = subgraph.addEdge (v, edge.getHeadVertex ());
							edgeNew.setLatency (edge.getLatency ());
						}
					}
			}
		}
		
		// remove subgraphs with only one vertex
		for (DAGraph subgraph : listGraphsToRemove)
			listGraphs.remove (subgraph);
		
		return listGraphs;
	}		
}
