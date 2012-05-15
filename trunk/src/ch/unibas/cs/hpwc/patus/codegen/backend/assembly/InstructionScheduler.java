package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import java.util.Collection;
import java.util.LinkedList;
import java.util.List;

import ch.unibas.cs.hpwc.patus.arch.IArchitectureDescription;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.analyze.DAGraph;
import ch.unibas.cs.hpwc.patus.graph.IVertex;
import ch.unibas.cs.hpwc.patus.graph.algorithm.CriticalPathLengthCalculator;
import ch.unibas.cs.hpwc.patus.graph.algorithm.GraphUtil;
import ch.unibas.cs.hpwc.patus.util.MathUtil;

/**
 * <p>Reorders the instructions, which are given as a dependence graph,
 * so that latencies between instructions are minimized.</p>
 * 
 * <p>The algorithms are described in the following paper:</p>
 * 
 * K. Wilken, J. Liu, M. Heffernan: Optimal Instruction Scheduling Using Integer Programming, PLDI 2000
 * 
 * @author Matthias-M. Christen
 */
public class InstructionScheduler
{
	///////////////////////////////////////////////////////////////////
	// Inner Types
	
	private static class NopInstruction implements IInstruction
	{
		@Override
		public void issue (StringBuilder sbResult)
		{
			// empty instruction: nothing to do
		}		
	}

	
	///////////////////////////////////////////////////////////////////
	// Member Variables
	
	/**
	 * The analysis graph
	 */
	private DAGraph m_graph;
	
	private IArchitectureDescription m_arch;
	
	private int m_nIssueRate;
	
	private int m_nLowerScheduleLengthBound;
	private int m_nUpperScheduleLengthBound;
		

	///////////////////////////////////////////////////////////////////
	// Implementation

	
	public InstructionScheduler (DAGraph graph, IArchitectureDescription arch)
	{
		m_graph = graph;
		m_arch = arch;
		
		m_nIssueRate = m_arch.getIssueRate ();

		m_nLowerScheduleLengthBound = 0;
		m_nUpperScheduleLengthBound = 0;
	}
	
	public InstructionList schedule ()
	{
		InstructionList il = new InstructionList ();
		
		computeScheduleLengthBounds ();
		if (m_nLowerScheduleLengthBound == m_nUpperScheduleLengthBound)
		{
			// the schedule is already optimal; no further actions necessary
			return il;//buildInstructionList ();
		}
		
		createStandardForm ();
		for (DAGraph subgraph : partitionGraph ())
		{
			InstructionRegionScheduler sched = new InstructionRegionScheduler (subgraph, m_arch);
			sched.schedule ();
		}
		
		return il;
	}
	
	/**
	 * Compute lower and upper bounds on the length of the schedule.
	 * The upper bound is computed from a critical path list scheduling
	 */
	protected void computeScheduleLengthBounds ()
	{
		int nCriticalPathLength = 0;
		Iterable<DAGraph.Vertex> itRoots = GraphUtil.getRootVertices (m_graph);
		Iterable<DAGraph.Vertex> itLeaves = GraphUtil.getLeafVertices (m_graph);
		CriticalPathLengthCalculator<DAGraph.Vertex, DAGraph.Edge, Integer> calc = new CriticalPathLengthCalculator<> (m_graph, Integer.class);
		for (DAGraph.Vertex vertRoot : itRoots)
			for (DAGraph.Vertex vertLeaf : itLeaves)
				nCriticalPathLength = Math.max (nCriticalPathLength, calc.getCriticalPathDistance (vertRoot, vertLeaf));
		m_nLowerScheduleLengthBound = 1 + Math.max (nCriticalPathLength, MathUtil.divCeil (m_graph.getVerticesCount (), m_nIssueRate) - 1);
		
		// TODO
		m_nUpperScheduleLengthBound = m_graph.getVerticesCount () * 10;
	}
		
	/**
	 * Create a DAG with exactly one root node and exactly one leaf node.
	 */
	protected void createStandardForm ()
	{
		// if there is more than one root nodes, create an artificial root nodes
		// which immediately precedes all the original root nodes
		Collection<DAGraph.Vertex> collRoots = GraphUtil.getRootVertices (m_graph);
		if (collRoots.size () > 1)
		{
			DAGraph.Vertex vertRootNew = new DAGraph.Vertex (new NopInstruction ());
			m_graph.addVertex (vertRootNew);
			
			for (DAGraph.Vertex vertRootOld : collRoots)
				m_graph.addEdge (vertRootNew, vertRootOld);
		}
		
		// if there is more than one root nodes, create an artificial root nodes
		// which immediately precedes all the original root nodes
		Collection<DAGraph.Vertex> collLeaves = GraphUtil.getLeafVertices (m_graph);
		if (collLeaves.size () > 1)
		{
			DAGraph.Vertex vertLeafNew = new DAGraph.Vertex (new NopInstruction ());
			m_graph.addVertex (vertLeafNew);
			
			for (DAGraph.Vertex vertLeafOld : collLeaves)
				m_graph.addEdge (vertLeafOld, vertLeafNew);
		}
	}
	
	protected List<DAGraph> partitionGraph ()
	{
		List<DAGraph> listGraphs = new LinkedList<> ();
		IVertex[] rgVerticesInTopologicalOrder = GraphUtil.getTopologicalSort (m_graph);
				
		// determine partition nodes and build subgraphs
		int nVertLatestIdx = 0;
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
			
			//for (DAGraph.Vertex w : GraphUtil.getSuccessors (m_graph, v))
			for (DAGraph.Edge edge : m_graph.getOutgoingEdges (v))
			{
				// check whether w is later in the topological order than "latest"
				for (int j = i; j < rgVerticesInTopologicalOrder.length; j++)
					if (rgVerticesInTopologicalOrder[j] == /*w*/ edge.getHeadVertex () && j > nVertLatestIdx)
						nVertLatestIdx = j;
			}
		}
		
		// add edges to the subgraphs
		for (DAGraph subgraph : listGraphs)
			for (DAGraph.Vertex v : subgraph.getVertices ())
				//for (DAGraph.Vertex w : GraphUtil.getSuccessors (m_graph, v))
				for (DAGraph.Edge edge : m_graph.getOutgoingEdges (v))
					subgraph.addEdge (v, /*w*/ edge.getHeadVertex ());
		
		return listGraphs;
	}		
}
