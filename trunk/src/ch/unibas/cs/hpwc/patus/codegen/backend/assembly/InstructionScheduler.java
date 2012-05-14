package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import java.util.Collection;
import java.util.LinkedList;
import java.util.List;

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
		}		
	}

	
	///////////////////////////////////////////////////////////////////
	// Member Variables
	
	/**
	 * The analysis graph
	 */
	private DAGraph m_graph;
	
	private int m_nIssueRate;
	
	private int m_nLowerScheduleLengthBound;
	private int m_nUpperScheduleLengthBound;
		

	///////////////////////////////////////////////////////////////////
	// Implementation

	
	public InstructionScheduler (DAGraph graph)
	{
		m_graph = graph;
		
		// TODO: issue rate from arch spec
		m_nIssueRate = 1;

		m_nLowerScheduleLengthBound = 0;
		m_nUpperScheduleLengthBound = 0;
	}
	
	public InstructionList schedule ()
	{
		InstructionList il = new InstructionList ();
		
		computeScheduleLengthBounds ();
		createStandardForm ();
		for (DAGraph subgraph : partitionGraph (m_graph))
		{
			// simplify the subgraph before solving the linear program
			CriticalPathLengthCalculator<DAGraph.Vertex, DAGraph.Edge, Integer> cpcalc = new CriticalPathLengthCalculator<> (subgraph, Integer.class);
			removeRedundantEdges (subgraph, cpcalc);
			linearizeRegions (subgraph);

			computeInitialScheduleBounds (subgraph, cpcalc);
			for (int nCurrentScheduleLength = m_nUpperScheduleLengthBound; nCurrentScheduleLength >= m_nLowerScheduleLengthBound; nCurrentScheduleLength--)
			{
				if (!solveILP (subgraph))
					break;
				reduceCurrentScheduleLength (subgraph);
			}
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
	
	@SuppressWarnings("static-method")
	protected List<DAGraph> partitionGraph (DAGraph graph)
	{
		List<DAGraph> listGraphs = new LinkedList<> ();
		IVertex[] rgVerticesInTopologicalOrder = GraphUtil.getTopologicalSort (graph);
				
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
			
			for (DAGraph.Vertex w : GraphUtil.getSuccessors (graph, v))
			{
				// check whether w is later in the topological order than "latest"
				for (int j = i; j < rgVerticesInTopologicalOrder.length; j++)
					if (rgVerticesInTopologicalOrder[j] == w && j > nVertLatestIdx)
						nVertLatestIdx = j;
			}
		}
		
		// add edges to the subgraphs
		for (DAGraph subgraph : listGraphs)
			for (DAGraph.Vertex v : subgraph.getVertices ())
				for (DAGraph.Vertex w : GraphUtil.getSuccessors (graph, v))
					subgraph.addEdge (v, w);
		
		return listGraphs;
	}
		
	@SuppressWarnings("static-method")
	protected void removeRedundantEdges (DAGraph graph, CriticalPathLengthCalculator<DAGraph.Vertex, DAGraph.Edge, Integer> cpcalc)
	{
		IVertex[] rgVerticesInTopologicalOrder = GraphUtil.getTopologicalSort (graph);
		
		for (int i = 0; i < rgVerticesInTopologicalOrder.length; i++)
		{
			Iterable<DAGraph.Vertex> itNeighbors = GraphUtil.getSuccessors (graph, (DAGraph.Vertex) rgVerticesInTopologicalOrder[i]);
			for (DAGraph.Vertex v : itNeighbors)
			{
				for (DAGraph.Vertex w : itNeighbors)
				{
					if (v != w)
					{
						DAGraph.Edge edgeIV = graph.getEdge ((DAGraph.Vertex) rgVerticesInTopologicalOrder[i], v);
						DAGraph.Edge edgeIW = graph.getEdge ((DAGraph.Vertex) rgVerticesInTopologicalOrder[i], w);
						if (edgeIW.getLatency () + cpcalc.getCriticalPathDistance (w, v) >= edgeIV.getLatency ())
							graph.removeEdge (edgeIV);
					}
				}
			}
		}
	}

	protected void linearizeRegions (DAGraph graph)
	{
		// TODO
	}

	protected void computeInitialScheduleBounds (DAGraph graph, CriticalPathLengthCalculator<DAGraph.Vertex, DAGraph.Edge, Integer> cpcalc)
	{
		Iterable<DAGraph.Vertex> itRoots = GraphUtil.getRootVertices (graph);
		Iterable<DAGraph.Vertex> itLeaves = GraphUtil.getLeafVertices (graph);
		
		for (DAGraph.Vertex v : graph.getVertices ())
		{
			int nCritPathDistFromRoots = 0;
			for (DAGraph.Vertex vertRoot : itRoots)
				nCritPathDistFromRoots = Math.max (nCritPathDistFromRoots, cpcalc.getCriticalPathDistance (vertRoot, v));
			
			int nCritPathDistToLeaves = 0;
			for (DAGraph.Vertex vertLeaf : itLeaves)
				nCritPathDistToLeaves = Math.max (nCritPathDistToLeaves, cpcalc.getCriticalPathDistance (v, vertLeaf));
			
			int nPredecessorsCount = GraphUtil.getPredecessors (graph, v).size ();
			int nSuccessorsCount = GraphUtil.getSuccessors (graph, v).size ();
			
			v.setScheduleBounds (
				1 + Math.max (nCritPathDistFromRoots, MathUtil.divCeil (1 + nPredecessorsCount, m_nIssueRate) - 1),
				m_nUpperScheduleLengthBound - Math.max (nCritPathDistToLeaves, MathUtil.divCeil (1 + nSuccessorsCount, m_nIssueRate) - 1)
			);
		}
	}
	
	@SuppressWarnings("static-method")
	protected void reduceCurrentScheduleLength (DAGraph graph)
	{
		for (DAGraph.Vertex v : graph.getVertices ())
			v.setScheduleBounds (v.getLowerScheduleBound (), v.getUpperScheduleBound () - 1);
	}

	/**
	 * Builds the ILP formulation from the graph and tries to solve the ILP.
	 * 
	 * @param graph
	 *            The (simplified) dependence analysis graph for which to build
	 *            the ILP
	 * @return <code>true</code> if the solver was able to solve the ILP,
	 *         <code>false</code> if no solution was found or the problem is
	 *         infeasible
	 */
	@SuppressWarnings("static-method")
	protected boolean solveILP (DAGraph graph)
	{
		return false;
	}
}
