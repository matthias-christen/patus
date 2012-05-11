package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.analyze.DAGraph;
import ch.unibas.cs.hpwc.patus.graph.IVertex;
import ch.unibas.cs.hpwc.patus.graph.algorithm.GraphUtil;

public class InstructionScheduler
{
	///////////////////////////////////////////////////////////////////
	// Member Variables
	
	/**
	 * The analysis graph
	 */
	private DAGraph m_graph;
		

	///////////////////////////////////////////////////////////////////
	// Implementation

	
	public InstructionScheduler (DAGraph graph)
	{
		m_graph = graph;
	}
	
	public InstructionList schedule ()
	{
		InstructionList il = new InstructionList ();
		
		createStandardForm ();
		for (DAGraph subgraph : partitionGraph (m_graph))
		{
			// simplify the subgraph before solving the linear program
			removeRedundantEdges (subgraph);
			linearizeRegions (subgraph);
			
			solveILP (subgraph);
		}
		
		return il;
	}
	
	protected void createStandardForm ()
	{
		
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
			
			for (DAGraph.Vertex w : GraphUtil.getNeighborsDirected (graph, v))
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
				for (DAGraph.Vertex w : GraphUtil.getNeighborsDirected (graph, v))
					subgraph.addEdge (v, w);
		
		return listGraphs;
	}
		
	@SuppressWarnings({ "static-method", "unchecked" })
	protected void removeRedundantEdges (DAGraph graph)
	{
		IVertex[] rgVerticesInTopologicalOrder = GraphUtil.getTopologicalSort (graph);
		Map<DAGraph.Vertex, Map<DAGraph.Vertex, Integer>> mapDistances = computeCriticalPathDistances (graph);
		
		for (int i = 0; i < rgVerticesInTopologicalOrder.length; i++)
		{
			Iterable<DAGraph.Vertex> itNeighbors = GraphUtil.getNeighborsDirected (graph, (DAGraph.Vertex) rgVerticesInTopologicalOrder[i]);
			for (DAGraph.Vertex v : itNeighbors)
			{
				for (DAGraph.Vertex w : itNeighbors)
				{
					if (v != w)
					{
						DAGraph.Edge edgeIV = graph.getEdge ((DAGraph.Vertex) rgVerticesInTopologicalOrder[i], v);
						DAGraph.Edge edgeIW = graph.getEdge ((DAGraph.Vertex) rgVerticesInTopologicalOrder[i], w);
						if (edgeIW.getLatency () + getCriticalPathDistance (mapDistances, w, v) >= edgeIV.getLatency ())
							graph.removeEdge (edgeIV);
					}
				}
			}
		}
	}
	
	protected Map<DAGraph.Vertex, Map<DAGraph.Vertex, Integer>> computeCriticalPathDistances (DAGraph graph)
	{
		// TODO
		return null;
	}
	
	@SuppressWarnings("static-method")
	protected int getCriticalPathDistance (Map<DAGraph.Vertex, Map<DAGraph.Vertex, Integer>> mapDistances, DAGraph.Vertex v1, DAGraph.Vertex v2)
	{
		Map<DAGraph.Vertex, Integer> map = mapDistances.get (v1);
		if (map == null)
			return Integer.MIN_VALUE;
		Integer nDist = map.get (v2);
		if (nDist == null)
			return Integer.MIN_VALUE;
		return nDist;
	}
	
	protected void linearizeRegions (DAGraph graph)
	{
		// TODO
	}
	
	protected void solveILP (DAGraph graph)
	{
		
	}
}
