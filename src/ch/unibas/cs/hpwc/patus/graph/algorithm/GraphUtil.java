package ch.unibas.cs.hpwc.patus.graph.algorithm;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ch.unibas.cs.hpwc.patus.graph.IEdge;
import ch.unibas.cs.hpwc.patus.graph.IGraph;
import ch.unibas.cs.hpwc.patus.graph.IVertex;

/**
 * A collection of graph-related utility functions.
 * 
 * @author Matthias-M. Christen
 */
public class GraphUtil
{
	public static class VertexWithDegree<V extends IVertex, E extends IEdge<V>>
	{
		private V m_vertex;
		private int m_nDegree;
		
		public VertexWithDegree (V vertex, IGraph<V, E> graph, EDegree mode)
		{
			m_vertex = vertex;
			m_nDegree = GraphUtil.getDegree (graph, vertex, mode);
		}

		public V getVertex ()
		{
			return m_vertex;
		}

		public int getDegree ()
		{
			return m_nDegree;
		}
	}
	
	public static class VertexDegreeComparator<V extends IVertex, E extends IEdge<V>, W extends VertexWithDegree<V, E>> implements Comparator<W>
	{
		private boolean m_bSortAscending;
		
		public VertexDegreeComparator (boolean bSortAscending)
		{
			m_bSortAscending = bSortAscending;
		}
		
		@Override
		public int compare (W v1, W v2)
		{
			return m_bSortAscending ? v1.getDegree () - v2.getDegree () : v2.getDegree () - v1.getDegree ();
		}		
	}
	
	public enum EDegree
	{
		IN_DEGREE,
		OUT_DEGREE,
		INOUT_DEGREE;
		
		public boolean isInDegree ()
		{
			return this.equals (IN_DEGREE) || this.equals (INOUT_DEGREE);
		}
		
		public boolean isOutDegree ()
		{
			return this.equals (OUT_DEGREE) || this.equals (INOUT_DEGREE);
		}
		
		public boolean isInOutDegree ()
		{
			return this.equals (INOUT_DEGREE);
		}
	}
	
	
	///////////////////////////////////////////////////////////////////
	// Implementation
	
	public static <V extends IVertex, E extends IEdge<V>> Collection<V> getRootVertices (IGraph<V, E> graph)
	{
		Set<V> setRoots = new HashSet<> ();
		
		for (V vertex : graph.getVertices ())
			setRoots.add (vertex);
		for (E edge : graph.getEdges ())
			setRoots.remove (edge.getHeadVertex ());			
		
		return setRoots;
	}
	
	public static <V extends IVertex, E extends IEdge<V>> Collection<V> getLeafVertices (IGraph<V, E> graph)
	{
		Set<V> setLeaves = new HashSet<> ();
		
		for (V vertex : graph.getVertices ())
			setLeaves.add (vertex);
		for (E edge : graph.getEdges ())
			setLeaves.remove (edge.getTailVertex ());			
		
		return setLeaves;
	}
	
	/**
	 * Finds the neighboring vertices of the vertex <code>vertex</code> in the
	 * directed graph <code>graph</code>.
	 * 
	 * @param graph
	 *            The directed graph to search for neighbors of
	 *            <code>vertex</code>
	 * @param vertex
	 *            The vertex whose neighbors to find
	 * @return An iterable over neighbors of <code>vertex</code>
	 */
	public static <V extends IVertex, E extends IEdge<V>> Collection<V> getSuccessors (IGraph<V, E> graph, V vertex)
	{
		Set<V> setSuccessors = new HashSet<> ();
		for (E edge : graph.getEdges ())
		{
			if (edge.getTailVertex ().equals (vertex))
				setSuccessors.add (edge.getHeadVertex ());
		}
		
		return setSuccessors;
	}
	
	public static <V extends IVertex, E extends IEdge<V>> Collection<V> getPredecessors (IGraph<V, E> graph, V vertex)
	{
		Set<V> setPredecessors = new HashSet<> ();
		for (E edge : graph.getEdges ())
		{
			if (edge.getHeadVertex ().equals (vertex))
				setPredecessors.add (edge.getTailVertex ());
		}
		
		return setPredecessors;
	}
	
	/**
	 * Finds the neighboring vertices of the vertex <code>vertex</code> in the
	 * undirected graph <code>graph</code>.
	 * 
	 * @param graph
	 *            The undirected graph to search for neighbors of
	 *            <code>vertex</code>
	 * @param vertex
	 *            The vertex whose neighbors to find
	 * @return An iterable over neighbors of <code>vertex</code>
	 */
	public static <V extends IVertex, E extends IEdge<V>> Collection<V> getNeighbors (IGraph<V, E> graph, V vertex)
	{
		Set<V> setNeighbors = new HashSet<> ();
		for (E edge : graph.getEdges ())
		{
			if (edge.getTailVertex ().equals (vertex))
				setNeighbors.add (edge.getHeadVertex ());
			if (edge.getHeadVertex ().equals (vertex))
				setNeighbors.add (edge.getTailVertex ());
		}
		
		return setNeighbors;
	}
	
	/**
	 * Returns the degree of vertex <code>vertex</code> in the graph
	 * <code>graph</code>.
	 * 
	 * @param graph
	 *            The graph to which the vertex <code>vertex</code> belongs
	 * @param vertex
	 *            The vertex whose degree to retrieve
	 * @return The degree of vertex <code>vertex</code>
	 */
	public static <V extends IVertex, E extends IEdge<V>> int getDegree (IGraph<V, E> graph, V vertex, EDegree mode)
	{
		int nDegree = 0;
		for (E edge : graph.getEdges ())
			if ((mode.isInDegree () && edge.getHeadVertex ().equals (vertex)) || (mode.isOutDegree () && edge.getTailVertex ().equals (vertex)))
				nDegree++;
		return nDegree;
	}
	
	public static <V extends IVertex, E extends IEdge<V>> Iterable<VertexWithDegree<V, E>> getVerticesWithDegree (IGraph<V, E> graph, EDegree mode)
	{
		List<VertexWithDegree<V, E>> listVertices = new ArrayList<> (graph.getVerticesCount ());		
		for (V vertex : graph.getVertices ())
			listVertices.add (new VertexWithDegree<> (vertex, graph, mode));
		return listVertices;
	}
	
	/**
	 * Creates a new iterable over vertices which iterates in ascending or
	 * descending order of degree over the graph's vertices.
	 * 
	 * @param graph
	 *            The graph whose vertices to sort
	 * @param bSortAscending
	 *            Determines whether the sort is done in ascending or descending
	 *            order
	 * @return A new iterable over the vertices of <code>graph</code> iterating
	 *         in ascending or descending order of the vertex degrees over the
	 *         graph's vertices
	 */
	public static <V extends IVertex, E extends IEdge<V>> Iterable<V> getVerticesSortedByDegree (IGraph<V, E> graph, EDegree mode, boolean bSortAscending)
	{
		List<VertexWithDegree<V, E>> listVertices = (List<VertexWithDegree<V, E>>) GraphUtil.getVerticesWithDegree (graph, mode);
		Collections.sort (listVertices, new VertexDegreeComparator<V, E, VertexWithDegree<V, E>> (bSortAscending));
		
		List<V> list = new ArrayList<> (graph.getVerticesCount ());
		for (VertexWithDegree<V, E> vertex : listVertices)
			list.add (vertex.getVertex ());
		
		return list;
	}
	
	private static class Int
	{
		int m_nValue = 0;
		
		public int getValue ()
		{
			return m_nValue;
		}
		
		public void increment ()
		{
			m_nValue++;
		}
	}

	/**
	 * <p>Sorts the <code>graph</code>'s vertices in a topological order and
	 * returns them as an array.</p>
	 * <p>See also: {@link http://en.wikipedia.org/wiki/Topological_sorting}</p>
	 * <pre>
	 * L &larr; Empty list that will contain the sorted nodes
	 * S &larr; Set of all nodes with no outgoing edges
	 * for each node n in S do
	 *   getTopologicalSortVisit(n)
	 * </pre>
	 * 
	 * @param graph
	 *            The graph to sort
	 * @return An array of <code>IVertex</code>s sorted in a topological order
	 */
	public static <V extends IVertex, E extends IEdge<V>> IVertex[] getTopologicalSort (IGraph<V, E> graph)
	{
		IVertex[] rgVertices = new IVertex[graph.getVerticesCount ()];
		
		// find vertices with no outgoing edges
		Iterable<V> setNoOutgoing = getLeafVertices (graph);
		
		// depth-first search
		Map<V, Boolean> map = new HashMap<> ();
		for (V v : setNoOutgoing)
			GraphUtil.getTopologicalSortVisit (graph, v, map, rgVertices, new Int ());
		
		return rgVertices;
	}
	
	/**
	 * <pre>
	 * function getTopologicalSortVisit(vertex v)
	 *   if v has not been visited yet then
	 *     mark v as visited
	 *     for each node w with an edge from w to v do
	 *       visit(w)
	 *     add v to L
	 * </pre>
	 * 
	 * @param graph
	 *            The graph
	 * @param vertex
	 *            The vertex v
	 * @param mapVisited
	 *            Flags indicating whether a vertex has already been visited
	 * @param rgList
	 *            The output list (L)
	 * @param nListIdx
	 *            The current index in the output list
	 * @return The next index in the output list
	 */
	private static <V extends IVertex, E extends IEdge<V>> void getTopologicalSortVisit (IGraph<V, E> graph, V vertex, Map<V, Boolean> mapVisited, IVertex[] rgList, Int nListIdx)
	{
		if (!mapVisited.containsKey (vertex))
		{
			mapVisited.put (vertex, Boolean.TRUE);
			
			// visit recursively (each vertex v with an edge from v to vertex)
			for (E edge : graph.getEdges ())
				if (edge.getHeadVertex ().equals (vertex))
					GraphUtil.getTopologicalSortVisit (graph, edge.getTailVertex (), mapVisited, rgList, nListIdx);

			rgList[nListIdx.getValue ()] = vertex;
			nListIdx.increment ();
		}
	}
	
	/**
	 * Determines whether <code>listVertices</code> is a valid topological sort
	 * of the graph <code>graph</code>.
	 * 
	 * @param graph
	 *            The graph
	 * @param listVertices
	 *            The vertex order to check
	 * @return <code>true</code> if <code>listVertices</code> is a valid
	 *         topological sort of the graph <code>graph</code>
	 */
	public static <V extends IVertex, E extends IEdge<V>> boolean isTopologicalOrder (IGraph<V, E> graph, Iterable<V> listVertices)
	{
		Set<V> setVisited = new HashSet<> ();
		
		for (V v : listVertices)
		{
			// the successors of the vertex v must not have been visited yet
			for (V vSucc : GraphUtil.getSuccessors (graph, v))
				if (setVisited.contains (vSucc))
					return false;
			
			setVisited.add (v);
		}
		
		return true;
	}
}
