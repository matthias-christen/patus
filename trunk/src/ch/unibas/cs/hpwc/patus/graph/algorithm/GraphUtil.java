package ch.unibas.cs.hpwc.patus.graph.algorithm;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
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
	private static class VertexWithDegree<V extends IVertex> implements Comparable<VertexWithDegree<V>>
	{
		private V m_vertex;
		private int m_nDegree;
		private boolean m_bSortAscending;
		
		public VertexWithDegree (V vertex, int nDegree, boolean bSortAscending)
		{
			m_vertex = vertex;
			m_nDegree = nDegree;
			m_bSortAscending = bSortAscending;
		}

		public V getVertex ()
		{
			return m_vertex;
		}

		public int getDegree ()
		{
			return m_nDegree;
		}

		@Override
		public int compareTo (VertexWithDegree<V> vertexOther)
		{
			return m_bSortAscending ? m_nDegree - vertexOther.getDegree () : vertexOther.getDegree () - m_nDegree;
		}
	}
	

	///////////////////////////////////////////////////////////////////
	// Implementation
	
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
	public static <V extends IVertex, E extends IEdge<V>> Iterable<V> getNeighborsDirected (IGraph<V, E> graph, V vertex)
	{
		Set<V> setNeighbors = new HashSet<> ();
		for (E edge : graph.getEdges ())
		{
			if (edge.getTailVertex ().equals (vertex))
				setNeighbors.add (edge.getHeadVertex ());
		}
		
		return setNeighbors;
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
	public static <V extends IVertex, E extends IEdge<V>> Iterable<V> getNeighborsUndirected (IGraph<V, E> graph, V vertex)
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
	 * Returns the degree of vertex <code>vertex</code> in the graph <code>graph</code>.
	 * @param graph
	 * @param vertex
	 * @return
	 */
	public static <V extends IVertex, E extends IEdge<V>> int getDegree (IGraph<V, E> graph, V vertex)
	{
		int nDegree = 0;
		for (E edge : graph.getEdges ())
		{
			if (edge.getTailVertex ().equals (vertex) || edge.getHeadVertex ().equals (vertex))
				nDegree++;
		}
		
		return nDegree;
	}
	
	/**
	 * 
	 * @param graph
	 * @param bSortAscending
	 * @return
	 */
	public static <V extends IVertex, E extends IEdge<V>> Iterable<V> getVerticesSortedByDegree (IGraph<V, E> graph, boolean bSortAscending)
	{
		List<VertexWithDegree<V>> listVertices = new ArrayList<> (graph.getVerticesCount ());
		
		for (V vertex : graph.getVertices ())
			listVertices.add (new VertexWithDegree<> (vertex, GraphUtil.getDegree (graph, vertex), bSortAscending));
		Collections.sort (listVertices);
		
		List<V> list = new ArrayList<> (graph.getVerticesCount ());
		for (VertexWithDegree<V> vertex : listVertices)
			list.add (vertex.getVertex ());
		
		return list;
	}
}
