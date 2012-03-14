package ch.unibas.cs.hpwc.patus.graph.algorithm;

import java.util.HashSet;
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
	///////////////////////////////////////////////////////////////////
	// Implementation
	
	/**
	 * Finds the neighboring vertices of the vertex <code>vertex</code> in the directed graph <code>graph</code>.
	 * @param graph The directed graph to search for neighbors of <code>vertex</code>
	 * @param vertex The vertex whose neighbors to find
	 * @return An iterable over neighbors of <code>vertex</code>
	 */
	public static <V extends IVertex, E extends IEdge<V>> Iterable<V> getNeighborsDirected (IGraph<V, E> graph, V vertex)
	{
		Set<V> setNeighbors = new HashSet<V> ();
		for (E edge : graph.getEdges ())
		{
			if (edge.getTailVertex ().equals (vertex))
				setNeighbors.add (edge.getHeadVertex ());
		}
		
		return setNeighbors;
	}

	/**
	 * Finds the neighboring vertices of the vertex <code>vertex</code> in the undirected graph <code>graph</code>.
	 * @param graph The undirected graph to search for neighbors of <code>vertex</code>
	 * @param vertex The vertex whose neighbors to find
	 * @return An iterable over neighbors of <code>vertex</code>
	 */
	public static <V extends IVertex, E extends IEdge<V>> Iterable<V> getNeighborsUndirected (IGraph<V, E> graph, V vertex)
	{
		Set<V> setNeighbors = new HashSet<V> ();
		for (E edge : graph.getEdges ())
		{
			if (edge.getTailVertex ().equals (vertex))
				setNeighbors.add (edge.getHeadVertex ());
			if (edge.getHeadVertex ().equals (vertex))
				setNeighbors.add (edge.getTailVertex ());
		}
		
		return setNeighbors;
	}
}
