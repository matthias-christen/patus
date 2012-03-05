package ch.unibas.cs.hpwc.patus.graph;

/**
 * A general graph data structure.
 * 
 * @author Matthias-M. Christen
 *
 * @param <V> The vertex type
 * @param <E> The edge type
 */
public interface IGraph<V extends IVertex, E extends IEdge<V>>
{
	/**
	 * Adds a new vertex to the graph.
	 * @param vertex The vertex to add
	 */
	public abstract void addVertex (V vertex);
	
	/**
	 * Adds a new edge to the graph.
	 * @param edge The edge to add
	 */
	public abstract void addEdge (E edge);
	
	/**
	 * Returns an iterable over all vertices in the graph.
	 * @return An iterable over the graph's vertices
	 */
	public abstract Iterable<V> getVertices ();
	
	/**
	 * Returns an iterable over all edges in the graph.
	 * @return An iterable over the graph's edges
	 */
	public abstract Iterable<E> getEdges ();
	
	/**
	 * Returns the number of vertices in the graph.
	 * @return The number of vertices
	 */
	public abstract int getVerticesCount ();
	
	/**
	 * Returns the number of edges in the graph.
	 * @return The number of edges
	 */
	public abstract int getEdgesCount ();
}
