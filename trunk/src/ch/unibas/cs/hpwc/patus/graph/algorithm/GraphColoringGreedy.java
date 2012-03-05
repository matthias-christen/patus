package ch.unibas.cs.hpwc.patus.graph.algorithm;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ch.unibas.cs.hpwc.patus.graph.IEdge;
import ch.unibas.cs.hpwc.patus.graph.IGraph;
import ch.unibas.cs.hpwc.patus.graph.IParametrizedVertex;

/**
 * Colors graphs using the greedy algorithm.
 * 
 * @author Matthias-M. Christen
 *
 * @param <V> The graph's vertex type
 * @param <E> The graph's edge type
 */
public class GraphColoringGreedy<V extends IParametrizedVertex<Integer>, E extends IEdge<V>>
{
	/**
	 * Colors the vertices of the graph <code>graph</code> using the greedy algorithm.
	 * @param graph The graph whose vertices to color
	 * @return The number of colors used for the coloring
	 */
	public static <V extends IParametrizedVertex<Integer>, E extends IEdge<V>> int run (IGraph<V, E> graph)
	{
		return new GraphColoringGreedy<V, E> (graph).run ();
	}

	
	///////////////////////////////////////////////////////////////////
	// Member Variables
	
	private IGraph<V, E> m_graph;
	
	private List<Integer> m_listColors;
	
	private int m_nLastColor;
	

	///////////////////////////////////////////////////////////////////
	// Implementation
	
	private GraphColoringGreedy (IGraph<V, E> graph)
	{
		m_graph = graph;
		m_listColors = new ArrayList<Integer> ();
		m_nLastColor = 0;
	}
	
	/**
	 * Runs the greedy vertex coloring algorithm.
	 * @return The number of colors used for the coloring
	 */
	private int run ()
	{
		for (V vertex : m_graph.getVertices ())
		{
			Collection<Integer> colors = getUnusedColors (vertex);
			if (colors.size () == 0)
			{
				// no available colors; add a new one
				m_listColors.add (m_nLastColor);
				vertex.setData (m_nLastColor);
				m_nLastColor++;
			}
			else
			{
				// set the vertex color (the first color from the collection of colors)
				vertex.setData (colors.iterator ().next ());
			}
		}
		
		return m_listColors.size ();
	}
	
	/**
	 * Returns a collection of possible candidate colors for vertex <code>vertex</code>
	 * @param vertices
	 * @return
	 */
	private Collection<Integer> getUnusedColors (V vertex)
	{
		Map<Integer, Boolean> mapColors = new HashMap<Integer, Boolean> ();
		for (int nCol : m_listColors)
			mapColors.put (nCol, Boolean.TRUE);
		
		for (V v : GraphUtil.getNeighborsUndirected (m_graph, vertex))
			if (mapColors.containsKey (v.getData ()))
				mapColors.remove (v.getData ());
		
		return mapColors.keySet ();
	}
}
