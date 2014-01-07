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
 * 
 * <p>See also <a href="http://code.google.com/p/annas/">http://code.google.com/p/annas/</a></p>
 */
public class GraphColoringGreedy<V extends IParametrizedVertex<Integer>, E extends IEdge<V>>
{
	/**
	 * Colors the vertices of the graph <code>graph</code> using the greedy
	 * algorithm.
	 * 
	 * @param graph
	 *            The graph whose vertices to color
	 * @return The number of colors used for the coloring
	 */
	public static <V extends IParametrizedVertex<Integer>, E extends IEdge<V>> int run (IGraph<V, E> graph)
	{
		return new GraphColoringGreedy<> (graph).run ();
	}

	
	///////////////////////////////////////////////////////////////////
	// Member Variables
	
	/**
	 * The graph to color
	 */
	private IGraph<V, E> m_graph;
	
	/**
	 * The current list of colors
	 */
	private List<Integer> m_listColors;
	
	/**
	 * The color which was inserted last
	 */
	private int m_nLastColor;
	

	///////////////////////////////////////////////////////////////////
	// Implementation
	
	private GraphColoringGreedy (IGraph<V, E> graph)
	{
		m_graph = graph;
		m_listColors = new ArrayList<> ();
		m_nLastColor = 0;
	}
	
	/**
	 * Runs the greedy vertex coloring algorithm.
	 * 
	 * @return The number of colors used for the coloring
	 */
	private int run ()
	{
		for (V vertex : GraphUtil.getVerticesSortedByDegree (m_graph, GraphUtil.EDegree.INOUT_DEGREE, false))
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
		Map<Integer, Boolean> mapColors = new HashMap<> ();
		for (int nCol : m_listColors)
			mapColors.put (nCol, Boolean.TRUE);
		
		for (V v : GraphUtil.getNeighbors (m_graph, vertex))
		{
			if (mapColors.containsKey (v.getData ()))
				mapColors.remove (v.getData ());
			if (mapColors.size () == 0)
				break;
		}
		
		return mapColors.keySet ();
	}
}
