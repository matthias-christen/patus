package ch.unibas.cs.hpwc.patus.graph;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import ch.unibas.cs.hpwc.patus.graph.algorithm.GraphUtil;

/**
 * A default graph implementation.
 * @author Matthias-M. Christen
 *
 * @param <V> The vertex type
 * @param <E> The edge type
 */
public class DefaultGraph<V extends IVertex, E extends IEdge<V>> implements IGraph<V, E>
{
	///////////////////////////////////////////////////////////////////
	// Member Variables
	
	protected Map<V, V> m_mapVertices;
	protected Set<E> m_setEdges;
	
	
	///////////////////////////////////////////////////////////////////
	// Implementation
	
	public DefaultGraph ()
	{
		m_mapVertices = new HashMap<> ();
		m_setEdges = new HashSet<> ();
	}
	
	public void addVertex (V v)
	{
		if (!m_mapVertices.containsKey (v))
			m_mapVertices.put (v, v);
	}
	
	public V findVertex (V v)
	{
		V vRes = m_mapVertices.get (v);
		if (vRes == null)
		{
			addVertex (v);
			return v;
		}
		
		return vRes;
	}
	
	@Override
	public void addEdge (E edge)
	{
		m_setEdges.add (edge);
	}

	@Override
	public Iterable<V> getVertices ()
	{
		return m_mapVertices.keySet ();
	}

	@Override
	public Iterable<E> getEdges ()
	{
		return m_setEdges;
	}

	@Override
	public int getVerticesCount ()
	{
		return m_mapVertices.size ();
	}

	@Override
	public int getEdgesCount ()
	{
		return m_setEdges.size ();
	}
	
	@Override
	public void removeAllVertices ()
	{
		m_mapVertices.clear ();
		removeAllEdges ();
	}
	
	public void removeEdge (E edge)
	{
		m_setEdges.remove (edge);
	}
	
	@Override
	public void removeAllEdges ()
	{
		m_setEdges.clear ();
	}
	
	@Override
	public String toString ()
	{
		StringBuilder sb = new StringBuilder (getClass ().getSimpleName ());
		sb.append (" {\n");
		
		for (V v : m_mapVertices.keySet ())
		{
			sb.append ('\t');
			sb.append (v.toString ());
			sb.append ("  --->  { ");

			boolean bFirst = true;
			for (V v1 : GraphUtil.getNeighborsDirected (this, v))
			{
				if (!bFirst)
					sb.append (", ");
				sb.append (v1.toString ());
				bFirst = false;
			}
			
			sb.append (" }\n");
		}
		sb.append ('}');

		return sb.toString ();
	}
}
