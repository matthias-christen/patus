package ch.unibas.cs.hpwc.patus.codegen.backend.assembly.analyze;

import ch.unibas.cs.hpwc.patus.graph.DefaultGraph;
import ch.unibas.cs.hpwc.patus.graph.IEdge;
import ch.unibas.cs.hpwc.patus.graph.IVertex;

public abstract class Graph<V extends IVertex, E extends IEdge<V>> extends DefaultGraph<V, E>
{
	///////////////////////////////////////////////////////////////////
	// Inner Types
	
	/**
	 * An edge in the analysis graph. 
	 */
	public static class Edge<V extends IVertex> implements IEdge<V>
	{
		private V m_vertex1;
		private V m_vertex2;
		
		public Edge (Graph<V, ? extends Edge<V>> graph, V v1, V v2)
		{
			m_vertex1 = graph.findVertex (v1);
			m_vertex2 = graph.findVertex (v2);
		}
		
		@Override
		public V getHeadVertex ()
		{
			return m_vertex1;
		}
		
		@Override
		public V getTailVertex ()
		{
			return m_vertex2;
		}

		@Override
		public int hashCode ()
		{
			final int nPrime = 31;
			int nResult = nPrime + ((m_vertex1 == null) ? 0 : m_vertex1.hashCode ());
			nResult = nPrime * nResult + ((m_vertex2 == null) ? 0 : m_vertex2.hashCode ());
			return nResult;
		}

		@Override
		public boolean equals (Object obj)
		{
			if (this == obj)
				return true;
			if (obj == null)
				return false;
			
			if (getClass () != obj.getClass ())
				return false;
			
			@SuppressWarnings("unchecked")
			Edge<V> other = (Edge<V>) obj;
			if (m_vertex1 == null)
			{
				if (other.m_vertex1 != null)
					return false;
			}
			else if (!m_vertex1.equals (other.m_vertex1))
				return false;
			if (m_vertex2 == null)
			{
				if (other.m_vertex2 != null)
					return false;
			}
			else if (!m_vertex2.equals (other.m_vertex2))
				return false;
			
			return true;
		}
		
		@Override
		public String toString ()
		{
			StringBuilder sb = new StringBuilder ("Edge: ");
			sb.append (m_vertex1.toString ());
			sb.append (" --> ");
			sb.append (m_vertex2.toString ());

			return sb.toString ();
		}
	}
		
	
	///////////////////////////////////////////////////////////////////
	// Implementation
	
	public Graph ()
	{
		super ();
	}
		
	public void addEdge (V v1, V v2)
	{
		addEdge (createEdge (v1, v2));
	}
	
	protected abstract E createEdge (V v1, V v2);	
}
