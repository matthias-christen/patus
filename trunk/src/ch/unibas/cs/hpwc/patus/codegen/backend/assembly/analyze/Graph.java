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
		private V m_vertexTail;
		private V m_vertexHead;
		
		public Edge (Graph<V, ? extends Edge<V>> graph, V vertexTail, V vertexHead)
		{
			m_vertexTail = graph.findVertex (vertexTail);
			m_vertexHead = graph.findVertex (vertexHead);
		}
		
		@Override
		public V getHeadVertex ()
		{
			return m_vertexHead;
		}
		
		@Override
		public V getTailVertex ()
		{
			return m_vertexTail;
		}

		@Override
		public int hashCode ()
		{
			final int nPrime = 31;
			int nResult = nPrime + ((m_vertexHead == null) ? 0 : m_vertexHead.hashCode ());
			nResult = nPrime * nResult + ((m_vertexTail == null) ? 0 : m_vertexTail.hashCode ());
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
			if (m_vertexTail == null)
			{
				if (other.m_vertexTail != null)
					return false;
			}
			else if (!m_vertexTail.equals (other.m_vertexTail))
				return false;
			if (m_vertexHead == null)
			{
				if (other.m_vertexHead != null)
					return false;
			}
			else if (!m_vertexHead.equals (other.m_vertexHead))
				return false;
			
			return true;
		}
		
		@Override
		public String toString ()
		{
			StringBuilder sb = new StringBuilder ("Edge: ");
			sb.append (m_vertexTail.toString ());
			sb.append (" --> ");
			sb.append (m_vertexHead.toString ());

			return sb.toString ();
		}
	}
		
	
	///////////////////////////////////////////////////////////////////
	// Implementation
	
	public Graph ()
	{
		super ();
	}
		
	public E addEdge (V vertexTail, V vertexHead)
	{
		E edge = createEdge (vertexTail, vertexHead);
		addEdge (edge);
		return edge;
	}
	
	protected abstract E createEdge (V vertexTail, V vertexHead);	
}
