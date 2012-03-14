package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import ch.unibas.cs.hpwc.patus.graph.IEdge;
import ch.unibas.cs.hpwc.patus.graph.IGraph;
import ch.unibas.cs.hpwc.patus.graph.IParametrizedVertex;
import ch.unibas.cs.hpwc.patus.graph.algorithm.GraphUtil;

/**
 * The graph resulting from the live analysis.
 * @author Matthias-M. Christen
 */
public class LAGraph implements IGraph<LAGraph.Vertex, LAGraph.Edge>
{
	///////////////////////////////////////////////////////////////////
	// Inner Types

	/**
	 * A vertex in the live analysis graph. 
	 */
	public static class Vertex implements IParametrizedVertex<Integer>
	{
		private IOperand m_operand;
		private int m_nColor;
		
		public Vertex (IOperand op)
		{
			m_operand = op;
			m_nColor = -1;
		}
		
		public IOperand getOperand ()
		{
			return m_operand;
		}
		
		public void setColor (int nColor)
		{
			m_nColor = nColor;
		}
		
		public int getColor ()
		{
			return m_nColor;
		}
		
		@Override
		public void setData (Integer nData)
		{
			setColor (nData);
		}

		@Override
		public Integer getData ()
		{
			return getColor ();
		}
		
		@Override
		public boolean equals (Object obj)
		{
			if (!(obj instanceof Vertex))
				return false;
			return m_operand.equals (((Vertex) obj).getOperand ());
		}
		
		@Override
		public int hashCode ()
		{
			return m_operand.hashCode ();
		}
		
		@Override
		public String toString ()
		{
			StringBuilder sb = new StringBuilder ("Vertex { op=");
			sb.append (m_operand);
			sb.append (", col=");
			sb.append (m_nColor);
			sb.append (" }");
			
			return sb.toString ();
		}
	}
	
	/**
	 * An edge in the live analysis graph. 
	 */
	public class Edge implements IEdge<Vertex>
	{
		private Vertex m_vertex1;
		private Vertex m_vertex2;
		
		public Edge (Vertex v1, Vertex v2)
		{
			m_vertex1 = findVertex (v1);
			m_vertex2 = findVertex (v2);
		}
		
		public Vertex getHeadVertex ()
		{
			return m_vertex1;
		}
		
		public Vertex getTailVertex ()
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
			
			Edge other = (Edge) obj;
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
	// Member Variables
	
	private Map<Vertex, Vertex> m_mapVertices;
	private Set<Edge> m_setEdges;
	
	
	///////////////////////////////////////////////////////////////////
	// Implementation
	
	public LAGraph ()
	{
		m_mapVertices = new HashMap<Vertex, Vertex> ();
		m_setEdges = new HashSet<LAGraph.Edge> ();
	}
	
	public void addVertex (Vertex v)
	{
		if (!m_mapVertices.containsKey (v))
			m_mapVertices.put (v, v);
	}
	
	public Vertex findVertex (Vertex v)
	{
		Vertex vRes = m_mapVertices.get (v);
		if (vRes == null)
		{
			addVertex (v);
			return v;
		}
		
		return vRes;
	}
	
	public void addEdge (Vertex v1, Vertex v2)
	{
		addEdge (new Edge (v1, v2));
	}

	@Override
	public void addEdge (Edge edge)
	{
		m_setEdges.add (edge);
	}

	@Override
	public Iterable<Vertex> getVertices ()
	{
		return m_mapVertices.keySet ();
	}

	@Override
	public Iterable<Edge> getEdges ()
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
	public String toString ()
	{
		StringBuilder sb = new StringBuilder ("LAGraph {\n");
		for (Vertex v : m_mapVertices.keySet ())
		{
			sb.append ('\t');
			sb.append (v.toString ());
			sb.append ("  --->  { ");

			boolean bFirst = true;
			for (Vertex v1 : GraphUtil.getNeighborsDirected (this, v))
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
