package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;

import ch.unibas.cs.hpwc.patus.graph.IEdge;
import ch.unibas.cs.hpwc.patus.graph.IGraph;
import ch.unibas.cs.hpwc.patus.graph.IParametrizedVertex;

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
	}
	
	/**
	 * An edge in the live analysis graph. 
	 */
	public static class Edge implements IEdge<Vertex>
	{
		private Vertex m_vertex1;
		private Vertex m_vertex2;
		
		public Edge (Vertex v1, Vertex v2)
		{
			m_vertex1 = v1;
			m_vertex2 = v2;
		}
		
		public Vertex getHeadVertex ()
		{
			return m_vertex1;
		}
		
		public Vertex getTailVertex ()
		{
			return m_vertex2;
		}
	}
	
	
	///////////////////////////////////////////////////////////////////
	// Member Variables
	
	private Set<Vertex> m_setVertices;
	private List<Edge> m_listEdges;
	
	
	///////////////////////////////////////////////////////////////////
	// Implementation
	
	public LAGraph ()
	{
		m_setVertices = new HashSet<LAGraph.Vertex> ();
		m_listEdges = new LinkedList<LAGraph.Edge> ();
	}
	
	public void addVertex (Vertex v)
	{
		m_setVertices.add (v);
	}
	
	public void addEdge (Vertex v1, Vertex v2)
	{
		addEdge (new Edge (v1, v2));
	}

	@Override
	public void addEdge (Edge edge)
	{
		m_listEdges.add (edge);
	}

	@Override
	public Iterable<Vertex> getVertices ()
	{
		return m_setVertices;
	}

	@Override
	public Iterable<Edge> getEdges ()
	{
		return m_listEdges;
	}

	@Override
	public int getVerticesCount ()
	{
		return m_setVertices.size ();
	}

	@Override
	public int getEdgesCount ()
	{
		return m_listEdges.size ();
	}
}
