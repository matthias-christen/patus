package ch.unibas.cs.hpwc.patus.codegen.backend.assembly.analyze;

import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.IInstruction;
import ch.unibas.cs.hpwc.patus.graph.IParametrizedEdge;
import ch.unibas.cs.hpwc.patus.graph.IVertex;

public class DAGraph extends Graph<DAGraph.Vertex, DAGraph.Edge>
{
	///////////////////////////////////////////////////////////////////
	// Inner Types

	/**
	 * A vertex in the live analysis graph. 
	 */
	public static class Vertex implements IVertex
	{
		private IInstruction m_instruction;

		private int m_nInitialLowerScheduleBound;
		private int m_nInitialUpperScheduleBound;

		private int m_nLowerScheduleBound;
		private int m_nUpperScheduleBound;
		
		private int m_nTemporaryLowerScheduleBound;
		private int m_nTemporaryUpperScheduleBound;

		
		public Vertex (IInstruction instr)
		{
			m_instruction = instr;
		}
		
		public IInstruction getInstruction ()
		{
			return m_instruction;
		}
		
		public void setScheduleBounds (int nLowerBound, int nUpperBound)
		{
			m_nTemporaryLowerScheduleBound = nLowerBound;
			m_nTemporaryUpperScheduleBound = nUpperBound;
		}
		
		public int getLowerScheduleBound ()
		{
			return m_nTemporaryLowerScheduleBound;
		}
		
		public int getUpperScheduleBound ()
		{
			return m_nTemporaryUpperScheduleBound;
		}
		
		public void commitBounds ()
		{
			m_nLowerScheduleBound = m_nTemporaryLowerScheduleBound;
			m_nUpperScheduleBound = m_nTemporaryUpperScheduleBound;
		}
		
		public void discardBounds ()
		{
			m_nTemporaryLowerScheduleBound = m_nLowerScheduleBound;
			m_nTemporaryUpperScheduleBound = m_nUpperScheduleBound;
		}

		public void setInitialScheduleBounds (int nLowerBound, int nUpperBound)
		{
			m_nInitialLowerScheduleBound = nLowerBound;
			m_nInitialUpperScheduleBound = nUpperBound;
			setScheduleBounds (m_nInitialLowerScheduleBound, m_nInitialUpperScheduleBound);
			commitBounds ();
		}
		
		public int getInitialLowerScheduleBound ()
		{
			return m_nInitialLowerScheduleBound;
		}
		
		public int getInitialUpperScheduleBound ()
		{
			return m_nInitialUpperScheduleBound;
		}

		@Override
		public boolean equals (Object obj)
		{
			if (!(obj instanceof Vertex))
				return false;
			return m_instruction.equals (((Vertex) obj).getInstruction ());
		}
		
		@Override
		public int hashCode ()
		{
			return m_instruction.hashCode ();
		}
		
		@Override
		public String toString ()
		{
			StringBuilder sb = new StringBuilder ("Vertex { instr=");
			sb.append (m_instruction.toString ());
			sb.append (" }");
			
			return sb.toString ();
		}
	}
	
	public class Edge extends Graph.Edge<DAGraph.Vertex> implements IParametrizedEdge<DAGraph.Vertex, Integer>
	{
		private int m_nLatency;
		
		public Edge (DAGraph.Vertex v1, DAGraph.Vertex v2)
		{
			super (DAGraph.this, v1, v2);
		}
				
		public void setLatency (int nLatency)
		{
			m_nLatency = nLatency;
		}
		
		public int getLatency ()
		{
			return m_nLatency;
		}
		
		@Override
		public void setData (Integer nData)
		{
			setLatency (m_nLatency);
		}

		@Override
		public Integer getData ()
		{
			return getLatency ();
		}
	}

	
	///////////////////////////////////////////////////////////////////
	// Member Variables
	
	private Map<DAGraph.Vertex, List<DAGraph.Edge>> m_mapOutgoingEdges;
	private Map<DAGraph.Vertex, List<DAGraph.Edge>> m_mapIncomingEdges;

	
	///////////////////////////////////////////////////////////////////
	// Implementation
	
	public DAGraph ()
	{
		m_mapOutgoingEdges = new HashMap<> ();
		m_mapIncomingEdges = new HashMap<> ();
	}

	@Override
	protected DAGraph.Edge createEdge (DAGraph.Vertex vertexHead, DAGraph.Vertex vertexTail)
	{
		DAGraph.Edge edge = new DAGraph.Edge (vertexHead, vertexTail);
		
		List<DAGraph.Edge> listOutgoing = m_mapOutgoingEdges.get (vertexTail);
		if (listOutgoing == null)
			m_mapOutgoingEdges.put (vertexTail, listOutgoing = new LinkedList<> ());
		listOutgoing.add (edge);
		
		List<DAGraph.Edge> listIncoming = m_mapIncomingEdges.get (vertexHead);
		if (listIncoming == null)
			m_mapIncomingEdges.put (vertexHead, listIncoming = new LinkedList<> ());
		listIncoming.add (edge);
		
		return edge;
	}
	
	public DAGraph.Edge getEdge (DAGraph.Vertex vertHead, DAGraph.Vertex vertexTail)
	{
		Iterable<DAGraph.Edge> itEdges = m_mapOutgoingEdges.get (vertHead);
		if (itEdges == null)
			return null;
		
		for (DAGraph.Edge edge : itEdges)
			if (edge.getTailVertex ().equals (vertexTail))
				return edge;
		
		return null;
	}
	
	/**
	 * Returns the collection of outgoing edges of vertex <code>v</code>.
	 * 
	 * @param v
	 *            The vertex whose outgoing edges are to be determined
	 * @return A collection of vertex <code>v</code>'s outgoing edges
	 */
	public Collection<DAGraph.Edge> getOutgoingEdges (DAGraph.Vertex v)
	{
		return m_mapOutgoingEdges.get (v);
	}

	/**
	 * Returns the collection of incoming edges of vertex <code>v</code>.
	 * 
	 * @param v
	 *            The vertex whose incoming edges are to be determined
	 * @return A collection of vertex <code>v</code>'s incoming edges
	 */
	public Collection<DAGraph.Edge> getIncomingEdges (DAGraph.Vertex v)
	{
		return m_mapIncomingEdges.get (v);
	}
	
	@Override
	public void removeEdge (Edge edge)
	{
		super.removeEdge (edge);
		
		List<DAGraph.Edge> listOutgoing = m_mapOutgoingEdges.get (edge.getTailVertex ());
		listOutgoing.remove (edge);
		
		List<DAGraph.Edge> listIncoming = m_mapIncomingEdges.get (edge.getHeadVertex ());
		listIncoming.remove (edge);
	}
}
