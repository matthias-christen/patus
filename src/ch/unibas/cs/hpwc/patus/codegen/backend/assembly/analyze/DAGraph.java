package ch.unibas.cs.hpwc.patus.codegen.backend.assembly.analyze;

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
		
		private int m_nLowerScheduleBound;
		private int m_nUpperScheduleBound;
		
		
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
			m_nLowerScheduleBound = nLowerBound;
			m_nUpperScheduleBound = nUpperBound;
		}
		
		public int getLowerScheduleBound ()
		{
			return m_nLowerScheduleBound;
		}
		
		public int getUpperScheduleBound ()
		{
			return m_nUpperScheduleBound;
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

	
	///////////////////////////////////////////////////////////////////
	// Implementation
	
	public DAGraph ()
	{
		m_mapOutgoingEdges = new HashMap<> ();
	}

	@Override
	protected DAGraph.Edge createEdge (DAGraph.Vertex vertexHead, DAGraph.Vertex vertexTail)
	{
		DAGraph.Edge edge = new DAGraph.Edge (vertexHead, vertexTail);
		
		List<DAGraph.Edge> listEdges = m_mapOutgoingEdges.get (vertexHead);
		if (listEdges == null)
			m_mapOutgoingEdges.put (vertexHead, listEdges = new LinkedList<> ());
		listEdges.add (edge);
		
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
}
