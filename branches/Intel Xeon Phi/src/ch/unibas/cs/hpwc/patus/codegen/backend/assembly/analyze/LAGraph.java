package ch.unibas.cs.hpwc.patus.codegen.backend.assembly.analyze;

import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.IOperand;
import ch.unibas.cs.hpwc.patus.graph.IParametrizedVertex;

/**
 * The graph resulting from the live analysis.
 * @author Matthias-M. Christen
 */
public class LAGraph extends Graph<LAGraph.Vertex, Graph.Edge<LAGraph.Vertex>>
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
	
	
	///////////////////////////////////////////////////////////////////
	// Implementation
	
	@SuppressWarnings({ "unchecked", "rawtypes" })
	@Override
	protected Graph.Edge createEdge (LAGraph.Vertex vertexTail, LAGraph.Vertex vertexHead)
	{
		return new Graph.Edge (this, vertexTail, vertexHead);
	}
}
