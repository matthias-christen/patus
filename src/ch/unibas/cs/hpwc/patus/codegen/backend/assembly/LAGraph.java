package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;

/**
 * The graph resulting from the live analysis.
 * @author Matthias-M. Christen
 */
public class LAGraph
{
	private static class Edge
	{
		private IOperand m_opVertex1;
		private IOperand m_opVertex2;
		
		public Edge (IOperand opVertex1, IOperand opVertex2)
		{
			m_opVertex1 = opVertex1;
			m_opVertex2 = opVertex2;
		}
	}
	
	
	private Set<IOperand> m_setNodes;
	private List<Edge> m_listEdges;
	
	public LAGraph ()
	{
		m_setNodes = new HashSet<IOperand> ();
		m_listEdges = new LinkedList<LAGraph.Edge> ();
	}
	
	public void addNode (IOperand op)
	{
		m_setNodes.add (op);
	}
	
	public void addEdge (IOperand op1, IOperand op2)
	{
		m_listEdges.add (new Edge (op1, op2));
	}
}
