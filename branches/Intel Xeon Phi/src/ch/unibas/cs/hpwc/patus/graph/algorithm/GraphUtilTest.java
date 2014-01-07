package ch.unibas.cs.hpwc.patus.graph.algorithm;

import static org.junit.Assert.assertEquals;

import org.junit.Before;
import org.junit.Test;

import ch.unibas.cs.hpwc.patus.graph.DefaultGraph;
import ch.unibas.cs.hpwc.patus.graph.IEdge;
import ch.unibas.cs.hpwc.patus.graph.IVertex;
import ch.unibas.cs.hpwc.patus.graph.algorithm.GraphUtil.EDegree;

public class GraphUtilTest
{
	public static class Vertex implements IVertex
	{
		private String m_strLabel;
		
		public Vertex (String strLabel)
		{
			m_strLabel = strLabel;
		}
		
		public String getLabel ()
		{
			return m_strLabel;
		}
		
		@Override
		public String toString ()
		{
			StringBuilder sb = new StringBuilder ("Vertex[");
			sb.append (m_strLabel);
			sb.append (']');
			return sb.toString ();
		}

		@Override
		public int hashCode ()
		{
			return m_strLabel == null ? 0 : m_strLabel.hashCode ();
		}

		@Override
		public boolean equals (Object obj)
		{
			if (this == obj)
				return true;
			if (obj == null)
				return false;
			if (!(obj instanceof Vertex))
				return false;
			
			Vertex other = (Vertex) obj;
			return m_strLabel == null ? other.m_strLabel == null : m_strLabel.equals (other.m_strLabel);
		}
	}
	
	public static class Edge implements IEdge<Vertex>
	{
		private Vertex m_vHead;
		private Vertex m_vTail;
		
		public Edge (Vertex vTail, Vertex vHead)
		{
			m_vHead = vHead;
			m_vTail = vTail;
		}

		@Override
		public Vertex getHeadVertex ()
		{
			return m_vHead;
		}

		@Override
		public Vertex getTailVertex ()
		{
			return m_vTail;
		}

		@Override
		public String toString ()
		{
			StringBuilder sb = new StringBuilder ("Edge[ ");
			sb.append (m_vTail.getLabel ());
			sb.append (" -> ");
			sb.append (m_vHead.getLabel ());
			sb.append (" ]");
			return sb.toString ();
		}

		@Override
		public int hashCode ()
		{
			return 31 * (m_vHead.hashCode () + 31 * m_vTail.hashCode ());
		}

		@Override
		public boolean equals (Object obj)
		{
			if (this == obj)
				return true;
			if (obj == null)
				return false;
			if (!(obj instanceof Edge))
				return false;
			
			Edge other = (Edge) obj;
			return m_vHead.equals (other.getHeadVertex ()) && m_vTail.equals (other.getTailVertex ());
		}
	}
	
	
	private DefaultGraph<Vertex, Edge> m_graph;
	private Vertex m_vA = new Vertex ("A");
	private Vertex m_vB = new Vertex ("B");
	private Vertex m_vC = new Vertex ("C");
	private Vertex m_vD = new Vertex ("D");
	private Vertex m_vE = new Vertex ("E");
	private Vertex m_vF = new Vertex ("F");

	@Before
	public void setUp () throws Exception
	{
		m_graph = new DefaultGraph<> ();
				
		m_graph.addVertex (m_vA);
		m_graph.addVertex (m_vB);
		m_graph.addVertex (m_vC);
		m_graph.addVertex (m_vD);
		m_graph.addVertex (m_vE);
		m_graph.addVertex (m_vF);
		
		//  A ---> B
		//  |\    ^  \
		//  | v  /   |
		//  |  E     |
		//  |   ^    |
		//  v    \   |
		//  C ---> D |
		//  |        |
		//  v        /
		//  F <------
		
		m_graph.addEdge (new Edge (m_vA, m_vB));
		m_graph.addEdge (new Edge (m_vA, m_vC));
		m_graph.addEdge (new Edge (m_vA, m_vE));
		m_graph.addEdge (new Edge (m_vB, m_vF));
		m_graph.addEdge (new Edge (m_vC, m_vD));
		m_graph.addEdge (new Edge (m_vC, m_vF));
		m_graph.addEdge (new Edge (m_vD, m_vE));
		m_graph.addEdge (new Edge (m_vE, m_vB));
	}

	@Test
	public void testGetDegree ()
	{
		assertEquals (0, GraphUtil.getDegree (m_graph, m_vA, GraphUtil.EDegree.IN_DEGREE));
		assertEquals (3, GraphUtil.getDegree (m_graph, m_vA, GraphUtil.EDegree.OUT_DEGREE));
		assertEquals (3, GraphUtil.getDegree (m_graph, m_vA, GraphUtil.EDegree.INOUT_DEGREE));

		assertEquals (2, GraphUtil.getDegree (m_graph, m_vB, GraphUtil.EDegree.IN_DEGREE));
		assertEquals (1, GraphUtil.getDegree (m_graph, m_vB, GraphUtil.EDegree.OUT_DEGREE));
		assertEquals (3, GraphUtil.getDegree (m_graph, m_vB, GraphUtil.EDegree.INOUT_DEGREE));

		assertEquals (1, GraphUtil.getDegree (m_graph, m_vC, GraphUtil.EDegree.IN_DEGREE));
		assertEquals (2, GraphUtil.getDegree (m_graph, m_vC, GraphUtil.EDegree.OUT_DEGREE));
		assertEquals (3, GraphUtil.getDegree (m_graph, m_vC, GraphUtil.EDegree.INOUT_DEGREE));

		assertEquals (1, GraphUtil.getDegree (m_graph, m_vD, GraphUtil.EDegree.IN_DEGREE));
		assertEquals (1, GraphUtil.getDegree (m_graph, m_vD, GraphUtil.EDegree.OUT_DEGREE));
		assertEquals (2, GraphUtil.getDegree (m_graph, m_vD, GraphUtil.EDegree.INOUT_DEGREE));

		assertEquals (2, GraphUtil.getDegree (m_graph, m_vE, GraphUtil.EDegree.IN_DEGREE));
		assertEquals (1, GraphUtil.getDegree (m_graph, m_vE, GraphUtil.EDegree.OUT_DEGREE));
		assertEquals (3, GraphUtil.getDegree (m_graph, m_vE, GraphUtil.EDegree.INOUT_DEGREE));

		assertEquals (2, GraphUtil.getDegree (m_graph, m_vF, GraphUtil.EDegree.IN_DEGREE));
		assertEquals (0, GraphUtil.getDegree (m_graph, m_vF, GraphUtil.EDegree.OUT_DEGREE));
		assertEquals (2, GraphUtil.getDegree (m_graph, m_vF, GraphUtil.EDegree.INOUT_DEGREE));
	}

	@Test
	public void testGetVerticesSortedByDegree ()
	{
		int nLastDegree = -1;
		for (Vertex v : GraphUtil.getVerticesSortedByDegree (m_graph, EDegree.IN_DEGREE, true))
		{
			int nDegree = GraphUtil.getDegree (m_graph, v, EDegree.IN_DEGREE);
			if (nLastDegree >= 0)
				assertEquals (true, nLastDegree <= nDegree);
			nLastDegree = nDegree;
		}
	}

	@Test
	public void testGetTopologicalSort ()
	{
		// TODO: implement test
		
		// property of topological sort:
		// "a topological sort  of a directed graph is a linear ordering of its vertices such that, for every edge uv, u comes before v in the ordering"
		
		for (IVertex v : GraphUtil.getTopologicalSort (m_graph))
			System.out.println (v);
	}
}
