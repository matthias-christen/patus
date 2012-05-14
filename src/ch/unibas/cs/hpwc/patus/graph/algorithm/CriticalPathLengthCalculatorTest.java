package ch.unibas.cs.hpwc.patus.graph.algorithm;

import org.junit.Before;
import org.junit.Test;

import ch.unibas.cs.hpwc.patus.graph.DefaultGraph;
import ch.unibas.cs.hpwc.patus.graph.IParametrizedEdge;
import ch.unibas.cs.hpwc.patus.graph.algorithm.GraphUtilTest.Vertex;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

public class CriticalPathLengthCalculatorTest
{
	public static class Edge extends GraphUtilTest.Edge implements IParametrizedEdge<GraphUtilTest.Vertex, Integer>
	{
		private Integer m_nData;
		
		public Edge (Vertex vertTail, Vertex vertHead, int nWeight)
		{
			super (vertTail, vertHead);
			setData (nWeight);
		}

		@Override
		public void setData (Integer nData)
		{
			m_nData = nData;
		}

		@Override
		public Integer getData ()
		{
			return m_nData;
		}
		
		@Override
		public String toString ()
		{
			StringBuilder sb = new StringBuilder (super.toString ());
			sb.append (" (");
			sb.append (m_nData);
			sb.append (')');
			return sb.toString ();
		}
	}
	
	private DefaultGraph<Vertex, Edge> m_graph;
	private Vertex m_vA = new Vertex ("A");
	private Vertex m_vB = new Vertex ("B");
	private Vertex m_vC = new Vertex ("C");
	private Vertex m_vD = new Vertex ("D");
	private Vertex m_vE = new Vertex ("E");
	private Vertex m_vF = new Vertex ("F");
	private Vertex m_vG = new Vertex ("G");
	private Vertex m_vH = new Vertex ("H");
	private Vertex m_vI = new Vertex ("I");
	private Vertex m_vJ = new Vertex ("J");
	
	private CriticalPathLengthCalculator<Vertex, Edge, Integer> m_calc;
	

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
		m_graph.addVertex (m_vG);
		m_graph.addVertex (m_vH);
		m_graph.addVertex (m_vI);
		m_graph.addVertex (m_vJ);
				
		m_graph.addEdge (new Edge (m_vA, m_vC, 1));
		m_graph.addEdge (new Edge (m_vA, m_vD, 1));
		m_graph.addEdge (new Edge (m_vB, m_vD, 1));
		m_graph.addEdge (new Edge (m_vB, m_vE, 3));
		m_graph.addEdge (new Edge (m_vB, m_vF, 2));
		m_graph.addEdge (new Edge (m_vC, m_vG, 2));
		m_graph.addEdge (new Edge (m_vD, m_vH, 2));
		m_graph.addEdge (new Edge (m_vD, m_vI, 4));
		m_graph.addEdge (new Edge (m_vE, m_vI, 2));
		m_graph.addEdge (new Edge (m_vF, m_vI, 5));
		m_graph.addEdge (new Edge (m_vG, m_vJ, 4));
		m_graph.addEdge (new Edge (m_vH, m_vG, 1));
		m_graph.addEdge (new Edge (m_vH, m_vJ, 2));
		m_graph.addEdge (new Edge (m_vI, m_vJ, 3));
		
		m_calc = new CriticalPathLengthCalculator<> (m_graph, Integer.class);
	}

	@Test
	public void testGetCriticalPathDistance ()
	{
		for (Vertex v : m_graph.getVertices ())
			System.out.println (StringUtil.concat (v.toString (), "  { ", m_calc.getCriticalPathDistance (v, m_vJ), " }"));
	}

}
