package ch.unibas.cs.hpwc.patus.graph.algorithm;

import java.lang.reflect.Constructor;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import ch.unibas.cs.hpwc.patus.graph.IGraph;
import ch.unibas.cs.hpwc.patus.graph.IParametrizedEdge;
import ch.unibas.cs.hpwc.patus.graph.IVertex;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

public class CriticalPathLengthCalculator<V extends IVertex, E extends IParametrizedEdge<V, N>, N extends Number>
{
	private IGraph<V, E> m_graph;
	private Class<? extends Number> m_clsNum;
	private Map<V, Map<V, N>> m_mapDistances;
	
	private N MINUS_INFINITY;
	private N ZERO;
	
	
	@SuppressWarnings("unchecked")
	public CriticalPathLengthCalculator (IGraph<V, E> graph, Class<? extends Number> clsNum)
	{
		m_graph = graph;
		m_clsNum = clsNum;
		m_mapDistances = new HashMap<> ();
		
		try
		{
			Constructor<? extends Number> constructor = m_clsNum.getConstructor (String.class);
			MINUS_INFINITY = (N) constructor.newInstance (String.valueOf (Integer.MIN_VALUE));
			ZERO = (N) constructor.newInstance ("0");
		}
		catch (Exception e)
		{
		}
	}
	
	/**
	 * Backflow algorithm
	 * @param vertEnd The end point
	 * @return A map with the critical path distances of all vertices to <code>vertEnd</code>
	 */
	protected Map<V, N> computeCriticalPathDistances (V vertEnd)
	{
		Map<V, N> map = new HashMap<> ();
		Set<V> setParents = new HashSet<> ();
		Set<V> setNextParents = null;
		
		setParents.add (vertEnd);
		map.put (vertEnd, ZERO);
		
		boolean bVertexFound = false;
		do
		{
			bVertexFound = false;
			setNextParents = new HashSet<> ();
			
			for (V vertParent : setParents)
			{
				for (E edge : m_graph.getEdges ())
				{
					if (edge.getHeadVertex ().equals (vertParent))
					{
						setNextParents.add (edge.getTailVertex ());
						updateMap (map, edge);
						bVertexFound = true;
					}
				}
			}
			
			setParents = setNextParents;
		} while (bVertexFound);
		
		return map;
	}
	
	protected void updateMap (Map<V, N> map, E edge)
	{
		N nMaxCurrentDistance = map.get (edge.getTailVertex ());
		N nMaxDistanceToHead = map.get (edge.getHeadVertex ());
		map.put (edge.getTailVertex (), getMaxDistance (nMaxCurrentDistance, nMaxDistanceToHead, edge.getData ()));
	}
	
	@SuppressWarnings("unchecked")
	protected N getMaxDistance (N nMaxDistCurrent, N nDistToHeadVertex, N nEdgeLength)
	{
		if (m_clsNum.equals (Integer.class))
		{
			int nMaxDistCurrent0 = nMaxDistCurrent == null ? 0 : (Integer) nMaxDistCurrent;
			int nDistToHeadVertex0 = nDistToHeadVertex == null ? 0 : (Integer) nDistToHeadVertex;
			return (N) new Integer (Math.max (nMaxDistCurrent0, nDistToHeadVertex0 + (Integer) nEdgeLength));
		}

		if (m_clsNum.equals (Long.class))
		{
			long nMaxDistCurrent0 = nMaxDistCurrent == null ? 0 : (Long) nMaxDistCurrent;
			long nDistToHeadVertex0 = nDistToHeadVertex == null ? 0 : (Long) nDistToHeadVertex;
			return (N) new Long (Math.max (nMaxDistCurrent0, nDistToHeadVertex0 + (Long) nEdgeLength));
		}

		if (m_clsNum.equals (Float.class))
		{
			float fMaxDistCurrent0 = nMaxDistCurrent == null ? 0 : (Float) nMaxDistCurrent;
			float fDistToHeadVertex0 = nDistToHeadVertex == null ? 0 : (Float) nDistToHeadVertex;
			return (N) new Float (Math.max (fMaxDistCurrent0, fDistToHeadVertex0 + (Float) nEdgeLength));
		}

		if (m_clsNum.equals (Double.class))
		{
			double fMaxDistCurrent0 = nMaxDistCurrent == null ? 0 : (Double) nMaxDistCurrent;
			double fDistToHeadVertex0 = nDistToHeadVertex == null ? 0 : (Double) nDistToHeadVertex;
			return (N) new Double (Math.max (fMaxDistCurrent0, fDistToHeadVertex0 + (Double) nEdgeLength));
		}
		
		throw new RuntimeException (StringUtil.concat (m_clsNum.getName (), " not supported"));
	}
	
	public N getCriticalPathDistance (V vertStart, V vertEnd)
	{
		Map<V, N> map = m_mapDistances.get (vertEnd);
		if (map == null)
		{
			// the distances haven't been computed for this end vertex yet
			m_mapDistances.put (vertEnd, map = computeCriticalPathDistances (vertEnd));
		}
		
		N nDistance = map.get (vertStart);
		if (nDistance == null)
			return MINUS_INFINITY;
		return nDistance;
	}
}
