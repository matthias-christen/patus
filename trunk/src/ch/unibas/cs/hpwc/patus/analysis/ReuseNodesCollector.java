package ch.unibas.cs.hpwc.patus.analysis;

import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import ch.unibas.cs.hpwc.patus.codegen.StencilNodeSet;
import ch.unibas.cs.hpwc.patus.representation.Index;
import ch.unibas.cs.hpwc.patus.representation.StencilNode;

/**
 * Finds sets of nodes which can be cyclically reused along a specific dimension. 
 * 
 * @author Matthias-M. Christen
 */
public class ReuseNodesCollector
{
	///////////////////////////////////////////////////////////////////
	// Inner Types
	
	private static class StencilNodeSetInfo
	{
		//private String m_strName;
		private int m_nMinCoord;
		private int m_nMaxCoord;
		
		public StencilNodeSetInfo (String strName)
		{
			//m_strName = strName;
			m_nMinCoord = Integer.MAX_VALUE;
			m_nMaxCoord = Integer.MIN_VALUE;
		}
		
		public void addCoord (int nCoordValue)
		{
			m_nMinCoord = Math.min (m_nMinCoord, nCoordValue);
			m_nMaxCoord = Math.max (m_nMaxCoord, nCoordValue);
		}
		
//		public String getName ()
//		{
//			return m_strName;
//		}

		public int getMinCoord ()
		{
			return m_nMinCoord;
		}

		public int getMaxCoord ()
		{
			return m_nMaxCoord;
		}
	}

	
	///////////////////////////////////////////////////////////////////
	// Member Variables
	
	private StencilNodeSet m_setNodes;
	
	private int m_nReuseDirection;
	
	private List<StencilNodeSet> m_listNodeClasses;
	

	///////////////////////////////////////////////////////////////////
	// Implementation
	
	public ReuseNodesCollector (StencilNodeSet setNodes, int nReuseDirection)
	{
		m_setNodes = setNodes;
		m_nReuseDirection = nReuseDirection;
				
		// categorize all stencil nodes into sets
		m_listNodeClasses = collectSets ();
		sortSets ();
	}
	
	/**
	 * Returns all the classes of stencil node sets, each set containing
	 * the stencil nodes differing in all coordinates, time indices, and vector indices,
	 * but not in the spatial index corresponding to the reuse direction (which was
	 * provided to the constructor of this class).
	 * @return An iterable over all the stencil node classes
	 */
	public Iterable<StencilNodeSet> getAllSets ()
	{
		return m_listNodeClasses;
	}

	/**
	 * 
	 * @param nMaxNodes
	 * @return
	 */
	public Iterable<StencilNodeSet> getSetsWithMaxNodesConstraint (int nMaxNodes)
	{
		return getSetsWithMaxNodesConstraint (nMaxNodes, 1);
	}

	/**
	 * 
	 * @param nMaxNodes
	 * @param nUnrollingFactor
	 * @return
	 */
	public Iterable<StencilNodeSet> getSetsWithMaxNodesConstraint (int nMaxNodes, int nUnrollingFactor)
	{
		List<StencilNodeSet> listResult = new LinkedList<StencilNodeSet> ();
		
		// greedily finds the sets whose ranges sum up to at max nMaxNodes
		// larger sets are preferred
		// (this is a variant of the subset sum problem)
		
		int nSum = 0;
		for (StencilNodeSet set : m_listNodeClasses)
		{
			int nRange = getRange (set) + nUnrollingFactor - 1;
			if (nSum + nRange <= nMaxNodes)
			{
				listResult.add (set);
				nSum += nRange;
			}
		}
		
		return listResult;
	}
		
	/**
	 * Categorizes all stencil nodes into sets.
	 * Stencil nodes in one sets can differ in the selected (the reuse) direction, but not in others.
	 * @return
	 */
	private List<StencilNodeSet> collectSets ()
	{
		Map<String, Map<Index, StencilNodeSet>> mapSets = new HashMap<String, Map<Index, StencilNodeSet>> ();
		List<StencilNodeSet> listSets = new LinkedList<StencilNodeSet> ();

		for (StencilNode node : m_setNodes)
		{
			// use the coordinates all dimensions except the reuse direction as a key
			Index idx = new Index (node.getIndex ());
			idx.getSpaceIndex ()[m_nReuseDirection] = 0;
			
			Map<Index, StencilNodeSet> map = mapSets.get (node.getName ());
			if (map == null)
				mapSets.put (node.getName (), map = new HashMap<Index, StencilNodeSet> ());
			
			StencilNodeSet set = map.get (idx);
			if (set == null)
			{
				map.put (idx, set = new StencilNodeSet ());
				listSets.add (set);

				set.setData (new StencilNodeSetInfo (node.getName ()));
			}
			
			set.add (node);
			((StencilNodeSetInfo) set.getData ()).addCoord (node.getSpaceIndex ()[m_nReuseDirection]);
		}
				
		return listSets;
	}
	
	/**
	 * Sorts the list of stencil node sets (<code>m_listNodeClases</code>) descendingly by range.
	 */
	private void sortSets ()
	{
		// sort the sets by descending range (= maxcoord - mincoord + 1)
		Collections.sort (m_listNodeClasses, new Comparator<StencilNodeSet> ()
		{
			@Override
			public int compare (StencilNodeSet set1, StencilNodeSet set2)
			{
				return getRange (set2) - getRange (set1);
			}
		});
	}
	
	/**
	 * Returns the range of stencil node set <code>set</code>, i.e., <code>maxcoord - mincoord + 1</code>.
	 * @param set The set of which to retrieve its range
	 * @return The range of set <code>set</code>
	 */
	private int getRange (StencilNodeSet set)
	{
		StencilNodeSetInfo info = (StencilNodeSetInfo) set.getData ();
		return info.getMaxCoord () - info.getMinCoord () + 1;
	}
}
