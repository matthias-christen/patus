package ch.unibas.cs.hpwc.patus.analysis;

import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.Expression;
import cetus.hir.IntegerLiteral;
import ch.unibas.cs.hpwc.patus.codegen.StencilNodeSet;
import ch.unibas.cs.hpwc.patus.representation.Index;
import ch.unibas.cs.hpwc.patus.representation.StencilNode;
import ch.unibas.cs.hpwc.patus.util.ExpressionUtil;

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
	
	private int m_nSIMDVectorLengthInReuseDirection;
	
	private List<StencilNodeSet> m_listNodeClasses;
	

	///////////////////////////////////////////////////////////////////
	// Implementation
	
	/**
	 * Constructs the reuse collector for the node set <code>setNodes</code>.
	 * Reuse is done in only one direction, <code>nReuseDirection</code>.
	 * 
	 * @param setNodes
	 *            The set of nodes from which to gather the ones that can be
	 *            reused in the direction <code>nReuseDirection</code>
	 * @param nReuseDirection
	 *            The direction of the reuse
	 * @param nSIMDVectorLengthInReuseDirection
	 *            The length of SIMD vectors in the reuse direction
	 */
	public ReuseNodesCollector (StencilNodeSet setNodes, int nReuseDirection, int nSIMDVectorLengthInReuseDirection)
	{
		m_setNodes = setNodes;
		m_nReuseDirection = nReuseDirection;
		m_nSIMDVectorLengthInReuseDirection = nSIMDVectorLengthInReuseDirection;
				
		// categorize all stencil nodes into sets
		m_listNodeClasses = collectSets ();
		sortSets ();
	}
	
	/**
	 * Returns all the classes of stencil node sets, each set containing
	 * the stencil nodes differing in all coordinates, time indices, and vector
	 * indices, but not in the spatial index corresponding to the reuse direction (which
	 * was provided to the constructor of this class).
	 * 
	 * @return An iterable over all the stencil node classes
	 */
	public Iterable<StencilNodeSet> getAllSets ()
	{
		return m_listNodeClasses;
	}

	/**
	 * 
	 * @param nMaxNodes
	 * @param nUnrollingFactor
	 * @return
	 */
	public Iterable<StencilNodeSet> getSetsWithMaxNodesConstraint (int nMaxNodes)
	{
		List<StencilNodeSet> listResult = new LinkedList<> ();
		
		// greedily finds the sets whose ranges sum up to at max nMaxNodes
		// larger sets are preferred
		// (this is a variant of the subset sum problem)
		
		int nSum = 0;
		for (StencilNodeSet set : m_listNodeClasses)
		{
			// only allow sets with more than one nodes
			if (set.size () < 2)
				continue;
			
			int nRange = getRange (set);
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
	 * Stencil nodes in one sets can differ in the selected (the reuse)
	 * direction, but not in others.
	 * 
	 * @return A list of stencil node sets categorized according to their
	 *         coordinates in non-reuse directions and (module SIMD vector
	 *         length) in reuse direction
	 */
	private List<StencilNodeSet> collectSets ()
	{
		Map<String, Map<Index, StencilNodeSet>> mapSets = new HashMap<> ();
		List<StencilNodeSet> listSets = new LinkedList<> ();

		for (StencilNode node : m_setNodes)
		{
			// use the coordinates all dimensions except the reuse direction as a key
			// in the reuse direction, the coordinates are reduced to sets mod the SIMD vector length,
			// so for scalars (m_nSIMDVectorLengthInReuseDirection == 1) each node in a line of points
			// in the reuse direction can be reused, but for vectors (m_nSIMDVectorLengthInReuseDirection > 1)
			// only every m_nSIMDVectorLengthInReuseDirection-th node falls into the same vector
			
			Index idx = new Index (node.getIndex ());
			
//			idx.getSpaceIndex ()[m_nReuseDirection] %= m_nSIMDVectorLengthInReuseDirection;
//			if (idx.getSpaceIndex ()[m_nReuseDirection] < 0)
//				idx.getSpaceIndex ()[m_nReuseDirection] += m_nSIMDVectorLengthInReuseDirection;
			
			Expression exprMod = ExpressionUtil.mod (idx.getSpaceIndex (m_nReuseDirection), new IntegerLiteral (m_nSIMDVectorLengthInReuseDirection));
			// make the modulus positive
			if (exprMod instanceof IntegerLiteral)
			{
				if (((IntegerLiteral) exprMod).getValue () < 0)
					exprMod = new IntegerLiteral (((IntegerLiteral) exprMod).getValue () + m_nSIMDVectorLengthInReuseDirection);
			}
			else
			{
				exprMod = new BinaryExpression (
					new BinaryExpression (exprMod, BinaryOperator.ADD, new IntegerLiteral (m_nSIMDVectorLengthInReuseDirection)),
					BinaryOperator.MODULUS,
					new IntegerLiteral (m_nSIMDVectorLengthInReuseDirection)
				);
			}
			idx.setSpaceIndex (m_nReuseDirection, exprMod);
			
			
			Map<Index, StencilNodeSet> map = mapSets.get (node.getName ());
			if (map == null)
				mapSets.put (node.getName (), map = new HashMap<> ());
			
			StencilNodeSet set = map.get (idx);
			if (set == null)
			{
				map.put (idx, set = new StencilNodeSet ());
				listSets.add (set);

				set.setData (new StencilNodeSetInfo (node.getName ()));
			}
			
			set.add (node);
			((StencilNodeSetInfo) set.getData ()).addCoord (ExpressionUtil.getIntegerValue (node.getIndex ().getSpaceIndex (m_nReuseDirection)));
		}
				
		return listSets;
	}
	
	/**
	 * Sorts the list of stencil node sets (<code>m_listNodeClases</code>)
	 * descendingly by range.
	 */
	private void sortSets ()
	{
		// sort the sets by descending range (= maxcoord - mincoord + 1)
		Collections.sort (m_listNodeClasses, new Comparator<StencilNodeSet> ()
		{
			@Override
			public int compare (StencilNodeSet set1, StencilNodeSet set2)
			{
				return ReuseNodesCollector.getRange (set2) - ReuseNodesCollector.getRange (set1);
			}
		});
	}
	
	/**
	 * Returns the range of stencil node set <code>set</code>, i.e.,
	 * <code>maxcoord - mincoord + 1</code>.
	 * 
	 * @param set
	 *            The set of which to retrieve its range
	 * @return The range of set <code>set</code>
	 */
	private static int getRange (StencilNodeSet set)
	{
		StencilNodeSetInfo info = (StencilNodeSetInfo) set.getData ();
		return info.getMaxCoord () - info.getMinCoord () + 1;
	}

	/**
	 * Adds new stencil nodes to the sets with more than one stencil nodes to
	 * account for
	 * unrolling in the dimension <code>nReuseDimension</code>.
	 * 
	 * @param nReuseDimension
	 *            The reuse dimension in which the unrolling will be done
	 * @param nUnrollFactor
	 *            The unroll factor
	 */
	public void addUnrollNodes (int nReuseDimension, int nUnrollFactor)
	{
		for (StencilNodeSet set : m_listNodeClasses)
		{
			// skip sets with only one stencil node
			if (set.size () < 2)
				continue;
			
			// get a node prototype of the set based on which the new nodes used for unrolling will be added
			StencilNode nodePrototype = set.iterator ().next ();
			
			// add new stencil nodes
			StencilNodeSetInfo info = (StencilNodeSetInfo) set.getData ();
			for (int i = 1; i < nUnrollFactor; i++)
			{
				StencilNode nodeNew = new StencilNode (nodePrototype);
//				nodeNew.getSpaceIndex ()[nReuseDimension] = info.getMaxCoord () + i;
				nodeNew.getIndex ().setSpaceIndex (nReuseDimension, info.getMaxCoord () + i);
				
				set.add (nodeNew);
			}
			
			// update the coordinate info
			info.addCoord (info.getMaxCoord () + nUnrollFactor - 1);
		}
	}
}
