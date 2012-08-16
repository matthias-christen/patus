/*******************************************************************************
 * Copyright (c) 2011 Matthias-M. Christen, University of Basel, Switzerland.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the GNU Lesser Public License v2.1
 * which accompanies this distribution, and is available at
 * http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
 * 
 * Contributors:
 *     Matthias-M. Christen, University of Basel, Switzerland - initial API and implementation
 ******************************************************************************/
package ch.unibas.cs.hpwc.patus.codegen;

import java.util.Arrays;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

import cetus.hir.Expression;
import cetus.hir.IntegerLiteral;
import ch.unibas.cs.hpwc.patus.representation.Index;
import ch.unibas.cs.hpwc.patus.representation.Stencil;
import ch.unibas.cs.hpwc.patus.representation.StencilNode;

/**
 * A set of {@link StencilNode}s with set operations.
 * 
 * @author Matthias-M. Christen
 */
public class StencilNodeSet implements Iterable<StencilNode>
{
	///////////////////////////////////////////////////////////////////
	// Inner Types

	public enum ENodeTypes
	{
		INPUT_NODES,
		OUTPUT_NODES,
		ALL_NODES
	}

	private final static Comparator<StencilNode> COMPARATOR = new Comparator<StencilNode> ()
	{
		@Override
		public int compare (StencilNode n1, StencilNode n2)
		{
			if (n1.getIndex ().getVectorIndex () != n2.getIndex ().getVectorIndex ())
				return n1.getIndex ().getVectorIndex () - n2.getIndex ().getVectorIndex ();

			if (n1.getIndex ().getTimeIndex () != n2.getIndex ().getTimeIndex ())
				return n1.getIndex ().getTimeIndex () - n2.getIndex ().getTimeIndex ();

			if (n1.getIndex ().getSpaceIndexEx ().length != n2.getIndex ().getSpaceIndexEx ().length)
			{
				//throw new RuntimeException ("Only nodes of same spatial dimension can be compared");
				return n1.getIndex ().getSpaceIndexEx ().length - n2.getIndex ().getSpaceIndexEx ().length;
			}
			
			if (n1.getIndex ().getSpaceIndexEx ().length == 0)
				return 0;

			int nIdx = 0;
			while (n1.getIndex ().getSpaceIndex (nIdx).equals (n2.getIndex ().getSpaceIndex (nIdx)))
			{
				nIdx++;

				// if the end is reached and all entries have been equal, the arrays are equal
				if (nIdx == n1.getIndex ().getSpaceIndexEx ().length)
					return 0;
			}

			Expression exprN1Idx = n1.getIndex ().getSpaceIndex (nIdx);
			Expression exprN2Idx = n2.getIndex ().getSpaceIndex (nIdx);
			if ((exprN1Idx instanceof IntegerLiteral) && (exprN2Idx instanceof IntegerLiteral))
				return (int) ((IntegerLiteral) exprN1Idx).getValue () - (int) ((IntegerLiteral) exprN2Idx).getValue ();
			return exprN1Idx.compareTo (exprN2Idx);
		}
	};


	///////////////////////////////////////////////////////////////////
	// Member Variables

	/**
	 * The actual underlying data structure
	 */
	private Set<StencilNode> m_set;
	
	/**
	 * Some application-specific data
	 */
	private Object m_objData;


	///////////////////////////////////////////////////////////////////
	// Implementation

	/**
	 * Constructs and empty {@link StencilNodeSet}.
	 */
	public StencilNodeSet ()
	{
		m_set = new TreeSet<> (StencilNodeSet.COMPARATOR);
	}

	/**
	 * Creates a new {@link StencilNodeSet} from a collection of stencil nodes.
	 * @param itStencilNodes
	 */
	public StencilNodeSet (Collection<StencilNode> itStencilNodes)
	{
		this ();
		for (StencilNode node : itStencilNodes)
			m_set.add (node);
	}

	/**
	 * Builds a new {@link StencilNodeSet} from an array of stencil nodes.
	 * @param rgNodes
	 */
	public StencilNodeSet (StencilNode... rgNodes)
	{
		this ();
		for (StencilNode node : rgNodes)
			m_set.add (node);
	}

	/**
	 * Copy constructor.
	 * @param set
	 */
	public StencilNodeSet (StencilNodeSet set)
	{
		this ();
		for (StencilNode node : set)
			m_set.add (node);
	}

	/**
	 * Constructs a new {@link StencilNodeSet} from a {@link Stencil}.
	 * @param stencil
	 * @param types
	 */
	public StencilNodeSet (Stencil stencil, ENodeTypes types)
	{
		this ();

		if (types == ENodeTypes.INPUT_NODES || types == ENodeTypes.ALL_NODES)
			for (StencilNode node : stencil)
				m_set.add (node);
		if (types == ENodeTypes.OUTPUT_NODES || types == ENodeTypes.ALL_NODES)
			for (StencilNode node : stencil.getOutputNodes ())
				m_set.add (node);
	}
	
	public void add (StencilNode node)
	{
		m_set.add (node);
	}
	
	public void add (Iterable<StencilNode> nodes)
	{
		for (StencilNode node : nodes)
			m_set.add (node);
	}
	
	public void clear ()
	{
		m_set.clear ();
	}
	
	public boolean isEmpty ()
	{
		return m_set.isEmpty ();
	}
	
	public boolean contains (StencilNode node)
	{
		return m_set.contains (node);
	}
	
	public boolean contains (Index index)
	{
		// the "equals" method of StencilNode also works for Index,
		// and "hashCode" of StencilNode delegates to Index
		
		return m_set.contains (index);
	}

	/**
	 * Creates the union of <code>this</code> set and <code>set</code>.
	 * @param set
	 * @return
	 */
	public StencilNodeSet union (StencilNodeSet set)
	{
		if (set == this)
			return this;

		StencilNodeSet setResult = new StencilNodeSet (m_set);
		for (StencilNode node : set)
			setResult.m_set.add (node);
		return setResult;
	}

	/**
	 * Restricts the stencil nodes in the set to nodes that have the specified time and vector indices.
	 * @param nTimeIndex The time index or <code>null</code> if the time index is not to be restricted
	 * @param nVectorIndex The vector index or <code>null</code> if no restriction on the vector index is desired
	 * @return A new set containing the stencil nodes restricted to <code>nTimeIndex</code> and <code>nVectorIndex</code>
	 */
	public StencilNodeSet restrict (Integer nTimeIndex, Integer nVectorIndex)
	{
		// no changes necessary if no restriction
		if (nTimeIndex == null && nVectorIndex == null)
			return this;

		// restrict the set
		StencilNodeSet setResult = new StencilNodeSet ();
		for (StencilNode node : this)
		{
			boolean bAccept = true;
			if (nTimeIndex != null && node.getIndex ().getTimeIndex () != nTimeIndex)
				bAccept = false;
			if (nVectorIndex != null && node.getIndex ().getVectorIndex () != nVectorIndex)
				bAccept = false;

			if (bAccept)
				setResult.m_set.add (node);
		}

		return setResult;
	}

	/**
	 * Applies the mask <code>mask</code> to the stencil nodes in the set.
	 * @param mask The mask to apply
	 * @return
	 */
	public StencilNodeSet applyMask (IMask mask)
	{
		StencilNodeSet setResult = new StencilNodeSet ();

		for (StencilNode node : this)
		{
			int[] rgCoords = node.getSpaceIndex ();
			int[] rgEquivClass = mask.apply (rgCoords);

			if (Arrays.equals (rgEquivClass, rgCoords))
				setResult.m_set.add (node);
			else
			{
				StencilNode nodeNew = new StencilNode (node);
				nodeNew.getIndex ().setSpaceIndex (rgEquivClass);
				setResult.m_set.add (nodeNew);
			}
		}

		return setResult;
	}

	/**
	 * Adds a spatial offset, <code>rgOffset</code> to all the stencil nodes in the set.
	 * @param rgOffset The offset to add
	 * @return A new set with new stencil nodes with coordinates offset by <code>rgOffset</code>
	 */
	public StencilNodeSet addSpatialOffset (int[] rgOffset)
	{
		StencilNodeSet setResult = new StencilNodeSet ();
		for (StencilNode node : this)
		{
			StencilNode nodeNew = new StencilNode (node);
			nodeNew.getIndex ().offsetInSpace (rgOffset);
			setResult.m_set.add (nodeNew);
		}
		return setResult;
	}

	/**
	 *
	 * @param nOffset
	 * @return
	 */
	public StencilNodeSet addTemporalOffset (int nOffset)
	{
		if (nOffset == 0)
			return this;

		StencilNodeSet setResult = new StencilNodeSet ();
		for (StencilNode node : this)
		{
			StencilNode nodeNew = new StencilNode (node);
			nodeNew.getIndex ().offsetInTime (nOffset);
			setResult.m_set.add (nodeNew);
		}
		return setResult;
	}

	/**
	 * Adds items along the <code>nDimension</code> axis such that all the nodes between the
	 * min and max node in the <code>nDimension</code> direction per time and vector index
	 * are covered.
	 * @param nDimension
	 * @return
	 */
	public StencilNodeSet fill (int nDimension)
	{
		StencilNodeSet setResult = new StencilNodeSet (m_set);
		Set<StencilNode> setPivots = new HashSet<> ();
		setPivots.addAll (m_set);

		StencilNode nodePivot = null;
		while (!setPivots.isEmpty ())
		{
			// find the next pivot node
			for ( ; ; )
			{
				// no pivot found anymore => we are done
				if (setPivots.isEmpty ())
					return setResult;

				// get the pivot and remove it from the pivot set
				nodePivot = setPivots.iterator ().next ();
				setPivots.remove (nodePivot);

				// make sure the node has sufficient dimensions, otherwise get another pivot
				if (nDimension < nodePivot.getIndex ().getSpaceIndexEx ().length)
					break;
			}

			// find all the nodes that are "comparable" to the pivot node and get the min and max coords
			int[] rgSpaceIndexNodePivot = nodePivot.getSpaceIndex ();
			int nMin = rgSpaceIndexNodePivot[nDimension];
			int nMax = nMin;
			Set<Integer> setExistingCoords = new TreeSet<> ();
			setExistingCoords.add (nMin);
			for (StencilNode node : m_set)
			{
				if (node == nodePivot)
					continue;

				// check whether time and vector indices are comparable
				if (node.getIndex ().getTimeIndex () != nodePivot.getIndex ().getTimeIndex ())
					continue;
				if (node.getIndex ().getVectorIndex () != nodePivot.getIndex ().getVectorIndex ())
					continue;

				// check whether the spatial coordinates are equal (except the coordinate in nDimension direction)
				int[] rgSpaceIndexNode = node.getSpaceIndex ();
				if (nDimension >= rgSpaceIndexNode.length)
					continue;

				boolean bIsComparable = true;
				for (int i = 0; i < Math.min (rgSpaceIndexNode.length, rgSpaceIndexNodePivot.length); i++)
				{
					if (i == nDimension)
						continue;
					if (rgSpaceIndexNode[i] != rgSpaceIndexNodePivot[i])
					{
						bIsComparable = false;
						break;
					}
				}
				if (!bIsComparable)
					continue;

				// nodes are comparable, compute new min and max
				nMin = Math.min (nMin, rgSpaceIndexNode[nDimension]);
				nMax = Math.max (nMax, rgSpaceIndexNode[nDimension]);
				setExistingCoords.add (rgSpaceIndexNode[nDimension]);
				setPivots.remove (node);
			}

			// add the nodes between min and max
			if (nMin != nMax)
			{
				for (int i = nMin + 1; i <= nMax - 1; i++)
					if (!setExistingCoords.contains (i))
					{
						StencilNode nodeNew = new StencilNode (nodePivot);
//						nodeNew.getIndex ().getSpaceIndex ()[nDimension] = i;
						nodeNew.getIndex ().setSpaceIndex (nDimension, new IntegerLiteral (i));
						setResult.m_set.add (nodeNew);
					}
			}
		}

		return setResult;
	}

	/**
	 * Returns the &quot;front&quot; stencil nodes in the direction <code>nDimension</code>.
	 * @param nDimension
	 * @return
	 */
	public StencilNodeSet getFront (int nDimension)
	{
		StencilNodeSet setResult = new StencilNodeSet (m_set);
		Set<StencilNode> setPivots = new HashSet<> ();
		setPivots.addAll (m_set);

		StencilNode nodePivot = null;
		while (!setPivots.isEmpty ())
		{
			// find the next pivot node
			while (nodePivot == null)
			{
				// no pivot found anymore => we are done
				if (setPivots.isEmpty ())
					return setResult;

				// get the pivot and remove it from the pivot set
				nodePivot = setPivots.iterator ().next ();
				setPivots.remove (nodePivot);

				// make sure the node has sufficient dimensions, otherwise get another pivot
				if (nDimension >= nodePivot.getIndex ().getSpaceIndexEx ().length)
					nodePivot = null;
			}

			// check the nodes in the set of remaining nodes: compare the pivot node to each of the remaining ones
			List<StencilNode> listRemove = new LinkedList<> ();
			boolean bPivotChanged = false;
			for (StencilNode node : setResult.m_set)
			{
				if (node == nodePivot)
					continue;

				// check whether time and vector indices are comparable
				if (node.getIndex ().getTimeIndex () != nodePivot.getIndex ().getTimeIndex ())
					continue;
				if (node.getIndex ().getVectorIndex () != nodePivot.getIndex ().getVectorIndex ())
					continue;

				// check whether the spatial coordinates are equal (except the coordinate in nDimension direction)
				int[] rgSpaceIndexNodePivot = nodePivot.getSpaceIndex ();
				int[] rgSpaceIndexNode = node.getSpaceIndex ();
				if (nDimension >= rgSpaceIndexNode.length)
					continue;

				boolean bIsComparable = true;
				for (int i = 0; i < Math.min (rgSpaceIndexNode.length, rgSpaceIndexNodePivot.length); i++)
				{
					if (i == nDimension)
						continue;
					if (rgSpaceIndexNode[i] != rgSpaceIndexNodePivot[i])
					{
						bIsComparable = false;
						break;
					}
				}
				if (!bIsComparable)
					continue;

				// nodes are comparable, i.e. have same indices and coordinates apart from the coordinate in dimension nDimension
				// remove the "smaller" node from the set
				if (rgSpaceIndexNode[nDimension] < rgSpaceIndexNodePivot[nDimension])
					listRemove.add (node);
				else if (rgSpaceIndexNode[nDimension] > rgSpaceIndexNodePivot[nDimension])
				{
					listRemove.add (nodePivot);
					nodePivot = node;
					bPivotChanged = true;
					setPivots.remove (nodePivot);
				}
			}

			// remove nodes that have been marked for removal
			for (StencilNode node : listRemove)
			{
				setResult.m_set.remove (node);
				setPivots.remove (node);
			}

			// request a new pivot node if it hasn't changed during stepping through the compare loop
			if (!bPivotChanged)
				nodePivot = null;
		}

		return setResult;
	}

	@Override
	public Iterator<StencilNode> iterator ()
	{
		return m_set.iterator ();
	}

	/**
	 * Returns the number of stencil nodes contained in the set.
	 * @return The set size
	 */
	public int size ()
	{
		return m_set.size ();
	}

	/**
	 * Returns an array containing all the temporal indices of the stencil
	 * nodes contained in <code>this</code> set.
	 * @return The array of temporal indices
	 */
	public int[] getTimeIndices ()
	{
		Set<Integer> set = new TreeSet<> ();
		for (StencilNode node : this)
			set.add (node.getIndex ().getTimeIndex ());

		int[] rgIndices = new int[set.size ()];
		int i = 0;
		for (int n : set)
			rgIndices[i++] = n;
		return rgIndices;
	}

	/**
	 * Finds the minimum time index of the nodes in the set.
	 * @return
	 */
	public int getMinimumTimeIndex ()
	{
		int nMin = Integer.MAX_VALUE;
		for (StencilNode n : m_set)
		{
			int nTimeIdx = n.getIndex ().getTimeIndex ();
			if (nTimeIdx < nMin)
				nMin = nTimeIdx;
		}

		return nMin;
	}

	/**
	 * Finds the maximum time index of the nodes in the set.
	 * @return
	 */
	public int getMaximumTimeIndex ()
	{
		int nMax = Integer.MIN_VALUE;
		for (StencilNode n : m_set)
		{
			int nTimeIdx = n.getIndex ().getTimeIndex ();
			if (nTimeIdx > nMax)
				nMax = nTimeIdx;
		}

		return nMax;
	}

	/**
	 * Returns an array containing all the vector indices of the stencil nodes
	 * contained in <code>this</code> set.
	 * @return The array of vector indices
	 */
	public int[] getVectorIndices ()
	{
		Set<Integer> set = new TreeSet<> ();
		for (StencilNode node : this)
			set.add (node.getIndex ().getVectorIndex ());

		int[] rgIndices = new int[set.size ()];
		int i = 0;
		for (int n : set)
			rgIndices[i++] = n;
		return rgIndices;
	}

	/**
	 * Returns the linear index of a stencil node with spatial index <code>rgSpatialIndex</code>.
	 * If no such node exists in the set, -1 is returned.
	 * @param rgSpatialIndex
	 * @return
	 */
	public int getLinearSpatialIndex (int[] rgSpatialIndex)
	{
		int nIdx = 0;
		for (StencilNode node : m_set)
		{
			if (Arrays.equals (node.getSpaceIndex (), rgSpatialIndex))
				return nIdx;
			nIdx++;
		}

		return -1;
	}
	
	/**
	 * Sets the application-specific data.
	 * @param objData The data to associate with the stencil node set
	 */
	public void setData (Object objData)
	{
		m_objData = objData;
	}
	
	/**
	 * Returns the application-specific data previously set by calling the
	 * {@link StencilNodeSet#setData(Object)} method.
	 * @return The application-specific data
	 */
	public Object getData ()
	{
		return m_objData;
	}

	@Override
	public boolean equals (Object obj)
	{
		if (!(obj instanceof StencilNodeSet))
			return false;

		StencilNodeSet setOther = (StencilNodeSet) obj;
		return m_set.equals (setOther.m_set);
	}

	@Override
	public int hashCode ()
	{
		return m_set.hashCode ();
	}

	@Override
	public String toString ()
	{
		Set<StencilNode> set = new TreeSet<> ();
		for (StencilNode node : m_set)
			set.add (node);

		StringBuilder sb = new StringBuilder ("{\n");
		for (StencilNode node : set)
		{
			sb.append ("\t");
			sb.append (node.toString ());
			sb.append ("\n");
		}
		sb.append ("}");
		return sb.toString ();
	}
}
