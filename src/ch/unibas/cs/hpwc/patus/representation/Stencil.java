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
package ch.unibas.cs.hpwc.patus.representation;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

import cetus.hir.ArrayAccess;
import cetus.hir.AssignmentExpression;
import cetus.hir.DepthFirstIterator;
import cetus.hir.Expression;
import cetus.hir.IntegerLiteral;
import cetus.hir.NameID;
import ch.unibas.cs.hpwc.patus.codegen.StencilNodeSet;
import ch.unibas.cs.hpwc.patus.geometry.Box;
import ch.unibas.cs.hpwc.patus.symbolic.ExpressionData;

/**
 * This class defines the stencil structure.
 * The structure consists of a set of input indices (dependencies) and a set of output
 * indices (vector components of the output).<br/>
 * The {@link Stencil} class provides utility methods to advance the stencil
 * structure in space and in time, methods to determine the &quot;boundary&quot;
 * indices and methods to determine whether indices lie within the dependencies
 * of the stencil structure.
 *
 * @author Matthias-M. Christen
 */
public class Stencil implements IStencilStructure, IStencilOperations
{
	///////////////////////////////////////////////////////////////////
	// Constants

	/**
	 * An empty list of nodes
	 */
	private final static List<StencilNode> EMPTY_NODE_LIST = new ArrayList<> ();


	///////////////////////////////////////////////////////////////////
	// Member Variables

	/**
	 * The actual stencil computation referencing {@link StencilNode}s.
	 */
	protected ExpressionData m_edStencilCalculation;

	/**
	 * The set of all input indices that are required to compute all vector
	 * components of the stencil
	 */
	protected StencilNodeSet m_setAllInputNodes;

	/**
	 * List of node sets; the list has one stencil node set per vector
	 * component of the output node
	 */
	protected List<StencilNodeSet> m_listInputNodeSets;

	/**
	 * The set of output indices of the stencil
	 */
	protected StencilNodeSet m_setOutputNodes;

	/**
	 * The number of dimensions in which the stencil is defined
	 */
	protected byte m_nDimensionality;

	/**
	 * The maximum time index that is encountered on the input nodes of the stencil
	 */
	protected int m_nMaxTimeIndex;


	/**
	 * The standard output space index
	 */
	private int[] m_rgOutputSpaceIndex;
	

	///////////////////////////////////////////////////////////////////
	// Implementation

	/**
	 * Constructs a new stencil object.
	 */
	public Stencil ()
	{
		m_edStencilCalculation = null;
		m_setAllInputNodes = new StencilNodeSet ();
		m_listInputNodeSets = new ArrayList<> ();
		m_setOutputNodes = new StencilNodeSet ();

		m_nDimensionality = 0;
		m_nMaxTimeIndex = 0;

		m_rgOutputSpaceIndex = new int[0];
	}

	/**
	 * Copy constructor.
	 * @param s The stencil to copy
	 */
	public Stencil (Stencil stencil)
	{
		this ();
		set (stencil);
	}

	@Override
	public void addInputNode (StencilNode node)
	{
		// ensure that the list is big enough
		for (int i = m_listInputNodeSets.size (); i <= node.getIndex ().getVectorIndex (); i++)
			m_listInputNodeSets.add (new StencilNodeSet ());

		if (node.getIndex ().getSpaceIndexEx ().length > m_nDimensionality)
			m_nDimensionality = (byte) node.getIndex ().getSpaceIndexEx ().length;

		// add the index to the sets
		// NOTE that the sets don't contain any duplicate nodes (even if added twice)
		m_listInputNodeSets.get (node.getIndex ().getVectorIndex ()).add (node);
		m_setAllInputNodes.add (node);
	}

	@Override
	public void addOutputNode (StencilNode node)
	{
		m_setOutputNodes.add (node);
	}

	@Override
	public void set (Stencil stencil)
	{
		// clear the stencil structure
		clear ();

		// add the input indices
		for (int i = 0; i < stencil.m_listInputNodeSets.size (); i++)
			for (StencilNode node : stencil.m_listInputNodeSets.get (i))
				addInputNode (new StencilNode (node));

		// add the output indices
		for (StencilNode nodeOut : stencil.getOutputNodes ())
			addOutputNode (new StencilNode (nodeOut));
	}

	@Override
	public void clear ()
	{
		m_setAllInputNodes.clear ();
		m_listInputNodeSets.clear ();
		m_setOutputNodes.clear ();
	}

	@Override
	public Set<Index> getIndexSet ()
	{
		Set<Index> set = new TreeSet<> ();
		for (StencilNode node : m_setAllInputNodes)
			set.add (node.getIndex ());
		return set;
	}

	@Override
	public Iterator<StencilNode> iterator ()
	{
		return m_setAllInputNodes.iterator ();
	}

	public int[] getOutputSpaceIndex ()
	{
		return m_rgOutputSpaceIndex;
	}

	@Override
	public Iterable<StencilNode> getOutputNodes ()
	{
		return m_setOutputNodes;
	}

	/**
	 * Determines whether the stencil is the empty stencil.
	 * 
	 * @return <code>true</code> iff the stencil is empty
	 */
	public boolean isEmpty ()
	{
		return !(m_setAllInputNodes.size () > 0 || (m_setAllInputNodes.size () == 0 && m_edStencilCalculation != null));
	}
	
	@Override
	public Iterable<Index> getOutputIndices ()
	{
		Set<Index> set = new TreeSet<> ();
		for (StencilNode node : m_setOutputNodes)
			set.add (node.getIndex ());
		return set;
	}

	@Override
	public Expression[] getSpatialOutputIndex ()
	{
		Expression[] rgPreviousSpaceIndex = null;
		for (StencilNode node : m_setOutputNodes)
		{
			// if the spatial indices differ, throw an exception
			if (rgPreviousSpaceIndex != null)
			{
				if (!Arrays.equals (node.getIndex ().getSpaceIndexEx (), rgPreviousSpaceIndex))
					throw new IllegalArgumentException ("The stencil can only have an output in a single output location.");
			}
			else
				rgPreviousSpaceIndex = new Expression[node.getIndex ().getSpaceIndexEx ().length];

			// copy the spatial index to the temporary array
			for (int i = 0; i < Math.min (node.getIndex ().getSpaceIndexEx ().length, rgPreviousSpaceIndex.length); i++)
				rgPreviousSpaceIndex[i] = node.getIndex ().getSpaceIndex (i).clone ();
		}

		return rgPreviousSpaceIndex;
	}

	@Override
	public StencilNodeSet getAllNodes ()
	{
		return m_setAllInputNodes.union (m_setOutputNodes);
	}

	@Override
	public Set<Index> getAllIndices ()
	{
		Set<Index> set = new TreeSet<> ();
		for (StencilNode node : m_setAllInputNodes)
			set.add (node.getIndex ());
		for (StencilNode node : m_setOutputNodes)
			set.add (node.getIndex ());
		return set;
	}

	@Override
	public Iterable<StencilNode> getNodeIteratorForOutputNode (StencilNode nodeOutput)
	{
		if (nodeOutput == null)
			return Stencil.EMPTY_NODE_LIST;
		return getNodeIteratorForVectorComponent (nodeOutput.getIndex ().getVectorIndex ());
	}

	@Override
	public Iterable<StencilNode> getNodeIteratorForOutputIndex (Index idxOutput)
	{
		if (idxOutput == null)
			return Stencil.EMPTY_NODE_LIST;
		return getNodeIteratorForVectorComponent (idxOutput.getVectorIndex ());
	}

	@Override
	public Iterable<StencilNode> getNodeIteratorForVectorComponent (int nVectorComponentIndex)
	{
		if (nVectorComponentIndex < 0 || nVectorComponentIndex >= m_listInputNodeSets.size ())
			return Stencil.EMPTY_NODE_LIST;
		return m_listInputNodeSets.get (nVectorComponentIndex);
	}

	@Override
	public int getNumberOfNodes ()
	{
		return m_setAllInputNodes.size ();
	}

	@Override
	public int getNumberOfNodes (StencilNode nodeOutput)
	{
		if (nodeOutput == null)
			return 0;
		return getNumberOfNodes (nodeOutput.getIndex ().getVectorIndex ());
	}

	@Override
	public int getNumberOfNodes (int nVectorComponentIndex)
	{
		if (nVectorComponentIndex < 0 || nVectorComponentIndex >= m_listInputNodeSets.size ())
			return 0;
		return m_listInputNodeSets.get (nVectorComponentIndex).size ();
	}

	@Override
	public int getNumberOfVectorComponents ()
	{
		return m_listInputNodeSets.size ();
	}

	@Override
	public byte getDimensionality ()
	{
		return m_nDimensionality;
	}

	public ExpressionData getExpressionData ()
	{
		return m_edStencilCalculation;
	}
	
	/**
	 * Returns the expression of a single stencil calculation,
	 * the right hand side of the stencil assignment statement.
	 * @return
	 */
	public Expression getExpression ()
	{
		if (m_edStencilCalculation == null)
			return null;
		
		if (m_edStencilCalculation.getExpression () instanceof AssignmentExpression)
			return ((AssignmentExpression) m_edStencilCalculation.getExpression ()).getRHS ();
		
		return m_edStencilCalculation.getExpression ();
	}

	/**
	 * Returns the number of FLOPs in this stencil operation.
	 * 
	 * @return The number of FLOPs in the stencil operation
	 */
	public int getFlopsCount ()
	{
		if (m_edStencilCalculation == null)
			return 0;
		return m_edStencilCalculation.getFlopsCount ();
	}

	/**
	 *
	 * @param exprStencilCalculation
	 */
	public void setExpression (ExpressionData exprStencilCalculation)
	{
		m_edStencilCalculation = exprStencilCalculation;
	}

	@Override
	public boolean contains (Index idx)
	{
		return m_setAllInputNodes.contains (idx);
	}

	@Override
	public void offsetInSpace (int[] rgSpaceOffset)
	{
		for (StencilNode node : m_setAllInputNodes)
			node.getIndex ().offsetInSpace (rgSpaceOffset);
		for (StencilNode node : m_setOutputNodes)
			node.getIndex ().offsetInSpace (rgSpaceOffset);
	}
	
	@Override
	public void offsetInSpace (Expression[] rgSpaceOffset)
	{
		for (StencilNode node : m_setAllInputNodes)
			node.getIndex ().offsetInSpace (rgSpaceOffset);
		for (StencilNode node : m_setOutputNodes)
			node.getIndex ().offsetInSpace (rgSpaceOffset);		
	}

	@Override
	public void advanceInSpace (int nDirection)
	{
		for (StencilNode node : m_setAllInputNodes)
			node.getIndex ().offsetInSpace (nDirection, 1);
		for (StencilNode node : m_setOutputNodes)
			node.getIndex ().offsetInSpace (nDirection, 1);
	}

	@Override
	public void offsetInTime (int nTimeOffset)
	{
		for (StencilNode node : m_setAllInputNodes)
			node.getIndex ().offsetInTime (nTimeOffset);
		for (StencilNode node : m_setOutputNodes)
			node.getIndex ().offsetInTime (nTimeOffset);
	}

	@Override
	public void advanceInTime ()
	{
		offsetInTime (1);
	}

	public Box getBoundingBox ()
	{
		return new Box (getMinSpaceIndex (), getMaxSpaceIndex ());
	}

	@Override
	public int[] getMinSpaceIndex ()
	{
		return IndexSetUtil.getMinSpaceIndex (getIndexSet ());
	}

	@Override
	public int[] getMinSpaceIndexByTimeIndex (int nTimeIndex)
	{
		return IndexSetUtil.getMinSpaceIndexByTimeIndex (getIndexSet (), nTimeIndex);
	}

	@Override
	public int[] getMinSpaceIndexByVectorIndex (int nVectorIndex)
	{
		return IndexSetUtil.getMinSpaceIndexByVectorIndex (getIndexSet (), nVectorIndex);
	}

	@Override
	public int[] getMinSpaceIndex (int nTimeIndex, int nVectorIndex)
	{
		return IndexSetUtil.getMinSpaceIndex (getAllIndices(), nTimeIndex, nVectorIndex);
	}

	@Override
	public int[] getMaxSpaceIndex ()
	{
		return IndexSetUtil.getMaxSpaceIndex (getIndexSet ());
	}

	@Override
	public int[] getMaxSpaceIndexByTimeIndex (int nTimeIndex)
	{
		return IndexSetUtil.getMaxSpaceIndexByTimeIndex (getIndexSet (), nTimeIndex);
	}

	@Override
	public int[] getMaxSpaceIndexByVectorIndex (int nVectorIndex)
	{
		return IndexSetUtil.getMaxSpaceIndexByVectorIndex (getIndexSet (), nVectorIndex);
	}

	@Override
	public int[] getMaxSpaceIndex (int nTimeIndex, int nVectorIndex)
	{
		return IndexSetUtil.getMaxSpaceIndex (getAllIndices(), nTimeIndex, nVectorIndex);
	}

	@Override
	public int getMinTimeIndex ()
	{
		return IndexSetUtil.getMinTimeIndex (getIndexSet ());
	}

	@Override
	public int getMaxTimeIndex ()
	{
		return IndexSetUtil.getMaxTimeIndex (getIndexSet ());
	}

	@Override
	public boolean isTimeblockingApplicable ()
	{
		Set<Integer> setOutputIndices = new HashSet<> ();
		for (StencilNode node : m_setOutputNodes)
			setOutputIndices.add (node.getIndex ().getVectorIndex ());

		for (StencilNode node : m_setAllInputNodes)
			if (node.getIndex ().isAdvanceableInTime ())
				if (!setOutputIndices.contains (node.getIndex ().getVectorIndex ()))
					return false;

		// TODO: check expression!

		return true;
	}

	@Override
	public String toString ()
	{
		StringBuilder sb = new StringBuilder ("Stencil expression:\n");
		if (m_edStencilCalculation != null)
			sb.append (m_edStencilCalculation.toString ());

		sb.append ("\n\nStencil indices:\n");
		if (m_setAllInputNodes.size () == 0)
			sb.append ("(empty stencil)");
		else
		{
			for (StencilNode node : m_setAllInputNodes)
			{
				sb.append (node.toString ());
				sb.append ('\n');
			}

			sb.append ("\n---->\n");
			for (StencilNode node : m_setOutputNodes)
			{
				sb.append (node.toString ());
				sb.append ('\n');
			}
		}

		return sb.toString ();
	}

	public String getStencilExpression ()
	{
		Expression expr = m_edStencilCalculation.getExpression ().clone ();
		for (DepthFirstIterator it = new DepthFirstIterator (expr); it.hasNext (); )
		{
			Object obj = it.next ();
			if (obj instanceof StencilNode)
			{
				StencilNode node = (StencilNode) obj;
				List<Expression> listIndices = new ArrayList<> ();
				for (Expression e : node.getIndex ().getSpaceIndexEx ())
					listIndices.add (e.clone ());
				listIndices.add (new IntegerLiteral (node.getIndex ().getTimeIndex ()));
				listIndices.add (new IntegerLiteral (node.getIndex ().getVectorIndex ()));
				node.swapWith (new ArrayAccess (new NameID (node.getName ()), listIndices));
			}
		}

		return expr.toString ();
	}
}
