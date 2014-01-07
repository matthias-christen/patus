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
import java.util.Iterator;
import java.util.List;

import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.Expression;
import cetus.hir.IntegerLiteral;
import ch.unibas.cs.hpwc.patus.symbolic.Symbolic;
import ch.unibas.cs.hpwc.patus.util.ExpressionUtil;


/**
 *
 * @author Matthias-M. Christen
 */
public class Index implements Comparable<Index>, ISpaceIndexable
{
	///////////////////////////////////////////////////////////////////
	// Member Variables

	/**
	 * The temporal index
	 */
	private int m_nTimeIndex;

	/**
	 * The spatial index
	 */
	private Expression[] m_rgSpaceIndex;

	/**
	 * The vector index
	 */
	private int m_nVectorIndex;

	/**
	 * Flag determining whether the index can be advanced in time.
	 * Typically, advanceable indices are part of the description of the
	 * solution field (if talking of stencils arising from PDE discretizations),
	 * non-advanceable indices are (PDE) coefficients.
	 */
	private boolean m_bIsAdvanceableInTime;


	///////////////////////////////////////////////////////////////////
	// Implementation

	/**
	 * Constructs a new index with default values.
	 */
	public Index ()
	{
		this (0, new int[] { }, 0, true);
	}

	/**
	 * Constructs a new index.
	 * 
	 * @param nTimeIndex
	 *            The index in time
	 * @param nSpaceIndex
	 *            The index in a 1D space
	 */
	public Index (int nTimeIndex, int nSpaceIndex)
	{
		this (nTimeIndex, new int[] { nSpaceIndex });
	}

	/**
	 * Constructs a new index.
	 * 
	 * @param nTimeIndex
	 *            The index in time
	 * @param nSpaceIndex
	 *            The index in a 1D space
	 * @param bIsAdvanceableInTime
	 *            Determines whether the index is advanceable in time.
	 *            i.e. whether the index belongs to the solution or is a
	 *            coefficient
	 */
	public Index (int nTimeIndex, int nSpaceIndex, boolean bIsAdvanceableInTime)
	{
		this (nTimeIndex, new int[] { nSpaceIndex }, bIsAdvanceableInTime);
	}

	/**
	 * Constructs a new index.
	 * 
	 * @param nTimeIndex
	 *            The index in time
	 * @param rgSpaceIndex
	 *            The index in a (rgSpaceIndex.length)-dimensional space
	 */
	public Index (int nTimeIndex, int[] rgSpaceIndex)
	{
		this (nTimeIndex, rgSpaceIndex, 0);
	}

	/**
	 * Constructs a new index.
	 * 
	 * @param nTimeIndex
	 *            The index in time
	 * @param rgSpaceIndex
	 *            The index in a (rgSpaceIndex.length)-dimensional space
	 * @param bIsAdvanceableInTime
	 *            Determines whether the index is advanceable in time.
	 *            i.e. whether the index belongs to the solution or is a
	 *            coefficient
	 */
	public Index (int nTimeIndex, int[] rgSpaceIndex, boolean bIsAdvanceableInTime)
	{
		this (nTimeIndex, rgSpaceIndex, 0, bIsAdvanceableInTime);
	}

	/**
	 * Constructs a new index.
	 * 
	 * @param nTimeIndex
	 *            The index in time
	 * @param rgSpaceIndex
	 *            The 1D index in space
	 * @param nVectorIndex
	 *            The vector index
	 */
	public Index (int nTimeIndex, int nSpaceIndex, int nVectorIndex)
	{
		this (nTimeIndex, new int[] { nSpaceIndex }, nVectorIndex);
	}

	/**
	 * Constructs a new index.
	 * 
	 * @param nTimeIndex
	 *            The index in time
	 * @param rgSpaceIndex
	 *            The 1D index in space
	 * @param nVectorIndex
	 *            The vector index
	 * @param bIsAdvanceableInTime
	 *            Determines whether the index is advanceable in time.
	 *            i.e. whether the index belongs to the solution or is a
	 *            coefficient
	 */
	public Index (int nTimeIndex, int nSpaceIndex, int nVectorIndex, boolean bIsAdvanceableInTime)
	{
		this (nTimeIndex, new int[] { nSpaceIndex }, nVectorIndex, bIsAdvanceableInTime);
	}

	/**
	 * Constructs a new index.
	 * 
	 * @param nTimeIndex
	 *            The index in time
	 * @param rgSpaceIndex
	 *            The index in space
	 * @param nVectorIndex
	 *            The vector index
	 */
	public Index (int nTimeIndex, int[] rgSpaceIndex, int nVectorIndex)
	{
		this (nTimeIndex, rgSpaceIndex, nVectorIndex, true);
	}

	/**
	 * Constructs a new index.
	 * 
	 * @param nTimeIndex
	 *            The index in time
	 * @param rgSpaceIndex
	 *            The index in space
	 * @param nVectorIndex
	 *            The vector index
	 * @param bIsAdvanceableInTime
	 *            Determines whether the index is advanceable in time.
	 *            i.e. whether the index belongs to the solution or is a
	 *            coefficient
	 */
	public Index (int nTimeIndex, int[] rgSpaceIndex, int nVectorIndex, boolean bIsAdvanceableInTime)
	{
		// copy the time index
		m_nTimeIndex = nTimeIndex;

		// copy the spatial index
		if (rgSpaceIndex == null)
			m_rgSpaceIndex = null;
		else
		{
			m_rgSpaceIndex = new Expression[rgSpaceIndex.length];
			for (int i = 0; i < rgSpaceIndex.length; i++)
				m_rgSpaceIndex[i] = new IntegerLiteral (rgSpaceIndex[i]);
		}

		// copy the vector index
		m_nVectorIndex = nVectorIndex;

		m_bIsAdvanceableInTime = bIsAdvanceableInTime;
	}
	
	public Index (int nTimeIndex, Expression[] rgSpaceIndex, int nVectorIndex, boolean bIsAdvanceableInTime)
	{
		// copy the time index
		m_nTimeIndex = nTimeIndex;

		// copy the spatial index
		if (rgSpaceIndex == null)
			m_rgSpaceIndex = null;
		else
		{
			m_rgSpaceIndex = new Expression[rgSpaceIndex.length];
			for (int i = 0; i < rgSpaceIndex.length; i++)
				m_rgSpaceIndex[i] = rgSpaceIndex[i].clone ();
		}

		// copy the vector index
		m_nVectorIndex = nVectorIndex;

		m_bIsAdvanceableInTime = bIsAdvanceableInTime;
	}

	/**
	 * Copy constructor.
	 * 
	 * @param index
	 *            The index to copy
	 */
	public Index (Index index)
	{
		this (index.getTimeIndex (), index.getSpaceIndexEx (), index.getVectorIndex (), index.isAdvanceableInTime ());
	}

	/**
	 * Returns the temporal component of the index.
	 * 
	 * @return The index describing the location of the plane in time
	 */
	public int getTimeIndex ()
	{
		return m_nTimeIndex;
	}

	/**
	 * Sets the index's time index.
	 * 
	 * @param nTimeIndex
	 *            The new time index
	 */
	public void setTimeIndex (int nTimeIndex)
	{
		m_nTimeIndex = nTimeIndex;
	}

	/**
	 * Sets the index's temporal index. The list must have exactly one entry,
	 * which must be either a number or an expression evaluating to a number.
	 * Otherwise an exception will be thrown.
	 * 
	 * @param listTimeIndex
	 *            The list of temporal indices
	 */
	public void setTimeIndex (List<?> listTimeIndex)
	{
		if (listTimeIndex == null || listTimeIndex.size () == 0)
			return;
		if (listTimeIndex.size () > 1)
			throw new RuntimeException ("The list of time indices must have exactly one entry");
		
		Object oIdx = listTimeIndex.iterator ().next ();
		if (oIdx instanceof Number)
			setTimeIndex (((Number) oIdx).intValue ());
		else if (oIdx instanceof Expression)
		{
			Integer nIdx = ExpressionUtil.getIntegerValueEx ((Expression) oIdx);
			if (nIdx == null)
				throw new RuntimeException ("The time index must be either a number or an expression evaluating to a number.");
			
			setTimeIndex (nIdx);
		}
		else
			throw new RuntimeException ("The time index must be either a number or an expression evaluating to a number.");
	}
	
	public void setTimeIndex (List<?> listTimeIndex, List<Expression> listOffset)
	{
		setTimeIndex (offsetIndex (listTimeIndex, listOffset));
	}

	/**
	 * Returns the spatial component of the index.
	 * 
	 * @return The part describing the location of the plane in space
	 */
	@Override
	public int[] getSpaceIndex ()
	{
		int[] rgIdx = new int[m_rgSpaceIndex.length];
		for (int i = 0; i < m_rgSpaceIndex.length; i++)
		{
			if (!(m_rgSpaceIndex[i] instanceof IntegerLiteral))
			{
				m_rgSpaceIndex[i] = Symbolic.simplify (m_rgSpaceIndex[i]);
				if (!(m_rgSpaceIndex[i] instanceof IntegerLiteral))
					throw new RuntimeException ("Not all indices are integer literals");
			}
			
			rgIdx[i] = (int) ((IntegerLiteral) m_rgSpaceIndex[i]).getValue ();
		}
		
		return rgIdx;
	}
	
	public Expression[] getSpaceIndexEx ()
	{
		return m_rgSpaceIndex;
	}

	/**
	 * Retrieves one coordinate of the spatial index.
	 * 
	 * @param nDim
	 *            The dimension for which to retrieve the spatial index
	 *            coordinate
	 * @return The spatial index in dimension <code>nDim</code>
	 * @throws IndexOutOfBoundsException
	 */
	public Expression getSpaceIndex (int nDim)
	{
		return m_rgSpaceIndex[nDim];
	}
	
	public void setSpaceIndex (int nDim, int nIdxValue)
	{
		Expression[] rgSpaceIdxOld = m_rgSpaceIndex;
		if (m_rgSpaceIndex == null || nDim >= m_rgSpaceIndex.length)
		{
			m_rgSpaceIndex = new Expression[nDim + 1];
			if (rgSpaceIdxOld != null)
			{
				System.arraycopy (rgSpaceIdxOld, 0, m_rgSpaceIndex, 0, rgSpaceIdxOld.length);
				for (int i = rgSpaceIdxOld.length; i <= nDim; i++)
					m_rgSpaceIndex[i] = new IntegerLiteral (0);
			}
		}
		
		m_rgSpaceIndex[nDim] = new IntegerLiteral (nIdxValue);
	}
	
	public void setSpaceIndex (int nDim, Expression exprIdxValue)
	{
		Expression[] rgSpaceIdxOld = m_rgSpaceIndex;
		if (m_rgSpaceIndex == null || nDim >= m_rgSpaceIndex.length)
		{
			m_rgSpaceIndex = new Expression[nDim + 1];
			if (rgSpaceIdxOld != null)
			{
				System.arraycopy (rgSpaceIdxOld, 0, m_rgSpaceIndex, 0, rgSpaceIdxOld.length);
				for (int i = rgSpaceIdxOld.length; i <= nDim; i++)
					m_rgSpaceIndex[i] = new IntegerLiteral (0);
			}
		}
		
		m_rgSpaceIndex[nDim] = exprIdxValue;		
	}

	/**
	 * Sets the index's space index.
	 * 
	 * @param rgSpaceIndex
	 *            The new spatial index
	 */
	public void setSpaceIndex (int[] rgSpaceIndex)
	{
		if (rgSpaceIndex == null)
		{
			m_rgSpaceIndex = null;
			return;
		}

		if (m_rgSpaceIndex == null || m_rgSpaceIndex.length != rgSpaceIndex.length)
			m_rgSpaceIndex = new Expression[rgSpaceIndex.length];
		
		for (int i = 0; i < rgSpaceIndex.length; i++)
			m_rgSpaceIndex[i] = new IntegerLiteral (rgSpaceIndex[i]);
	}
	
	public void setSpaceIndex (Expression[] rgSpaceIndex)
	{
		if (rgSpaceIndex == null)
		{
			m_rgSpaceIndex = null;
			return;
		}
		
		if (m_rgSpaceIndex == null || m_rgSpaceIndex.length != rgSpaceIndex.length)
			m_rgSpaceIndex = new Expression[rgSpaceIndex.length];
		
		for (int i = 0; i < rgSpaceIndex.length; i++)
			m_rgSpaceIndex[i] = rgSpaceIndex[i].clone ();
	}

	/**
	 * Sets the index's space index.
	 * 
	 * @param listSpaceIndex
	 *            The new spatial index
	 */
	public void setSpaceIndex (List<?> listSpaceIndex)
	{
		if (listSpaceIndex == null)
		{
			m_rgSpaceIndex = null;
			return;
		}

		if (m_rgSpaceIndex == null || m_rgSpaceIndex.length != listSpaceIndex.size ())
			m_rgSpaceIndex = new Expression[listSpaceIndex.size ()];
		int i = 0;
		for (Object oIdx : listSpaceIndex)
		{
			if (oIdx instanceof Number)
				m_rgSpaceIndex[i] = new IntegerLiteral (((Number) oIdx).intValue ());
			else if (oIdx instanceof Expression)
				m_rgSpaceIndex[i] = Symbolic.simplify ((Expression) oIdx);
			else
				throw new RuntimeException ("Entries in the index space list must be either numbers or expressions.");
			
			i++;
		}
	}
	
	/**
	 * Sets the index's spatial coordinates to <code>listSpaceInex</code> +
	 * <code>listOffsets</code> (elementwise addition).
	 * 
	 * @param listSpaceIndex
	 *            The list of indices
	 * @param listOffsets
	 *            The list of offsets
	 */
	public void setSpaceIndex (List<?> listSpaceIndex, List<Expression> listOffsets)
	{
		setSpaceIndex (offsetIndex (listSpaceIndex, listOffsets));
	}
	
	/**
	 * Returns the vector component of the index.
	 * 
	 * @return The vector part of the index
	 */
	public int getVectorIndex ()
	{
		return m_nVectorIndex;
	}

	/**
	 * Sets the index's vector index.
	 * 
	 * @param nVectorIndex
	 *            The new vector index
	 */
	public void setVectorIndex (int nVectorIndex)
	{
		m_nVectorIndex = nVectorIndex;
	}

	public boolean isAdvanceableInTime ()
	{
		return m_bIsAdvanceableInTime;
	}

	public void setAdvanceableInTime (boolean bIsAdvanceableInTime)
	{
		m_bIsAdvanceableInTime = bIsAdvanceableInTime;
	}

	/**
	 * Offsets the index by the relative indices in <code>idxTemplate</code>.
	 * <b>Note</b> that this function alters the index.
	 * 
	 * @param idxTemplate
	 *            The offset template
	 * @return A new index that is offset by <code>idxTemplate</code> compared
	 *         to the original index
	 */
	public void offset (Index idxTemplate)
	{
		// offset time and space indices
		// NOTE: the vector index isn't changed
		offsetInTime (idxTemplate.getTimeIndex ());
		offsetInSpace (idxTemplate.getSpaceIndex ());
	}

	/**
	 * Offsets the index by <code>nTimeOffset</code> in time.
	 * 
	 * @param nTimeOffset
	 *            The temporal offset
	 */
	public void offsetInTime (int nTimeOffset)
	{
		// offset the time index
		if (m_bIsAdvanceableInTime)
			m_nTimeIndex += nTimeOffset;
	}

	/**
	 * Offsets the index by <code>rgSpaceOffset</code> in space.
	 * 
	 * @param rgSpaceOffset
	 *            The spatial offset
	 */
	public void offsetInSpace (int[] rgSpaceOffset)
	{
		if (m_rgSpaceIndex == null || m_rgSpaceIndex.length == 0)
			return;

		// ensure that the space index array is large enough
		if (rgSpaceOffset.length > m_rgSpaceIndex.length)
		{
			Expression[] rgTmpSpaceIndex = new Expression[rgSpaceOffset.length];
			System.arraycopy (m_rgSpaceIndex, 0, rgTmpSpaceIndex, 0, m_rgSpaceIndex.length);
			
			for (int i = m_rgSpaceIndex.length; i < rgSpaceOffset.length; i++)
				rgTmpSpaceIndex[i] = new IntegerLiteral (0);

			m_rgSpaceIndex = rgTmpSpaceIndex;
		}

		// offset the spatial index
		for (int i = 0; i < rgSpaceOffset.length; i++)
			m_rgSpaceIndex[i] = ExpressionUtil.add (m_rgSpaceIndex[i], new IntegerLiteral (rgSpaceOffset[i]));
	}
	
	public void offsetInSpace (Expression[] rgSpaceOffset)
	{
		if (m_rgSpaceIndex == null || m_rgSpaceIndex.length == 0)
			return;

		// ensure that the space index array is large enough
		if (rgSpaceOffset.length > m_rgSpaceIndex.length)
		{
			Expression[] rgTmpSpaceIndex = new Expression[rgSpaceOffset.length];
			System.arraycopy (m_rgSpaceIndex, 0, rgTmpSpaceIndex, 0, m_rgSpaceIndex.length);
			
			for (int i = m_rgSpaceIndex.length; i < rgSpaceOffset.length; i++)
				rgTmpSpaceIndex[i] = new IntegerLiteral (0);

			m_rgSpaceIndex = rgTmpSpaceIndex;
		}

		// offset the spatial index
		for (int i = 0; i < rgSpaceOffset.length; i++)
			m_rgSpaceIndex[i] = ExpressionUtil.add (m_rgSpaceIndex[i], rgSpaceOffset[i].clone ());		
	}

	/**
	 * Offsets the index by <code>nSpaceOffset</code> steps in direction
	 * <code>nDirection</code> (this is the number of the axis) in space.
	 * 
	 * @param nDirection
	 *            The direction (the number of the axis) of the offset
	 * @param nSpaceOffset
	 *            The space offset
	 */
	public void offsetInSpace (int nDirection, int nSpaceOffset)
	{
		if (m_rgSpaceIndex == null)
			return;

		// ensure that the space index array is large enough
		if (nDirection >= m_rgSpaceIndex.length)
		{
			Expression[] rgTmpSpaceIndex = new Expression[nDirection + 1];
			System.arraycopy (m_rgSpaceIndex, 0, rgTmpSpaceIndex, 0, m_rgSpaceIndex.length);
			
			for (int i = m_rgSpaceIndex.length; i <= nDirection; i++)
				rgTmpSpaceIndex[i] = new IntegerLiteral (0);

			m_rgSpaceIndex = rgTmpSpaceIndex;
		}

		// offset the spatial index
		m_rgSpaceIndex[nDirection] = ExpressionUtil.add (m_rgSpaceIndex[nDirection], new IntegerLiteral (nSpaceOffset));
	}
	
	private static List<Expression> offsetIndex (List<?> listSpaceIndex, List<Expression> listOffsets)
	{
		List<Expression> listResult = new ArrayList<> (listSpaceIndex.size ());

		Iterator<?> itSpaceIdx = listSpaceIndex.iterator ();
		Iterator<Expression> itOffset = listOffsets == null ? null : listOffsets.iterator ();
		
		while (itSpaceIdx.hasNext () && itOffset != null && itOffset.hasNext ())
		{
			Object oIdx = itSpaceIdx.next ();
			Expression exprOffset = itOffset.next ();
			
			Expression exprIdx = null;
			if (oIdx instanceof Number)
				exprIdx = new IntegerLiteral (((Number) oIdx).longValue ());
			else if (oIdx instanceof Expression)
				exprIdx = (Expression) oIdx;
			else
				throw new RuntimeException ("Entries in the index space list must be either numbers or expressions.");

			listResult.add (new BinaryExpression (exprIdx.clone (), BinaryOperator.ADD, exprOffset.clone ()));
		}
		
		while (itSpaceIdx.hasNext ())
		{
			Object oIdx = itSpaceIdx.next ();
			
			if (oIdx instanceof Number)
				listResult.add (new IntegerLiteral (((Number) oIdx).longValue ()));
			else if (oIdx instanceof Expression)
				listResult.add ((Expression) oIdx);
			else
				throw new RuntimeException ("Entries in the index space list must be either numbers or expressions.");
		}

		return listResult;
	}


	///////////////////////////////////////////////////////////////////
	// Object Overrides

	@Override
	public String toString ()
	{
		StringBuffer sb = new StringBuffer ("[t=");
		sb.append (m_nTimeIndex);
		if (m_bIsAdvanceableInTime)
			sb.append ('^');
		sb.append (", s=");

		if (m_rgSpaceIndex != null)
		{
			if (m_rgSpaceIndex.length == 1)
				sb.append (m_rgSpaceIndex[0]);
			else
			{
				sb.append ('(');
				for (int i = 0; i < m_rgSpaceIndex.length; i++)
				{
					if (i > 0)
						sb.append (", ");
					sb.append (m_rgSpaceIndex[i]);
				}
				sb.append (')');
			}
		}
		else
			sb.append ("(null)");

		sb.append ("][");
		sb.append (m_nVectorIndex);
		sb.append (']');

		return sb.toString ();
	}

	@Override
	public boolean equals (Object obj)
	{
		if (!(obj instanceof Index))
			return false;

		Index idx = (Index) obj;

		// check that the space indices are equal
		if (m_rgSpaceIndex == null)
		{
			if (idx.m_rgSpaceIndex != null)
				if (idx.m_rgSpaceIndex.length != 0)
					return false;
		}
		else
		{
			if (idx.m_rgSpaceIndex == null)
				return false;

			if (m_rgSpaceIndex.length != idx.m_rgSpaceIndex.length)
				return false;
			for (int i = 0; i < m_rgSpaceIndex.length; i++)
				if (!m_rgSpaceIndex[i].equals (idx.m_rgSpaceIndex[i]))
					return false;
		}

		// check time and vector indices
		return m_nTimeIndex == idx.getTimeIndex () && m_nVectorIndex == idx.getVectorIndex () && m_bIsAdvanceableInTime == idx.isAdvanceableInTime ();
	}

	@Override
	public int hashCode ()
	{
		int nHashCode = m_nTimeIndex * 1000000 + m_nVectorIndex * 2 + (m_bIsAdvanceableInTime ? 1 : 0);

		if (m_rgSpaceIndex != null)
		{
			int nMultiplicator = 10;
			for (Expression expr : m_rgSpaceIndex)
			{
				nHashCode += expr.hashCode () * nMultiplicator;
				nMultiplicator *= 3;
			}
		}

		return nHashCode;
	}


	///////////////////////////////////////////////////////////////////
	// Comparable Implementation

	@Override
	public int compareTo (Index idx)
	{
		if (m_nTimeIndex != idx.getTimeIndex ())
			return m_nTimeIndex - idx.getTimeIndex ();
		int nResult = IndexSetUtil.SPACE_INDEX_COMPARATOR.compare (m_rgSpaceIndex, idx.getSpaceIndexEx ());
		if (nResult != 0)
			return nResult;
		if (m_nVectorIndex != idx.getVectorIndex ())
			return m_nVectorIndex - idx.getVectorIndex ();
		return (m_bIsAdvanceableInTime ? 1 : 0) - (idx.isAdvanceableInTime () ? 1 : 0);
	}
}
