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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import cetus.hir.Expression;
import ch.unibas.cs.hpwc.patus.representation.ISpaceIndexable;
import ch.unibas.cs.hpwc.patus.util.IntArray;

public abstract class AbstractMask implements IMask
{
	///////////////////////////////////////////////////////////////////
	// Member Variables

	private Expression[] m_rgExpressions;

	private int[] m_rgMask;
	private int[] m_rgMaskIndices;


	///////////////////////////////////////////////////////////////////
	// Implementation

	public AbstractMask (Expression[] rgExpressions)
	{
		m_rgExpressions = rgExpressions;
		m_rgMask = null;
	}

	protected void init ()
	{
		if (m_rgMask == null)
		{
			m_rgMask = createMask (m_rgExpressions);

			// compute the dimension of the quotient space and the indices that are masked
			int nQuotientSpaceDimension = 0;
			for (int nBit : m_rgMask)
				if (nBit != 0)
					nQuotientSpaceDimension++;
			m_rgMaskIndices = new int[nQuotientSpaceDimension];
			int j = 0;
			for (int i = 0; i < m_rgMask.length; i++)
				if (m_rgMask[i] != 0)
				{
					m_rgMaskIndices[j] = i;
					j++;
				}
		}
	}

	private int[] getMask ()
	{
		init ();
		return m_rgMask;
	}

	private int[] getMaskIndices ()
	{
		init ();
		return m_rgMaskIndices;
	}

	abstract protected int[] createMask (Expression[] rgExpressions);

	@Override
	public Map<IntArray, List<ISpaceIndexable>> getEquivalenceClasses (Iterable<? extends ISpaceIndexable> itInput)
	{
		Map<IntArray, List<ISpaceIndexable>> mapEquivClasses = new HashMap<> ();
		for (ISpaceIndexable index : itInput)
		{
			int[] rgEquivClass = getEquivalenceClass (index);
			IntArray arrEquivClass = new IntArray (rgEquivClass);

			List<ISpaceIndexable> listItems = mapEquivClasses.get (arrEquivClass);
			if (listItems == null)
				mapEquivClasses.put (arrEquivClass, listItems = new ArrayList<> ());
			listItems.add (index);
		}

		return mapEquivClasses;
	}

	@Override
	public int[] getEquivalenceClass (ISpaceIndexable index)
	{
		return apply (index.getSpaceIndex ());
	}

	@Override
	public int[] apply (int[] rgIndex)
	{
		int[] rgVec = new int[rgIndex.length];
		Arrays.fill (rgVec, 0);
		for (int i = 0; i < getMaskIndices ().length; i++)
		{
			if (getMaskIndices ()[i] < rgIndex.length)
				rgVec[getMaskIndices ()[i]] = rgIndex[getMaskIndices ()[i]];
		}

		return rgVec;
	}

	/**
	 * Returns a copy of the mask vector.
	 * @return
	 */
	public int[] getVector ()
	{
		int[] rgMask = new int[getMask ().length];
		System.arraycopy (getMask (), 0, rgMask, 0, rgMask.length);
		return rgMask;
	}

	/**
	 * Returns the dimensionality of an object to which the mask has been applied.
	 * Note that, if the mask is constructed from an iterator, as the iterators are
	 * cosets in the quotient space &Sigma;/mask, the dimensionality of the
	 * iterator is returned by {@link AbstractMask#getCodimension()}.
	 * @return
	 */
	public int getDimension ()
	{
		return getMaskIndices ().length;
	}

	/**
	 *
	 * @return
	 */
	public int getCodimension ()
	{
		return getMask ().length - getMaskIndices ().length;
	}

	@Override
	public String toString ()
	{
		StringBuilder sb = new StringBuilder ();
		for (int i = 0; i < getMask ().length; i++)
		{
			sb.append ("[ ");
			for (int j = 0; j < getMask ().length; j++)
			{
				if (i == j)
				{
					sb.append (getMask ()[i]);
					sb.append (' ');
				}
				else
					sb.append ("  ");
			}
			sb.append ("]\n");
		}

		return sb.toString ();
	}
}
