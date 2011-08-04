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

import cetus.hir.Expression;
import ch.unibas.cs.hpwc.patus.ast.SubdomainIterator;
import ch.unibas.cs.hpwc.patus.geometry.Vector;
import ch.unibas.cs.hpwc.patus.util.ExpressionUtil;

public class ReuseMask extends AbstractMask
{
	///////////////////////////////////////////////////////////////////
	// Member Variables

	private int m_nReuseDimension;


	///////////////////////////////////////////////////////////////////
	// Implementation

	public ReuseMask (SubdomainIterator it)
	{
		this (it.getIteratorSubdomain ().getBox ().getSize ().getCoords ());
	}

	public ReuseMask (Vector v)
	{
		this (v.getCoords ());
	}

	public ReuseMask (Expression[] rgExpressions)
	{
		super (rgExpressions);
	}

	@Override
	protected int[] createMask (Expression[] rgExpressions)
	{
		m_nReuseDimension = -1;

		int[] rgMask = new int[rgExpressions.length];
		Arrays.fill (rgMask, 0);

		for (int i = 0; i < rgExpressions.length; i++)
		{
			if (ExpressionUtil.isValue (rgExpressions[i], 1))
			{
				rgMask[i] = 1;
				m_nReuseDimension = i;
				break;
			}
		}

		return rgMask;
	}

	/**
	 * Returns the dimension in which the memory objects are reused or -1 if there is no
	 * data reuse.
	 * @return
	 */
	public int getReuseDimension ()
	{
		// make sure the mask has been created
		init ();

		return m_nReuseDimension;
	}
}
