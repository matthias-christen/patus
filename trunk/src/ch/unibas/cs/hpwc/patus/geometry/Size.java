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
package ch.unibas.cs.hpwc.patus.geometry;

import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.Expression;
import ch.unibas.cs.hpwc.patus.symbolic.Symbolic;
import ch.unibas.cs.hpwc.patus.util.ExpressionUtil;


/**
 *
 * @author Matthias-M. Christen
 */
public class Size extends Vector
{
	///////////////////////////////////////////////////////////////////
	// Implementation

	/**
	 * Creates an size object with zero dimensions.
	 */
	public Size ()
	{
		//ensureDimensionality (GlobalConstants.DIMENSIONALITY);
	}

	/**
	 * Creates a size object with dimensionality <code>nDimensionality</code>.
	 * @param nDimensionality The dimensionality of the size object
	 */
	public Size (byte nDimensionality)
	{
		ensureDimensionality (nDimensionality);
	}

	/**
	 * Creates a size object with dimensions set to the elements of <code>rgSize</code>.
	 * @param rgSize The dimensions
	 */
 	public Size (int... rgSize)
	{
		setCoords (rgSize);
	}

	/**
	 * Creates a size object with dimensions set to the elements of <code>rgSize</code>.
	 * @param rgSize The dimensions
	 */
	public Size (Expression... rgSize)
	{
		setCoords (rgSize);
	}

	/**
	 *
	 * @param ptMin
	 * @param ptMax
	 */
	public Size (Point ptMin, Point ptMax)
	{
		ensureDimensionality ((byte) Math.max (ptMin.getDimensionality (), ptMax.getDimensionality ()));
		for (int i = 0; i < getDimensionality (); i++)
		{
			m_rgCoords[i] = Symbolic.simplify (
				ExpressionUtil.increment (new BinaryExpression (ptMax.getCoord (i).clone (), BinaryOperator.SUBTRACT, ptMin.getCoord (i).clone ())),
				Symbolic.ALL_VARIABLES_INTEGER
			);
		}
	}

	public Size (Vector vecSize)
	{
		setCoords (vecSize.m_rgCoords);
	}

	public void addBorder (Border border)
	{
		if (border == null)
			return;

		m_bCoordsSimplified = false;
		for (int i = 0; i < getDimensionality (); i++)
		{
			m_rgCoords[i] = new BinaryExpression (
				m_rgCoords[i].clone (),
				BinaryOperator.ADD,
				new BinaryExpression (border.getMin ().getCoord (i).clone (), BinaryOperator.ADD, border.getMax ().getCoord (i).clone ())
			);
		}
	}

	@Override
	public Size clone ()
	{
		return new Size (this);
	}

	public Expression getVolume ()
	{
		return ExpressionUtil.product (m_rgCoords);
	}
}
