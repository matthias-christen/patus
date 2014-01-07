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

import cetus.hir.Expression;
import ch.unibas.cs.hpwc.patus.util.ExpressionUtil;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 *
 * @author Matthias-M. Christen
 */
public class Border extends Primitive
{
	///////////////////////////////////////////////////////////////////
	// Member Variables

	/**
	 * The left/lower/front border (the border with the smaller coordinates)
	 * This will be subtracted from the lower end of the box.
	 */
	private Size m_sizeMin;

	/**
	 * The right/upper/back border (the border with the larger coordinates).
	 * This will be added to the upper end of the box.
	 */
	private Size m_sizeMax;


	///////////////////////////////////////////////////////////////////
	// Implementation

	public Border ()
	{
		setMin (new Size ());
		setMax (new Size ());
	}

	public Border (byte nDimensionality)
	{
		setMin (new Size (nDimensionality));
		setMax (new Size (nDimensionality));
	}

	/**
	 * Constructs a new border object from min and max sizes.
	 * @param sizeMin
	 * @param sizeMax
	 */
	public Border (Size sizeMin, Size sizeMax)
	{
		setMin (sizeMin);
		setMax (sizeMax);
	}

	public Size getMin ()
	{
		return m_sizeMin;
	}

	public void setMin (Size sizeMin)
	{
		m_sizeMin = sizeMin;
		ensureDimensionality (m_sizeMin.getDimensionality ());
	}

	public Size getMax ()
	{
		return m_sizeMax;
	}

	public void setMax (Size sizeMax)
	{
		m_sizeMax = sizeMax;
		ensureDimensionality (m_sizeMax.getDimensionality ());
	}

	/**
	 * Scales the border by a factor <code>exprScale</code>.
	 * @param exprScale
	 */
	public void scale (Expression exprScale)
	{
		if (exprScale == null || ExpressionUtil.isValue (exprScale, 1))
			return;
		m_sizeMin.scale (exprScale);
		m_sizeMax.scale (exprScale);
	}

	@Override
	public String toString ()
	{
		return StringUtil.concat (
			"[ min=", m_sizeMin == null ? "" : m_sizeMin.toString (),
			", max=", m_sizeMax == null ? "" : m_sizeMax.toString (), " ]");
	}
}
