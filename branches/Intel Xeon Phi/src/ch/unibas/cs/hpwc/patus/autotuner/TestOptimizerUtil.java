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
package ch.unibas.cs.hpwc.patus.autotuner;


import java.util.ArrayList;
import java.util.List;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.Expression;
import cetus.hir.IntegerLiteral;
import cetus.hir.NameID;

public class TestOptimizerUtil
{
	///////////////////////////////////////////////////////////////////
	// Member Variables

	private List<Expression> m_listConstraints;
	private int[] m_rgLowerBounds;
	private int[] m_rgUpperBounds;


	///////////////////////////////////////////////////////////////////
	// Implementation

	@Before
	public void setUp () throws Exception
	{
		m_listConstraints = new ArrayList<> (1);
		m_listConstraints.add (
			new BinaryExpression (
				new BinaryExpression (new NameID ("$1"), BinaryOperator.MULTIPLY, new BinaryExpression (new NameID ("$2"), BinaryOperator.MULTIPLY, new NameID ("$3"))),
				BinaryOperator.COMPARE_LE,
				new IntegerLiteral (1024)));

		m_rgLowerBounds = new int[] { 0, 0, 0 };
		m_rgUpperBounds = new int[] { 128, 128, 128 };
	}

	@Test
	public void testWithinBounds1 ()
	{
		Assert.assertTrue (OptimizerUtil.isWithinBounds (new int[] { 8, 8, 8 }, m_rgLowerBounds, m_rgUpperBounds, m_listConstraints));
	}

	@Test
	public void testWithinBounds2 ()
	{
		Assert.assertTrue (OptimizerUtil.isWithinBounds (new int[] { 1, 4, 128 }, m_rgLowerBounds, m_rgUpperBounds, m_listConstraints));
	}

	@Test
	public void testWithinBounds3 ()
	{
		Assert.assertTrue (OptimizerUtil.isWithinBounds (new int[] { 8, 16, 8 }, m_rgLowerBounds, m_rgUpperBounds, m_listConstraints));
	}

	@Test
	public void testWithinBounds4 ()
	{
		Assert.assertFalse (OptimizerUtil.isWithinBounds (new int[] { 16, 16, 16 }, m_rgLowerBounds, m_rgUpperBounds, m_listConstraints));
	}

	@Test
	public void testWithinBounds5 ()
	{
		Assert.assertFalse (OptimizerUtil.isWithinBounds (new int[] { 8, 32, 16 }, m_rgLowerBounds, m_rgUpperBounds, m_listConstraints));
	}
}
