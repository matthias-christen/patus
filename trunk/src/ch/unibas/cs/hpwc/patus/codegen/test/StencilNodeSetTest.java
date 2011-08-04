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
package ch.unibas.cs.hpwc.patus.codegen.test;


import junit.framework.Assert;

import org.junit.Before;
import org.junit.Test;

import cetus.hir.Expression;
import cetus.hir.IntegerLiteral;
import cetus.hir.Specifier;
import ch.unibas.cs.hpwc.patus.codegen.ProjectionMask;
import ch.unibas.cs.hpwc.patus.codegen.StencilNodeSet;
import ch.unibas.cs.hpwc.patus.representation.Index;
import ch.unibas.cs.hpwc.patus.representation.StencilNode;

public class StencilNodeSetTest
{
	///////////////////////////////////////////////////////////////////
	// Member Variables

	private StencilNodeSet m_set1;
	private StencilNodeSet m_set2;
	private StencilNodeSet m_set3;


	///////////////////////////////////////////////////////////////////
	// Implementation

	@Before
	public void setUp () throws Exception
	{
		m_set1 = new StencilNodeSet (
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 0, 0, 0 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 1, 0, 0 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { -1, 0, 0 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 0, 1, 0 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 0, -1, 0 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 0, 0, 1 }, 0)),
			new StencilNode ("c", Specifier.FLOAT, new Index (-1, new int[] { 0, 0, 0 }, 1)),
			new StencilNode ("c", Specifier.FLOAT, new Index (-1, new int[] { 0, 0, 1 }, 1)),
			new StencilNode ("c", Specifier.FLOAT, new Index (-1, new int[] { 0, 0, 0 }, 2)),
			new StencilNode ("c", Specifier.FLOAT, new Index (-2, new int[] { 0, 0, 0 }, 2)));

		m_set2 = new StencilNodeSet (
			new StencilNode ("T", Specifier.FLOAT, new Index (0, new int[] { 0, 0, 0 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { -1, 0, 0 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 0, -1, 0 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 0, 0, -1 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-2, new int[] { 1, 0, 0 }, 0)));

		m_set3 = new StencilNodeSet (
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { -1, 0, 1 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { -3, 0, -2 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 1, 0, 1 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { -1, 0, 0 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 2, 0, 0 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { -2, 0, -1 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { -1, 0, -1 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 0, 0, 1 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 1, 0, -1 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 0, 0, -2 }, 0)));
	}

	@Test
	public void union1 ()
	{
		StencilNodeSet setResult = m_set1.union (m_set2);

		StencilNodeSet setExpected = new StencilNodeSet (
			new StencilNode ("T", Specifier.FLOAT, new Index (0, new int[] { 0, 0, 0 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 0, 0, 0 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 1, 0, 0 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { -1, 0, 0 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 0, 1, 0 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 0, -1, 0 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 0, 0, 1 }, 0)),
			new StencilNode ("c", Specifier.FLOAT, new Index (-1, new int[] { 0, 0, 0 }, 1)),
			new StencilNode ("c", Specifier.FLOAT, new Index (-1, new int[] { 0, 0, 1 }, 1)),
			new StencilNode ("c", Specifier.FLOAT, new Index (-1, new int[] { 0, 0, 0 }, 2)),
			new StencilNode ("c", Specifier.FLOAT, new Index (-2, new int[] { 0, 0, 0 }, 2)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 0, 0, -1 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-2, new int[] { 1, 0, 0 }, 0)));
		Assert.assertEquals (setExpected, setResult);
	}

	@Test
	public void union2 ()
	{
		StencilNodeSet setResult = m_set1.union (m_set3);

		StencilNodeSet setExpected = new StencilNodeSet (
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 0, 0, 0 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 1, 0, 0 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { -1, 0, 0 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 0, 1, 0 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 0, -1, 0 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 0, 0, 1 }, 0)),
			new StencilNode ("c", Specifier.FLOAT, new Index (-1, new int[] { 0, 0, 0 }, 1)),
			new StencilNode ("c", Specifier.FLOAT, new Index (-1, new int[] { 0, 0, 1 }, 1)),
			new StencilNode ("c", Specifier.FLOAT, new Index (-1, new int[] { 0, 0, 0 }, 2)),
			new StencilNode ("c", Specifier.FLOAT, new Index (-2, new int[] { 0, 0, 0 }, 2)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { -1, 0, 1 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { -3, 0, -2 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 1, 0, 1 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 2, 0, 0 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { -2, 0, -1 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { -1, 0, -1 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 1, 0, -1 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 0, 0, -2 }, 0)));

		Assert.assertEquals (setExpected, setResult);
	}

	@Test
	public void restrict1 ()
	{
		StencilNodeSet setResult = m_set1.restrict (0, null);
		Assert.assertEquals (new StencilNodeSet (), setResult);
	}

	@Test
	public void restrict2 ()
	{
		StencilNodeSet setResult = m_set1.restrict (-1, null);

		StencilNodeSet setExpected = new StencilNodeSet (
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 0, 0, 0 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 1, 0, 0 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { -1, 0, 0 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 0, 1, 0 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 0, -1, 0 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 0, 0, 1 }, 0)),
			new StencilNode ("c", Specifier.FLOAT, new Index (-1, new int[] { 0, 0, 0 }, 1)),
			new StencilNode ("c", Specifier.FLOAT, new Index (-1, new int[] { 0, 0, 1 }, 1)),
			new StencilNode ("c", Specifier.FLOAT, new Index (-1, new int[] { 0, 0, 0 }, 2)));
		Assert.assertEquals (setExpected, setResult);
	}

	@Test
	public void restrict3 ()
	{
		StencilNodeSet setResult = m_set1.restrict (null, 1);

		StencilNodeSet setExpected = new StencilNodeSet (
			new StencilNode ("c", Specifier.FLOAT, new Index (-1, new int[] { 0, 0, 0 }, 1)),
			new StencilNode ("c", Specifier.FLOAT, new Index (-1, new int[] { 0, 0, 1 }, 1)));
		Assert.assertEquals (setExpected, setResult);
	}

	@Test
	public void restrict4 ()
	{
		StencilNodeSet setResult = m_set1.restrict (-1, 0);

		StencilNodeSet setExpected = new StencilNodeSet (
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 0, 0, 0 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 1, 0, 0 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { -1, 0, 0 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 0, 1, 0 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 0, -1, 0 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 0, 0, 1 }, 0)));
		Assert.assertEquals (setExpected, setResult);
	}

	@Test
	public void front1 ()
	{
		StencilNodeSet setResult = m_set1.getFront (0);

		StencilNodeSet setExpected = new StencilNodeSet (
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 1, 0, 0 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 0, 1, 0 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 0, -1, 0 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 0, 0, 1 }, 0)),
			new StencilNode ("c", Specifier.FLOAT, new Index (-1, new int[] { 0, 0, 0 }, 1)),
			new StencilNode ("c", Specifier.FLOAT, new Index (-1, new int[] { 0, 0, 1 }, 1)),
			new StencilNode ("c", Specifier.FLOAT, new Index (-1, new int[] { 0, 0, 0 }, 2)),
			new StencilNode ("c", Specifier.FLOAT, new Index (-2, new int[] { 0, 0, 0 }, 2)));
		Assert.assertEquals (setExpected, setResult);
	}

	@Test
	public void front2 ()
	{
		StencilNodeSet setResult = m_set1.getFront (1);

		StencilNodeSet setExpected = new StencilNodeSet (
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 1, 0, 0 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { -1, 0, 0 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 0, 1, 0 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 0, 0, 1 }, 0)),
			new StencilNode ("c", Specifier.FLOAT, new Index (-1, new int[] { 0, 0, 0 }, 1)),
			new StencilNode ("c", Specifier.FLOAT, new Index (-1, new int[] { 0, 0, 1 }, 1)),
			new StencilNode ("c", Specifier.FLOAT, new Index (-1, new int[] { 0, 0, 0 }, 2)),
			new StencilNode ("c", Specifier.FLOAT, new Index (-2, new int[] { 0, 0, 0 }, 2)));
		Assert.assertEquals (setExpected, setResult);
	}

	@Test
	public void front3 ()
	{
		StencilNodeSet setResult = m_set3.getFront (0);

		StencilNodeSet setExpected = new StencilNodeSet (
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 1, 0, 1 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 2, 0, 0 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 1, 0, -1 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 0, 0, -2 }, 0)));
		Assert.assertEquals (setExpected, setResult);
	}

	@Test
	public void front4 ()
	{
		StencilNodeSet setResult = m_set3.getFront (1);

		StencilNodeSet setExpected = new StencilNodeSet (
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { -1, 0, 1 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 0, 0, 1 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 1, 0, 1 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { -1, 0, 0 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 2, 0, 0 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { -2, 0, -1 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { -1, 0, -1 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 1, 0, -1 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { -3, 0, -2 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 0, 0, -2 }, 0)));
		Assert.assertEquals (setExpected, setResult);
	}

	@Test
	public void front5 ()
	{
		StencilNodeSet setResult = m_set3.getFront (2);

		StencilNodeSet setExpected = new StencilNodeSet (
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { -1, 0, 1 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { -3, 0, -2 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 1, 0, 1 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 2, 0, 0 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { -2, 0, -1 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 0, 0, 1 }, 0)));
		Assert.assertEquals (setExpected, setResult);
	}

	@Test
	public void mask1 ()
	{
		StencilNodeSet setResult = m_set1.applyMask (
			new ProjectionMask (new Expression[] { new IntegerLiteral (0), new IntegerLiteral (1), new IntegerLiteral (1) }));

		StencilNodeSet setExpected = new StencilNodeSet (
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 0, 0, 0 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 0, 1, 0 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 0, -1, 0 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 0, 0, 1 }, 0)),
			new StencilNode ("c", Specifier.FLOAT, new Index (-1, new int[] { 0, 0, 0 }, 1)),
			new StencilNode ("c", Specifier.FLOAT, new Index (-1, new int[] { 0, 0, 1 }, 1)),
			new StencilNode ("c", Specifier.FLOAT, new Index (-1, new int[] { 0, 0, 0 }, 2)),
			new StencilNode ("c", Specifier.FLOAT, new Index (-2, new int[] { 0, 0, 0 }, 2)));
		Assert.assertEquals (setExpected, setResult);
	}

	@Test
	public void mask2 ()
	{
		StencilNodeSet setResult = m_set1.applyMask (
			new ProjectionMask (new Expression[] { new IntegerLiteral (0), new IntegerLiteral (0), new IntegerLiteral (1) }));

		StencilNodeSet setExpected = new StencilNodeSet (
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 0, 0, 0 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 0, 0, 1 }, 0)),
			new StencilNode ("c", Specifier.FLOAT, new Index (-1, new int[] { 0, 0, 0 }, 1)),
			new StencilNode ("c", Specifier.FLOAT, new Index (-1, new int[] { 0, 0, 1 }, 1)),
			new StencilNode ("c", Specifier.FLOAT, new Index (-1, new int[] { 0, 0, 0 }, 2)),
			new StencilNode ("c", Specifier.FLOAT, new Index (-2, new int[] { 0, 0, 0 }, 2)));
		Assert.assertEquals (setExpected, setResult);
	}

	@Test
	public void fill1 ()
	{
		StencilNodeSet setResult = m_set1.fill (0);
		Assert.assertEquals (m_set1, setResult);
	}

	@Test
	public void fill2 ()
	{
		StencilNodeSet setResult = m_set3.fill (0);
		StencilNodeSet setExpected = new StencilNodeSet (
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { -1, 0, 1 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 0, 0, 1 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 1, 0, 1 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { -1, 0, 0 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 0, 0, 0 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 1, 0, 0 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 2, 0, 0 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { -2, 0, -1 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { -1, 0, -1 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 0, 0, -1 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 1, 0, -1 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { -3, 0, -2 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { -2, 0, -2 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { -1, 0, -2 }, 0)),
			new StencilNode ("T", Specifier.FLOAT, new Index (-1, new int[] { 0, 0, -2 }, 0)));
		Assert.assertEquals (setExpected, setResult);
	}
}
