package ch.unibas.cs.hpwc.patus.util;

import static org.junit.Assert.*;

import org.junit.Test;

public class MathUtilTest
{
	/*
	@Test
	public void testGetGCDIntInt ()
	{
		fail ("Not yet implemented");
	}

	@Test
	public void testGetGCDIntArray ()
	{
		fail ("Not yet implemented");
	}

	@Test
	public void testGetLCMIntInt ()
	{
		fail ("Not yet implemented");
	}

	@Test
	public void testGetLCMIntArray ()
	{
		fail ("Not yet implemented");
	}

	@Test
	public void testIsPowerOfTwo ()
	{
		fail ("Not yet implemented");
	}

	@Test
	public void testSgn ()
	{
		fail ("Not yet implemented");
	}
	*/

	@Test
	public void testLog2 ()
	{
		assertEquals (0, MathUtil.log2 (1));
		assertEquals (1, MathUtil.log2 (2));
		assertEquals (1, MathUtil.log2 (3));
		assertEquals (2, MathUtil.log2 (4));
		assertEquals (2, MathUtil.log2 (6));
		assertEquals (3, MathUtil.log2 (8));
		assertEquals (4, MathUtil.log2 (16));
		assertEquals (9, MathUtil.log2 (1023));
		assertEquals (10, MathUtil.log2 (1024));
		assertEquals (10, MathUtil.log2 (1025));
		assertEquals (14, MathUtil.log2 (16384));
	}
}
