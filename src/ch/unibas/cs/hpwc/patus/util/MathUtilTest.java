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
	
	@Test
	public void testGetPrevPower2 ()
	{
		for (int i = 1; i < 100; i++)
			System.out.println (i + " => " + MathUtil.getPrevPower2 (i));
		
		
		assertEquals (1, MathUtil.getPrevPower2 (1));
		assertEquals (2, MathUtil.getPrevPower2 (2));
		assertEquals (2, MathUtil.getPrevPower2 (3));
		assertEquals (4, MathUtil.getPrevPower2 (4));
		assertEquals (4, MathUtil.getPrevPower2 (5));
		assertEquals (4, MathUtil.getPrevPower2 (7));
		assertEquals (8, MathUtil.getPrevPower2 (8));
		assertEquals (8, MathUtil.getPrevPower2 (9));
		assertEquals (64, MathUtil.getPrevPower2 (100));
		assertEquals (128, MathUtil.getPrevPower2 (200));
		assertEquals (128, MathUtil.getPrevPower2 (200));
		assertEquals (1024, MathUtil.getPrevPower2 (1025));
		assertEquals (1024, MathUtil.getPrevPower2 (2025));
		assertEquals (2048, MathUtil.getPrevPower2 (2125));
		assertEquals (4096, MathUtil.getPrevPower2 (4444));
		assertEquals (65536, MathUtil.getPrevPower2 (100000));
		assertEquals (524288, MathUtil.getPrevPower2 (1000000));
		assertEquals (8388608, MathUtil.getPrevPower2 (10000000));
		assertEquals (67108864, MathUtil.getPrevPower2 (100000000));
		assertEquals (536870912, MathUtil.getPrevPower2 (1000000000));
	}
}
