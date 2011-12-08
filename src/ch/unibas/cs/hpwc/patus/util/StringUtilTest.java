package ch.unibas.cs.hpwc.patus.util;

import static org.junit.Assert.*;

import org.junit.Test;

public class StringUtilTest
{
	@Test
	public void testTrimLeft ()
	{
		assertEquals ("test", StringUtil.trimLeft ("test", new char[] { '_' }));
		assertEquals ("test", StringUtil.trimLeft ("_test", new char[] { '_' }));
		assertEquals ("test", StringUtil.trimLeft ("__test", new char[] { '_' }));
		assertEquals ("test", StringUtil.trimLeft ("_#test", new char[] { '_', '#' }));
		assertEquals ("", StringUtil.trimLeft ("___", new char[] { '_' }));
		assertEquals ("test_", StringUtil.trimLeft ("_test_", new char[] { '_' }));
	}
	
	@Test
	public void testTrimRight ()
	{
		assertEquals ("test", StringUtil.trimRight ("test", new char[] { '_' }));
		assertEquals ("test", StringUtil.trimRight ("test__", new char[] { '_' }));
		assertEquals ("test", StringUtil.trimRight ("test#_", new char[] { '_', '#' }));
		assertEquals ("", StringUtil.trimRight ("___", new char[] { '_' }));		
		assertEquals ("_test", StringUtil.trimRight ("_test_", new char[] { '_' }));
	}
}
