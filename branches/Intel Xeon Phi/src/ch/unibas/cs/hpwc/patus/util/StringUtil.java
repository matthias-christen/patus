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
package ch.unibas.cs.hpwc.patus.util;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

import cetus.hir.Expression;

/**
 *
 * @author Matthias-M. Christen
 */
public class StringUtil
{
	///////////////////////////////////////////////////////////////////
	// Implementation

	public static StringBuilder joinAsBuilder (Iterable<?> coll, String strGlue, StringBuilder sb)
	{
		boolean bIsFirst = true;
		for (Object o : coll)
		{
			if (!bIsFirst)
				sb.append (strGlue);
			sb.append (o == null ? "null" : o.toString ());
			bIsFirst = false;
		}

		return sb;
	}

	public static <T> StringBuilder joinAsBuilder (T[] rgArray, String strGlue, StringBuilder sb)
	{
		boolean bIsFirst = true;
		for (Object o : rgArray)
		{
			if (!bIsFirst)
				sb.append (strGlue);
			sb.append (o == null ? "null" : o.toString ());
			bIsFirst = false;
		}

		return sb;
	}

	public static StringBuilder joinAsBuilder (int[] rgArray, String strGlue, StringBuilder sb)
	{
		boolean bIsFirst = true;
		for (int n : rgArray)
		{
			if (!bIsFirst)
				sb.append (strGlue);
			sb.append (n);
			bIsFirst = false;
		}

		return sb;
	}

	public static StringBuilder joinAsBuilder (Iterable<?> coll, String strGlue)
	{
		return StringUtil.joinAsBuilder (coll, strGlue, new StringBuilder ());
	}

	public static <T> StringBuilder joinAsBuilder (T[] rgArray, String strGlue)
	{
		return StringUtil.joinAsBuilder (rgArray, strGlue, new StringBuilder ());
	}

	public static StringBuilder joinAsBuilder (int[] rgArray, String strGlue)
	{
		return StringUtil.joinAsBuilder (rgArray, strGlue, new StringBuilder ());
	}

	public static String join (Iterable<?> coll, String strGlue)
	{
		return StringUtil.joinAsBuilder (coll, strGlue).toString ();
	}

	public static <T> String join (T[] rgArray, String strGlue)
	{
		return StringUtil.joinAsBuilder (rgArray, strGlue).toString ();
	}

	public static String join (int[] rgArray, String strGlue)
	{
		return StringUtil.joinAsBuilder (rgArray, strGlue).toString ();
	}

	/**
	 * Concatenates the parts <code>rgParts</code>.
	 * @param rgParts The parts to concatenate
	 * @return <code>rgParts</code> concatenated
	 */
	public static String concat (Object... rgParts)
	{
		StringBuilder sb = new StringBuilder ();
		for (Object objPart : rgParts)
			if (objPart != null)
				sb.append (objPart.toString ());
		return sb.toString ();
	}

	/**
	 * Makes a camel toe version of <code>strOrig</code>, e.g.,
	 * transforms &quot;camel_toe&quot; into &quot;camelToe&quot;.
	 * @param strOrig The original string
	 * @return The camel toe version of <code>strOrig</code>
	 */
	public static String toCamelToe (String strOrig)
	{
		StringBuilder sb = new StringBuilder ();
		boolean bMakeNextUpper = false;
		for (int i = 0; i < strOrig.length (); i++)
		{
			char c = strOrig.charAt (i);
			if (c == '_')
			{
				bMakeNextUpper = true;
				continue;
			}

			if (bMakeNextUpper)
			{
				sb.append (Character.toUpperCase (c));
				bMakeNextUpper = false;
			}
			else
				sb.append (c);
		}

		return sb.toString ();
	}

	public static String toString (Object obj)
	{
		if (obj instanceof Map<?, ?>)
		{
			Map<?, ?> map = (Map<?, ?>) obj;
			StringBuilder sb = new StringBuilder ("[\n");
			for (Object objKey : map.keySet ())
			{
				sb.append ('\t');
				sb.append (StringUtil.toString (objKey));
				sb.append (" --> ");
				sb.append (StringUtil.toString (map.get (objKey)));
				sb.append ('\n');
			}
			sb.append ("]");
			return sb.toString ();
		}

		if (obj instanceof List<?>)
		{
			List<?> list = (List<?>) obj;
			StringBuilder sb = new StringBuilder ("[\n");
			for (Object objElem : list)
			{
				sb.append ('\t');
				sb.append (StringUtil.toString (objElem));
				sb.append ('\n');
			}
			sb.append ("]");
			return sb.toString ();
		}

		if (obj instanceof int[])
			return Arrays.toString ((int[]) obj);
		if (obj instanceof short[])
			return Arrays.toString ((short[]) obj);
		if (obj instanceof byte[])
			return Arrays.toString ((byte[]) obj);
		if (obj instanceof long[])
			return Arrays.toString ((long[]) obj);
		if (obj instanceof float[])
			return Arrays.toString ((float[]) obj);
		if (obj instanceof double[])
			return Arrays.toString ((double[]) obj);
		if (obj instanceof String[])
			return Arrays.toString ((String[]) obj);
		if (obj instanceof boolean[])
			return Arrays.toString ((boolean[]) obj);
		if (obj instanceof char[])
			return Arrays.toString ((char[]) obj);
		if (obj instanceof Expression[])
			return StringUtil.concat ("[ ", StringUtil.join ((Expression[]) obj, ", "), " ]");
		
		// default
		return obj.toString ();
	}

	public static String trimLeft (String s, char[] rgTrim)
	{
		int nIdx = 0;
		String strTrim = new String (rgTrim);
		
		while (nIdx < s.length () && strTrim.indexOf (s.charAt (nIdx)) >= 0)
			nIdx++;
		
		return s.substring (nIdx);
	}
	
	public static String trimRight (String s, char[] rgTrim)
	{
		int nIdx = s.length () - 1;
		String strTrim = new String (rgTrim);
		
		while (nIdx >= 0 && strTrim.indexOf (s.charAt (nIdx)) >= 0)
			nIdx--;
		
		return s.substring (0, nIdx + 1);		
	}
	
	public static String padRight (String s, int nNumTotalChars)
	{
		if (s.length () > nNumTotalChars)
			return s.substring (0, nNumTotalChars);
		return String.format (StringUtil.concat ("%1$-", nNumTotalChars, "s"), s);
	}

	public static String num2IdStr (int n)
	{
		if (n >= 0)
			return String.valueOf (n);
		return "m" + String.valueOf (-n);
	}

	public static String num2IdStr (long n)
	{
		if (n >= 0)
			return String.valueOf (n);
		return "m" + String.valueOf (-n);
	}
}
