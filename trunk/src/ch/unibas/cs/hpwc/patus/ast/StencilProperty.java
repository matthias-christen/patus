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
package ch.unibas.cs.hpwc.patus.ast;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

import cetus.hir.NameID;
import ch.unibas.cs.hpwc.patus.util.CodeGeneratorUtil;
import ch.unibas.cs.hpwc.patus.util.StringUtil;



/**
 *
 * @author Matthias-M. Christen
 */
public class StencilProperty
{
	///////////////////////////////////////////////////////////////////
	// Constants

	private final static Pattern PATTERN_DOMAINSIZE = Pattern.compile ("([u-z](\\d*))\\_((min)|(max))");
	private final static Matcher MATCHER_DOMAINSIZE = PATTERN_DOMAINSIZE.matcher ("");


	///////////////////////////////////////////////////////////////////
	// Implementation

	/**
	 *
	 * @param strProperty
	 * @return
	 * @throws NoSuchFieldException
	 */
	public static NameID get (String strProperty) throws NoSuchFieldException
	{
		if ("t_max".equals (strProperty))
			return StencilProperty.getMaxTime ();

		MATCHER_DOMAINSIZE.reset (strProperty);
		if (MATCHER_DOMAINSIZE.matches ())
		{
			if ("min".equals (MATCHER_DOMAINSIZE.group (3)))
				return StencilProperty.getDomainMinIdentifier (CodeGeneratorUtil.getDimensionFromName (MATCHER_DOMAINSIZE.group (1)));
			if ("max".equals (MATCHER_DOMAINSIZE.group (3)))
				return StencilProperty.getDomainMaxIdentifier (CodeGeneratorUtil.getDimensionFromName (MATCHER_DOMAINSIZE.group (1)));
		}

		throw new NoSuchFieldException (StringUtil.concat (strProperty, " is no valid stencil property"));
	}

	/**
	 *
	 * @param nDimension
	 * @return
	 */
	public static NameID getDomainMinIdentifier (int nDimension)
	{
		return new NameID (StringUtil.concat (CodeGeneratorUtil.getDimensionName (nDimension), "_min"));
	}

	/**
	 *
	 * @param nDimension
	 * @return
	 */
	public static NameID getDomainMaxIdentifier (int nDimension)
	{
		return new NameID (StringUtil.concat (CodeGeneratorUtil.getDimensionName (nDimension), "_max"));
	}

	/**
	 * Returns the identifier for the maximum time placeholder.
	 * @return
	 */
	public static NameID getMaxTime ()
	{
		return new NameID ("t_max");
	}

	public static void main (String[] args) throws Exception
	{
		System.out.println (CodeGeneratorUtil.getDimensionFromName ("x"));
		System.out.println (CodeGeneratorUtil.getDimensionFromName ("y"));
		System.out.println (CodeGeneratorUtil.getDimensionFromName ("u"));
		System.out.println (CodeGeneratorUtil.getDimensionFromName ("x0"));
		System.out.println (CodeGeneratorUtil.getDimensionFromName ("x5"));

		System.out.println (StencilProperty.get ("x_min").getName ());
		System.out.println (StencilProperty.get ("w_min").getName ());
		System.out.println (StencilProperty.get ("x0_min").getName ());
		System.out.println (StencilProperty.get ("x6_min").getName ());
		System.out.println (StencilProperty.get ("x10_min").getName ());
		System.out.println (StencilProperty.get ("x5_max").getName ());
		System.out.println (StencilProperty.get ("bb_lx").getName ());
		System.out.println (StencilProperty.get ("bb_uv").getName ());
		System.out.println (StencilProperty.get ("bb_uu").getName ());
		System.out.println (StencilProperty.get ("bb_lx4").getName ());
		System.out.println (StencilProperty.get ("bb_lx12").getName ());
		//System.out.println (StencilProperty.get ("bc_lx12").getName ());
		//System.out.println (StencilProperty.get ("bb_x8").getName ());
		//System.out.println (StencilProperty.get ("bb_vx").getName ());
		System.out.println (StencilProperty.get ("bb_ly2").getName ());
	}
}
