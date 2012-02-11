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

import cetus.hir.AnnotationStatement;
import cetus.hir.CodeAnnotation;
import cetus.hir.IntegerLiteral;
import cetus.hir.NameID;
import cetus.hir.Specifier;
import ch.unibas.cs.hpwc.patus.arch.TypeBaseIntrinsicEnum;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 * Defines HIR objects for globals (global functions and variables).
 * @author Matthias-M. Christen
 */
public class Globals
{
	///////////////////////////////////////////////////////////////////
	// Constants

	public static final IntegerLiteral ZERO = new IntegerLiteral (0);
	public static final IntegerLiteral ONE = new IntegerLiteral (1);
	
	public static final Specifier[] BASE_DATATYPES = new Specifier[] { Specifier.FLOAT, Specifier.DOUBLE };


	// ----------------------------------------------------------------
	// Strategy Code

	public static final NameID FNX_STENCIL = new NameID ("stencil");
	public static final NameID FNX_INITIALIZE = new NameID ("initialize");


	// ----------------------------------------------------------------
	// Generated Code

	public static final NameID NUMBER_OF_THREADS = new NameID (TypeBaseIntrinsicEnum.NUMTHREADS.value ());
	public static final NameID THREAD_NUMBER = new NameID (TypeBaseIntrinsicEnum.THREADID.value ());

	public static final NameID FNX_BARRIER = new NameID (TypeBaseIntrinsicEnum.BARRIER.value ());
	public static final NameID FNX_MIN = new NameID (TypeBaseIntrinsicEnum.MIN.value ());
	public static final NameID FNX_MAX = new NameID (TypeBaseIntrinsicEnum.MAX.value ());
	public static final NameID FNX_FMA = new NameID (TypeBaseIntrinsicEnum.FMA.value ());
	public static final NameID FNX_FMS = new NameID (TypeBaseIntrinsicEnum.FMS.value ());
	public static final NameID FNX_MALLOC = new NameID (TypeBaseIntrinsicEnum.MALLOC.value ());

	public static final Specifier SPECIFIER_INDEX = Specifier.INT;
	public static final Specifier SPECIFIER_SIZE = Specifier.INT;


	public static final AnnotationStatement ANNOTATION_DISPAYPERFORMANCE_START = new AnnotationStatement (new CodeAnnotation ("#ifdef DISPLAY_PERFORMANCE"));
	public static final AnnotationStatement ANNOTATION_DISPAYPERFORMANCE_END = new AnnotationStatement (new CodeAnnotation ("#endif"));


	///////////////////////////////////////////////////////////////////
	// Implementation

	public static NameID getInitializeFunction (String strStencilName, boolean bMakeFortranCompatible)
	{
		String strInitializeFnxName = StringUtil.concat (FNX_INITIALIZE.getName (), "_", strStencilName);
		if (bMakeFortranCompatible)
			return new NameID (Globals.createFortranName (strInitializeFnxName));
		return new NameID (strInitializeFnxName);
	}

	public static String createFortranName (String strStencilKernelName)
	{
		return StringUtil.concat (strStencilKernelName, "_");
	}
	
	/**
	 * Determines whether the specifier <code>specType</code> is a base type specifier.
	 * @param specType The specifier to examine
	 * @return <code>true</code> iff the specifier <code>specType</code> is a base data type specifier
	 */
	public static boolean isBaseDatatype (Specifier specType)
	{
		for (Specifier s : BASE_DATATYPES)
			if (s.equals (specType))
				return true;
		return false;
	}
}
