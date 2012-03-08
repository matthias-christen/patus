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

import java.util.HashMap;
import java.util.Map;

import cetus.hir.AnnotationStatement;
import cetus.hir.BinaryOperator;
import cetus.hir.CodeAnnotation;
import cetus.hir.IntegerLiteral;
import cetus.hir.NameID;
import cetus.hir.Specifier;
import cetus.hir.UnaryOperator;
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
	
	private static final Map<Object, TypeBaseIntrinsicEnum> MAP_INTRINSICS = new HashMap<Object, TypeBaseIntrinsicEnum> ();
	private static final Map<TypeBaseIntrinsicEnum, String[]> MAP_INTRINSICPARAMS = new HashMap<TypeBaseIntrinsicEnum, String[]> ();
	
	/**
	 * Generic &quot;left hand side&quot; argument
	 */
	public static final String ARGNAME_LHS = "lhs";
	
	/**
	 * Generic &quot;right hand side&quot; argument
	 */
	public static final String ARGNAME_RHS = "rhs";


	///////////////////////////////////////////////////////////////////
	// Implementation
	
	static
	{
		// operator names / operators  --->  TypeBaseIntrinsicEnum
		MAP_INTRINSICS.put ("+", TypeBaseIntrinsicEnum.PLUS);
		MAP_INTRINSICS.put ("-", TypeBaseIntrinsicEnum.MINUS);
		MAP_INTRINSICS.put ("*", TypeBaseIntrinsicEnum.MULTIPLY);
		MAP_INTRINSICS.put ("/", TypeBaseIntrinsicEnum.DIVIDE);
		
		MAP_INTRINSICS.put (BinaryOperator.ADD, TypeBaseIntrinsicEnum.PLUS);
		MAP_INTRINSICS.put (BinaryOperator.SUBTRACT, TypeBaseIntrinsicEnum.MINUS);
		MAP_INTRINSICS.put (BinaryOperator.MULTIPLY, TypeBaseIntrinsicEnum.MULTIPLY);
		MAP_INTRINSICS.put (BinaryOperator.DIVIDE, TypeBaseIntrinsicEnum.DIVIDE);
		
		MAP_INTRINSICS.put (UnaryOperator.PLUS, TypeBaseIntrinsicEnum.UNARY_PLUS);
		MAP_INTRINSICS.put (UnaryOperator.MINUS, TypeBaseIntrinsicEnum.UNARY_MINUS);
		
		// base intrinsic arguments 
		String[] rgDefaultBinary = new String[] { Globals.ARGNAME_LHS, Globals.ARGNAME_RHS };
		MAP_INTRINSICPARAMS.put (TypeBaseIntrinsicEnum.PLUS, rgDefaultBinary);
		MAP_INTRINSICPARAMS.put (TypeBaseIntrinsicEnum.MINUS, rgDefaultBinary);
		MAP_INTRINSICPARAMS.put (TypeBaseIntrinsicEnum.MULTIPLY, rgDefaultBinary);
		MAP_INTRINSICPARAMS.put (TypeBaseIntrinsicEnum.DIVIDE, rgDefaultBinary);
		
		String[] rgFMAParam = new String[] { "summand", "factor1", "factor2" };
		MAP_INTRINSICPARAMS.put (TypeBaseIntrinsicEnum.FMA, rgFMAParam);
		MAP_INTRINSICPARAMS.put (TypeBaseIntrinsicEnum.FMS, rgFMAParam);
	}

	/**
	 * Returns a {@link NameID} for the initialization function.
	 * @param strStencilName The name of the stencil
	 * @param bMakeFortranCompatible Flag indicating whether to make the initialization function name
	 * 	compatible with Fortran
	 * @return A {@link NameID} for the initialization function
	 */
	public static NameID getInitializeFunction (String strStencilName, boolean bMakeFortranCompatible)
	{
		String strInitializeFnxName = StringUtil.concat (FNX_INITIALIZE.getName (), "_", strStencilName);
		if (bMakeFortranCompatible)
			return new NameID (Globals.createFortranName (strInitializeFnxName));
		return new NameID (strInitializeFnxName);
	}

	/**
	 * Creates a Fortran-compatible name.
	 * @param strStencilKernelName The original function name
	 * @return A Fortran-compatible version of the original function name <code>strStencilKernelName</code>
	 */
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

	/**
	 * Returns the {@link TypeBaseIntrinsicEnum} corresponding to <code>strOperationOrBaseName</code>
	 * or <code>null</code> if no corresponding intrinsic exists.
	 * @param strOperationOrBaseName The operation as a string or the intrinsic's base name
	 * @return
	 */
	public static TypeBaseIntrinsicEnum getIntrinsicBase (String strOperationOrBaseName)
	{
		return MAP_INTRINSICS.get (strOperationOrBaseName);
	}

	public static TypeBaseIntrinsicEnum getIntrinsicBase (UnaryOperator op)
	{
		return MAP_INTRINSICS.get (op);
	}

	public static TypeBaseIntrinsicEnum getIntrinsicBase (BinaryOperator op)
	{
		return MAP_INTRINSICS.get (op);
	}
	
	/**
	 * Returns the expected list of arguments for the base intrinsic <code>t</code>.
	 * The list returned by this function is to be converted to the actual argument list
	 * required by the intrinsic as defined in the architecture description (<code>architecture.xml</code>).
	 * @param t
	 * @return
	 */
	public static String[] getIntrinsicArguments (TypeBaseIntrinsicEnum t)
	{
		return MAP_INTRINSICPARAMS.get (t);
	}
}
