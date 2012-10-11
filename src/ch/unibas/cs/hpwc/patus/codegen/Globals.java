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
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

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
	
	public static final int NUM_THREADS = Runtime.getRuntime ().availableProcessors ();
	public static final ExecutorService EXECUTOR_SERVICE = Executors.newFixedThreadPool (NUM_THREADS);
	

	public static final IntegerLiteral ZERO = new IntegerLiteral (0);
	public static final IntegerLiteral ONE = new IntegerLiteral (1);
	
	public static final Specifier[] BASE_DATATYPES = new Specifier[] { Specifier.FLOAT, Specifier.DOUBLE };


	// ----------------------------------------------------------------
	// Strategy Code

	public static final NameID FNX_STENCIL = new NameID ("stencil");
	public static final NameID FNX_INITIALIZE = new NameID ("initialize");
	
	/**
	 * The suffix appended to the parameterized initialization/stencil kernel functions
	 */
	public static final String PARAMETRIZED_FUNCTION_SUFFIX = "_parm";


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
	
	public static final NameID FNX_VECTOR_REDUCE_SUM = new NameID (TypeBaseIntrinsicEnum.VECTOR_REDUCE_SUM.value ());
	public static final NameID FNX_VECTOR_REDUCE_PRODUCT = new NameID (TypeBaseIntrinsicEnum.VECTOR_REDUCE_PRODUCT.value ());
	public static final NameID FNX_VECTOR_REDUCE_MIN = new NameID (TypeBaseIntrinsicEnum.VECTOR_REDUCE_MIN.value ());
	public static final NameID FNX_VECTOR_REDUCE_MAX = new NameID (TypeBaseIntrinsicEnum.VECTOR_REDUCE_MAX.value ());

	public static final Specifier SPECIFIER_INDEX = Specifier.INT;
	public static final Specifier SPECIFIER_SIZE = Specifier.INT;

	public static final AnnotationStatement ANNOTATION_DISPAYPERFORMANCE_START = new AnnotationStatement (new CodeAnnotation ("#ifdef DISPLAY_PERFORMANCE"));
	public static final AnnotationStatement ANNOTATION_DISPAYPERFORMANCE_END = new AnnotationStatement (new CodeAnnotation ("#endif"));
	
	private static final Map<Object, TypeBaseIntrinsicEnum> MAP_INTRINSICS = new HashMap<> ();
	private static final Map<String, String[]> MAP_INTRINSICPARAMS = new HashMap<> ();
	
	/**
	 * Generic &quot;left hand side&quot; argument
	 */
	public static final String ARGNAME_LHS = "lhs";
	
	/**
	 * Generic &quot;right hand side&quot; argument
	 */
	public static final String ARGNAME_RHS = "rhs";
	
	public static final String ARGNAME_RESULT = "=result";


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
		MAP_INTRINSICPARAMS.put (TypeBaseIntrinsicEnum.PLUS.value (), rgDefaultBinary);
		MAP_INTRINSICPARAMS.put (TypeBaseIntrinsicEnum.MINUS.value (), rgDefaultBinary);
		MAP_INTRINSICPARAMS.put (TypeBaseIntrinsicEnum.MULTIPLY.value (), rgDefaultBinary);
		MAP_INTRINSICPARAMS.put (TypeBaseIntrinsicEnum.DIVIDE.value (), rgDefaultBinary);
		
		String[] rgFMAParam = new String[] { "summand", "factor1", "factor2" };
		MAP_INTRINSICPARAMS.put (TypeBaseIntrinsicEnum.FMA.value (), rgFMAParam);
		MAP_INTRINSICPARAMS.put (TypeBaseIntrinsicEnum.FMS.value (), rgFMAParam);
	}

	/**
	 * Returns the name of the initialization function for the stencil named
	 * <code>strStencilName</code>.
	 * 
	 * @param strStencilName
	 *            The name of the stencil
	 * @return The name of the initialization function
	 */
	public static String getInitializeFunctionName (String strStencilName)
	{
		return StringUtil.concat (FNX_INITIALIZE.getName (), "_", strStencilName);
	}

	/**
	 * Creates a Fortran-compatible name.
	 * 
	 * @param strStencilKernelName
	 *            The original function name
	 * @return A Fortran-compatible version of the original function name
	 *         <code>strStencilKernelName</code>
	 */
	public static String createFortranName (String strStencilKernelName)
	{
		return StringUtil.concat (strStencilKernelName, "_");
	}
	
	/**
	 * Determines whether the specifier <code>specType</code> is a base type
	 * specifier.
	 * 
	 * @param specType
	 *            The specifier to examine
	 * @return <code>true</code> iff the specifier <code>specType</code> is a
	 *         base data type specifier
	 */
	public static boolean isBaseDatatype (Specifier specType)
	{
		for (Specifier s : BASE_DATATYPES)
			if (s.equals (specType))
				return true;
		return false;
	}

	/**
	 * Returns the {@link TypeBaseIntrinsicEnum} corresponding to
	 * <code>strOperationOrBaseName</code> or <code>null</code> if no
	 * corresponding intrinsic exists.
	 * 
	 * @param strOperationOrBaseName
	 *            The operation as a string or the intrinsic's base name
	 * @return The {@link TypeBaseIntrinsicEnum} for
	 *         <code>strOperationOrBaseName</code>
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
	 * Returns the expected list of arguments for the base intrinsic
	 * <code>t</code>. The list returned by this function is to be converted to
	 * the actual argument list required by the intrinsic as defined in the
	 * architecture description (<code>architecture.xml</code>).
	 * 
	 * @param t
	 *            The base intrinsic
	 * @return The argument list for the intrinsic <code>t</code>
	 */
	public static String[] getIntrinsicArguments (TypeBaseIntrinsicEnum t)
	{
		return MAP_INTRINSICPARAMS.get (t.value ());
	}
	
	/**
	 * Returns the expected list of arguments for the base intrinsic with name
	 * <code>strIntrinsicBaseName</code>. The list returned by this function is to be converted to
	 * the actual argument list required by the intrinsic as defined in the
	 * architecture description (<code>architecture.xml</code>).
	 * 
	 * @param t
	 *            The base intrinsic
	 * @return The argument list for the intrinsic <code>strIntrinsicBaseName</code>
	 */
	public static String[] getIntrinsicArguments (String strIntrinsicBaseName)
	{
		return MAP_INTRINSICPARAMS.get (strIntrinsicBaseName);
	}
	
	/**
	 * Determines whether the operation corresponding to the intrinsic with base
	 * name <code>strIntrinsicBaseName</code> is commutative.
	 * 
	 * @param strIntrinsicBaseName
	 *            The base name of the intrinsic
	 * @return <code>true</code> iff the operation corresponding to
	 *         <code>strIntrinsicBaseName</code> is commutative
	 */
	public static boolean isIntrinsicCommutative (String strIntrinsicBaseName)
	{
		return TypeBaseIntrinsicEnum.PLUS.value ().equals (strIntrinsicBaseName) ||
			TypeBaseIntrinsicEnum.MULTIPLY.value ().equals (strIntrinsicBaseName);
	}
	
	/**
	 * Determines whether the arguments with indices <code>nArg1Idx</code> and
	 * <code>nArg2Idx</code> can be swapped without changing the result of the
	 * computation.
	 * 
	 * @param strIntrinsicBaseName
	 *            The base name of the intrinsic
	 * @param nArg1Idx
	 *            The index of the one argument
	 * @param nArg2Idx
	 *            The index of another argument
	 * @return <code>true</code> iff it is legal to swap the arguments with
	 *         indices <code>nArg1Idx</code> and <code>nArg2Idx</code>
	 */
	public static boolean canSwapIntrinsicArguments (String strIntrinsicBaseName, int nArg1Idx, int nArg2Idx)
	{
		if (nArg1Idx == nArg2Idx)
			return true;
		
		if ((nArg1Idx == 0 || nArg1Idx == 1) && (nArg2Idx == 0 || nArg2Idx == 1))
		{
			if (Globals.isIntrinsicCommutative (strIntrinsicBaseName))
				return true;
		}
		
		if (TypeBaseIntrinsicEnum.FMA.equals (strIntrinsicBaseName) || TypeBaseIntrinsicEnum.FMS.equals (strIntrinsicBaseName))
			return (nArg1Idx == 1 || nArg1Idx == 2) && (nArg2Idx == 1 || nArg2Idx == 2);
		
		return false;
	}
}
