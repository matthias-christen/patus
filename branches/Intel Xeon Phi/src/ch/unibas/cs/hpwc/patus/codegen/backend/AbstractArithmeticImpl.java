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
package ch.unibas.cs.hpwc.patus.codegen.backend;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.log4j.Logger;

import cetus.hir.ArrayAccess;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.Expression;
import cetus.hir.FunctionCall;
import cetus.hir.NameID;
import cetus.hir.Specifier;
import cetus.hir.UnaryExpression;
import cetus.hir.UnaryOperator;
import ch.unibas.cs.hpwc.patus.arch.IArchitectureDescription;
import ch.unibas.cs.hpwc.patus.arch.TypeArchitectureType.Intrinsics.Intrinsic;
import ch.unibas.cs.hpwc.patus.arch.TypeBaseIntrinsicEnum;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;
import ch.unibas.cs.hpwc.patus.codegen.Globals;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.Argument;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.Arguments;
import ch.unibas.cs.hpwc.patus.util.CodeGeneratorUtil;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 * Provides a default implementation for arithmetic operations for
 * the architecture-specific code generation.
 * The default implementation checks the XML hardware specification for
 * intrinsics and uses them as function calls. If no intrinsics are
 * provided, the standard operators are used.
 *
 * @author Matthias-M. Christen
 */
public abstract class AbstractArithmeticImpl implements IArithmetic
{
	///////////////////////////////////////////////////////////////////
	// Constants

	private final static Logger LOGGER = Logger.getLogger (AbstractArithmeticImpl.class);


	///////////////////////////////////////////////////////////////////
	// Member Variables

	protected CodeGeneratorSharedObjects m_data;


	///////////////////////////////////////////////////////////////////
	// Implementation

	public AbstractArithmeticImpl (CodeGeneratorSharedObjects data)
	{
		m_data = data;
	}

	@Override
	public Expression createExpression (Expression exprIn, Specifier specDatatype, boolean bVectorize)
	{
		// get hardware description and data type of the expression
		IArchitectureDescription hw = m_data.getArchitectureDescription ();

		// if no SIMD (i.e. the SIMD vector length is 1) there's nothing to be done
		if (hw.getSIMDVectorLength (specDatatype) == 1 || !bVectorize)
			return exprIn;

		if (exprIn instanceof UnaryExpression)
			return generateUnaryExpression ((UnaryExpression) exprIn, specDatatype, bVectorize);
		if (exprIn instanceof BinaryExpression)
			return generateBinaryExpression ((BinaryExpression) exprIn, specDatatype, bVectorize);
		if (exprIn instanceof FunctionCall)
			return generateFunctionCall ((FunctionCall) exprIn, specDatatype, bVectorize);
		if (exprIn instanceof ArrayAccess)
			return new ArrayAccess (
				createExpression (((ArrayAccess) exprIn).getArrayName (), specDatatype, bVectorize),
				((ArrayAccess) exprIn).getIndices ());
		return exprIn.clone ();
	}

	/**
	 * Invokes the method named <code>strMethod</code> (which must return an
	 * {@link Expression})
	 * with the arguments <code>rgObjArgs</code> on <code>this</code> object.
	 * 
	 * @param strMethod
	 *            The method to invoke
	 * @param rgObjArgs
	 *            The arguments
	 * @return The resulting expression from the method invocation, or
	 *         <code>null</code> if something went wrong
	 */
	private Expression invoke (String strMethod, Object... rgObjArgs)
	{
		// we only accept invocations to "internal" functions, which all have the signature { Expression }+ Specifier Boolean
		if (rgObjArgs.length < 3)
			return null;
		if (!(rgObjArgs[rgObjArgs.length - 2] instanceof Specifier))
			return null;
		if (!(rgObjArgs[rgObjArgs.length - 1] instanceof Boolean))
			return null;
		for (int i = 0; i < rgObjArgs.length - 2; i++)
			if (!(rgObjArgs[i] instanceof Expression))
				return null;

		Expression exprResult = null;
		if (strMethod != null)
		{
			Method m = null;
			try
			{
				Class<?>[] rgArgClasses = new Class<?>[rgObjArgs.length];
				for (int i = 0; i < rgObjArgs.length - 2; i++)
					rgArgClasses[i] = Expression.class;
				rgArgClasses[rgObjArgs.length - 2] = Specifier.class;
				rgArgClasses[rgObjArgs.length - 1] = boolean.class;

				m = IBackend.class.getMethod (strMethod, rgArgClasses);
			}
			catch (SecurityException e)
			{
			}
			catch (NoSuchMethodException e)
			{
			}
			if (m != null)
			{
				try
				{
					exprResult = (Expression) m.invoke (m_data.getCodeGenerators ().getBackendCodeGenerator (), rgObjArgs);
				}
				catch (IllegalArgumentException e)
				{
				}
				catch (IllegalAccessException e)
				{
				}
				catch (InvocationTargetException e)
				{
				}
			}
		}

		return exprResult;
	}

	/**
	 * Replace a unary expression.
	 * @param ue The unary expression to replace
	 * @param specDatatype The datatype of the expression
	 * @return
	 */
	private Expression generateUnaryExpression (UnaryExpression ue, Specifier specDatatype, boolean bVectorize)
	{
		String strMethod = null;
		if (ue.getOperator ().equals (UnaryOperator.PLUS))
			strMethod = "unary_plus";
		else if (ue.getOperator ().equals (UnaryOperator.MINUS))
			strMethod = "unary_minus";

		// calculate
		Expression exprResult = invoke (strMethod, ue.getExpression (), specDatatype, bVectorize);
		return exprResult == null ? ue : exprResult;
	}

	private Expression internalGenerateUnaryExpression (UnaryOperator op, Expression expr, Specifier specDatatype, boolean bVectorize)
	{
		Intrinsic intrinsic = m_data.getArchitectureDescription ().getIntrinsic (op, specDatatype);
		Expression exprNew = createExpression (expr, specDatatype, bVectorize);
		return intrinsic != null ? new FunctionCall (new NameID (intrinsic.getName ()), CodeGeneratorUtil.expressions (exprNew)) : new UnaryExpression (op, exprNew);
	}

	/**
	 * Replace a binary expression.
	 * @param be
	 * @param specDatatype
	 * @param bCanSwapWith
	 * @return
	 */
	private Expression generateBinaryExpression (BinaryExpression be, Specifier specDatatype, boolean bVectorize)
	{
		String strMethod = null;
		if (be.getOperator ().equals (BinaryOperator.ADD))
			strMethod = TypeBaseIntrinsicEnum.PLUS.value ();
		else if (be.getOperator ().equals (BinaryOperator.SUBTRACT))
			strMethod = TypeBaseIntrinsicEnum.MINUS.value ();
		else if (be.getOperator ().equals (BinaryOperator.MULTIPLY))
			strMethod = TypeBaseIntrinsicEnum.MULTIPLY.value ();
		else if (be.getOperator ().equals (BinaryOperator.DIVIDE))
			strMethod = TypeBaseIntrinsicEnum.DIVIDE.value ();

		// calculate
		Expression exprResult = invoke (strMethod, be.getLHS (), be.getRHS (), specDatatype, bVectorize);
		return exprResult == null ? be : exprResult;
	}

	private Expression internalGenerateBinaryExpression (BinaryOperator op, Expression expr1, Expression expr2, Specifier specDatatype, boolean bVectorize)
	{
		Intrinsic intrinsic = m_data.getArchitectureDescription ().getIntrinsic (op, specDatatype);
		Expression exprNew1 = createExpression (expr1, specDatatype, bVectorize);
		Expression exprNew2 = createExpression (expr2, specDatatype, bVectorize);

		return intrinsic != null ?
			new FunctionCall (new NameID (intrinsic.getName ()), CodeGeneratorUtil.expressions (exprNew1, exprNew2)) :
			new BinaryExpression (exprNew1, op, exprNew2);
	}

	/**
	 * Replace a function call.
	 * @param fc
	 * @param specDatatSpecifier
	 * @param bCanSwapWith
	 * @return
	 */
	@SuppressWarnings("unchecked")
	private Expression generateFunctionCall (FunctionCall fc, Specifier specDatatype, boolean bVectorize)
	{
		// try to calculate with class methods
		Object[] rgArgs = new Object[fc.getArguments ().size () + 2];
		fc.getArguments ().toArray (rgArgs);
		rgArgs[rgArgs.length - 2] = specDatatype;
		rgArgs[rgArgs.length - 1] = bVectorize;

		Expression exprResult = invoke (fc.getName ().toString (), rgArgs);
		if (exprResult == null)
		{
			Intrinsic intrinsic = m_data.getArchitectureDescription ().getIntrinsic (fc, specDatatype);
			if (intrinsic != null)
			{
				List<Expression> listArgs = new ArrayList<> (fc.getArguments ().size ());
				for (Expression exprArg : (List<Expression>) fc.getArguments ())
					listArgs.add (createExpression (exprArg.clone (), specDatatype, bVectorize));

				exprResult = new FunctionCall (new NameID (intrinsic.getName ()), listArgs);
			}
		}

		return exprResult == null ? fc : exprResult;
	}

//	private static int findIndex (String[] rgHaystack, String strNeedle)
//	{
//		for (int i = 0; i < rgHaystack.length; i++)
//			if (rgHaystack[i].equals (strNeedle))
//				return i;
//		return -1;
//	}

	private Expression internalGenerateFunctionCall (String strFunctionName, Specifier specDatatype, boolean bVectorize, Expression[] rgArguments, String[] rgArgNames)
	{
		Intrinsic intrinsic = m_data.getArchitectureDescription ().getIntrinsic (new FunctionCall (new NameID (strFunctionName)), specDatatype);

		// create the list of arguments; permute according to the definition of the intrinsic's "argument" attribute
		//List<Expression> listArgs = new ArrayList<> (rgArguments.length);
		Expression[] rgIntrinsicArgExprs = new Expression[rgArguments.length];

		boolean bArgsFilled = false;
		if (intrinsic != null && intrinsic.getArguments () != null && !"".equals (intrinsic.getArguments ()))
		{
			if (rgArgNames == null)
			{
				LOGGER.warn (StringUtil.concat ("No argument names provided in the code generator for the function ",
					strFunctionName, ", but argument names are given in the architecture description"));
			}
			else
			{
				Argument[] rgIntrinsicArgs = Arguments.parseArguments (intrinsic.getArguments ());
				for (int i = 0; i < rgArguments.length; i++)
				{
					Argument arg = Arguments.getNamedArgument (rgIntrinsicArgs, rgArgNames[i]);
					if (arg == null)
					{
						LOGGER.error (StringUtil.concat ("The argument provided '", rgArgNames[i],
							"' defined in the architecture description doesn't match any of the function arguments. Admissible function arguments are: ",
							Arrays.toString (rgArgNames)));
					}
					else
						rgIntrinsicArgExprs[arg.getNumber ()] = createExpression (rgArguments[i], specDatatype, bVectorize);
				}
				
//				// match the arguments
//				for (String strExpectedArg : intrinsic.getArguments ().split (","))
//				{
//					// find the index
//					int nIdx = AbstractArithmeticImpl.findIndex (rgArgNames, strExpectedArg);
//					if (nIdx < 0)
//					{
//						LOGGER.error (StringUtil.concat ("The expected argument '", strExpectedArg,
//							"' defined in the architecture description doesn't match any of the function arguments. Admissible function arguments are: ",
//							Arrays.toString (rgArgNames)));
//					}
//					else
//						listArgs.add (createExpression (rgArguments[nIdx], specDatatype, bVectorize));
//				}

				bArgsFilled = true;
			}
		}

		if (!bArgsFilled)
		{
			for (int i = 0; i < rgArguments.length; i++)
				rgIntrinsicArgExprs[i] = createExpression (rgArguments[i], specDatatype, bVectorize);
//			for (Expression exprArg : rgArguments)
//				listArgs.add (createExpression (exprArg, specDatatype, bVectorize));
		}

		return new FunctionCall (intrinsic != null ? new NameID (intrinsic.getName ()) : new NameID (strFunctionName), Arrays.asList (rgIntrinsicArgExprs));
	}

	@Override
	public Expression unary_plus (Expression expr, Specifier spec, boolean bVectorize)
	{
		// unary plus does nothing...
		return expr;
	}

	@Override
	public Expression unary_minus (Expression expr, Specifier spec, boolean bVectorize)
	{
		// return null to tell that the method is not implement specifically, and the default implementation
		// (using the XML hardware configuration) is to be used

		return internalGenerateUnaryExpression (UnaryOperator.MINUS, expr, spec, bVectorize);
	}


	@Override
	public Expression plus (Expression expr1, Expression expr2, Specifier spec, boolean bVectorize)
	{
		return internalGenerateBinaryExpression (BinaryOperator.ADD, expr1, expr2, spec, bVectorize);
	}

	@Override
	public Expression minus (Expression expr1, Expression expr2, Specifier spec, boolean bVectorize)
	{
		return internalGenerateBinaryExpression (BinaryOperator.SUBTRACT, expr1, expr2, spec, bVectorize);
	}

	@Override
	public Expression multiply (Expression expr1, Expression expr2, Specifier spec, boolean bVectorize)
	{
		return internalGenerateBinaryExpression (BinaryOperator.MULTIPLY, expr1, expr2, spec, bVectorize);
	}

	@Override
	public Expression divide (Expression expr1, Expression expr2, Specifier spec, boolean bVectorize)
	{
		return internalGenerateBinaryExpression (BinaryOperator.DIVIDE, expr1, expr2, spec, bVectorize);
	}

	@Override
	public Expression sqrt (Expression expr, Specifier spec, boolean bVectorize)
	{
		return internalGenerateFunctionCall ("sqrt", spec, bVectorize, new Expression[] { expr }, null);
	}

	@Override
	public Expression fma (Expression exprSummand, Expression exprFactor1, Expression exprFactor2, Specifier spec, boolean bVectorize)
	{
		return internalGenerateFunctionCall (
			Globals.FNX_FMA.getName (),
			spec,
			bVectorize,
			new Expression[] { exprSummand, exprFactor1, exprFactor2 },
			Globals.getIntrinsicArguments (TypeBaseIntrinsicEnum.FMA)
		);
	}

	@Override
	public Expression fms (Expression exprSummand, Expression exprFactor1, Expression exprFactor2, Specifier spec, boolean bVectorize)
	{
		return internalGenerateFunctionCall (
			Globals.FNX_FMS.getName (),
			spec,
			bVectorize,
			new Expression[] { exprSummand, exprFactor1, exprFactor2 },
			Globals.getIntrinsicArguments (TypeBaseIntrinsicEnum.FMS)
		);
	}
	
	@Override
	public Expression vector_reduce_sum (Expression expr, Specifier specDatatype)
	{
		return internalGenerateFunctionCall (Globals.FNX_VECTOR_REDUCE_SUM.getName (), specDatatype, true, new Expression[] { expr }, null);
	}

	@Override
	public Expression vector_reduce_product (Expression expr, Specifier specDatatype)
	{
		return internalGenerateFunctionCall (Globals.FNX_VECTOR_REDUCE_PRODUCT.getName (), specDatatype, true, new Expression[] { expr }, null);
	}

	@Override
	public Expression vector_reduce_min (Expression expr, Specifier specDatatype)
	{
		return internalGenerateFunctionCall (Globals.FNX_VECTOR_REDUCE_MIN.getName (), specDatatype, true, new Expression[] { expr }, null);
	}

	@Override
	public Expression vector_reduce_max (Expression expr, Specifier specDatatype)
	{
		return internalGenerateFunctionCall (Globals.FNX_VECTOR_REDUCE_MAX.getName (), specDatatype, true, new Expression[] { expr }, null);
	}
}
