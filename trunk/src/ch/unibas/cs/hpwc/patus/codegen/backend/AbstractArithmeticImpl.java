package ch.unibas.cs.hpwc.patus.codegen.backend;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;

import cetus.hir.ArrayAccess;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.Expression;
import cetus.hir.FunctionCall;
import cetus.hir.IDExpression;
import cetus.hir.NameID;
import cetus.hir.Specifier;
import cetus.hir.UnaryExpression;
import cetus.hir.UnaryOperator;
import ch.unibas.cs.hpwc.patus.arch.IArchitectureDescription;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;
import ch.unibas.cs.hpwc.patus.util.CodeGeneratorUtil;

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
	 * Invokes the method named <code>strMethod</code> (which must return an {@link Expression})
	 * with the arguments <code>rgObjArgs</code> on <code>this</code> object.
	 * @param strMethod The method to invoke
	 * @param rgObjArgs The arguments
	 * @return The resulting expression from the method invocation, or <code>null</code>
	 * 	if something went wrong
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
			strMethod = "plus";
		else if (ue.getOperator ().equals (UnaryOperator.MINUS))
			strMethod = "minus";

		// calculate
		Expression exprResult = invoke (strMethod, ue.getExpression (), specDatatype, bVectorize);
		return exprResult == null ? ue : exprResult;
	}

	private Expression internalGenerateUnaryExpression (UnaryOperator op, Expression expr, Specifier specDatatype, boolean bVectorize)
	{
		IDExpression opFnx = m_data.getArchitectureDescription ().getIntrinsicName (op, specDatatype);
		Expression exprNew = createExpression (expr, specDatatype, bVectorize);
		return opFnx != null ? new FunctionCall (opFnx, CodeGeneratorUtil.expressions (exprNew)) : new UnaryExpression (op, exprNew);
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
			strMethod = "add";
		else if (be.getOperator ().equals (BinaryOperator.SUBTRACT))
			strMethod = "subtract";
		else if (be.getOperator ().equals (BinaryOperator.MULTIPLY))
			strMethod = "multiply";
		else if (be.getOperator ().equals (BinaryOperator.DIVIDE))
			strMethod = "divide";

		// calculate
		Expression exprResult = invoke (strMethod, be.getLHS (), be.getRHS (), specDatatype, bVectorize);
		return exprResult == null ? be : exprResult;
	}

	private Expression internalGenerateBinaryExpression (BinaryOperator op, Expression expr1, Expression expr2, Specifier specDatatype, boolean bVectorize)
	{
		IDExpression opFnx = m_data.getArchitectureDescription ().getIntrinsicName (op, specDatatype);
		Expression exprNew1 = createExpression (expr1, specDatatype, bVectorize);
		Expression exprNew2 = createExpression (expr2, specDatatype, bVectorize);
		return opFnx != null ? new FunctionCall (opFnx.clone (), CodeGeneratorUtil.expressions (exprNew1, exprNew2)) : new BinaryExpression (exprNew1, op, exprNew2);
	}

	/**
	 * Replace a function call.
	 * @param fc
	 * @param specDatatSpecifier
	 * @param bCanSwapWith
	 * @return
	 */
	private Expression generateFunctionCall (FunctionCall fc, Specifier specDatatype, boolean bVectorize)
	{
		// try to calculate with class methods
		Expression exprResult = invoke (fc.getName ().toString (), fc.getArguments ().toArray (), bVectorize);
		return exprResult == null ? fc : exprResult;
	}

	private Expression internalGenerateFunctionCall (String strFunctionName, Specifier specDatatype, boolean bVectorize, Expression... rgArguments)
	{
		IDExpression op = m_data.getArchitectureDescription ().getIntrinsicName (new FunctionCall (new NameID (strFunctionName)), specDatatype);
		List<Expression> listArgs = new ArrayList<Expression> (rgArguments.length);
		for (Expression exprArg : rgArguments)
			listArgs.add (createExpression (exprArg, specDatatype, bVectorize));
		return new FunctionCall (op != null ? op : new NameID (strFunctionName), listArgs);
	}

	@Override
	public Expression plus (Expression expr, Specifier spec, boolean bVectorize)
	{
		// unary plus does nothing...
		return expr;
	}

	@Override
	public Expression minus (Expression expr, Specifier spec, boolean bVectorize)
	{
		// return null to tell that the method is not implement specifically, and the default implementation
		// (using the XML hardware configuration) is to be used

		return internalGenerateUnaryExpression (UnaryOperator.MINUS, expr, spec, bVectorize);
	}


	@Override
	public Expression add (Expression expr1, Expression expr2, Specifier spec, boolean bVectorize)
	{
		return internalGenerateBinaryExpression (BinaryOperator.ADD, expr1, expr2, spec, bVectorize);
	}

	@Override
	public Expression subtract (Expression expr1, Expression expr2, Specifier spec, boolean bVectorize)
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
		return internalGenerateFunctionCall ("sqrt", spec, bVectorize, expr);
	}


	@Override
	public Expression fma (Expression expr1, Expression expr2, Expression expr3, Specifier spec, boolean bVectorize)
	{
		return internalGenerateFunctionCall ("fma", spec, bVectorize, expr1, expr2, expr3);
	}
}
