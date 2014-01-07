package ch.unibas.cs.hpwc.patus.grammar.stencil2;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import cetus.hir.ArrayAccess;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.DepthFirstIterator;
import cetus.hir.Expression;
import cetus.hir.FloatLiteral;
import cetus.hir.IDExpression;
import cetus.hir.IntegerLiteral;
import cetus.hir.Literal;
import cetus.hir.NameID;
import ch.unibas.cs.hpwc.patus.codegen.CodeGenerationOptions;
import ch.unibas.cs.hpwc.patus.representation.StencilCalculation;
import ch.unibas.cs.hpwc.patus.symbolic.ExpressionData;
import ch.unibas.cs.hpwc.patus.symbolic.Symbolic;
import ch.unibas.cs.hpwc.patus.util.ExpressionUtil;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

public class ArithmeticUtil
{
	private CodeGenerationOptions m_options;
	private Errors m_errors;

	
	public ArithmeticUtil (CodeGenerationOptions options, Errors errors)
	{
		m_options = options;
		m_errors = errors;
	}
	
	@SuppressWarnings("static-method")
	public Literal createLiteral (double fValue, boolean bIsIntegerLiteral)
	{
		return bIsIntegerLiteral ? new IntegerLiteral ((long) fValue) : new FloatLiteral (fValue);
	}
	
	public Literal getConstantValue (String strIdentifier, Map<String, Literal> mapConstants)
	{
		if (mapConstants == null)
			return null;
	       
		Literal litValue = mapConstants.get (strIdentifier);
		return litValue == null ? null : litValue.clone ();
	}
	
	/**
	 * Create a balanced sum expression.
	 */
	public ExpressionData sum (List<ExpressionData> listSummands, boolean bIsInteger)
	{
		List<ExpressionData> listSummandsSimplified = new ArrayList<> (listSummands.size ());
		double fSum = 0;
		for (ExpressionData expr : listSummands)
		{
			if (ExpressionUtil.isNumberLiteral (expr.getExpression ()))
				fSum += ExpressionUtil.getFloatValue (expr.getExpression ());
			else
				listSummandsSimplified.add (expr);
		}
		
		ExpressionData exprExplicitSum = new ExpressionData (createLiteral (fSum, bIsInteger), 0, Symbolic.EExpressionType.EXPRESSION);
		if (listSummandsSimplified.size () == 0)
			return exprExplicitSum;
		
		if (fSum != 0)
			listSummandsSimplified.add (exprExplicitSum);
			
		if (m_options.getBalanceBinaryExpressions ())
			return balancedBinaryExpression (listSummandsSimplified, BinaryOperator.ADD);
			
		// don't balance expressions
		return leftToRightBinaryExpression (listSummandsSimplified, BinaryOperator.ADD);
	}
	
	/**
	 * Create a balanced product expression.
	 */
	public ExpressionData product (List<ExpressionData> listFactors, boolean bIsInteger)
	{
		List<ExpressionData> listFactorsSimplified = new ArrayList<> (listFactors.size ());
		double fProduct = 1;
		for (ExpressionData expr : listFactors)
		{
			if (ExpressionUtil.isNumberLiteral (expr.getExpression ()))
				fProduct *= ExpressionUtil.getFloatValue (expr.getExpression ());
			else
				listFactorsSimplified.add (expr);
		}
		
		ExpressionData exprExplicitProduct = new ExpressionData (createLiteral (fProduct, bIsInteger), 0, Symbolic.EExpressionType.EXPRESSION);
		if (listFactorsSimplified.size () == 0)
			return exprExplicitProduct;
		
		if (fProduct != 1)
			listFactorsSimplified.add (exprExplicitProduct);
			
		if (m_options.getBalanceBinaryExpressions ())
			return balancedBinaryExpression (listFactorsSimplified, BinaryOperator.MULTIPLY);

		// don't balance expressions
		return leftToRightBinaryExpression (listFactorsSimplified, BinaryOperator.MULTIPLY);

	}
	
	public ExpressionData balancedBinaryExpression (List<ExpressionData> listOperands, BinaryOperator op)
	{
		if (listOperands.size () == 0)
			return new ExpressionData (new IntegerLiteral (0), 0, Symbolic.EExpressionType.EXPRESSION);
		if (listOperands.size () == 1)
			return listOperands.get (0);
			
		ExpressionData exprLeft = balancedBinaryExpression (listOperands.subList (0, listOperands.size () / 2), op);
		ExpressionData exprRight = balancedBinaryExpression (listOperands.subList (listOperands.size () / 2, listOperands.size ()), op);

		return new ExpressionData (
			new BinaryExpression (exprLeft.getExpression (), op, exprRight.getExpression ()),
			exprLeft.getFlopsCount () + 1 + exprRight.getFlopsCount (),
			Symbolic.EExpressionType.EXPRESSION);
	}
	
	@SuppressWarnings("static-method")
	public ExpressionData leftToRightBinaryExpression (List<ExpressionData> listOperands, BinaryOperator op)
	{
		Expression exprSum = null;
		int nFlops = 0;
		for (ExpressionData expr : listOperands)
		{
			if (exprSum == null)
				exprSum = expr.getExpression ();
			else
			{
				exprSum = new BinaryExpression (exprSum.clone (), op, expr.getExpression ());
				nFlops++;
			}
			nFlops += expr.getFlopsCount ();
		}
		
		return new ExpressionData (exprSum, nFlops, Symbolic.EExpressionType.EXPRESSION);	
	}
	
	public ExpressionData subtract (ExpressionData expr1, ExpressionData expr2, boolean bIsInteger)
	{
		if (ExpressionUtil.isNumberLiteral (expr1.getExpression ()) && ExpressionUtil.isNumberLiteral (expr2.getExpression ()))
		{
			return new ExpressionData (
				createLiteral (ExpressionUtil.getFloatValue (expr1.getExpression ()) - ExpressionUtil.getFloatValue (expr2.getExpression ()), bIsInteger),
				0,
				Symbolic.EExpressionType.EXPRESSION);
		}
			
		return new ExpressionData (
			new BinaryExpression (expr1.getExpression (), BinaryOperator.SUBTRACT, expr2.getExpression ()),
			expr1.getFlopsCount () + 1 + expr2.getFlopsCount (),
			Symbolic.EExpressionType.EXPRESSION);
	}
	
	public ExpressionData divide (ExpressionData expr1, ExpressionData expr2, boolean bIsInteger)
	{
		if (ExpressionUtil.isNumberLiteral (expr1.getExpression ()) && ExpressionUtil.isNumberLiteral (expr2.getExpression ()))
		{
			return new ExpressionData (
				createLiteral (ExpressionUtil.getFloatValue (expr1.getExpression ()) / ExpressionUtil.getFloatValue (expr2.getExpression ()), bIsInteger),
				0,
				Symbolic.EExpressionType.EXPRESSION);
		}
			
		return new ExpressionData (
			new BinaryExpression (expr1.getExpression (), BinaryOperator.DIVIDE, expr2.getExpression ()),
			expr1.getFlopsCount () + 1 + expr2.getFlopsCount (),
			Symbolic.EExpressionType.EXPRESSION);
	}
	
	public ExpressionData modulus (ExpressionData expr1, ExpressionData expr2, boolean bIsInteger)
	{
		if (ExpressionUtil.isNumberLiteral (expr1.getExpression ()) && ExpressionUtil.isNumberLiteral (expr2.getExpression ()))
		{
			return new ExpressionData (
				createLiteral (ExpressionUtil.getIntegerValue (expr1.getExpression ()) % ExpressionUtil.getIntegerValue (expr2.getExpression ()), bIsInteger),
				0,
				Symbolic.EExpressionType.EXPRESSION);
		}
			
		return new ExpressionData (
			new BinaryExpression (expr1.getExpression (), BinaryOperator.MODULUS, expr2.getExpression ()),
			expr1.getFlopsCount () + 1 + expr2.getFlopsCount (),
			Symbolic.EExpressionType.EXPRESSION);
	}

	/**
	 * Determines whether the function named <code>strFunctionName</code> with the arguments <code>listArgs</code>
	 * is a compile time reduction (i.e., one that gets expanded at compile time, e.g.
	 * <code>{ i=-1..1, j=-1..1, k=-1..1 } sum(U[x+i, y+j, z+k])</code>
	 * (which is a short way of writing <code>U[x-1,y-1,z-1]+U[x,y-1,z-1]+U[x+1,y-1,z-1]+U[x-1,y,z-1]+...</code>)
	 */
	public boolean isCompileTimeReduction (String strFunctionName, List<Expression> listArgs, LocalVars lv, Token la)
	{
		if ((!"sum".equals (strFunctionName) && !"product".equals (strFunctionName)) || lv == null)
			return false;
			
		if (listArgs.size () > 1)
			m_errors.SemErr (la.line, la.col, "Recuction functions require exactly one argument.");
			
		// check whether the local vars appear in the argument list
		return lv.containsLocalVariable (listArgs.get (0));
	}
	
	public ExpressionData expandCompileTimeReduction (String strFunctionName, List<Expression> listArgs, LocalVars lv, Token la)
	{
		BinaryOperator op = null;
		int nUnit = 0;
		if ("sum".equals (strFunctionName))
			op = BinaryOperator.ADD;
		else if ("product".equals (strFunctionName))
		{
			op = BinaryOperator.MULTIPLY;
			nUnit = 1;
		}
		else
			m_errors.SemErr (la.line, la.col, StringUtil.concat ("\"", strFunctionName, "\" is not a reduction function."));

		Expression exprProto = listArgs.get (0);
		Expression expr = null;
		
		int nFlopsCount = 0;
		for (ExpressionData ed : lv.expand (exprProto))
		{
			// check for unity elements
			if (ed.getExpression () instanceof IntegerLiteral)
			{
				// nothing to do if the term is 0 (for sums) or 1 (for products)
				if (((IntegerLiteral) ed.getExpression ()).getValue () == nUnit)
					continue;
			}

			// add the one flop for the reduction if not the first term of the sum/product
			if (expr != null)
				nFlopsCount++;
			nFlopsCount += ed.getFlopsCount ();

			// build the expression
			expr = expr == null ? ed.getExpression () : new BinaryExpression (expr, op, ed.getExpression ());
		}
					
		return new ExpressionData (expr, nFlopsCount, Symbolic.EExpressionType.EXPRESSION);
	}
	
	/**
	 * Converts all the {@link IntegerLiteral}s in the
	 * <code>listExpressions</code> to <code>int</code>s or returns
	 * <code>null</code> if not all entries are {@link IntegerLiteral}s.
	 * 
	 * @param listExpressions
	 *            The list to convert to an array of <code>int</code>s
	 * @return The values of the {@link IntegerLiteral}s in
	 *         <code>listExpressions</code> as an array of <code>int</code>s or
	 *         <code>null</code> if not all entries are {@link IntegerLiteral}s
	 */
	@SuppressWarnings("static-method")
	public int[] asIntArray (List<Expression> listExpressions)
	{
		int[] rgValues = new int[listExpressions.size ()];
		int i = 0;
		
		for (Expression expr : listExpressions)
		{
			Expression exprResult = cetus.hir.Symbolic.simplify (expr);
			if (!(exprResult instanceof IntegerLiteral))
				return null;
			rgValues[i++] = (int) ((IntegerLiteral) exprResult).getValue ();
			
			// this does not simplify integer quotients
			//Integer n = ExpressionUtil.getIntegerValueEx (expr);
			//if (n == null)
			//	return null;
			//rgValues[i++] = n;
		}
		
		return rgValues;
	}
	
	@SuppressWarnings("static-method")
	public String getIndexedIdentifier (String strId, int[] rgIdx)
	{
		StringBuilder sb = new StringBuilder (strId);
		for (int nIdx : rgIdx)
		{
			sb.append ('_');
			if (nIdx < 0)
				sb.append ('m');
			sb.append (Math.abs (nIdx));
		}
		
		return sb.toString ();
	}
	
	public ExpressionData replaceIndexedScalars (ExpressionData expr, Map<String, StencilCalculation.ParamType> mapScalars, Map<String, Literal> mapConstants, Token la)
	{
		return new ExpressionData (replaceIndexedScalars (expr.getExpression (), mapScalars, mapConstants, la), expr.getFlopsCount (), expr.getType ());
	}

	public Expression replaceIndexedScalars (Expression expr, Map<String, StencilCalculation.ParamType> mapScalars, Map<String, Literal> mapConstants, Token la)
	{
		Expression exprResult = expr.clone ();
		for (DepthFirstIterator it = new DepthFirstIterator (exprResult); it.hasNext (); )
		{
			Object o = it.next ();
			
			if (o instanceof ArrayAccess)
			{
				ArrayAccess aa = (ArrayAccess) o;
				Expression exprName = aa.getArrayName ();
				if (exprName instanceof IDExpression)
				{
					String strName = ((IDExpression) exprName).getName ();
					if (mapScalars.containsKey (strName))
					{
						int[] rgIdx = asIntArray (aa.getIndices ());
						if (rgIdx != null)
						{
							String strSubstName = getIndexedIdentifier (strName, rgIdx);
							if (!mapScalars.containsKey (strSubstName))
								throwOutOfBoundsError (aa, strName, rgIdx, mapScalars, la);
							
							Literal litValue = getConstantValue (strSubstName, mapConstants);
							Expression exprReplacement = litValue != null ? litValue : new NameID (strSubstName);
							
							if (exprResult == o)
								return exprReplacement;
							else
								aa.swapWith (exprReplacement);
						}
					}
				}
			}
		}
		
		return exprResult;
	}
	
	private void throwOutOfBoundsError (Expression expr, String strBaseName, int[] rgIdx, Map<String, StencilCalculation.ParamType> mapScalars, Token la)
	{
		StencilCalculation.ParamType pt = mapScalars.get (strBaseName);
		
		int i = 0;
		for (Range range : pt.getRanges ())
		{
			if (rgIdx[i] < range.getStart () || rgIdx[i] > range.getEnd ())
				m_errors.SemErr (la.line, la.col, StringUtil.concat ("Index in dimension ", i, " of \"", expr.toString (), "\" out of bounds: should be in ", range.toString (), "."));
			i++;
		}		
	}
}
