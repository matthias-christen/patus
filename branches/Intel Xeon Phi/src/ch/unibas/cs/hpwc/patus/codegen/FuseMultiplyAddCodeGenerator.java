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

import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.BreadthFirstIterator;
import cetus.hir.Expression;
import cetus.hir.FunctionCall;
import cetus.hir.Specifier;
import cetus.hir.Traversable;
import ch.unibas.cs.hpwc.patus.arch.TypeBaseIntrinsicEnum;
import ch.unibas.cs.hpwc.patus.util.CodeGeneratorUtil;

/**
 * This class replaces subexpressions that can be computed by a fused
 * multiply-add with a call to <code>fma</code>.
 *
 * @author Matthias-M. Christen
 */
public class FuseMultiplyAddCodeGenerator
{
	///////////////////////////////////////////////////////////////////
	// Member Variables

	private CodeGeneratorSharedObjects m_data;
	
	private Map<Specifier, Boolean> m_mapHasFMA;
	private Map<Specifier, Boolean> m_mapHasFMS;


	///////////////////////////////////////////////////////////////////
	// Implementation

	public FuseMultiplyAddCodeGenerator (CodeGeneratorSharedObjects data)
	{
		m_data = data;

		m_mapHasFMA = new HashMap<> ();
		m_mapHasFMS = new HashMap<> ();
	}
	
	private boolean hasFMA (Specifier specDatatype)
	{
		return hasIntrinsic (TypeBaseIntrinsicEnum.FMA.value (), specDatatype, m_mapHasFMA);
	}
	
	private boolean hasFMS (Specifier specDatatype)
	{
		return hasIntrinsic (TypeBaseIntrinsicEnum.FMS.value (), specDatatype, m_mapHasFMS);		
	}
	
	private boolean hasIntrinsic (String strFnx, Specifier specDatatype, Map<Specifier, Boolean> map)
	{
		Boolean bHasIntrinsic = map.get (specDatatype);
		if (bHasIntrinsic != null)
			return bHasIntrinsic;
		
		boolean bHasIntr = m_data.getArchitectureDescription ().getIntrinsic (strFnx, specDatatype) != null;
		map.put (specDatatype, bHasIntr);
		return bHasIntr;
	}
	
	private Expression createFMA (boolean bIsAdd, Expression exprSummand, Expression exprFactor1, Expression exprFactor2,
		Specifier specDatatype, boolean bResolveToIntrinsics)
	{
		if (bIsAdd)
		{
			// fused multiply-add
			
			if (hasFMA (specDatatype))
			{
				if (bResolveToIntrinsics)
				{
					return m_data.getCodeGenerators ().getBackendCodeGenerator ().fma (
						exprSummand.clone (), exprFactor1.clone (), exprFactor2.clone (), specDatatype, false);
				}
				
				// don't resolve to an intrinsic; use the generic function
				return new FunctionCall (
					Globals.FNX_FMA.clone (),
					CodeGeneratorUtil.expressions (exprSummand.clone (), exprFactor1.clone (), exprFactor2.clone ())
				);
			}
			
			// no FMA intrinsic defined
			return null;
		}
		else
		{
			// fused multiply-subtract

			if (hasFMS (specDatatype))
			{
				if (bResolveToIntrinsics)
					return m_data.getCodeGenerators ().getBackendCodeGenerator ().fms (
						exprSummand.clone (), exprFactor1.clone (), exprFactor2.clone (), specDatatype, false);
				
				// don't resolve to an intrinsic; use the generic function
				return new FunctionCall (
					Globals.FNX_FMS.clone (),
					CodeGeneratorUtil.expressions (exprSummand.clone (), exprFactor1.clone (), exprFactor2.clone ()));
			}
			
			// no FMS intrinsic defined
			return null;
		}
	}
	
	
	/**
	 * Creates a new expression in which additions and multiplies are replaced
	 * by a call to a &qout;fused multiply-add&quot; whenever possible.
	 * 
	 * @param expression
	 *            The expression in which to substitute multiply and add
	 *            operations by a fused multiply-add
	 * @param specDatatype
	 *            The datatype of the expression
	 * @param bResolveToIntrinsics
	 *            Specifies whether FMA/FMS calls are to be resolved to
	 *            intrinsics.
	 *            If set to <code>true</code>, the intrinsic as defined in the
	 *            architecture description will be used.
	 *            If set to <code>false</code>, a generic function call named
	 *            {@link TypeBaseIntrinsicEnum#FMA} or
	 *            {@link TypeBaseIntrinsicEnum#FMS}, respectively, with
	 *            arguments ("summand", "factor1", "factor2") (as defined in
	 *            {@link Globals}) will be used.
	 * @return A new expression containing fused multiply-adds
	 */
	public Expression applyFMAs (Expression expression, Specifier specDatatype, boolean bResolveToIntrinsics)
	{
		if (!hasFMA (specDatatype) && !hasFMS (specDatatype))
			return expression;
		
		boolean bFMAFound = false;
		Expression exprNew = expression.clone ();

		for (BreadthFirstIterator it = new BreadthFirstIterator (exprNew); it.hasNext (); )
		{
			Traversable tvbTop = (Traversable) it.next ();

			// search subtrees
			if (tvbTop instanceof BinaryExpression)
			{
				BinaryExpression bexprTop = (BinaryExpression) tvbTop;

				// check whether the operation of the top node is an add
				// (if not, we can't apply a FMA to this node...)
				boolean bIsAdd = BinaryOperator.ADD.equals (bexprTop.getOperator ());
				boolean bIsSubtract = BinaryOperator.SUBTRACT.equals (bexprTop.getOperator ());
				if (bIsAdd || bIsSubtract)
				{
					// check whether the FMA is applicable with the left branch being a multiply
					if (bexprTop.getLHS () instanceof BinaryExpression)
					{
						BinaryExpression bexprLeft = (BinaryExpression) bexprTop.getLHS ();
						if (BinaryOperator.MULTIPLY.equals (bexprLeft.getOperator ()))
						{
							// conditions are met: create the FMA call
							Expression exprFMA = createFMA (bIsAdd, bexprTop.getRHS (), bexprLeft.getLHS (), bexprLeft.getRHS (), specDatatype, bResolveToIntrinsics);

							if (exprFMA != null)
							{
								if (bexprTop == exprNew)
									exprNew = exprFMA;
								else
									bexprTop.swapWith (exprFMA);

								bFMAFound = true;
								break;
							}
						}
					}

					// if we come here, the left branch wasn't a multiply
					// check whether the right one is
					// note that this version only works for addition
					if ((bexprTop.getRHS () instanceof BinaryExpression) && bIsAdd)
					{
						BinaryExpression bexprRight = (BinaryExpression) bexprTop.getRHS ();
						if (BinaryOperator.MULTIPLY.equals (bexprRight.getOperator ()))
						{
							Expression exprFMA = createFMA (bIsAdd, bexprTop.getLHS (), bexprRight.getLHS (), bexprRight.getRHS (), specDatatype, bResolveToIntrinsics);

							if (exprFMA != null)
							{
								if (bexprTop == exprNew)
									exprNew = exprFMA;
								else
									bexprTop.swapWith (exprFMA);

								bFMAFound = true;
								break;
							}							
						}
					}
				}
			}
		}

		// recursively apply if an FMA has been found and replaced
		if (bFMAFound)
			return applyFMAs (exprNew, specDatatype, bResolveToIntrinsics);

		return exprNew;
	}


	///////////////////////////////////////////////////////////////////
	// Testing

//	public static void main (String[] args) throws Exception
//	{
//		Expression e = ExpressionParser.parseExpression ("a*T*T+b*T+c+e*U+f*V");
//		System.out.println (e);
//		System.out.println (FuseMultiplyAddCodeGenerator.applyFMAs (e));
//	}
}
