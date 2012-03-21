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

import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.BreadthFirstIterator;
import cetus.hir.Expression;
import cetus.hir.Specifier;
import cetus.hir.Traversable;
import ch.unibas.cs.hpwc.patus.arch.IArchitectureDescription;
import ch.unibas.cs.hpwc.patus.arch.TypeBaseIntrinsicEnum;

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


	///////////////////////////////////////////////////////////////////
	// Implementation

	public FuseMultiplyAddCodeGenerator (CodeGeneratorSharedObjects data)
	{
		m_data = data;
	}
	
	/**
	 * Creates a new expression in which additions and multiplies are replaced
	 * by a call to a &qout;fused multiply-add&quot; whenever possible.
	 * 
	 * @param expression
	 *            The expression in which to substitute multiply and add
	 *            operations by a fused multiply-add
	 * @return A new expression containing fused multiply-adds
	 */
	public Expression applyFMAs (Expression expression, Specifier specDatatype)
	{
		IArchitectureDescription arch = m_data.getArchitectureDescription ();
		boolean bHasFMA = arch.getIntrinsic (TypeBaseIntrinsicEnum.FMA.value (), specDatatype) != null;
		boolean bHasFMS = arch.getIntrinsic (TypeBaseIntrinsicEnum.FMS.value (), specDatatype) != null;
		if (!bHasFMA && !bHasFMS)
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

							Expression exprFMA = bIsAdd ?
								(bHasFMA ?
									m_data.getCodeGenerators ().getBackendCodeGenerator ().fma (
										bexprTop.getRHS ().clone (), bexprLeft.getLHS ().clone (), bexprLeft.getRHS ().clone (), specDatatype, false) :
									bexprTop
								) :
								(bHasFMS ?
									m_data.getCodeGenerators ().getBackendCodeGenerator ().fms (
										bexprTop.getRHS ().clone (), bexprLeft.getLHS ().clone (), bexprLeft.getRHS ().clone (), specDatatype, false) :
									bexprTop
								);

							if (bexprTop == exprNew)
								exprNew = exprFMA;
							else
								bexprTop.swapWith (exprFMA);

							bFMAFound = true;
							break;
						}
					}

					// if we come here, the left branch wasn't a multiply
					// check whether the right one is
					if (bexprTop.getRHS () instanceof BinaryExpression)
					{
						BinaryExpression bexprRight = (BinaryExpression) bexprTop.getRHS ();
						if (BinaryOperator.MULTIPLY.equals (bexprRight.getOperator ()))
						{
							Expression exprFMA = bIsAdd ?
								(bHasFMA ?
									m_data.getCodeGenerators ().getBackendCodeGenerator ().fma (
										bexprTop.getLHS ().clone (), bexprRight.getLHS ().clone (), bexprRight.getRHS ().clone (), specDatatype, false) :
									bexprTop
								) :
								(bHasFMS ?
									m_data.getCodeGenerators ().getBackendCodeGenerator ().fms (
										bexprTop.getLHS ().clone (), bexprRight.getLHS ().clone (), bexprRight.getRHS ().clone (), specDatatype, false) :
									bexprTop
								);

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

		// recursively apply if an FMA has been found and replaced
		if (bFMAFound)
			return applyFMAs (exprNew, specDatatype);

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
