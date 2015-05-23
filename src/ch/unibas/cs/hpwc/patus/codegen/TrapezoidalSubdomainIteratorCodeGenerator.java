/*******************************************************************************
 * Copyright (c) 2015 Matthias-M. Christen, University of Lugano, Switzerland.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the GNU Lesser Public License v2.1
 * which accompanies this distribution, and is available at
 * http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
 * 
 * Contributors:
 *     Matthias-M. Christen, University of Lugano, Switzerland - initial API and implementation
 ******************************************************************************/
package ch.unibas.cs.hpwc.patus.codegen;

import cetus.hir.AssignmentExpression;
import cetus.hir.AssignmentOperator;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.CompoundStatement;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.ForLoop;
import cetus.hir.Identifier;
import cetus.hir.IntegerLiteral;
import cetus.hir.Statement;
import cetus.hir.Traversable;
import ch.unibas.cs.hpwc.patus.analysis.StrategyAnalyzer;
import ch.unibas.cs.hpwc.patus.ast.RangeIterator;
import ch.unibas.cs.hpwc.patus.ast.StatementList;
import ch.unibas.cs.hpwc.patus.ast.StatementListBundle;
import ch.unibas.cs.hpwc.patus.ast.SubdomainIterator;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;
import ch.unibas.cs.hpwc.patus.codegen.options.StencilLoopUnrollingConfiguration;
import ch.unibas.cs.hpwc.patus.geometry.Box;
import ch.unibas.cs.hpwc.patus.geometry.Size;
import ch.unibas.cs.hpwc.patus.symbolic.Symbolic;
import ch.unibas.cs.hpwc.patus.util.ExpressionUtil;

/**
 * Implements the code generator for subdomain iterators, also handling
 * synchronous or asynchronous data transfers.
 *
 * @author Matthias-M. Christen
 */
public class TrapezoidalSubdomainIteratorCodeGenerator extends AbstractSubdomainIteratorCodeGenerator
{
	///////////////////////////////////////////////////////////////////
	// Constants and Static Types

	//private final static Logger LOGGER = Logger.getLogger (TrapezoidalSubdomainIteratorCodeGenerator.class);


	///////////////////////////////////////////////////////////////////
	// Inner Types

	protected class CodeGenerator extends AbstractSubdomainIteratorCodeGenerator.CodeGenerator
	{
		///////////////////////////////////////////////////////////////////
		// Member Variables


		///////////////////////////////////////////////////////////////////
		// Implementation

		public CodeGenerator (SubdomainIterator sgit, CodeGeneratorRuntimeOptions options)
		{
			super (sgit, options);
		}

		/**
		 * Gets the maximum unrolling factors per dimension (depending on the domain of the iterator).
		 */
		public void initializeMaximumUnrollingFactors ()
		{
			byte nDimensionality = m_sdIterator.getDomainIdentifier ().getDimensionality ();
			Size sizeDomain = m_sdIterator.getDomainIdentifier ().getSubdomain ().getSize ();

			m_rgMaxUnrollingFactorPerDimension = new int[nDimensionality];

			Expression exprCoord = sizeDomain.getCoord (0);
			if (ExpressionUtil.isNumberLiteral (exprCoord))
				m_rgMaxUnrollingFactorPerDimension[0] = ExpressionUtil.getIntegerValue (exprCoord);
			else
				m_rgMaxUnrollingFactorPerDimension[0] = StencilLoopUnrollingConfiguration.NO_UNROLLING_LIMIT;

			// don't allow unrolling in y, z, ... dimensions
			for (int i = 1; i < nDimensionality; i++)
				m_rgMaxUnrollingFactorPerDimension[i] = 1;
		}
		
		/**
		 * Creates the lower loop bound for the <code>for</code> loop in
		 * dimension <code>nDim</code>.
		 * 
		 * @param nDim
		 *            The dimension
		 * @return The lower loop bound
		 */
		public Expression getLowerLoopBound (int nDim)
		{
			return m_data.getData().getGeneratedIdentifiers().getDimensionMinIdentifier(m_sdIterator.getIterator(), nDim).clone();
		}

		/**
		 * Creates the upper loop bound for the <code>for</code> loop in
		 * dimension <code>nDim</code>.
		 * 
		 * @param nDim
		 *            The dimension
		 * @return The upper loop bound
		 */
		public Expression getUpperLoopBound (int nDim)
		{
			return m_data.getData().getGeneratedIdentifiers().getDimensionMaxIdentifier(m_sdIterator.getIterator(), nDim).clone();
		}
		
		private Statement generateAssignMin (int nDim)
		{
			return new ExpressionStatement(new AssignmentExpression(
				m_data.getData().getGeneratedIdentifiers().getDimensionMinIdentifier (m_sdIterator.getIterator (), nDim).clone (),
				AssignmentOperator.NORMAL,
				generateBoundInitialization (nDim, 0)
			));
		}

		private Statement generateAssignMax (int nDim)
		{
			return new ExpressionStatement(new AssignmentExpression(
				m_data.getData().getGeneratedIdentifiers().getDimensionMaxIdentifier (m_sdIterator.getIterator (), nDim).clone (),
				AssignmentOperator.NORMAL,
				generateBoundInitialization (nDim, 1)
			));
		}

		/**
		 * var x_start = xmin[0] + t * s_xt_a + (z - xmin[2]) * s_xz_a + (y_start - xmin[1]) * s_xy_a;
		 * 
		 * @param nDim
		 * @param nIdx
		 * @return
		 */
		private Expression generateBoundInitialization (int nDim, int nIdx)
		{
			byte nDimensionality = m_data.getStencilCalculation().getDimensionality();
			Box box = m_sdIterator.getDomainIdentifier ().getSubdomain ().getBox ();
			
			RangeIterator temporalIterator = m_data.getCodeGenerators().getStrategyAnalyzer().getEnclosingTemporalLoop(m_sdIterator);
			
			Expression exprResult = new BinaryExpression(
				nIdx == 0 ?
					box.getMin ().getCoord (nDim).clone () :
					box.getMax ().getCoord (nDim).clone (),
				BinaryOperator.ADD,
				new BinaryExpression(
					temporalIterator.getLoopIndex().clone(), //m_idTemporalIdx.clone(),
					BinaryOperator.MULTIPLY,
					m_rgSlopes[nDim][nDimensionality - nDim - 1][nIdx].clone()
				)
			);	
			
			for (int i = nDimensionality - 1; i > nDim; i--)
			{
				exprResult = new BinaryExpression(
					exprResult,
					BinaryOperator.ADD,
					new BinaryExpression(
						new BinaryExpression(
							i == nDim + 1 ?
								m_data.getData ().getGeneratedIdentifiers ().getDimensionMinIdentifier(m_sdIterator.getIterator (), i).clone() :
								m_data.getData ().getGeneratedIdentifiers ().getDimensionIndexIdentifier (m_sdIterator.getIterator (), i).clone(),
							BinaryOperator.SUBTRACT,
							nIdx == 0 ?
								box.getMin ().getCoord (i).clone () :
								box.getMax ().getCoord (i).clone ()
						),
						BinaryOperator.MULTIPLY,
						m_rgSlopes[nDim][i - nDim - 1][nIdx].clone()
					)
				);
			}
			
			return exprResult;
		}
		
		private Statement generateIncrementMin (int nDim)
		{
			Identifier idMin = m_data.getData ().getGeneratedIdentifiers ().getDimensionMinIdentifier (m_sdIterator.getIterator (), nDim).clone ();
			return new ExpressionStatement (new AssignmentExpression(
				idMin,
				AssignmentOperator.ADD,
				m_rgSlopes[nDim][0][0].clone ()
			));
		}
		
		private Statement generateIncrementMax (int nDim)
		{
			Identifier idMax = m_data.getData ().getGeneratedIdentifiers ().getDimensionMaxIdentifier (m_sdIterator.getIterator (), nDim).clone ();
			return new ExpressionStatement (new AssignmentExpression(
				idMax,
				AssignmentOperator.ADD,
				m_rgSlopes[nDim][0][1].clone ()
			));
		}

		/**
		 * Creates a single {@link ForLoop} corresponding to dimension
		 * <code>nDim</code> of the {@link SubdomainIterator} and adds it to the
		 * <code>cmpstmtOuter</code> statement.
		 * 
		 * @param nDim
		 *            The dimension for which to create the {@link ForLoop}
		 * @param slGenerated
		 *            The {@link StatementList} to which the generated
		 *            {@link ForLoop} will be added
		 * @param exprStartOffset
		 *            The offset to the start value for the loop index. If set
		 *            to {@link SubdomainIteratorCodeGenerator#NULL_EXPRESSION},
		 *            the initialization statement will be omitted
		 * @param exprNegEndOffset
		 *            The negative offset to the end value for the loop index
		 * @param bIsOutermostLoopOfInnerNest
		 *            Specifies whether the loop for this dimension is the
		 *            outer-most loop of the &quot;regular&quot; loop nest
		 *            structure and subsequentially will be treated as reference
		 *            for unrolling (if the nest contains a stencil call)
		 * 
		 * @return The generated loop or <code>null</code> if no loop was
		 *         created for the dimension
		 */
		public StatementListBundle generateIteratorForDimension (
			int nDim, StatementListBundle slGenerated, Expression exprStartOffset, Expression exprNegEndOffset, int nUnrollingFactor)
		{
			// prepare loop creation
			Identifier idIdx = m_data.getData ().getGeneratedIdentifiers ().getDimensionIndexIdentifier (m_sdIterator.getIterator (), nDim);

			Expression exprStart = exprStartOffset == TrapezoidalSubdomainIteratorCodeGenerator.NULL_EXPRESSION ? null : getLowerLoopBound (nDim);
			if (exprStartOffset != null && exprStartOffset != TrapezoidalSubdomainIteratorCodeGenerator.NULL_EXPRESSION)
				exprStart = new BinaryExpression (exprStart, BinaryOperator.ADD, exprStartOffset);

			Expression exprEnd = getUpperLoopBound (nDim);
			if (exprNegEndOffset != null && !ExpressionUtil.isZero (exprNegEndOffset))
				exprEnd = Symbolic.simplify (new BinaryExpression (exprEnd, BinaryOperator.SUBTRACT, exprNegEndOffset));

			// account for loop unrolling
			Expression exprMainLoopStep = new IntegerLiteral (nUnrollingFactor);

			// account for SIMD
			int nSIMDVectorLength = m_data.getCodeGenerators ().getStencilCalculationCodeGenerator ().getLcmSIMDVectorLengths ();
			if (m_bContainsStencilCall && nDim == 0 && (isStencilCalculation () || m_data.getOptions ().useNativeSIMDDatatypes ()))
				exprMainLoopStep = ExpressionUtil.product (exprMainLoopStep.clone (), new IntegerLiteral (nSIMDVectorLength));

			// list to which the loop statements will be added
			StatementListBundle slbStatements = new StatementListBundle ();
			
			// add the pre-loop statements
			if (nDim > 0)
			{
				slbStatements.addStatement (generateAssignMin (nDim - 1));
				slbStatements.addStatement (generateAssignMax (nDim - 1));
			}

			// create the loops
			CompoundStatement cmpstmtMainLoopBody = new CompoundStatement ();
			if (hasSIMDPrologueAndEpilogueLoops (nDim))
				createLoopWithProAndEpi (slbStatements, idIdx, exprStart, exprEnd, exprMainLoopStep, cmpstmtMainLoopBody);
			else
			{
				// create the main loop if the dimension of the domain is > 1
				createDefaultLoop (slbStatements, idIdx, exprStart, exprEnd, exprMainLoopStep, cmpstmtMainLoopBody);
			}

			// add the post-loop statements
			slbStatements.addStatement (generateIncrementMin (nDim));
			slbStatements.addStatement (generateIncrementMax (nDim));
			
			return slbStatements;
		}
	}


	///////////////////////////////////////////////////////////////////
	// Member Variables

	private Identifier[][][] m_rgSlopes;


	///////////////////////////////////////////////////////////////////
	// Implementation

	public TrapezoidalSubdomainIteratorCodeGenerator (final ICodeGenerator cgStencilCodeGenerator, CodeGeneratorSharedObjects data, Identifier[][][] rgSlopes)
	{
		super (new ICodeGenerator()
		{
			@Override
			public StatementListBundle generate(Traversable trvInput, CodeGeneratorRuntimeOptions options)
			{
				if (!StrategyAnalyzer.isStencilCall (trvInput))
					throw new RuntimeException ("The trapezoidal subdomain iterator doesn't contain a stencil call");
				
				Expression exprStencil = null;
				if (trvInput instanceof Expression)
					exprStencil = (Expression) trvInput;
				else if (trvInput instanceof ExpressionStatement)
					exprStencil = ((ExpressionStatement) trvInput).getExpression();

				if (exprStencil == null)
					throw new RuntimeException ("No stencil call found directly in the trapezoidal subdomain iterator");

				return cgStencilCodeGenerator.generate (exprStencil.clone(), options);
			}
		}, data);
		
		m_rgSlopes = rgSlopes;
	}
	
	@Override
	protected CodeGenerator createCodeGenerator (Traversable trvInput, CodeGeneratorRuntimeOptions options)
	{
		return new CodeGenerator ((SubdomainIterator) trvInput, options);
	}
}
