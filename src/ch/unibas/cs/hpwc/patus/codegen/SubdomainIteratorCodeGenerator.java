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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import org.apache.log4j.Logger;

import cetus.hir.AnnotationStatement;
import cetus.hir.AssignmentExpression;
import cetus.hir.AssignmentOperator;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.CompoundStatement;
import cetus.hir.DepthFirstIterator;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.ForLoop;
import cetus.hir.IDExpression;
import cetus.hir.Identifier;
import cetus.hir.IntegerLiteral;
import cetus.hir.NameID;
import cetus.hir.NullStatement;
import cetus.hir.SizeofExpression;
import cetus.hir.Statement;
import cetus.hir.Traversable;
import cetus.hir.Typecast;
import cetus.hir.UnaryExpression;
import cetus.hir.UnaryOperator;
import cetus.hir.UserSpecifier;
import cetus.hir.VariableDeclaration;
import cetus.hir.VariableDeclarator;
import ch.unibas.cs.hpwc.patus.analysis.StrategyAnalyzer;
import ch.unibas.cs.hpwc.patus.ast.IStatementList;
import ch.unibas.cs.hpwc.patus.ast.IndexBoundsCalculationInsertionAnnotation;
import ch.unibas.cs.hpwc.patus.ast.Parameter;
import ch.unibas.cs.hpwc.patus.ast.StatementList;
import ch.unibas.cs.hpwc.patus.ast.StatementListBundle;
import ch.unibas.cs.hpwc.patus.ast.SubdomainIdentifier;
import ch.unibas.cs.hpwc.patus.ast.SubdomainIterator;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;
import ch.unibas.cs.hpwc.patus.codegen.options.StencilLoopUnrollingConfiguration;
import ch.unibas.cs.hpwc.patus.geometry.Size;
import ch.unibas.cs.hpwc.patus.representation.StencilNode;
import ch.unibas.cs.hpwc.patus.symbolic.Symbolic;
import ch.unibas.cs.hpwc.patus.util.CodeGeneratorUtil;
import ch.unibas.cs.hpwc.patus.util.DomainPointEnumerator;
import ch.unibas.cs.hpwc.patus.util.ExpressionUtil;
import ch.unibas.cs.hpwc.patus.util.StatementListBundleUtil;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 * Implements the code generator for subdomain iterators, also handling
 * synchronous or asynchronous data transfers.
 *
 * @author Matthias-M. Christen
 */
public class SubdomainIteratorCodeGenerator implements ICodeGenerator
{
	///////////////////////////////////////////////////////////////////
	// Constants and Static Types

	private final static Logger LOGGER = Logger.getLogger (SubdomainIteratorCodeGenerator.class);

	/**
	 * If the start expression in
	 * {@link CodeGenerator#generateIteratorForDimension(int, StatementListBundle, Expression, Expression, int)}
	 * is set to {@link SubdomainIteratorCodeGenerator#NULL_EXPRESSION}, an empty statement is created for
	 * the initialization statement
	 */
	private final static Expression NULL_EXPRESSION = new NameID ("__null__");

	private final static String TAG_MAINLOOP = "mainloop";
	private final static String TAG_PROEPILOOP = "proepiloop";


	///////////////////////////////////////////////////////////////////
	// Inner Types

	private class CodeGenerator
	{
		///////////////////////////////////////////////////////////////////
		// Member Variables

		private CodeGeneratorRuntimeOptions m_options;

		/**
		 * The subdomain iterator for which code is generated
		 */
		private SubdomainIterator m_sdIterator;

		//private int m_nMaxTimestep;
		private int m_nReuseDim;

		private boolean m_bHasNestedLoops;
		private boolean m_bContainsStencilCall;
		private boolean m_bIsEligibleForStencilLoopUnrolling;
		private boolean m_bLoadData;

		private int[] m_rgMaxUnrollingFactorPerDimension;

		private Identifier m_idCounter;


		///////////////////////////////////////////////////////////////////
		// Implementation

		public CodeGenerator (SubdomainIterator sgit, CodeGeneratorRuntimeOptions options)
		{
			m_options = options;

			m_sdIterator = sgit;

			// calculated values...
			m_bHasNestedLoops = StrategyAnalyzer.hasNestedLoops (m_sdIterator);
			m_bContainsStencilCall = StrategyAnalyzer.directlyContainsStencilCall (m_sdIterator);

			// do loop unrolling? (only if we generate code for the stencil computation
			// and only if the structure is "eligible" (i.e., the loop contains exactly a stencil call))
			m_bIsEligibleForStencilLoopUnrolling = false;
			if (options.hasValue (CodeGeneratorRuntimeOptions.OPTION_STENCILCALCULATION, CodeGeneratorRuntimeOptions.VALUE_STENCILCALCULATION_STENCIL))
				m_bIsEligibleForStencilLoopUnrolling = StrategyAnalyzer.isEligibleForStencilLoopUnrolling (m_sdIterator);

			m_bLoadData = m_data.getCodeGenerators ().getStrategyAnalyzer ().isDataLoadedInIterator (m_sdIterator, m_data.getArchitectureDescription ());
//XXX
//m_bLoadData = false;
//m_bLoadData = true;

			// get the maximum unrolling factors
			initializeMaximumUnrollingFactors ();
		}

		/**
		 * Gets the maximum unrolling factors per dimension (depending on the domain of the iterator).
		 */
		private void initializeMaximumUnrollingFactors ()
		{
			byte nDimensionality = m_sdIterator.getDomainIdentifier ().getDimensionality ();
			Size sizeDomain = m_sdIterator.getDomainIdentifier ().getSubdomain ().getSize ();

			m_rgMaxUnrollingFactorPerDimension = new int[nDimensionality];

			for (int i = 0; i < nDimensionality; i++)
			{
				Expression exprCoord = sizeDomain.getCoord (i);
				if (ExpressionUtil.isNumberLiteral (exprCoord))
					m_rgMaxUnrollingFactorPerDimension[i] = ExpressionUtil.getIntegerValue (exprCoord);
				else
					m_rgMaxUnrollingFactorPerDimension[i] = StencilLoopUnrollingConfiguration.NO_UNROLLING_LIMIT;
			}
		}

		/**
		 * Generates the C code implementing the subdomain iterator.
		 */
		public StatementListBundle generate ()
		{
			if (m_bLoadData)
				return generateWithDataTransfers ();

			return generateInner (m_sdIterator.getIterator ().getDimensionality (), new StatementListBundle ());
		}
		
		private boolean isStencilCalculation ()
		{
			return m_options.hasValue (CodeGeneratorRuntimeOptions.OPTION_STENCILCALCULATION, CodeGeneratorRuntimeOptions.VALUE_STENCILCALCULATION_STENCIL);
		}

		/**
		 * Generates the C code for the loop nest.
		 * @param nStartDim
		 */
		private StatementListBundle generateInner (int nStartDim, StatementListBundle slbGeneratedParent)
		{
			Set<StencilLoopUnrollingConfiguration> setUnrollingConfigs = new HashSet<StencilLoopUnrollingConfiguration> ();
			if (m_bIsEligibleForStencilLoopUnrolling)
			{
				// create the unrolling configurations
				DomainPointEnumerator dpe = new DomainPointEnumerator ();
				for (int i = 0; i < nStartDim; i++)
					dpe.addDimension (0, m_data.getOptions ().getUnrollingFactors ().length - 1);

				for (int[] rgUnrollingIndices : dpe)
				{
					StencilLoopUnrollingConfiguration config = new StencilLoopUnrollingConfiguration ();
					for (int i = 0; i < rgUnrollingIndices.length; i++)
					{
						config.setUnrollingForDimension (
							i,
							m_data.getOptions ().getUnrollingFactors ()[rgUnrollingIndices[i]],
							m_rgMaxUnrollingFactorPerDimension[i]);
					}

					setUnrollingConfigs.add (config);
				}
			}
			else
			{
				// loop is not eligible for unrolling: add a non-unroll configuration
				setUnrollingConfigs.add (new StencilLoopUnrollingConfiguration ());
			}

			Parameter param = new Parameter (StringUtil.concat ("_unroll_", m_sdIterator.getIterator ().getName ()));

			// create a code branch for each unrolling configuration
			for (StencilLoopUnrollingConfiguration config : setUnrollingConfigs)
			{
				// generate the code for the loop nest
				StatementListBundle slbLoopNest = new StatementListBundle ();
				slbLoopNest.addStatementAtTop (new AnnotationStatement (new IndexBoundsCalculationInsertionAnnotation (m_sdIterator)));

				recursiveGenerateInner (slbLoopNest, nStartDim - 1, false, config);

				// TODO: is this correct? what if there are more than 1 codes with new params?
				if (slbLoopNest.size () == 1)
					slbGeneratedParent.addStatement (slbLoopNest.getDefault (), param, config.toInteger ());
				else
					StatementListBundleUtil.addToLoopBody (slbGeneratedParent, slbLoopNest);
			}

			return slbGeneratedParent;
		}

		/**
		 *
		 * @param slbParent
		 * @param nDimension
		 * @param bHasParentLoop Flag indicating whether the parent structure is a loop
		 * 	(to whose body the current structure will be added); if the parent is not a
		 *  loop (<code>bHasParentLoop == false</code>), the current structure is just
		 *  appended to the parent structure
		 * @param config
		 */
		private void recursiveGenerateInner (StatementListBundle slbParent, int nDimension, boolean bHasParentLoop, StencilLoopUnrollingConfiguration config)
		{
			if (nDimension == 0 && isAssemblyUsedForInnerMost ())
			{
				// this is an innermost loop (containing a stencil computation), and an assembly code
				// generator, which generates code for the innermost loop and the contained stencil expression,
				// was specified
				
				generateInnerMostAssembly ();
			}
			else
				recursiveGenerateInnerDefault (slbParent, nDimension, bHasParentLoop, config);
		}
			
		private void recursiveGenerateInnerDefault (StatementListBundle slbParent, int nDimension, boolean bHasParentLoop, StencilLoopUnrollingConfiguration config)
		{			
			// do we need prologue and epilogue loops?
			// we don't need a clean up loop in the inner most loop if we have prologue and epilogue loops (bUseNativeSIMD == false)
			boolean bHasProEpiLoops =
				m_data.getArchitectureDescription ().useSIMD () && !m_data.getOptions ().useNativeSIMDDatatypes () && isStencilCalculation ();
			boolean bHasCleanupLoops = nDimension > 0 || (nDimension == 0 && !bHasProEpiLoops);

			if (nDimension >= 0)
			{
				int nUnrollFactor = config.getUnrollingFactor (nDimension);

				// unrolled loop
				StatementListBundle slbInnerLoop = generateIteratorForDimension (
					nDimension,
					slbParent,
					null,
					bHasCleanupLoops ? new IntegerLiteral (nUnrollFactor - 1) : null,
					nUnrollFactor
				);

				recursiveGenerateInner (
					slbInnerLoop == null ? slbParent : slbInnerLoop,
					nDimension - 1,
					slbInnerLoop != null,
					config
				);
				
				if (slbInnerLoop != null)
				{
					if (bHasParentLoop)
						StatementListBundleUtil.addToLoopBody (slbParent, slbInnerLoop);
					else
						slbParent.addStatements (slbInnerLoop);
				}

				// cleanup loop
				if (nUnrollFactor > 1 && bHasCleanupLoops)
				{
					slbInnerLoop = generateIteratorForDimension (nDimension, slbParent, SubdomainIteratorCodeGenerator.NULL_EXPRESSION, null, 1);

					StencilLoopUnrollingConfiguration configCleanup = config.clone ();
					configCleanup.setUnrollingForDimension (nDimension, 1, 1);

					recursiveGenerateInner (
						slbInnerLoop == null ? slbParent : slbInnerLoop,
						nDimension - 1,
						slbInnerLoop != null,
						configCleanup
					);
					
					if (slbInnerLoop != null)
					{
						if (bHasParentLoop)
							StatementListBundleUtil.addToLoopBody (slbParent, slbInnerLoop);
						else
							slbParent.addStatements (slbInnerLoop);
					}
				}
			}
			else
			{
				// create the stencil calculation statement

				// make sure that the statements to compute the indices are added
				m_data.getData ().getMemoryObjectManager ().resetIndices ();
				m_data.getCodeGenerators ().getUnrollGeneratedIdentifiers ().reset ();

				// set the code generator options and let the stencil code generator generate the code for the stencil calculation
				CodeGeneratorRuntimeOptions options = m_options.clone ();
				options.setOption (CodeGeneratorRuntimeOptions.OPTION_STENCILLOOPUNROLLINGFACTOR, config);

				// has prologue/epilogue loops?
				if (bHasProEpiLoops)
				{
					// main loop
					SubdomainIteratorCodeGenerator.LOGGER.debug (StringUtil.concat ("Creating main loop for ", m_sdIterator.getLoopHeadAnnotation (), " (no native SIMD datatypes; has prologue/epilogue loops)"));
					StatementListBundleUtil.addToLoopBody (slbParent, SubdomainIteratorCodeGenerator.TAG_MAINLOOP, m_cgParent.generate (m_sdIterator.getLoopBody ().clone (), options));

					// create the loop body for the prologue and epilogue loops
					m_data.getData ().getMemoryObjectManager ().resetIndices ();
					m_data.getCodeGenerators ().getUnrollGeneratedIdentifiers ().reset ();

					// create the code generation options for the prologue and epilogue loops
					// no loop unrolling in the unit stride direction, no vectorization
					CodeGeneratorRuntimeOptions optionsProEpi = m_options.clone ();
					StencilLoopUnrollingConfiguration configUnrollProEpi = config.clone ();
					configUnrollProEpi.setUnrollingForDimension (0, 1, 1);
					optionsProEpi.setOption (CodeGeneratorRuntimeOptions.OPTION_STENCILLOOPUNROLLINGFACTOR, configUnrollProEpi);
					optionsProEpi.setOption (CodeGeneratorRuntimeOptions.OPTION_NOVECTORIZE, Boolean.TRUE);

					StatementListBundleUtil.addToLoopBody (slbParent, SubdomainIteratorCodeGenerator.TAG_PROEPILOOP, m_cgParent.generate (m_sdIterator.getLoopBody (), optionsProEpi));
				}
				else
				{
					SubdomainIteratorCodeGenerator.LOGGER.debug (StringUtil.concat ("Creating main loop for ", m_sdIterator.getLoopHeadAnnotation (), " (native SIMD datatypes; no prologue/epilogue loops)"));
					StatementListBundleUtil.addToLoopBody (slbParent, m_cgParent.generate (m_sdIterator.getLoopBody (), options));
				}
			}
		}

		/**
		 * Creates the lower loop bound for the <code>for</code> loop in dimension <code>nDim</code>.
		 * @param nDim The dimension
		 * @return The lower loop bound
		 */
		private Expression getLowerLoopBound (int nDim)
		{
			if (m_sdIterator.getDomainSubdomain ().isBaseGrid ())
				return m_sdIterator.getDomainIdentifier ().getSubdomain ().getBox ().getMin ().getCoord (nDim).clone ();

			Expression exprMin = m_data.getData ().getGeneratedIdentifiers ().getDimensionIndexIdentifier (m_sdIterator.getDomainIdentifier (), nDim).clone ();
			if (exprMin == null)
				exprMin = m_sdIterator.getDomainSubdomain ().getBox ().getMin ().getCoord (nDim).clone ();
			
			return exprMin;
		}

		/**
		 * Creates the upper loop bound for the <code>for</code> loop in dimension <code>nDim</code>.
		 * @param nDim The dimension
		 * @return The upper loop bound
		 */
		private Expression getUpperLoopBound (int nDim)
		{
			if (m_sdIterator.getDomainSubdomain ().isBaseGrid ())
				return ExpressionUtil.increment (m_sdIterator.getDomainIdentifier ().getSubdomain ().getBox ().getMax ().getCoord (nDim).clone ());

			Expression exprMax = m_data.getData ().getGeneratedIdentifiers ().getDimensionMaxIdentifier (m_sdIterator.getDomainIdentifier (), nDim).clone ();
			if (exprMax == null)
				exprMax = ExpressionUtil.increment (m_sdIterator.getDomainSubdomain ().getBox ().getMax ().getCoord (nDim).clone ());

			return exprMax;
		}

		/**
		 * Creates and assigns a named maximum identifier.
		 * @param nDim The dimension for which to create the identifier
		 * @return
		 */
		private Statement generateNamedMaximumIdentifier (int nDim)
		{
			Identifier idLoopIdx = m_data.getData ().getGeneratedIdentifiers ().getDimensionIndexIdentifier (m_sdIterator.getIterator (), nDim).clone ();
			Identifier idMax = m_data.getData ().getGeneratedIdentifiers ().getDimensionMaxIdentifier (m_sdIterator.getIterator (), nDim).clone ();

			return new ExpressionStatement (new AssignmentExpression (
				idMax,
				AssignmentOperator.NORMAL,
				ExpressionUtil.min (
					new BinaryExpression (
						idLoopIdx.clone (),
						BinaryOperator.ADD,
						m_sdIterator.getIteratorSubdomain ().getBox ().getSize ().getCoord (nDim).clone ()),
					// maximum grid index + 1 (+1 since the for loops don't include the maximum)
					ExpressionUtil.increment (m_data.getStencilCalculation ().getDomainSize ().getMax ().getCoord (nDim).clone ()))
				)
			);
		}

		/**
		 * Creates a single {@link ForLoop} corresponding to dimension <code>nDim</code> of the
		 * {@link SubdomainIterator} and adds it to the <code>cmpstmtOuter</code> statement.
		 *
		 * @param nDim The dimension for which to create the {@link ForLoop}
		 * @param slGenerated The {@link StatementList} to which the generated {@link ForLoop} will be added
		 * @param exprStartOffset The offset to the start value for the loop index. If set to
		 * 	{@link SubdomainIteratorCodeGenerator#NULL_EXPRESSION}, the initialization statement will be omitted
		 * @param exprNegEndOffset The negative offset to the end value for the loop index
		 * @param bIsOutermostLoopOfInnerNest Specifies whether the loop for this dimension is the outer-most
		 * 	loop of the &quot;regular&quot; loop nest structure and subsequentially will be treated as reference
		 * 	for unrolling (if the nest contains a stencil call)
		 *
		 * @return The generated loop or <code>null</code> if no loop was created for the dimension
		 */
		private StatementListBundle generateIteratorForDimension (int nDim, StatementListBundle slGenerated, Expression exprStartOffset, Expression exprNegEndOffset, int nUnrollingFactor)
		{
			Expression exprIteratorSize = m_sdIterator.getIteratorSubdomain ().getBox ().getSize ().getCoord (nDim);
			Expression exprDomainSize = m_sdIterator.getDomainSubdomain ().getBox ().getSize ().getCoord (nDim);

			// prepare loop creation
			Identifier idIdx = m_data.getData ().getGeneratedIdentifiers ().getDimensionIndexIdentifier (m_sdIterator.getIterator (), nDim);

			// create the loop
			// >    for <it.loopidx>_<dim> = <min> .. <max> by <step_dim> {                <--- this loop
			// >        <it.loopidx>_<dim>_max = <min> + <step_dim>                        <--- add this at the top of the compound statement which will become the body of the next loop
			// >        for <it.loopidx>_<dim-1> = <min'> .. <max'> by <step_(dim-1)> {    <--- the next loop
			// >            ...
			// >        }
			// >    }

			// if bHasSIMDPrologueAndEpilogueLoops:
			// >    # prologue loop
			// >    for <it.loopidx>_<dim-1> = <min'> .. <min'> + <prologue_length>
			// >        <non-vectorized code>
			// >    # vectorized main loop
			// >    for <it.loopidx>_<dim-1>  <=  <max'> - <SIMD_vector_length> + 1  by  <SIMD_vector_length>
			// >        <vectorized code>
			// >    # epilogue loop
			// >    for <it.loopidx>_<dim-1>  <=  <max'>
			// >        <non-vectorized code>

			Expression exprStart = exprStartOffset == SubdomainIteratorCodeGenerator.NULL_EXPRESSION ? null : getLowerLoopBound (nDim);
			if (exprStartOffset != null && exprStartOffset != SubdomainIteratorCodeGenerator.NULL_EXPRESSION)
				exprStart = new BinaryExpression (exprStart, BinaryOperator.ADD, exprStartOffset);

			// don't do anything if we don't need a loop for dimension nDim:
			// * domain size is 1 in dimension nDim or
			// * iterator and domain size are equal
			if (ExpressionUtil.isValue (exprDomainSize, 1) || exprDomainSize.equals (exprIteratorSize))
			{
				if (exprStart != null)
					StatementListBundleUtil.addToLoopBody (slGenerated, new ExpressionStatement (new AssignmentExpression (idIdx.clone (), AssignmentOperator.NORMAL, exprStart)));
				if (m_bHasNestedLoops)
					StatementListBundleUtil.addToLoopBody (slGenerated, generateNamedMaximumIdentifier (nDim));

				// TODO: location can be chosen better; cf. ThreadCodeGenerator#getIndexCalculationLocation [unify codes]

				return null;
			}

			Expression exprEnd = getUpperLoopBound (nDim);
			if (exprNegEndOffset != null && !ExpressionUtil.isZero (exprNegEndOffset))
				exprEnd = Symbolic.simplify (new BinaryExpression (exprEnd, BinaryOperator.SUBTRACT, exprNegEndOffset));

			Expression exprMainLoopStep = exprIteratorSize;

			// account for loop unrolling
			if (nUnrollingFactor != 1)
				exprMainLoopStep = ExpressionUtil.product (exprMainLoopStep.clone (), new IntegerLiteral (nUnrollingFactor));

			// account for SIMD
			int nSIMDVectorLength = m_data.getCodeGenerators ().getStencilCalculationCodeGenerator ().getLcmSIMDVectorLengths ();
			if (m_bContainsStencilCall && nDim == 0 && (isStencilCalculation () || m_data.getOptions ().useNativeSIMDDatatypes ()))
				exprMainLoopStep = ExpressionUtil.product (exprMainLoopStep.clone (), new IntegerLiteral (nSIMDVectorLength));

			// list to which the loop statements will be added
			StatementListBundle slbStatements = new StatementListBundle ();

			// create the loops
			CompoundStatement cmpstmtMainLoopBody = new CompoundStatement ();
			if (hasSIMDPrologueAndEpilogueLoops (nDim))
				createLoopWithProAndEpi (slbStatements, idIdx, exprStart, exprEnd, exprMainLoopStep, cmpstmtMainLoopBody);
			else
			{
				// create the main loop if the dimension of the domain is > 1
				createDefaultLoop (slbStatements, idIdx, exprStart, exprEnd, exprMainLoopStep, cmpstmtMainLoopBody);
			}

			// create and set the new named maximum point
			if (m_bHasNestedLoops)
				cmpstmtMainLoopBody.addStatement (generateNamedMaximumIdentifier (nDim));
			return slbStatements;
		}
		
		/**
		 * <p>Determines whether prologue and epilogue loops should be generated.</p>
		 * If no native SIMD datatypes are used, create prologue and epilogue loops
		 * prologue and epilogue loops are only generated if:
		 * <ul>
		 * <li>the loop contains a stencil call</li>
		 * <li>the loop is the inner most loop of a loop nest (<code>nDim == 0</code>)</li>
		 * <li>SIMD is used with no native SIMD data types</li>
		 * <li>within the stencil calculation (not within the initialization, for instance)</li>
		 * </ul>
		 * 
		 * @param nDim The dimension of the loop
		 * @return <code>true</code> iff prologue and epilogue loop are to be generated for the loop in dimension <code>nDim</code>
		 */
		private boolean hasSIMDPrologueAndEpilogueLoops (int nDim)
		{
			return m_bContainsStencilCall &&
				nDim == 0 &&
				m_data.getArchitectureDescription ().useSIMD () &&
				!m_data.getOptions ().useNativeSIMDDatatypes () &&
				isStencilCalculation ();
		}
		
		/**
		 * Determines whether inline assembly is to be used for the innermost loop
		 * @return <code>true</code> iff inline assembly is to be used for the innermost loop
		 */
		private boolean isAssemblyUsedForInnerMost ()
		{
			return m_bContainsStencilCall && m_data.getCodeGenerators ().getBackendCodeGenerator ().hasAssemblyCodeGenerator () && isStencilCalculation ();
		}
		
		private void generateInnerMostAssembly ()
		{
			throw new RuntimeException ("Not implemented");
		}
		
		/**
		 * Returns a statement initializing the loop index <code>idIdx</code> with the start expression <code>exprStart</code>.
		 * @param idIdx The loop index
		 * @param exprStart The start value or <code>null</code> if no initialization is desired
		 * @return The statement initializing the loop index
		 */
		private Statement getStartStatement (IDExpression idIdx, Expression exprStart)
		{
			return exprStart == null ?
				new NullStatement () :
				new ExpressionStatement (new AssignmentExpression (idIdx.clone (), AssignmentOperator.NORMAL, exprStart));			
		}
		
		/**
		 * Returns the condition expression comparing the loop index to the end value.
		 * @param idIdx The loop index
		 * @param exprEnd The end value
		 * @return The loop condition expression
		 */
		private Expression getConditionExpression (IDExpression idIdx, Expression exprEnd)
		{
			return new BinaryExpression (idIdx.clone (), BinaryOperator.COMPARE_LT, exprEnd);
		}
		
		/**
		 * Creates the default loop<br/>
		 * <code>for (<i>idIdx</i> = <i>exprStart</i>; <i>idIdx</i> &lt; <i>exprEnd</i>; <i>idIdx</i> += <i>exprMainLoopStep</i>) { <i>stmtMainLoopBody</i> }</code>
		 * @param slbStatements The {@link StatementListBundle} to which the generated code is added
		 * @param idIdx The loop index
		 * @param exprStart The start value
		 * @param exprEnd The end value
		 * @param exprMainLoopStep The step
		 * @param stmtMainLoopBody The loop body
		 */
		private void createDefaultLoop (StatementListBundle slbStatements, IDExpression idIdx, Expression exprStart, Expression exprEnd, Expression exprMainLoopStep, Statement stmtMainLoopBody)
		{
			slbStatements.addStatement (new ForLoop (
				getStartStatement (idIdx, exprStart),
				getConditionExpression (idIdx, exprEnd),
				new AssignmentExpression (idIdx.clone (), AssignmentOperator.ADD, exprMainLoopStep.clone ()),
				stmtMainLoopBody)
			);			
		}
		
		/**
		 * Creates a loop with a prologue and an epilogue loop.
		 * @param slbStatements The {@link StatementListBundle} to which the generated code is added
		 * @param idIdx The loop index
		 * @param exprStart The starting expression
		 * @param exprEnd The end expression
		 * @param exprMainLoopStep The step
		 * @param stmtMainLoopBody The loop body
		 */
		private void createLoopWithProAndEpi (StatementListBundle slbStatements,
			IDExpression idIdx, Expression exprStart, Expression exprEnd, Expression exprMainLoopStep,
			Statement stmtMainLoopBody)
		{
			// create the prologue loop
			Expression exprStartPrologue = null;
			if (exprStart == null)
			{
				VariableDeclarator decl = new VariableDeclarator (new NameID (StringUtil.concat (idIdx.getName (), "_start")));
				slbStatements.addDeclaration (new VariableDeclaration (Globals.SPECIFIER_INDEX, decl));
				exprStartPrologue = new Identifier (decl);
			}
			else
				exprStartPrologue = exprStart.clone ();

			slbStatements.addStatement (
				new ForLoop (
					getStartStatement (idIdx, exprStart),
					new BinaryExpression (
						idIdx.clone (),
						BinaryOperator.COMPARE_LT,
						getPrologueEnd (exprStartPrologue, exprEnd, slbStatements)),
					new UnaryExpression (UnaryOperator.POST_INCREMENT, idIdx.clone ()),
					new CompoundStatement ()
				),
				SubdomainIteratorCodeGenerator.TAG_PROEPILOOP
			);

			// create the main loop if the dimension of the domain is > 1
			slbStatements.addStatement (
				new ForLoop (
					new NullStatement (),
					new BinaryExpression (
						idIdx.clone (),
						BinaryOperator.COMPARE_LT,
						Symbolic.simplify (new BinaryExpression (
							exprEnd.clone (),
							BinaryOperator.SUBTRACT,
							ExpressionUtil.decrement (exprMainLoopStep.clone ())
						))),
					new AssignmentExpression (idIdx.clone (), AssignmentOperator.ADD, exprMainLoopStep.clone ()),
					stmtMainLoopBody
				),
				SubdomainIteratorCodeGenerator.TAG_MAINLOOP
			);

			// create the epilogue loop
			slbStatements.addStatement (
				new ForLoop (
					new NullStatement (),
					getConditionExpression (idIdx, exprEnd),
					new UnaryExpression (UnaryOperator.POST_INCREMENT, idIdx.clone ()),
					new CompoundStatement ()
				),
				SubdomainIteratorCodeGenerator.TAG_PROEPILOOP
			);	
		}

		/**
		 * prologue length = (align_restrict - (((uintptr_t) &u[idx_min] + align_restrict - 1) & (uintptr_t) (align_restrict - 1))) / sizeof (Type)
		 * @return
		 */
		private Expression getPrologueEnd (Expression exprPrologueStart, Expression exprEnd, IStatementList slb)
		{
			// find output nodes
			StencilNode node = null;
			for (StencilNode n : m_data.getStencilCalculation ().getStencilBundle ().getFusedStencil ().getOutputNodes ())
			{
				if (node == null)
					node = n;
				else
				{
					// check whether indices and datatypes are compatible
					if (!Arrays.equals (node.getSpaceIndex (), n.getSpaceIndex ()) || !node.getSpecifier ().equals (n.getSpecifier ()))
						throw new RuntimeException ("Stencil output nodes are not compatible; can't generate code with SIMD vector length > 1 and no-native type usage.");
				}
			}

			// make sure that an output node has been found
			if (node == null)
				throw new RuntimeException ("No output stencil node found. Code generation aborted.");

			// initialize loop index for the index calculation
			SubdomainIdentifier sdidIterator = m_sdIterator.getIterator ();
			slb.addStatement (new ExpressionStatement (new AssignmentExpression (
				m_data.getData ().getGeneratedIdentifiers ().getDimensionIndexIdentifier (sdidIterator, 0).clone (),
				AssignmentOperator.NORMAL,
				getLowerLoopBound (0))));

			// create an expression accessing the grid
			m_data.getData ().getMemoryObjectManager ().resetIndices ();
			m_data.getCodeGenerators ().getUnrollGeneratedIdentifiers ().reset ();
			Expression exprGridAccess = m_data.getData ().getMemoryObjectManager ().getMemoryObjectExpression (
				sdidIterator, node, null, true, true, false, slb, m_options);

			// get rid of the outer-most type cast
			for (DepthFirstIterator it = new DepthFirstIterator (exprGridAccess); it.hasNext (); )
			{
				Object obj = it.next ();
				if (obj instanceof Typecast)
				{
					try
					{
						exprGridAccess = ((Expression) ((Typecast) obj).getChildren ().get (0)).clone ();
					}
					catch (Exception e)
					{
					}

					break;
				}
			}

			// add an address-of operator if there isn't one yet
			if (!(exprGridAccess instanceof UnaryExpression) || !((UnaryExpression) exprGridAccess).getOperator ().equals (UnaryOperator.ADDRESS_OF))
				exprGridAccess = new UnaryExpression (UnaryOperator.ADDRESS_OF, exprGridAccess);

			int nAlignRestrict = m_data.getArchitectureDescription ().getAlignmentRestriction (node.getSpecifier ());

			// create a new variable that will hold the prologue length
			VariableDeclarator decl = new VariableDeclarator (CodeGeneratorUtil.createNameID ("prologueend", m_nPrologueLengthIdentifiersCount++));
			m_data.getData ().addDeclaration (new VariableDeclaration (Globals.SPECIFIER_INDEX, decl));
			Identifier idPrologueEnd = new Identifier (decl);

			// create the statement calculating the prologue end:
			//     prologue_length = (align_restrict - (((uintptr_t) &u[idx_min] + align_restrict - 1) & (uintptr_t) (align_restrict - 1))) / sizeof (Type)
			// the prologue end is calculated as
			//     prologue_end = min (prologue_start + prologue_length, exprEnd)
			slb.addStatement (new ExpressionStatement (new AssignmentExpression (
				idPrologueEnd,
				AssignmentOperator.NORMAL,
				ExpressionUtil.min (
					new BinaryExpression (
						exprPrologueStart,
						BinaryOperator.ADD,
						new BinaryExpression (
							new BinaryExpression (
								new IntegerLiteral (nAlignRestrict),
								BinaryOperator.SUBTRACT,
								new BinaryExpression (
									new BinaryExpression (
										new Typecast (
											CodeGeneratorUtil.specifiers (new UserSpecifier (new NameID ("uintptr_t"))),
											exprGridAccess
										),
										BinaryOperator.ADD,
										new Typecast (
											CodeGeneratorUtil.specifiers (new UserSpecifier (new NameID ("uintptr_t"))),
											new IntegerLiteral (nAlignRestrict - 1)
										)
									),
									BinaryOperator.BITWISE_AND,
									new IntegerLiteral (nAlignRestrict - 1)
								)
							),
							BinaryOperator.DIVIDE,
							new SizeofExpression (CodeGeneratorUtil.specifiers (node.getSpecifier ()))
						)
					),
					exprEnd.clone ()
				)
			)));

			return idPrologueEnd.clone ();
		}

		/**
		 *
		 */
		private StatementListBundle generateWithDataTransfers ()
		{
			StatementListBundle slbGenerated = new StatementListBundle ();

			//m_nMaxTimestep = m_data.getCodeGenerators ().getStrategyAnalyzer ().getMaximumTimstepOfTemporalIterator (m_sgIterator);
			m_nReuseDim = new ReuseMask (m_sdIterator).getReuseDimension ();

			// allocate memory objects for the subdomain iterator
			m_data.getCodeGenerators ().getDatatransferCodeGenerator ().allocateLocalMemoryObjects (m_sdIterator, m_options);

			// add the iteration counter
			m_idCounter = m_data.getData ().getGeneratedIdentifiers ().getLoopCounterIdentifier (m_sdIterator.getIterator ());
			slbGenerated.addStatement (new ExpressionStatement (new AssignmentExpression (m_idCounter.clone (), AssignmentOperator.NORMAL, new IntegerLiteral (0))));

			// create the outer loops that are not affected by data reuse
			StatementListBundle slbAddLoop = slbGenerated;
			for (int j = m_sdIterator.getIterator ().getDimensionality () - 1; j >= m_nReuseDim + 1; j--)
			{
				//XXX check this!!!!!!!!!!!!!! (generateIteratorForDimension now returns loop, not loop body)
				StatementListBundle slbAddLoopOuter = generateIteratorForDimension (j, slbAddLoop, null, null, 1);
				if (slbAddLoopOuter != null)
					slbAddLoop = slbAddLoopOuter;
			}

			// create the inner loops
			StatementListBundle slbInner = new StatementListBundle ();
			if (m_data.getArchitectureDescription ().supportsAsynchronousIO (m_sdIterator.getParallelismLevel ()))
				generateAsynchronousIterator (slbInner);
			else
				generateSynchronousIterator (slbInner);

			StatementListBundleUtil.addToLoopBody (slbAddLoop, slbInner);
			return slbAddLoop;
		}

		/**
		 *
		 * @param slbOuter
		 */
		private void generateSynchronousIterator (StatementListBundle slbOuter)
		{
			// get required data
			DatatransferCodeGenerator dtcg = m_data.getCodeGenerators ().getDatatransferCodeGenerator ();
			MemoryObjectManager mgr = m_data.getData ().getMemoryObjectManager ();
			SubdomainIdentifier sdid = m_sdIterator.getIterator ();
			int nReuseDim = m_nReuseDim == -1 ? m_sdIterator.getDomainIdentifier ().getDimensionality () - 1 : m_nReuseDim;

	        // load the full memory object set for timestep 0
	        // except the "front" memory objects
			// ast.add (load (M, *))
			dtcg.loadData (mgr.getBackStencilNodes (sdid), m_sdIterator, slbOuter, m_options);

			// ast.add (ast_reuse_loop = create_iterator_dim (v, d_reuse))
			//XXX check this!!!!!!!!!!!!!! (generateIteratorForDimension now returns loop, not loop body)
			StatementListBundle slbLoop = generateIteratorForDimension (nReuseDim, slbOuter, null, null, 1);

	        // load the "front" memory objects
			// ast_reuse_loop.add (load (M[0][][], *))
			StatementListBundle slbLoopBody = new StatementListBundle (new ArrayList<Statement> ());
			dtcg.loadData (mgr.getFrontStencilNodes (sdid), m_sdIterator, slbLoopBody, m_options);

			// wait for data to arrive
			dtcg.waitFor (mgr.getFrontStencilNodes (sdid), m_sdIterator, slbLoopBody, m_options);

	        // create the inner loops
			// create_inner (v, d_reuse, ast_reuse_loop)
			generateInner (nReuseDim, slbLoopBody);

	        // write the result back
			// ast_reuse_loop.add (store (M[t_max][][], *))
			dtcg.storeData (mgr.getOutputStencilNodes (sdid), m_sdIterator, slbLoopBody, m_options);

            // ast.add ("i++")
			slbLoopBody.addStatement (generateIncrementLoopCounter ());

			if (slbLoop == null)
				//StatementListBundleUtil.addToLoopBody (slbOuter, slbLoopBody);
				slbOuter.addStatements (slbLoopBody);
			else
			{
				StatementListBundleUtil.addToLoopBody (slbLoop, slbLoopBody);
				//StatementListBundleUtil.addToLoopBody (slbOuter, slbLoop);
				slbOuter.addStatements (slbLoop);
			}

			// TODO: non-temporal write (skipping cache/local mem)
		}

		/**
		 * Generates the statement to increment the loop counter.
		 * @return The statement incrementing the loop counter
		 */
		private Statement generateIncrementLoopCounter ()
		{
			return new ExpressionStatement (new UnaryExpression (UnaryOperator.PRE_INCREMENT, m_idCounter.clone ()));
		}

		/**
		 *
		 * @param cmpstmtOuter
		 */
		private void generateAsynchronousIterator (StatementListBundle slOuter)
		{
			if (true)
				throw new RuntimeException ("SubgridIteratorCodeGenerator.generateAsynchronousIterator not implemented");

			//TODO: complete code

			// -- pipeline filling phase

            // load the full memory object set
            // ast.add (load (M[0][all][], "i % 4"))
//			XXX
            // ast.add ("i++")
			slOuter.addStatement (generateIncrementLoopCounter ());

            // preload "front" memory objects of the next iteration
            // ast.add (load (M[0][][], "i % "))
//			XXX

            // wait for the data to arrive and compute
            // ast.add (wait_for ("(i - 1) % "))
//			XXX
            // create_inner (v, d_reuse - 1, ast)

            Identifier idIdx = m_data.getData ().getGeneratedIdentifiers ().getDimensionIndexIdentifier (m_sdIterator.getIterator (), m_nReuseDim);
            Identifier idIdxMax = m_data.getData ().getGeneratedIdentifiers ().getDimensionMaxIdentifier (m_sdIterator.getIterator (), m_nReuseDim);
            Expression exprOffset = m_sdIterator.getIteratorSubdomain ().getBox ().getSize ().getCoord (m_nReuseDim);
            slOuter.addStatement (new ExpressionStatement (new AssignmentExpression (
            	idIdx.clone (), AssignmentOperator.NORMAL, getLowerLoopBound (m_nReuseDim))));
            slOuter.addStatement (new ExpressionStatement (new AssignmentExpression (
            	idIdxMax.clone (),
            	AssignmentOperator.NORMAL,
            	new BinaryExpression (idIdx.clone (), BinaryOperator.ADD, exprOffset.clone ()))));

            generateInner (m_nReuseDim, slOuter);
            // ast.add ("i++")
            slOuter.addStatement (generateIncrementLoopCounter ());


            // -- pipeline working phase

            // ast.add (ast_reuse_loop = create_iterator_dim (v, d_reuse))
            //XXX check this!!!!!!!!!!!!!! (generateIteratorForDimension now returns loop, not loop body)
            StatementListBundle slReuseLoop = generateIteratorForDimension (m_nReuseDim, slOuter, exprOffset.clone (), exprOffset.clone (), 1);

            // preload the "front" memory objects, store results
            // ast_reuse_loop.add (load (M[][][], ""))
//			XXX

            // ast_reuse_loop.add (store ())
//			XXX

            // wait for the data loaded previously and compute
            // ast_reuse_loop.add (wait_for (""))
//			XXX

            // create_inner (v, d_reuse - 1, ast_reuse_loop)
            generateInner (m_nReuseDim, slReuseLoop);

            // ast_reuse_loop.add ("i++")
            slOuter.addStatement (generateIncrementLoopCounter ());


            // -- pipeline draining phase

            // wait for the data loaded previously and compute
            // ast.add (store (M[][][], ""))
//			XXX

            // ast.add (wait_for (""))
//			XXX

            // create_inner (v, d_reuse - 1, ast)
            Expression exprUpperBound = getUpperLoopBound (m_nReuseDim);
            slOuter.addStatement (new ExpressionStatement (new AssignmentExpression (
            	idIdx.clone (),
            	AssignmentOperator.NORMAL,
            	new BinaryExpression (exprUpperBound.clone (), BinaryOperator.SUBTRACT, exprOffset.clone ()))));
            slOuter.addStatement (new ExpressionStatement (new AssignmentExpression (
            	idIdxMax.clone (), AssignmentOperator.NORMAL, exprUpperBound.clone ())));

            generateInner (m_nReuseDim, slOuter);

            // ast.add ("i++")
            slOuter.addStatement (generateIncrementLoopCounter ());

            // store last results
            // ast.add (store (M[][][], ""))
//			XXX
		}
	}


	///////////////////////////////////////////////////////////////////
	// Member Variables

	/**
	 * The parent code generator that is invoked to generate the code for the loop body
	 */
	private ICodeGenerator m_cgParent;

	/**
	 * The data object shared among the code generators
	 */
	private CodeGeneratorSharedObjects m_data;

	protected int m_nPrologueLengthIdentifiersCount;


	///////////////////////////////////////////////////////////////////
	// Implementation

	public SubdomainIteratorCodeGenerator (ICodeGenerator cgParent, CodeGeneratorSharedObjects data)
	{
		m_cgParent = cgParent;
		m_data = data;
		m_nPrologueLengthIdentifiersCount = 0;
	}

	@Override
	public StatementListBundle generate (Traversable trvInput, CodeGeneratorRuntimeOptions options)
	{
		if (SubdomainIteratorCodeGenerator.LOGGER.isDebugEnabled ())
			SubdomainIteratorCodeGenerator.LOGGER.debug (StringUtil.concat ("Generating code with options ", options.toString ()));

		if (!(trvInput instanceof SubdomainIterator))
			throw new RuntimeException (StringUtil.concat (getClass ().getName (), " can only be used to generate code for SubgridIterators."));

		CodeGenerator cg = new CodeGenerator ((SubdomainIterator) trvInput, options);

		StatementListBundle slGenerated = cg.generate ();
		m_data.getData ().getMemoryObjectManager ().resetIndices ();

		return slGenerated;
	}
}
