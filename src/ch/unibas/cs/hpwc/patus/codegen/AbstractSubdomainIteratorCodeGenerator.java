package ch.unibas.cs.hpwc.patus.codegen;

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
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.InnermostLoopCodeGenerator;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;
import ch.unibas.cs.hpwc.patus.codegen.options.StencilLoopUnrollingConfiguration;
import ch.unibas.cs.hpwc.patus.geometry.Size;
import ch.unibas.cs.hpwc.patus.representation.StencilNode;
import ch.unibas.cs.hpwc.patus.symbolic.Symbolic;
import ch.unibas.cs.hpwc.patus.util.CodeGeneratorUtil;
import ch.unibas.cs.hpwc.patus.util.ExpressionUtil;
import ch.unibas.cs.hpwc.patus.util.IntArray;
import ch.unibas.cs.hpwc.patus.util.StatementListBundleUtil;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

public abstract class AbstractSubdomainIteratorCodeGenerator implements ICodeGenerator
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
	protected final static Expression NULL_EXPRESSION = new NameID ("__null__");

	protected final static String TAG_MAINLOOP = "mainloop";
	protected final static String TAG_PROEPILOOP = "proepiloop";

	
	///////////////////////////////////////////////////////////////////
	// Inner Types

	public abstract class CodeGenerator
	{
		///////////////////////////////////////////////////////////////////
		// Member Variables

		public CodeGeneratorRuntimeOptions m_options;

		/**
		 * The subdomain iterator for which code is generated
		 */
		public SubdomainIterator m_sdIterator;

		//private int m_nMaxTimestep;
		public int m_nReuseDim;

		public boolean m_bHasNestedLoops;
		public boolean m_bContainsStencilCall;
		public boolean m_bIsEligibleForStencilLoopUnrolling;

		public int[] m_rgMaxUnrollingFactorPerDimension;

		public Identifier m_idCounter;


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

			// get the maximum unrolling factors
			initializeMaximumUnrollingFactors ();
		}

		/**
		 * Gets the maximum unrolling factors per dimension (depending on the domain of the iterator).
		 */
		public void initializeMaximumUnrollingFactors ()
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
			return generateInner (m_sdIterator.getIterator ().getDimensionality (), new StatementListBundle ());
		}
		
		public boolean isStencilCalculation ()
		{
			return m_options.hasValue (CodeGeneratorRuntimeOptions.OPTION_STENCILCALCULATION, CodeGeneratorRuntimeOptions.VALUE_STENCILCALCULATION_STENCIL);
		}

		/**
		 * Generates the C code for the loop nest.
		 * @param nStartDim
		 */
		public StatementListBundle generateInner (int nStartDim, StatementListBundle slbGeneratedParent)
		{
			Parameter param = m_bIsEligibleForStencilLoopUnrolling ?
				new Parameter (StringUtil.concat ("_unroll_", m_sdIterator.getIterator ().getName ())) :
				null;

			// create a code branch for each unrolling configuration
			for (StencilLoopUnrollingConfiguration config :	m_data.getOptions ().getStencilLoopUnrollingConfigurations (
				nStartDim, m_rgMaxUnrollingFactorPerDimension, m_bIsEligibleForStencilLoopUnrolling))
			{
				// generate the code for the loop nest
				StatementListBundle slbLoopNest = new StatementListBundle ();
				slbLoopNest.addStatementAtTop (new AnnotationStatement (new IndexBoundsCalculationInsertionAnnotation (m_sdIterator)));

				recursiveGenerateInner (slbLoopNest, nStartDim - 1, false, config);

				if (slbLoopNest.size () == 1)
					slbGeneratedParent.addStatement (slbLoopNest.getDefault (), param, config.toInteger ());
				else
					slbGeneratedParent.addStatements (slbLoopNest, param, config.toInteger ());
			}

			return slbGeneratedParent;
		}

		/**
		 * Recursively generates a nested loop.
		 * @param slbParent
		 * @param nDimension
		 * @param bHasParentLoop Flag indicating whether the parent structure is a loop
		 * 	(to whose body the current structure will be added); if the parent is not a
		 *  loop (<code>bHasParentLoop == false</code>), the current structure is just
		 *  appended to the parent structure
		 * @param config
		 */
		public void recursiveGenerateInner (StatementListBundle slbParent, int nDimension, boolean bHasParentLoop, StencilLoopUnrollingConfiguration config)
		{
			if (nDimension == 0 && isSpecializedCGUsedForInnerMost ())
			{
				// this is an innermost loop (containing a stencil computation)
				// if a specialized code generator for the innermost loop and the contained stencil expression
				// was specified, use it				
				generateInnerMost (slbParent, bHasParentLoop, config);
			}
			else
				recursiveGenerateInnerDefault (slbParent, nDimension, bHasParentLoop, config);
		}
			
		public void recursiveGenerateInnerDefault (StatementListBundle slbParent, int nDimension, boolean bHasParentLoop, StencilLoopUnrollingConfiguration config)
		{			
			// do we need prologue and epilogue loops?
			// we don't need a clean up loop in the inner most loop if we have prologue and epilogue loops (bUseNativeSIMD == false)
			boolean bDoBoundaryChecks = m_options.getBooleanValue (CodeGeneratorRuntimeOptions.OPTION_DOBOUNDARYCHECKS, false);
			boolean bHasProEpiLoops = m_data.getArchitectureDescription ().useSIMD () && !m_data.getOptions ().useNativeSIMDDatatypes () &&
				isStencilCalculation () && m_bContainsStencilCall && !bDoBoundaryChecks;
			boolean bHasCleanupLoops = nDimension > 0 || (nDimension == 0 && !bHasProEpiLoops);

			if (nDimension >= 0)
			{
				int nUnrollFactor = bDoBoundaryChecks ? 1 : config.getUnrollingFactor (nDimension);

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
					slbInnerLoop = generateIteratorForDimension (
						nDimension,
						slbParent,
						AbstractSubdomainIteratorCodeGenerator.NULL_EXPRESSION,
						null,
						1
					);

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
				createStencilCalculation (slbParent, bHasProEpiLoops, config);
			}
		}
		
		private void createStencilCalculation (StatementListBundle slbParent, boolean bHasProEpiLoops, StencilLoopUnrollingConfiguration config)
		{
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
				AbstractSubdomainIteratorCodeGenerator.LOGGER.debug (StringUtil.concat (
					"Creating main loop for ",
					m_sdIterator.getLoopHeadAnnotation (),
					" (no native SIMD datatypes; has prologue/epilogue loops)")
				);
				
				StatementListBundleUtil.addToLoopBody (
					slbParent,
					AbstractSubdomainIteratorCodeGenerator.TAG_MAINLOOP,
					m_cgParent.generate (m_sdIterator.getLoopBody ().clone (), options)
				);

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

				StatementListBundleUtil.addToLoopBody (
					slbParent,
					AbstractSubdomainIteratorCodeGenerator.TAG_PROEPILOOP,
					m_cgParent.generate (m_sdIterator.getLoopBody (), optionsProEpi)
				);
			}
			else
			{
				AbstractSubdomainIteratorCodeGenerator.LOGGER.debug (StringUtil.concat (
					"Creating main loop for ",
					m_sdIterator.getLoopHeadAnnotation (),
					" (native SIMD datatypes; no prologue/epilogue loops)")
				);
				
				StatementListBundleUtil.addToLoopBody (
					slbParent,
					m_cgParent.generate (m_sdIterator.getLoopBody ().clone (), options)
				);
			}			
		}

		/**
		 * Creates the lower loop bound for the <code>for</code> loop in
		 * dimension <code>nDim</code>.
		 * 
		 * @param nDim
		 *            The dimension
		 * @return The lower loop bound
		 */
		abstract public Expression getLowerLoopBound (int nDim);

		/**
		 * Creates the upper loop bound for the <code>for</code> loop in
		 * dimension <code>nDim</code>.
		 * 
		 * @param nDim
		 *            The dimension
		 * @return The upper loop bound
		 */
		abstract public Expression getUpperLoopBound (int nDim);


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
		 *            to {@link AbstractSubdomainIteratorCodeGenerator#NULL_EXPRESSION},
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
		public abstract StatementListBundle generateIteratorForDimension (
			int nDim, StatementListBundle slGenerated, Expression exprStartOffset, Expression exprNegEndOffset, int nUnrollingFactor);

		
		/**
		 * <p>
		 * Determines whether prologue and epilogue loops should be generated.
		 * </p>
		 * If no native SIMD datatypes are used, create prologue and epilogue
		 * loops prologue and epilogue loops are only generated if:
		 * <ul>
		 * 	<li>the loop contains a stencil call</li>
		 * 	<li>the loop is the inner most loop of a loop nest (
		 * 		<code>nDim == 0</code>)</li>
		 * 	<li>SIMD is used with no native SIMD data types</li>
		 * 	<li>within the stencil calculation (not within the initialization,
		 * 		for instance)</li>
		 * </ul>
		 * 
		 * @param nDim
		 *            The dimension of the loop
		 * @return <code>true</code> iff prologue and epilogue loop are to be
		 *         generated for the loop in dimension <code>nDim</code>
		 */
		public boolean hasSIMDPrologueAndEpilogueLoops (int nDim)
		{
			return m_bContainsStencilCall &&
				nDim == 0 &&
				m_data.getArchitectureDescription ().useSIMD () &&
				!m_data.getOptions ().useNativeSIMDDatatypes () &&
				isStencilCalculation () &&
				!m_options.getBooleanValue (CodeGeneratorRuntimeOptions.OPTION_DOBOUNDARYCHECKS, false);
		}
		
		/**
		 * Determines whether inline assembly is to be used for the innermost
		 * loop.
		 * 
		 * @return <code>true</code> iff inline assembly is to be used for the
		 *         innermost loop
		 */
		public boolean isSpecializedCGUsedForInnerMost ()
		{
			return m_bContainsStencilCall &&
				m_data.getCodeGenerators ().getInnermostLoopCodeGenerator () != null &&
				isStencilCalculation () &&
				!m_options.getBooleanValue (CodeGeneratorRuntimeOptions.OPTION_DOBOUNDARYCHECKS, false);
		}
		
		/**
		 * Generates code for the innermost loop containing a stencil call using a specialized
		 * innermost loop code generator.
		 * @param slbParent
		 */
		private void generateInnerMost (StatementListBundle slbParent, boolean bHasParentLoop, StencilLoopUnrollingConfiguration config)
		{
			// set code generator options
			m_options.setOption (InnermostLoopCodeGenerator.OPTION_INLINEASM_UNROLLFACTOR, config.getUnrollingFactor (0));
				
			// generate the code
			byte nDimensionality = m_sdIterator.getIterator ().getDimensionality ();
			Set<IntArray> setUnrollings = new HashSet<> ();
			for (int[] rgOffset : config.getConfigurationSpace (nDimensionality))
			{
				IntArray arr = new IntArray (rgOffset, true);
				arr.set (0, 0);
				setUnrollings.add (arr);
			}
			
			// clear the index cache before invoking the inner-most loop CG
			m_data.getData ().getMemoryObjectManager ().clear ();
			
			// generate one version of the code for each unrolling configuration
			// TODO: since only the base grid addresses change, cache generated versions of the code
			StatementListBundle slbInnerLoop = null;
			for (IntArray arr : setUnrollings)
			{
				CodeGeneratorRuntimeOptions options = m_options.clone ();
				options.setOption (CodeGeneratorRuntimeOptions.OPTION_INNER_UNROLLINGCONFIGURATION, arr.get ());
				
				StatementListBundle slb = m_data.getCodeGenerators ().getInnermostLoopCodeGenerator ().generate (m_sdIterator, options);
				
				if (slb != null)
				{
					if (slbInnerLoop == null)
						slbInnerLoop = new StatementListBundle ();
					slbInnerLoop.addStatements (slb);
				}
			}
			
			// add the code to the parent loop
			if (slbInnerLoop != null)
			{
				SubdomainIdentifier sdidIterator = m_sdIterator.getIterator ();
				slbInnerLoop.addStatementAtTop (new ExpressionStatement (new AssignmentExpression (
					m_data.getData ().getGeneratedIdentifiers ().getDimensionIndexIdentifier (sdidIterator, 0).clone (),
					AssignmentOperator.NORMAL,
					getLowerLoopBound (0)
				)));

				if (bHasParentLoop)
					StatementListBundleUtil.addToLoopBody (slbParent, slbInnerLoop);
				else
					slbParent.addStatements (slbInnerLoop);
			}
		}
		
		/**
		 * Returns a statement initializing the loop index <code>idIdx</code>
		 * with the start expression <code>exprStart</code>.
		 * 
		 * @param idIdx
		 *            The loop index
		 * @param exprStart
		 *            The start value or <code>null</code> if no initialization
		 *            is desired
		 * @return The statement initializing the loop index
		 */
		public Statement getStartStatement (IDExpression idIdx, Expression exprStart)
		{
			return exprStart == null ?
				new NullStatement () :
				new ExpressionStatement (new AssignmentExpression (idIdx.clone (), AssignmentOperator.NORMAL, exprStart));			
		}

		
		/**
		 * Returns the condition expression comparing the loop index to the end
		 * value.
		 * 
		 * @param idIdx
		 *            The loop index
		 * @param exprEnd
		 *            The end value
		 * @return The loop condition expression
		 */
		public Expression getConditionExpression (IDExpression idIdx, Expression exprEnd)
		{
			return new BinaryExpression (idIdx.clone (), BinaryOperator.COMPARE_LT, exprEnd);
		}

		
		/**
		 * Creates the default loop<br/>
		 * <code>for (<i>idIdx</i> = <i>exprStart</i>; <i>idIdx</i> &lt; <i>exprEnd</i>; <i>idIdx</i> += <i>exprMainLoopStep</i>) { <i>stmtMainLoopBody</i> }</code>
		 * 
		 * @param slbStatements
		 *            The {@link StatementListBundle} to which the generated
		 *            code is added
		 * @param idIdx
		 *            The loop index
		 * @param exprStart
		 *            The start value
		 * @param exprEnd
		 *            The end value
		 * @param exprMainLoopStep
		 *            The step
		 * @param stmtMainLoopBody
		 *            The loop body
		 */
		public void createDefaultLoop (StatementListBundle slbStatements, IDExpression idIdx, Expression exprStart, Expression exprEnd, Expression exprMainLoopStep, Statement stmtMainLoopBody)
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
		 * 
		 * @param slbStatements
		 *            The {@link StatementListBundle} to which the generated
		 *            code is added
		 * @param idIdx
		 *            The loop index
		 * @param exprStart
		 *            The starting expression
		 * @param exprEnd
		 *            The end expression
		 * @param exprMainLoopStep
		 *            The step
		 * @param stmtMainLoopBody
		 *            The loop body
		 */
		public void createLoopWithProAndEpi (StatementListBundle slbStatements,
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
				AbstractSubdomainIteratorCodeGenerator.TAG_PROEPILOOP
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
				AbstractSubdomainIteratorCodeGenerator.TAG_MAINLOOP
			);

			// create the epilogue loop
			slbStatements.addStatement (
				new ForLoop (
					new NullStatement (),
					getConditionExpression (idIdx, exprEnd),
					new UnaryExpression (UnaryOperator.POST_INCREMENT, idIdx.clone ()),
					new CompoundStatement ()
				),
				AbstractSubdomainIteratorCodeGenerator.TAG_PROEPILOOP
			);	
		}

		/**
		 * prologue length = (align_restrict - (((uintptr_t) &u[idx_min] + align_restrict - 1) & (uintptr_t) (align_restrict - 1))) / sizeof (Type)
		 * @return
		 */
		public Expression getPrologueEnd (Expression exprPrologueStart, Expression exprEnd, IStatementList slb)
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
	}

	
	///////////////////////////////////////////////////////////////////
	// Member Variables

	/**
	 * The parent code generator that is invoked to generate the code for the loop body
	 */
	protected ICodeGenerator m_cgParent;

	/**
	 * The data object shared among the code generators
	 */
	protected CodeGeneratorSharedObjects m_data;

	protected int m_nPrologueLengthIdentifiersCount;


	///////////////////////////////////////////////////////////////////
	// Implementation

	public AbstractSubdomainIteratorCodeGenerator (ICodeGenerator cgParent, CodeGeneratorSharedObjects data)
	{
		m_cgParent = cgParent;
		m_data = data;
		m_nPrologueLengthIdentifiersCount = 0;
	}
	
	protected abstract CodeGenerator createCodeGenerator (Traversable trvInput, CodeGeneratorRuntimeOptions options);

	@Override
	public StatementListBundle generate (Traversable trvInput, CodeGeneratorRuntimeOptions options)
	{
		if (AbstractSubdomainIteratorCodeGenerator.LOGGER.isDebugEnabled ())
			AbstractSubdomainIteratorCodeGenerator.LOGGER.debug (StringUtil.concat ("Generating code with options ", options.toString ()));

		if (!(trvInput instanceof SubdomainIterator))
			throw new RuntimeException (StringUtil.concat (getClass ().getName (), " can only be used to generate code for SubgridIterators."));

		CodeGenerator cg = createCodeGenerator ((SubdomainIterator) trvInput, options);

		StatementListBundle slbGenerated = cg.generate ();
		m_data.getData ().getMemoryObjectManager ().resetIndices ();

		return slbGenerated;
	}
}
