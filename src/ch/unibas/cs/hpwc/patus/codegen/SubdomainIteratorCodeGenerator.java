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

import org.apache.log4j.Logger;

import cetus.hir.AssignmentExpression;
import cetus.hir.AssignmentOperator;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.CompoundStatement;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.ForLoop;
import cetus.hir.IDExpression;
import cetus.hir.Identifier;
import cetus.hir.IntegerLiteral;
import cetus.hir.NullStatement;
import cetus.hir.Statement;
import cetus.hir.Traversable;
import cetus.hir.UnaryExpression;
import cetus.hir.UnaryOperator;
import ch.unibas.cs.hpwc.patus.ast.StatementList;
import ch.unibas.cs.hpwc.patus.ast.StatementListBundle;
import ch.unibas.cs.hpwc.patus.ast.SubdomainIdentifier;
import ch.unibas.cs.hpwc.patus.ast.SubdomainIterator;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;
import ch.unibas.cs.hpwc.patus.symbolic.Symbolic;
import ch.unibas.cs.hpwc.patus.util.ExpressionUtil;
import ch.unibas.cs.hpwc.patus.util.StatementListBundleUtil;

/**
 * Implements the code generator for subdomain iterators, also handling
 * synchronous or asynchronous data transfers.
 *
 * @author Matthias-M. Christen
 */
public class SubdomainIteratorCodeGenerator extends AbstractSubdomainIteratorCodeGenerator
{
	///////////////////////////////////////////////////////////////////
	// Constants and Static Types

	private final static Logger LOGGER = Logger.getLogger (SubdomainIteratorCodeGenerator.class);


	///////////////////////////////////////////////////////////////////
	// Inner Types

	protected class CodeGenerator extends AbstractSubdomainIteratorCodeGenerator.CodeGenerator
	{
		///////////////////////////////////////////////////////////////////
		// Member Variables

		private boolean m_bLoadData;


		///////////////////////////////////////////////////////////////////
		// Implementation

		public CodeGenerator (SubdomainIterator sgit, CodeGeneratorRuntimeOptions options)
		{
			super (sgit, options);

			m_bLoadData = m_data.getCodeGenerators ().getStrategyAnalyzer ().isDataLoadedInIterator (m_sdIterator, m_data.getArchitectureDescription ());
//XXX
//m_bLoadData = false;
//m_bLoadData = true;
		}

		/**
		 * Generates the C code implementing the subdomain iterator.
		 */
		public StatementListBundle generate ()
		{
			if (m_bLoadData)
				return generateWithDataTransfers ();

			return super.generate();
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
			if (m_sdIterator.getDomainSubdomain ().isBaseGrid ())
				return m_sdIterator.getDomainIdentifier ().getSubdomain ().getBox ().getMin ().getCoord (nDim).clone ();

			Expression exprMin = m_data.getData ().getGeneratedIdentifiers ().getDimensionIndexIdentifier (m_sdIterator.getDomainIdentifier (), nDim).clone ();
			if (exprMin == null)
				exprMin = m_sdIterator.getDomainSubdomain ().getBox ().getMin ().getCoord (nDim).clone ();
			
			return exprMin;
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
		public StatementListBundle generateIteratorForDimension (
			int nDim, StatementListBundle slGenerated, Expression exprStartOffset, Expression exprNegEndOffset, int nUnrollingFactor)
		{
			Expression exprIteratorSize = m_sdIterator.getIteratorSubdomain ().getBox ().getSize ().getCoord (nDim);
			Expression exprDomainSize = m_sdIterator.getDomainSubdomain ().getBox ().getSize ().getCoord (nDim);
			
			boolean bDoBoundaryChecks = m_options.getBooleanValue (CodeGeneratorRuntimeOptions.OPTION_DOBOUNDARYCHECKS, false);

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

			Expression exprStart = exprStartOffset == AbstractSubdomainIteratorCodeGenerator.NULL_EXPRESSION ? null : getLowerLoopBound (nDim);
			if (exprStartOffset != null && exprStartOffset != AbstractSubdomainIteratorCodeGenerator.NULL_EXPRESSION)
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
			if (m_bContainsStencilCall && nDim == 0 && (isStencilCalculation () || m_data.getOptions ().useNativeSIMDDatatypes ()) && !bDoBoundaryChecks)
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
			slbGenerated.addStatement (new ExpressionStatement (new AssignmentExpression (
				m_idCounter.clone (), AssignmentOperator.NORMAL, new IntegerLiteral (0))));

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
		 * 
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
		@SuppressWarnings("unused")
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
	// Implementation

	public SubdomainIteratorCodeGenerator (ICodeGenerator cgParent, CodeGeneratorSharedObjects data)
	{
		super (cgParent, data);
	}

	@Override
	protected CodeGenerator createCodeGenerator (Traversable trvInput, CodeGeneratorRuntimeOptions options)
	{
		return new CodeGenerator ((SubdomainIterator) trvInput, options);
	}
}
