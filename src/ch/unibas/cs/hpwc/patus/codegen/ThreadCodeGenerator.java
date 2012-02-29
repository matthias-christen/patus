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
import java.util.List;

import cetus.hir.AccessExpression;
import cetus.hir.AnnotationStatement;
import cetus.hir.AssignmentExpression;
import cetus.hir.AssignmentOperator;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.CompoundStatement;
import cetus.hir.DeclarationStatement;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.FunctionCall;
import cetus.hir.IDExpression;
import cetus.hir.Identifier;
import cetus.hir.IfStatement;
import cetus.hir.IntegerLiteral;
import cetus.hir.Literal;
import cetus.hir.NameID;
import cetus.hir.Statement;
import cetus.hir.SymbolTools;
import cetus.hir.Traversable;
import cetus.hir.VariableDeclarator;
import ch.unibas.cs.hpwc.patus.analysis.LoopAnalyzer;
import ch.unibas.cs.hpwc.patus.analysis.StrategyAnalyzer;
import ch.unibas.cs.hpwc.patus.ast.IndexBoundsCalculationInsertionAnnotation;
import ch.unibas.cs.hpwc.patus.ast.Loop;
import ch.unibas.cs.hpwc.patus.ast.RangeIterator;
import ch.unibas.cs.hpwc.patus.ast.StatementListBundle;
import ch.unibas.cs.hpwc.patus.ast.SubdomainIterator;
import ch.unibas.cs.hpwc.patus.codegen.backend.IIndexing.IIndexingLevel;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;
import ch.unibas.cs.hpwc.patus.geometry.Point;
import ch.unibas.cs.hpwc.patus.geometry.Size;
import ch.unibas.cs.hpwc.patus.symbolic.ExpressionOptimizer;
import ch.unibas.cs.hpwc.patus.symbolic.NotConvertableException;
import ch.unibas.cs.hpwc.patus.symbolic.Symbolic;
import ch.unibas.cs.hpwc.patus.util.CodeGeneratorUtil;
import ch.unibas.cs.hpwc.patus.util.ExpressionUtil;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 * Generates the function executed by a single thread.
 *
 * @author Matthias-M. Christen
 */
public class ThreadCodeGenerator
{
	///////////////////////////////////////////////////////////////////
	// Constants

	//private final static Logger LOGGER = Logger.getLogger (ThreadCodeGenerator.class);

	
	///////////////////////////////////////////////////////////////////
	// Member Variables

	/**
	 * The shared data containing all the information for the code generator
	 * (including the stencil calculation and the strategy)
	 */
	private CodeGeneratorSharedObjects m_data;

	
	///////////////////////////////////////////////////////////////////
	// Implementation

	/**
	 *
	 * @param strategy
	 * @param stencil
	 * @param cgThreadFunction
	 */
	public ThreadCodeGenerator (CodeGeneratorSharedObjects data)
	{
		m_data = data;
	}

	/**
	 * Generates the code executed per thread.
	 */
	public CompoundStatement generate (Statement stmtInput, CodeGeneratorRuntimeOptions options)
	{
		// create the function body
		CompoundStatement cmpstmtFunctionBody = new CompoundStatement ();
		generateLoop (stmtInput, cmpstmtFunctionBody, options);

		return cmpstmtFunctionBody;
	}

	/**
	 * Generates the C loop nest.
	 *
	 * @param stmtInput
	 *            The statement to process
	 * @param cmpstmtOutput
	 *            The output structure to which the generated code will be
	 *            appended
	 * @param nLoopNestDepth
	 *            The current loop nesting depth
	 */
	private void generateLoop (Statement stmtInput, CompoundStatement cmpstmtOutput, CodeGeneratorRuntimeOptions options)
	{
		for (Traversable t : stmtInput.getChildren ())
		{
			// add the strategy code as a comment
			if (!(t instanceof DeclarationStatement))
				CodeGeneratorUtil.addComment (cmpstmtOutput, t.toString ());

			// add the code
			if (t instanceof RangeIterator)
				generateRangeIterator ((RangeIterator) t, stmtInput, cmpstmtOutput, options);
			else if (t instanceof SubdomainIterator)
				generateSubdomainIterator ((SubdomainIterator) t, stmtInput, cmpstmtOutput, options);
			else if (t instanceof Statement && !(t instanceof DeclarationStatement))
				cmpstmtOutput.addStatement (((Statement) t).clone ());
		}
	}

	/**
	 * Creates the C code for a range iterator.
	 *
	 * @param loop
	 * @param stmtInput
	 * @param cmpstmtOutput
	 */
	protected void generateRangeIterator (RangeIterator loop, Statement stmtInput, CompoundStatement cmpstmtOutput, CodeGeneratorRuntimeOptions options)
	{
		// create the code for the children
		CompoundStatement cmpstmtLoopBody = new CompoundStatement ();
		cmpstmtLoopBody.setParent (cmpstmtOutput);
		generateLoop (loop, cmpstmtLoopBody, options);

		Expression exprTripCount = LoopAnalyzer.getConstantTripCount (loop.getStart (), loop.getEnd (), loop.getStep ());
		if (exprTripCount != null && ExpressionUtil.isValue (exprTripCount, 1))
		{
			// trip count is 1: omit the loop
			StatementListBundle slb = new StatementListBundle (cmpstmtOutput);

			cmpstmtLoopBody.setParent (null);
			slb.addStatement (cmpstmtLoopBody);
			m_data.getData ().getMemoryObjectManager ().swapMemoryObjectPointers (null, slb, options);
		}
		else
		{
			if (!loop.isSequential ())
			{
				// replace a parallel loop by the part a thread executes
				switch (m_data.getCodeGenerators ().getBackendCodeGenerator ().getThreading ())
				{
				case MULTI:
					// calculate linearized block indices from the linearized
					// thread index
					generateMultiCoreRangeIterator (loop, cmpstmtLoopBody, cmpstmtOutput, options);
					break;

				case MANY:
					// calculate ND block indices from an ND thread index
					generateManyCoreRangeIterator (loop, cmpstmtLoopBody, cmpstmtOutput, options);
					break;

				default:
					throw new RuntimeException (StringUtil.concat ("Code generation for ", m_data.getCodeGenerators ().getBackendCodeGenerator ().getThreading (), " not implemented"));
				}
			}
			else
			{
				// sequential loop: leave the loop untouched
				RangeIterator loopGenerated = (RangeIterator) loop.clone ();
				loopGenerated.setLoopBody (cmpstmtLoopBody);
				cmpstmtOutput.addStatement (loopGenerated);
			}

			// generate the pointer swapping code if this is the inner most
			// temporal loop
			if (m_data.getCodeGenerators ().getStrategyAnalyzer ().isInnerMostTemporalLoop (loop))
			{
				StatementListBundle slbLoopBody = new StatementListBundle (cmpstmtLoopBody);
				m_data.getData ().getMemoryObjectManager ().swapMemoryObjectPointers (loop, slbLoopBody, options);
			}
		}
	}

	/**
	 * Replace a parallel loop by the part a thread executes:
	 *
	 * <pre>
	 * for i = start .. end [by stride] parallel [#thds] [schedule chunksize]
	 * |    &lt;Code(i)>
	 * +--
	 *
	 * 	                             ==>
	 *
	 * for j = (start + rank * stride * chunksize) .. end  by (stride * chunksize * #thds)
	 * |    maxi = j + stride * chunksize - 1
	 * |    for i = j .. maxi  by stride
	 * |    |    &lt;Code(i)>
	 * +--  +--
	 * </pre>
	 *
	 * default chunksize = ceil((end - start + 1) / (stride * #thds)) </pre>
	 *
	 * @param loop
	 * @param cmpstmtLoopBody
	 * @param cmpstmtOutput
	 */
	protected void generateMultiCoreRangeIterator (RangeIterator loop, CompoundStatement cmpstmtLoopBody, CompoundStatement cmpstmtOutput, CodeGeneratorRuntimeOptions options)
	{
		// loop index for the outer loop
		Identifier idJ = SymbolTools.getTemp (cmpstmtOutput, loop.getLoopIndex (false).getSymbol ().getTypeSpecifiers (), loop.getLoopIndex ().getName ());

		// stride for the outer loop
		// strideJ = stride * chunksize * #thds
		Identifier idStrideJ = SymbolTools.getTemp (cmpstmtOutput, loop.getLoopIndex (false).getSymbol ().getTypeSpecifiers (), "stride");
		cmpstmtOutput.addStatement (new ExpressionStatement (new AssignmentExpression (
			idStrideJ,
			AssignmentOperator.NORMAL,
			ExpressionUtil.product (loop.getStep (), loop.getChunkSize (0), m_data.getCodeGenerators ().getLoopCodeGenerator ().getNumberOfThreadsInDimension (loop, 0, options))
		)));

		CompoundStatement cmpstmtLoopOuterBody = new CompoundStatement ();

		// generate the "end" for the inner i loop:
		// idx_max = j + stride * chunksize - 1
		Identifier idEndI = SymbolTools.getTemp (cmpstmtOutput, loop.getLoopIndex (false).getSymbol ().getTypeSpecifiers (), "max");
		cmpstmtLoopOuterBody.addStatement (new ExpressionStatement (new AssignmentExpression (
			idEndI,
			AssignmentOperator.NORMAL,
			ExpressionUtil.sum (idJ, ExpressionUtil.product (loop.getStep (), loop.getChunkSize (0)), new IntegerLiteral (-1))
		)));

		// create the inner loop
		cmpstmtLoopOuterBody.addStatement (new RangeIterator (loop.getLoopIndex (), idJ, idEndI, loop.getStep (), cmpstmtLoopBody.clone (), loop.getParallelismLevel ()));

		// outer loop
		cmpstmtOutput.addStatement (new RangeIterator (
			idJ,
			ExpressionUtil.sum (
				loop.getStart (),
				ExpressionUtil.product (
					/* Globals.getThreadNumber () */m_data.getCodeGenerators ().getIndexCalculator ().calculateIndicesFromHardwareIndices (new Size (Integer.MAX_VALUE), cmpstmtLoopOuterBody, options)[0],
					loop.getStep (),
					loop.getChunkSize (0)
				)
			),
			loop.getEnd (),
			idStrideJ,
			cmpstmtLoopOuterBody,
			loop.getParallelismLevel ()
		));

		// add synchronization point
		Statement stmtBarrier = m_data.getCodeGenerators ().getBackendCodeGenerator ().getBarrier (loop.getParallelismLevel ());
		if (stmtBarrier != null)
			cmpstmtOutput.addStatement (stmtBarrier);
	}

	/**
	 *
	 * @param loop
	 * @param cmpstmtLoopBody
	 * @param cmpstmtOutput
	 */
	protected void generateManyCoreRangeIterator (RangeIterator loop, CompoundStatement cmpstmtLoopBody, CompoundStatement cmpstmtOutput, CodeGeneratorRuntimeOptions options)
	{
		// for now, simply call the multicore implementation...
		generateMultiCoreRangeIterator (loop, cmpstmtLoopBody, cmpstmtOutput, options);
	}

	protected void generateSubdomainIterator (
		SubdomainIterator loop, Statement stmtInput, CompoundStatement cmpstmtOutput, CodeGeneratorRuntimeOptions options)
	{
		// create the code for the children
		CompoundStatement cmpstmtLoopBody = new CompoundStatement ();
		cmpstmtLoopBody.setParent (cmpstmtOutput);
		generateLoop (loop, cmpstmtLoopBody, options);

		// add the annotation telling where to add index bounds calculations
		CodeGeneratorUtil.addStatementAtTop (cmpstmtLoopBody, new AnnotationStatement (new IndexBoundsCalculationInsertionAnnotation (loop)));

		// determine whether the loop contains a stencil call
		boolean bContainsStencilCall = StrategyAnalyzer.directlyContainsStencilCall (loop);

		// // if the loop entails data transfers, let the
		// SubdomainIteratorCodeGenerator do the parallelization
		if (!loop.isSequential () /* && !m_data.getCodeGenerators ().getStrategyAnalyzer ().isDataLoadedInIterator (loop, m_data.getArchitectureDescription ()) */)
		{
			// replace a parallel loop by the part a thread executes
			switch (m_data.getCodeGenerators ().getBackendCodeGenerator ().getThreading ())
			{
			case MULTI:
				// calculate linearized block indices from the linearized thread index
				generateMultiCoreSubdomainIterator (loop, cmpstmtLoopBody, cmpstmtOutput, bContainsStencilCall, options);
				break;

			case MANY:
				// calculate ND block indices from an ND thread index
				generateManyCoreSubdomainIterator (loop, cmpstmtLoopBody, cmpstmtOutput, bContainsStencilCall, options);
				break;

			default:
				throw new RuntimeException (StringUtil.concat ("Code generation for ", m_data.getCodeGenerators ().getBackendCodeGenerator ().getThreading (), " not implemented"));
			}
			
//			generateMultiCoreSubdomainIterator (loop, cmpstmtLoopBody, cmpstmtOutput, bContainsStencilCall, options);
		}
		else
		{
			// leave the loop untouched
			Loop loopGenerated = loop.clone ();
			loopGenerated.setLoopBody (cmpstmtLoopBody);
			cmpstmtOutput.addStatement (loopGenerated);
		}
	}

	/**
	 * replace a parallel loop by the part a thread executes:
	 *
	 * [ Convention: x is unit stride ]
	 *
	 * <pre>
	 * for subdomain v(sx, sy, sz) in u parallel [schedule chunksize]
	 * |    &lt;Code(v)>
	 * +-
	 *
	 *                             ==>
	 *
	 * numblocks = prod_{i} (ceil (iterator_domain.sx_{i} / iterator.sx_{i}))
	 * for j = thdid .. numblocks - 1  by chunksize * #thds
	 * |    // loop over blocks in a chunk
	 * |    for i = j .. j + chunksize - 1
	 * |    |    calculate ref addr @i
	 * |    |    &lt;Code(@i)>
	 * +-   +-
	 * </pre>
	 *
	 * if sequential, #thds=1, chunksize=1, hence: (when unrolling the inner
	 * loop, this will be automatically created)
	 *
	 * <pre>
	 * numblocks = prod_{i} (ceil (iterator_domain.sx_{i} / iterator.sx_{i}))
	 * for i = 0 .. numblocks - 1
	 * |    calculate ref addr @i
	 * |    &lt;Code(@i)>
	 * +-
	 *
	 *
	 *
	 * for plane p in v parallel [#thds] [schedule chunksize]
	 *     &lt;Code(p)>
	 *
	 *                             ==>
	 *
	 *
	 * for point p in v parallel [#thds] [schedule chunksize]
	 *     &lt;Code(p)>
	 *
	 *                             ==>
	 *
	 * </pre>
	 *
	 * @param loop
	 * @param cmpstmtLoopBody
	 * @param cmpstmtOutput
	 * @param bContainsStencilCall
	 */
	protected void generateMultiCoreSubdomainIterator0 (
		SubdomainIterator loop, CompoundStatement cmpstmtLoopBody, CompoundStatement cmpstmtOutput, boolean bContainsStencilCall,
		CodeGeneratorRuntimeOptions options)
	{
		// compute the number of blocks
		// int xxx_blocks_cnt = prod_i (ceil (iterator_domain.sx(i) / iterator.sx(i)))
		Identifier idNumBlocks = m_data.getData ().getGeneratedIdentifiers ().getNumBlocksIdentifier (loop.getIterator ());

		// TODO: data transfers


		// create loop indices and the inner loop if the chunksize is not 1
		// idJ is the index of the outer loop
		Identifier idJ = null;
		Loop loopInner = null;
		boolean bHasNonUnitChunk = Symbolic.isTrue (new BinaryExpression (loop.getChunkSize (0), BinaryOperator.COMPARE_EQ, new IntegerLiteral (1)), null) != Symbolic.ELogicalValue.TRUE; 
		if (bHasNonUnitChunk)
		{
			// create the loop indices for the outer loop
			NameID nidJ = new NameID (StringUtil.concat (loop.getIterator ().getName (), "_idxouter"));
			idJ = new Identifier (new VariableDeclarator (Globals.SPECIFIER_INDEX, nidJ));

			//*
			loopInner = new RangeIterator (
				m_data.getData ().getGeneratedIdentifiers ().getIndexIdentifier (loop.getIterator ()).clone (),
				idJ,
				new BinaryExpression (idJ, BinaryOperator.ADD, ExpressionUtil.decrement (loop.getChunkSize (0))),
				new IntegerLiteral (bContainsStencilCall ? m_data.getCodeGenerators ().getStencilCalculationCodeGenerator ().getLcmSIMDVectorLengths () : 1),
				null,	// start with an empty body; it will be assigned later (since RangeIterator clones the body)
				loop.getParallelismLevel ()
			);//*/
/*
			Subdomain sgDomain = new Subdomain (null, Subdomain.ESubdomainType.SUBDOMAIN, new Size (loop.getChunkSize ()));
			SubdomainIdentifier sdidDomain = new SubdomainIdentifier ("a", sgDomain);
			SubdomainIdentifier sdidIterator = new SubdomainIdentifier ("b", new Subdomain (sgDomain, Subdomain.ESubdomainType.POINT, new Point (idJ.clone ()), new Size (1)));
			loopInner = new SubdomainIterator (
				sdidIterator,
				sdidDomain,
				new Border ((byte) 1),
				1,
				Globals.ONE.clone (),
				null,
				loop.getParallelismLevel ());
//*/
		}
		else
			idJ = m_data.getData ().getGeneratedIdentifiers ().getIndexIdentifier (loop.getIterator ()).clone ();


		/////////////////////////////////
		// add bound variables

		// add an assignment to the loop body that calculates the box bounds
		// 		subdomain = domain + i * iterator
		// and the actual code to calculate the lower and upper bounds

		Size sizeIterator = loop.getIteratorSubdomain ().getSize ();
		byte nDim = loop.getDomainSubdomain ().getBox ().getDimensionality ();
		SubdomainGeneratedIdentifiers ids = m_data.getData ().getGeneratedIdentifiers ();

		// calculate the multi-dimensional index
		CompoundStatement cmpstmtBoundsCalculations = new CompoundStatement ();
		Identifier[] rgMin = new Identifier[nDim];
		Expression[] rgNumBlocks = new Expression[nDim];
		for (int i = 0; i < nDim; i++)
		{
			// create the minimum identifiers
			rgMin[i] = ids.getDimensionIndexIdentifier (loop.getIterator (), i);

			// calculate the number of blocks per dimension
			rgNumBlocks[i] = ExpressionUtil.ceil (
				loop.getDomainSubdomain ().getSize ().getCoord (i).clone (),
				sizeIterator.getCoord (i).clone ());
		}

		// create a multi-dimensional index from the 1-dimensional loop index ids.getIndexIdentifier (loop.getIterator ())
		m_data.getCodeGenerators ().getIndexCalculator ().calculateOneToMulti (
			ids.getIndexIdentifier (loop.getIterator ()).clone (),
			rgMin, rgNumBlocks, null, cmpstmtBoundsCalculations);

		// multiply the indices by the size of the grid and calculate the upper bounds
		for (int i = 0; i < nDim; i++)
		{
			// adjust the lower bound
			cmpstmtBoundsCalculations.addStatement (new ExpressionStatement (new AssignmentExpression (
				rgMin[i].clone (),
				AssignmentOperator.NORMAL,
				new BinaryExpression (
					new BinaryExpression (rgMin[i].clone (), BinaryOperator.MULTIPLY, sizeIterator.getCoord (i).clone ()),
					BinaryOperator.ADD,
					loop.getDomainSubdomain ().getLocalCoordinates ().getCoord (i).clone ()	// TODO: check whether this is correct for subdomains
				)
			)));

			// upper bounds
			Identifier idMax = ids.getDimensionMaxIdentifier (loop.getIterator (), i).clone ();
			cmpstmtBoundsCalculations.addStatement (new ExpressionStatement (new AssignmentExpression (
				idMax,
				AssignmentOperator.NORMAL,
				ExpressionUtil.min (
					// calculated maximum
					new BinaryExpression (rgMin[i].clone (), BinaryOperator.ADD, sizeIterator.getCoord (i).clone ()),
					// maximum grid index + 1 (+1 since the for loops don't include the maximum)
					new BinaryExpression (m_data.getStencilCalculation ().getDomainSize ().getMax ().getCoord (i).clone (), BinaryOperator.ADD, Globals.ONE.clone ())
				)
			)));
		}

		// determine the location where the index calculation is inserted
		// (the number-of-blocks calculation doesn't depend on the loop index (v_idx), but the index bounds (v_x_min, v_x_max, ...) do
		CompoundStatement cmpstmtIndexCalculation = getIndexCalculationLocation (loop, cmpstmtLoopBody);

		Expression exprNumBlocks = loop.getNumberOfBlocks ();
		try
		{
			exprNumBlocks = ExpressionOptimizer.optimize (exprNumBlocks, Symbolic.ALL_VARIABLES_INTEGER);
		}
		catch (NotConvertableException e)
		{
			// something went wrong when trying to optimize the expression... ignore and use the original
		}

		Statement stmtInit = new ExpressionStatement (new AssignmentExpression (idNumBlocks, AssignmentOperator.NORMAL, exprNumBlocks));
		if (cmpstmtIndexCalculation == null)
			m_data.getData ().addInitializationStatement (stmtInit);
		else
			CodeGeneratorUtil.addStatements (cmpstmtIndexCalculation, stmtInit);

		CodeGeneratorUtil.addStatementsAtTop (cmpstmtLoopBody, cmpstmtBoundsCalculations);

		// add the body to the inner loop after the last modification
		if (loopInner != null)
			loopInner.setLoopBody (cmpstmtLoopBody);


		/////////////////////////////////
		// create the outer loop

		Expression exprStartOuter = /*Globals.getThreadNumber ()*/ m_data.getCodeGenerators ().getIndexCalculator ().calculateIndicesFromHardwareIndices (new Size (Integer.MAX_VALUE), cmpstmtOutput, options)[0];
		if (bHasNonUnitChunk)
			exprStartOuter = new BinaryExpression (loop.getChunkSize (0), BinaryOperator.MULTIPLY, exprStartOuter);
		
 		RangeIterator loopOuter = new RangeIterator (
			idJ,
			exprStartOuter,
			new BinaryExpression (idNumBlocks.clone (), BinaryOperator.SUBTRACT, Globals.ONE.clone ()),
			Symbolic.simplify (
				ExpressionUtil.product (
					loop.getChunkSize (0),
					m_data.getCodeGenerators ().getLoopCodeGenerator ().getNumberOfThreadsInDimension (loop, 0, options), /*loop.getNumberOfThreads (),*/
					// account for SIMD if there is no inner loop and the loop contains a stencil call
					new IntegerLiteral (loopInner == null && bContainsStencilCall ?
						m_data.getCodeGenerators ().getStencilCalculationCodeGenerator ().getLcmSIMDVectorLengths () : 1)
				),
				Symbolic.ALL_VARIABLES_INTEGER
			),
			loopInner == null ? cmpstmtLoopBody : loopInner,
			loop.getParallelismLevel ());

		// add the loop to the output statement
		cmpstmtOutput.addStatement (loopOuter);

		// synchronize
		Statement stmtBarrier = m_data.getCodeGenerators ().getBackendCodeGenerator ().getBarrier (loop.getParallelismLevel ());
		if (stmtBarrier != null)
			cmpstmtOutput.addStatement (stmtBarrier);
	}
	
	/**
	 * Returns the maximum indexing dimension in the chosen hardware architecture.
	 * @return The maximum indexing dimension
	 */
	private int getMaxIndexDimension ()
	{
		int nMaxIndexDimension = 0;
		for (int i = 0; i < m_data.getCodeGenerators ().getBackendCodeGenerator ().getIndexingLevelsCount (); i++)
		{
			IIndexingLevel level = m_data.getCodeGenerators ().getBackendCodeGenerator ().getIndexingLevel (i);
			nMaxIndexDimension = Math.max (nMaxIndexDimension, level.getDimensionality ());
		}

		return nMaxIndexDimension;
	}
	
	private Expression getIdentifier (Expression exprOrig, String strIdentifier, StatementListBundle slb, CodeGeneratorRuntimeOptions options)
	{
		if (exprOrig instanceof IDExpression || exprOrig instanceof Literal || exprOrig instanceof AccessExpression)
			return exprOrig;
		
		return m_data.getCodeGenerators ().getConstantGeneratedIdentifiers ().getConstantIdentifier (
			exprOrig, strIdentifier, Globals.SPECIFIER_SIZE, slb, null, options);
	}
	
	/**
	 * Adds the bounds variables to the original loop body.
	 * @param loop
	 * @param stmtOrignalBody
	 * @param nDimStart
	 * @param nDimEnd
	 * @return
	 */
	private CompoundStatement createLoopBody (SubdomainIterator loop, Statement stmtOrignalBody, int nDimStart, int nDimEnd, boolean bUseBlockIndices)
	{
		CompoundStatement cmpstmtBody = new CompoundStatement ();

		// add an assignment to the loop body that calculates the box bounds
		// 		subdomain = domain + i * iterator
		// and the actual code to calculate the lower and upper bounds

		Size sizeIterator = loop.getIteratorSubdomain ().getSize ();
		SubdomainGeneratedIdentifiers ids = m_data.getData ().getGeneratedIdentifiers ();

		// calculate the multi-dimensional index
		Identifier[] rgMin = new Identifier[nDimEnd - nDimStart + 1];
		Expression[] rgNumBlocks = new Expression[nDimEnd - nDimStart + 1];
		for (int i = nDimStart; i <= nDimEnd; i++)
		{
			// create the minimum identifiers
			rgMin[i - nDimStart] = ids.getDimensionIndexIdentifier (loop.getIterator (), i);

			// calculate the number of blocks per dimension
			if (bUseBlockIndices)
			{
				rgNumBlocks[i - nDimStart] = ExpressionUtil.ceil (
					loop.getDomainSubdomain ().getSize ().getCoord (i).clone (),
					sizeIterator.getCoord (i).clone ());
			}
		}

		// create a multi-dimensional index from the 1-dimensional loop index ids.getIndexIdentifier (loop.getIterator ())
		m_data.getCodeGenerators ().getIndexCalculator ().calculateOneToMulti (
			//ids.getIndexIdentifier (loop.getIterator ()).clone (),
			ids.getDimensionBlockIndexIdentifier (loop.getIterator (), nDimStart).clone (),
			rgMin, rgNumBlocks, null, cmpstmtBody);
		
		// multiply the indices by the size of the grid and calculate the upper bounds
		for (int i = nDimStart; i <= nDimEnd; i++)
		{
			// adjust the lower bound
			cmpstmtBody.addStatement (new ExpressionStatement (new AssignmentExpression (
				rgMin[i - nDimStart].clone (),
				AssignmentOperator.NORMAL,
				new BinaryExpression (
					bUseBlockIndices ?
						new BinaryExpression (rgMin[i - nDimStart].clone (), BinaryOperator.MULTIPLY, sizeIterator.getCoord (i).clone ()) :
						rgMin[i - nDimStart].clone (),
					BinaryOperator.ADD,
					loop.getDomainSubdomain ().getLocalCoordinates ().getCoord (i).clone ()	// TODO: check whether this is correct for subdomains
				)
			)));
			
			// upper bounds
			Identifier idMax = ids.getDimensionMaxIdentifier (loop.getIterator (), i).clone ();
			cmpstmtBody.addStatement (new ExpressionStatement (new AssignmentExpression (
				idMax,
				AssignmentOperator.NORMAL,
				ExpressionUtil.min (
					// calculated maximum
					new BinaryExpression (rgMin[i - nDimStart].clone (), BinaryOperator.ADD, sizeIterator.getCoord (i).clone ()),
					// maximum grid index + 1 (+1 since the for loops don't include the maximum)
					new BinaryExpression (m_data.getStencilCalculation ().getDomainSize ().getMax ().getCoord (i).clone (), BinaryOperator.ADD, Globals.ONE.clone ())
				)
			)));
		}
		
		cmpstmtBody.addStatement (stmtOrignalBody.clone ());
		return cmpstmtBody;
	}
	
	/**
	 * Generates a double loop nest for a single dimension. The outer loop
	 * iterates over chunks, the inner over chunked blocks. If the chunk is 1,
	 * only one loop is generated.
	 * 
	 * @param sdit
	 *            The original subdomain iterator
	 * @param exprStartOrig
	 *            The start of the loop, typically dependent on the thread ID
	 *            (for round-robin parallelization)
	 * @param exprEndOrig
	 *            The end of the domain
	 * @param exprBlockStepOrig
	 *            The step by which the index is increased in each iteration:
	 *            the size of the block in the respective dimension
	 * @param exprNumThreadsInDimOrig
	 *            The number of threads in the dimension for which to create the
	 *            loops
	 * @param stmtLoopBody
	 *            The loop body
	 * @param rgIndices
	 *            An array into which the index for the dimension for which the
	 *            loops are created is written
	 * @param cmpstmtInitializations
	 *            The compound statement to which initializations will be added
	 * @param nDimStart
	 *            The start dimension
	 * @param nDimEnd
	 *            The end dimension
	 * @param bUseBlockIndices
	 *            Flag specifying whether the loop indices are block indices.
	 *            Set to <code>true</code> if multiple dimensions are emulated
	 *            at once in one loop nest (i.e. for the last indexing dimension
	 *            if there are strictly less indexing dimensions than the
	 *            dimensionality)
	 * @param bContainsStencilCall
	 *            Flag specifying whether the subdomain iterator contains a
	 *            stencil call (and we therefore need to take care of modifying
	 *            the loop bounds for SIMD)
	 * @param options
	 *            Runtime code generation options
	 * 
	 * @return The generated loop nest
	 */
	private Statement generateLoopsForDim (SubdomainIterator sdit,
		Expression exprStartOrig, Expression exprEndOrig, Expression exprBlockStepOrig, Expression exprNumThreadsInDimOrig,
		Statement stmtLoopBody,
		Expression[] rgIndices, CompoundStatement cmpstmtInitializations, int nDimStart, int nDimEnd,
		boolean bUseBlockIndices, boolean bContainsStencilCall,
		CodeGeneratorRuntimeOptions options)
	{
		SubdomainGeneratedIdentifiers ids = m_data.getData ().getGeneratedIdentifiers ();
		CompoundStatement cmpstmtLoopBody = createLoopBody (sdit, stmtLoopBody, nDimStart, nDimEnd, bUseBlockIndices);
						
		StatementListBundle slbInit = new StatementListBundle (cmpstmtInitializations);
		
		// create identifiers for the start/end/num threads expressions if they are no IDExpressions or literals
		Expression exprStart = getIdentifier (exprStartOrig, "start", slbInit, options);
		Expression exprEnd = getIdentifier (exprEndOrig, "end", slbInit, options);
		Expression exprBlockStep = getIdentifier (exprBlockStepOrig, "step", slbInit, options);
		Expression exprNumThreadsInDim = getIdentifier (exprNumThreadsInDimOrig, "numthds", slbInit, options);
		
		// determine whether the outer loop is needed
		// (i.e., if the number of threads in the dimension (the step of the outer loop) is strictly less than the end expression)
		boolean bNoHasOuterLoop = Symbolic.isTrue (
			new BinaryExpression (exprNumThreadsInDimOrig.clone (), BinaryOperator.COMPARE_GE, exprEndOrig.clone ()), Symbolic.ALL_VARIABLES_INTEGER) == Symbolic.ELogicalValue.FALSE;

		if (bNoHasOuterLoop)
		{
			rgIndices[nDimStart] = exprStart.clone ();
			return cmpstmtLoopBody;
		}
		
		// compute the total chunk size for the dimensions being processed
		Expression exprChunkSize = sdit.getChunkSize (nDimStart);
		for (int i = nDimStart + 1; i <= nDimEnd; i++)
		{
			Expression exprChunkSizeForDim = sdit.getChunkSize (i);
			if (!ExpressionUtil.isValue (exprChunkSizeForDim, 1))
				exprChunkSize = new BinaryExpression (exprChunkSize, BinaryOperator.MULTIPLY, exprChunkSizeForDim);
		}
		
		// determine whether the inner loop is needed
		// (i.e., if there is a non-unit chunk)
		boolean bHasNonUnitChunk = Symbolic.isTrue (
			new BinaryExpression (exprChunkSize.clone (), BinaryOperator.COMPARE_EQ, new IntegerLiteral (1)), null) != Symbolic.ELogicalValue.TRUE;
		
		Identifier idInnerLoopIdx = ids.getDimensionBlockIndexIdentifier (sdit.getIterator (), nDimStart);
		Identifier idOuterLoopIdx = null;
		rgIndices[nDimStart] = idInnerLoopIdx.clone ();

		// create the inner loop
		RangeIterator itLoopInner = null;
		if (bHasNonUnitChunk)
		{
			idOuterLoopIdx = new Identifier (new VariableDeclarator (
				Globals.SPECIFIER_INDEX,
				new NameID (StringUtil.concat (idInnerLoopIdx.getName (), "_idxouter"))));
			
			// step: account for SIMD or just 1 if no stencil call/SIMD
			int nInnerStep = bContainsStencilCall ?
				m_data.getCodeGenerators ().getStencilCalculationCodeGenerator ().getLcmSIMDVectorLengths () : 1;

			itLoopInner = new RangeIterator (
				idInnerLoopIdx.clone (),
				idOuterLoopIdx.clone (),
				
				// end = min (global_end, idxouter + chunk*blockstep - 1)
				Symbolic.simplify (new FunctionCall (Globals.FNX_MIN.clone (), CodeGeneratorUtil.expressions (
					exprEnd,
					new BinaryExpression (
						idOuterLoopIdx.clone (),
						BinaryOperator.ADD,
						ExpressionUtil.decrement (new BinaryExpression (exprChunkSize.clone (), BinaryOperator.MULTIPLY, exprBlockStep.clone ())))
					)
				)),
					
				// step
				nInnerStep == 1 ? exprBlockStep.clone () : new BinaryExpression (exprBlockStep.clone (), BinaryOperator.MULTIPLY, new IntegerLiteral (nInnerStep)),
					
				cmpstmtLoopBody,
				sdit.getParallelismLevel ()
			);
		}
		else
			idOuterLoopIdx = idInnerLoopIdx;
			
		// create the outer loop
		Expression exprOuterStep = getIdentifier (Symbolic.simplify (
			new BinaryExpression (exprNumThreadsInDim.clone (), BinaryOperator.MULTIPLY, new BinaryExpression (exprChunkSize.clone (), BinaryOperator.MULTIPLY, exprBlockStep.clone ()))),
			"stepouter", slbInit, options);
		
		return new RangeIterator (idOuterLoopIdx,
			bHasNonUnitChunk ? new BinaryExpression (exprStart.clone (), BinaryOperator.MULTIPLY, exprChunkSize.clone ()) : exprStart,
			exprEnd,
			exprOuterStep,
			itLoopInner == null ? cmpstmtLoopBody : itLoopInner,
			sdit.getParallelismLevel ()); 
	}	

	/**
	 * Multiplies the numbers of blocks in dimensions <code>nStartDim</code>, <code>nStartDim+1</code>, ..., dimensionality.
	 * @param loop The loop for which the calculate the number of blocks
	 * @param nStartDim The start dimension
	 * @return The total number of blocks in dimensions <code>nStartDim</code>, <code>nStartDim+1</code>, ..., dimensionality
	 */
	private Expression calculateNumberOfBlocksInMissingDims (SubdomainIterator loop, int nStartDim)
	{
		Expression exprTotalNumBlocks = null;
		
		for (int i = nStartDim; i < loop.getDomainIdentifier ().getDimensionality (); i++)
		{
			Expression exprNumBlocks = loop.getNumberOfBlocksInDimension (i);//ExpressionUtil.ceil (loop.getNumberOfBlocksInDimension (i), loop.getChunkSize (i));
			if (!ExpressionUtil.isValue (exprNumBlocks, 1))
			{
				if (exprTotalNumBlocks == null)
					exprTotalNumBlocks = exprNumBlocks;
				else
					exprTotalNumBlocks = new BinaryExpression (exprTotalNumBlocks, BinaryOperator.MULTIPLY, exprNumBlocks);
			}			
		}

		return exprTotalNumBlocks == null ? Globals.ONE.clone () : Symbolic.optimizeExpression (exprTotalNumBlocks);
	}
	
	/**
	 * 
	 * @param loop
	 * @param nStartDim
	 * @param nEndDim
	 * @return
	 */
	private Expression calculateDomainSize (SubdomainIterator loop, int nStartDim, int nEndDim)
	{
		Expression exprSize = loop.getTotalDomainSubdomain ().getSize ().getCoord (nStartDim).clone ();
		for (int i = nStartDim + 1; i <= nEndDim; i++)
			exprSize = new BinaryExpression (exprSize, BinaryOperator.MULTIPLY, loop.getTotalDomainSubdomain ().getSize ().getCoord (i).clone ());
		return exprSize;
	}
	
	/**
	 * 
	 * @param loop
	 * @param cmpstmtLoopBody
	 * @param cmpstmtOutput
	 * @param bContainsStencilCall
	 * @param options
	 */
	protected void generateMultiCoreSubdomainIterator (
		SubdomainIterator loop, CompoundStatement cmpstmtLoopBody, CompoundStatement cmpstmtOutput, boolean bContainsStencilCall,
		CodeGeneratorRuntimeOptions options)
	{
		IndexCalculatorCodeGenerator iccg = m_data.getCodeGenerators ().getIndexCalculator ();
		
		// TODO: data transfers
		
		// get the start and end parallelism level of the subdomain iterator "loop":
		// if "loop" is the inner-most parallel subdomain iterator it spans all the "remaining"
		// parallelism levels
		int nParallelismLevelStart = loop.getParallelismLevel ();
		int nParallelismLevelEnd = nParallelismLevelStart;
		if (m_data.getCodeGenerators ().getStrategyAnalyzer ().isInnerMostParallelLoop (loop))
			nParallelismLevelEnd = m_data.getCodeGenerators ().getBackendCodeGenerator ().getIndexingLevelsCount ();
		
		// create an array which will contain all the loop indices used in nested iterators or the stencil calculation.
		// these loop indices identify the point for which the stencil is calculated or the starting points for nested
		// subdomain iterators.
		// the array is filled by "generateLoopsForDim"
		int nMaxIndexDimension = Math.min (m_data.getStencilCalculation ().getDimensionality (), getMaxIndexDimension ());
		Expression[] rgIndices = new Expression[nMaxIndexDimension];
		
		// generate the loop nest
		Statement stmtBody = cmpstmtLoopBody;
		int nEnd = m_data.getStencilCalculation ().getDimensionality () > nMaxIndexDimension ? nMaxIndexDimension - 1 : nMaxIndexDimension;
		
		// all indexing dimensions if all of them are to be treated the same, or all but the
		// last if the last has to emulate missing dimensions
		for (int i = 0; i < nEnd; i++)
		{
			// create the loops for dimension i
			stmtBody = generateLoopsForDim (
				loop,
				
				// start
				new BinaryExpression (
					iccg.calculateHardwareIndicesToOne (i, nParallelismLevelStart, nParallelismLevelEnd),
					BinaryOperator.MULTIPLY,
					loop.getIteratorSubdomain ().getSize ().getCoord (i).clone ()
				),
				
				// end
				// decrement because "end" is expected to be inclusive in generateLoopsForDim
				ExpressionUtil.decrement (loop.getTotalDomainSubdomain ().getSize ().getCoord (i).clone ()),
				
				// block step
				loop.getIteratorSubdomain ().getSize ().getCoord (i),
				
				// number of hardware entities used
				// determines the step in the outer loop
				iccg.calculateTotalHardwareSize (i, nParallelismLevelStart, nParallelismLevelEnd),
				
				stmtBody,
				rgIndices,
				cmpstmtOutput,
				i,
				i,
				false,
				bContainsStencilCall,
				options
			);
		}
		
		// last indexing dimension, if it has to emulate multiple dimensions
		if (m_data.getStencilCalculation ().getDimensionality () > nMaxIndexDimension)
		{
			// calculate the end expression
			Expression exprEnd = calculateNumberOfBlocksInMissingDims (loop, nMaxIndexDimension - 1);
				
			// create the loops for dimension i
			stmtBody = generateLoopsForDim (
				loop,
				iccg.calculateHardwareIndicesToOne (nMaxIndexDimension - 1, nParallelismLevelStart, nParallelismLevelEnd),
				ExpressionUtil.decrement (exprEnd),	// decrement because "end" is expected to be inclusive in generateLoopsForDim
				Globals.ONE.clone (),
				iccg.calculateTotalHardwareSize (nMaxIndexDimension - 1, nParallelismLevelStart, nParallelismLevelEnd),
				stmtBody, rgIndices, cmpstmtOutput,
				nMaxIndexDimension - 1, loop.getDomainIdentifier ().getDimensionality () - 1,
				true, bContainsStencilCall, options);			
		}
				
		// add the loop to the output statement
		cmpstmtOutput.addStatement (stmtBody);
		
		// TODO: transfer data back

		// synchronize
		Statement stmtBarrier = m_data.getCodeGenerators ().getBackendCodeGenerator ().getBarrier (loop.getParallelismLevel ());
		if (stmtBarrier != null)
			cmpstmtOutput.addStatement (stmtBarrier);
	}

	private void loadData (SubdomainIterator loop, CompoundStatement cmpstmtLoopBody, CodeGeneratorRuntimeOptions options)
	{
		// has data transfers?
		boolean bHasDatatransfers = m_data.getCodeGenerators ().getStrategyAnalyzer ().isDataLoadedInIterator (loop, m_data.getArchitectureDescription ());

		// add data transfer code if required
		if (bHasDatatransfers)
		{
			if (true)
				throw new RuntimeException ("Not implemented");

			DatatransferCodeGenerator dtcg = m_data.getCodeGenerators ().getDatatransferCodeGenerator ();
			MemoryObjectManager mgr = m_data.getData ().getMemoryObjectManager ();

			// allocate memory objects
			dtcg.allocateLocalMemoryObjects (loop, options);

			// load and wait
			StatementListBundle slbLoad = new StatementListBundle (new ArrayList<Statement> ());
			StencilNodeSet setInputNodes = mgr.getInputStencilNodes (loop.getIterator ());

			dtcg.loadData (setInputNodes, loop, slbLoad, options);
			dtcg.waitFor (setInputNodes, loop, slbLoad, options);

			CodeGeneratorUtil.addStatementsAtTop (cmpstmtLoopBody, slbLoad.getDefaultList ().getStatementsAsList ());
		}		
	}
	
	private void storeData (SubdomainIterator loop, CompoundStatement cmpstmtLoopBody, CodeGeneratorRuntimeOptions options)
	{
		// has data transfers?
		boolean bHasDatatransfers = m_data.getCodeGenerators ().getStrategyAnalyzer ().isDataLoadedInIterator (loop, m_data.getArchitectureDescription ());

		// add datatransfer code if required: store
		if (bHasDatatransfers)
		{
			if (true)
				throw new RuntimeException ("Not implemented");

			DatatransferCodeGenerator dtcg = m_data.getCodeGenerators ().getDatatransferCodeGenerator ();
			MemoryObjectManager mgr = m_data.getData ().getMemoryObjectManager ();

			// store
			StatementListBundle slbStore = new StatementListBundle (new ArrayList<Statement> ());
			dtcg.storeData (mgr.getOutputStencilNodes (loop.getIterator ()), loop, slbStore, options);
			CodeGeneratorUtil.addStatements (cmpstmtLoopBody, slbStore.getDefaultList ().getStatementsAsList ());
		}
	}
	
	private Statement generateManyCoreLoopForDim (SubdomainIterator loop,
		Statement stmtLoopBody,
		Expression[] rgBlockIndices,
		int nDim, boolean bHasNestedLoops, boolean bContainsStencilCall,
		CodeGeneratorRuntimeOptions options)
	{
		SubdomainGeneratedIdentifiers ids = m_data.getData ().getGeneratedIdentifiers ();
		Point ptDomainBase = loop.getDomainSubdomain ().getBaseGridCoordinates ();
		
		Expression exprIteratorSizeInDim = loop.getIteratorSubdomain ().getBox ().getSize ().getCoord (nDim);
		Expression exprIteratorSizeWithChunk = ExpressionUtil.isValue (loop.getChunkSize (nDim), 1) ?
			exprIteratorSizeInDim : new BinaryExpression (exprIteratorSizeInDim.clone (), BinaryOperator.MULTIPLY, loop.getChunkSize (nDim));

		// v_*_min = u_*_min + idx_* * v_*_size
		Identifier idMin = ids.getDimensionIndexIdentifier (loop.getIterator (), nDim);
		Expression exprMin = Symbolic.optimizeExpression (new BinaryExpression (
			ptDomainBase.getCoord (nDim).clone (),
			BinaryOperator.ADD,
			new BinaryExpression (rgBlockIndices[nDim].clone (), BinaryOperator.MULTIPLY, exprIteratorSizeWithChunk.clone ())));
		
		// v_*_max = v_*_min + v_*_size
		Identifier idMax = null;
		Expression exprMax = null;
		if (bHasNestedLoops)
		{
			idMax = ids.getDimensionMaxIdentifier (loop.getIterator (), nDim).clone ();
			exprMax = new BinaryExpression (idMin.clone (), BinaryOperator.ADD, exprIteratorSizeWithChunk.clone ());

			// account for SIMD
			if (bContainsStencilCall && nDim == 0)
			{
				int nSIMDVectorLength = m_data.getCodeGenerators ().getStencilCalculationCodeGenerator ().getLcmSIMDVectorLengths ();
				if (nSIMDVectorLength > 1)
					exprMax = ExpressionUtil.ceil (exprMax, new IntegerLiteral (nSIMDVectorLength));
			}
			
			exprMax = Symbolic.simplify (exprMax, Symbolic.ALL_VARIABLES_INTEGER);
		}
		
		// prepare the new loop body
		CompoundStatement cmpstmtNewLoopBody = null;
		if (stmtLoopBody instanceof CompoundStatement && !(stmtLoopBody instanceof Loop))
			cmpstmtNewLoopBody = (CompoundStatement) stmtLoopBody;
		else
		{
			cmpstmtNewLoopBody = new CompoundStatement ();
			CodeGeneratorUtil.addStatements (cmpstmtNewLoopBody, stmtLoopBody.clone ());
		}
		
		if (ExpressionUtil.isValue (loop.getChunkSize (nDim), 1))
		{
			// default chunk size; no loop needed
			if (bHasNestedLoops)
				CodeGeneratorUtil.addStatementAtTop (cmpstmtNewLoopBody, new ExpressionStatement (new AssignmentExpression (idMax, AssignmentOperator.NORMAL, exprMax)));
			CodeGeneratorUtil.addStatementAtTop (cmpstmtNewLoopBody, new ExpressionStatement (new AssignmentExpression (idMin, AssignmentOperator.NORMAL, exprMin)));

			return cmpstmtNewLoopBody;
		}
		else
		{
			// non-default chunk size
			cmpstmtNewLoopBody.setParent (null);
			if (bHasNestedLoops)
				CodeGeneratorUtil.addStatementAtTop (cmpstmtNewLoopBody, new ExpressionStatement (new AssignmentExpression (idMax, AssignmentOperator.NORMAL, exprMax)));
			
			// create identifiers for the start/end/num threads expressions if they are no IDExpressions or literals
			StatementListBundle slbLoop = new StatementListBundle ();
			Expression exprStart = getIdentifier (exprMin, "start", slbLoop, options);
			

			slbLoop.addStatement (new RangeIterator (
				idMin.clone (),
				exprStart.clone (),
				new BinaryExpression (
					exprStart.clone (),
					BinaryOperator.ADD,
					ExpressionUtil.decrement (exprIteratorSizeWithChunk)),
				exprIteratorSizeInDim,
				cmpstmtNewLoopBody,
				loop.getParallelismLevel ()
			));
			
			return slbLoop.getDefault ();
		}
	}
	
	/**
	 * Generates a C version of a subdomain iterator for a &quot;many-core&quot;
	 * architecture. The generated code is added to <code>cmpstmtOutput</code>.
	 *
	 * @param loop
	 * @param cmpstmtLoopBody
	 * @param cmpstmtOutput
	 * @param bContainsStencilCall
	 */
	protected void generateManyCoreSubdomainIterator (SubdomainIterator loop, CompoundStatement cmpstmtLoopBody, CompoundStatement cmpstmtOutput, boolean bContainsStencilCall, CodeGeneratorRuntimeOptions options)
	{
		loadData (loop, cmpstmtLoopBody, options);
		
		int nDim = loop.getDomainIdentifier ().getDimensionality ();

		// adjust the size of the blocks if chunking is used
//		Size sizeIterator = loop.getIteratorSubdomain ().getBox ().getSize ();
//		for (int i = 0; i < sizeIterator.getDimensionality (); i++)
//			if (!ExpressionUtil.isValue (loop.getChunkSize (i), 1))
//				sizeIterator.setCoord (i, new BinaryExpression (sizeIterator.getCoord (0), BinaryOperator.MULTIPLY, loop.getChunkSize (i)));
		
		// expression checking whether v_*_min is within the parent iterator's bounds
		SubdomainGeneratedIdentifiers ids = m_data.getData ().getGeneratedIdentifiers ();
		Expression exprInBounds = null;
		for (int i = 0; i < nDim; i++)
		{
			Expression exprInBoundsLocal = new BinaryExpression (ids.getDimensionIndexIdentifier (loop.getIterator (), i).clone (), BinaryOperator.COMPARE_LE, loop.getDomainSubdomain ().getBox ().getMax ().getCoord (i).clone ());
			exprInBounds = exprInBounds == null ? exprInBoundsLocal : new BinaryExpression (exprInBounds, BinaryOperator.LOGICAL_AND, exprInBoundsLocal);
		}
		Statement stmtLoopBody = new IfStatement (exprInBounds, cmpstmtLoopBody.clone ());
		
		// calculate the block indices from the thread indices, add auxiliary
		// calculations to the initialization block (=> null)
		Expression[] rgBlockIndices = m_data.getCodeGenerators ().getIndexCalculator ().calculateIndicesFromHardwareIndices (loop.getNumberOfBlocksPerDimension (), null, options);
		boolean bHasNestedLoops = StrategyAnalyzer.hasNestedLoops (loop);

		for (int i = 0; i < nDim; i++)
			stmtLoopBody = generateManyCoreLoopForDim (loop, stmtLoopBody, rgBlockIndices, i, bHasNestedLoops, bContainsStencilCall, options);
		cmpstmtOutput.addStatement (stmtLoopBody);

//		// add the created bound to the generated code
//		// if there is a non-default chunk size (i.e. != 1), the first variable
//		// is captured in a loop, hence only add the bound variables except the first
//		List<Statement> listStmtBounds = new ArrayList<Statement> ();
//		for (int i = bHasNondefaultChunkSize ? 1 : 0; i < rgExprBounds.length; i++)
//			listStmtBounds.add ()[bHasNondefaultChunkSize ? i - 1 : i] = new ExpressionStatement (rgExprBounds[i]));
//
//		// determine the location where the index calculations are to be inserted
//		CompoundStatement cmpstmtIndexCalculation = getIndexCalculationLocation (loop, cmpstmtLoopBody);
//		if (cmpstmtIndexCalculation == null)
//			m_data.getData ().addInitializationStatements (rgStmtBounds);
//		else
//			CodeGeneratorUtil.addStatements (cmpstmtIndexCalculation, rgStmtBounds);
//
//		storeData (loop, cmpstmtLoopBody, options);
//		
//		// add guards to prevent execution if out of bounds
//		CompoundStatement cmpstmtNewLoopBody = null;
//		if (exprInBounds != null)
//		{
//			cmpstmtNewLoopBody = new CompoundStatement ();
//			cmpstmtLoopBody.setParent (null);
//			cmpstmtNewLoopBody.addStatement (new IfStatement (exprInBounds, cmpstmtLoopBody));
//		}
//		else
//			cmpstmtNewLoopBody = cmpstmtLoopBody;
//		
//
//		// create the loop for handling chunks
//		if (bHasNondefaultChunkSize)
//		{
//			Identifier idMin = ids.getDimensionIndexIdentifier (loop.getIterator (), 0);
//			Expression exprMin = ((AssignmentExpression) rgExprBounds[0]).getRHS ();
//
//			cmpstmtOutput.addStatement (new RangeIterator (
//				idMin.clone (),
//				exprMin.clone (),
//				new BinaryExpression (exprMin.clone (),	BinaryOperator.ADD, ExpressionUtil.decrement (loop.getChunkSize (0))),
//				Globals.ONE.clone (),
//				cmpstmtNewLoopBody,
//				loop.getParallelismLevel ()
//			));
//		}
//		else
//			cmpstmtOutput.addStatement (cmpstmtNewLoopBody.clone ());
		
		// synchronize
		Statement stmtBarrier = m_data.getCodeGenerators ().getBackendCodeGenerator ().getBarrier (loop.getParallelismLevel ());
		if (stmtBarrier != null)
			cmpstmtOutput.addStatement (stmtBarrier);
	}

	/**
	 * Determines the location at which the index calculations for loop
	 * <code>loop</code> are inserted.
	 *
	 * @param loop
	 * @param cmpstmtDefaultLocation
	 * @return The compound statement at which the index calculations are
	 *         inserted. <code>null</code> is returned if the statement is to be
	 *         added in the initialization statement of the kernel function (
	 *         {@link CodeGeneratorData#addInitializationStatement(Statement)}).
	 */
	private CompoundStatement getIndexCalculationLocation (SubdomainIterator loop, CompoundStatement cmpstmtDefaultLocation)
	{
		// find the subdomain iterator iterating over the domain of the subdomain iterator loop
		// and place the index calculations (i.e., the bounds for loop) within that subdomain
		// iterator (since the child subdomain iterator loop only depends on the domain of the
		// parent subdomain iterator)

		List<IndexBoundsCalculationInsertionAnnotation> listAnnotations = IndexBoundsCalculationInsertionAnnotation.getIndexBoundCalculationLocationsFor (loop);
		if (listAnnotations == null)
			return null;// m_data.getData ().getInitializationStatement ();

		if (listAnnotations.size () == 1)
		{
			Traversable trvAnnotationStatement = ((Traversable) listAnnotations.get (0).getAnnotatable ());

			// get the CompoundStatement parent of the annotation statement
			CompoundStatement cmpstmtParent = null;
			for (Traversable trvParent = trvAnnotationStatement.getParent ();; trvParent = trvParent.getParent ())
			{
				if (trvParent == null)
					return null;// m_data.getData ().getInitializationStatement ();
				if (trvParent instanceof CompoundStatement)
				{
					cmpstmtParent = (CompoundStatement) trvParent;
					break;
				}
			}

			if (cmpstmtParent == null)
				return cmpstmtDefaultLocation;

			// establish the insertion point
			boolean bAnnotationStatementFound = false;
			for (Traversable trv : cmpstmtParent.getChildren ())
			{
				if (bAnnotationStatementFound)
				{
					if (trv instanceof AnnotationStatement)
						continue;
					if (trv instanceof CompoundStatement)
						return (CompoundStatement) trv;

					// no compound statement found => insert a new one
					CompoundStatement cmpstmt = new CompoundStatement ();
					cmpstmtParent.addStatementAfter ((Statement) trvAnnotationStatement, cmpstmt);
					return cmpstmt;
				}
				if (trv == trvAnnotationStatement)
					bAnnotationStatementFound = true;
			}

			// no compound statement found after the annotation: add a new
			// compound statement at the end
			CompoundStatement cmpstmt = new CompoundStatement ();
			cmpstmtParent.addStatementAfter ((Statement) trvAnnotationStatement, cmpstmt);
			return cmpstmt;
		}

		return cmpstmtDefaultLocation;
	}
}
