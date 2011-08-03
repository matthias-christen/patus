package ch.unibas.cs.hpwc.patus.codegen;

import java.util.ArrayList;
import java.util.List;

import org.apache.log4j.Logger;

import cetus.hir.AnnotationStatement;
import cetus.hir.AssignmentExpression;
import cetus.hir.AssignmentOperator;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.CompoundStatement;
import cetus.hir.DeclarationStatement;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.Identifier;
import cetus.hir.IfStatement;
import cetus.hir.IntegerLiteral;
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
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;
import ch.unibas.cs.hpwc.patus.geometry.Box;
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

	private final static Logger LOGGER = Logger.getLogger (ThreadCodeGenerator.class);

	
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
			ExpressionUtil.product (loop.getStep (), loop.getChunkSize (), m_data.getCodeGenerators ().getLoopCodeGenerator ().getNumberOfThreads (loop, options))
		)));

		CompoundStatement cmpstmtLoopOuterBody = new CompoundStatement ();

		// generate the "end" for the inner i loop:
		// idx_max = j + stride * chunksize - 1
		Identifier idEndI = SymbolTools.getTemp (cmpstmtOutput, loop.getLoopIndex (false).getSymbol ().getTypeSpecifiers (), "max");
		cmpstmtLoopOuterBody.addStatement (new ExpressionStatement (new AssignmentExpression (
			idEndI,
			AssignmentOperator.NORMAL,
			ExpressionUtil.sum (idJ, ExpressionUtil.product (loop.getStep (), loop.getChunkSize ()), new IntegerLiteral (-1))
		)));

		// create the inner loop
		cmpstmtLoopOuterBody.addStatement (new RangeIterator (loop.getLoopIndex (), idJ, idEndI, loop.getStep (), cmpstmtLoopBody.clone (), loop.getParallelismLevel ()));

		// outer loop
		cmpstmtOutput.addStatement (new RangeIterator (
			idJ,
			ExpressionUtil.sum (
				loop.getStart (),
				ExpressionUtil.product (
					/* Globals.getThreadNumber () */m_data.getCodeGenerators ().getIndexCalculator ().calculateIndices (new Size (Integer.MAX_VALUE), cmpstmtLoopOuterBody, options)[0],
					loop.getStep (),
					loop.getChunkSize ()
				)
			),
			loop.getEnd (),
			idStrideJ,
			cmpstmtLoopOuterBody,
			loop.getParallelismLevel ()
		));

		// add synchronization point
		Statement stmtBarrier = m_data.getCodeGenerators ().getBackendCodeGenerator ().getBarrier ();
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
	protected void generateMultiCoreSubdomainIterator (
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
		boolean bHasNonUnitChunk = Symbolic.isTrue (new BinaryExpression (loop.getChunkSize (), BinaryOperator.COMPARE_EQ, new IntegerLiteral (1)), null) != Symbolic.ELogicalValue.TRUE; 
		if (bHasNonUnitChunk)
		{
			// create the loop indices for the outer loop
			NameID nidJ = new NameID (StringUtil.concat (loop.getIterator ().getName (), "_idxouter"));
			idJ = new Identifier (new VariableDeclarator (Globals.SPECIFIER_INDEX, nidJ));

			//*
			loopInner = new RangeIterator (
				m_data.getData ().getGeneratedIdentifiers ().getIndexIdentifier (loop.getIterator ()).clone (),
				idJ,
				new BinaryExpression (idJ, BinaryOperator.ADD, ExpressionUtil.decrement (loop.getChunkSize ())),
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

		Expression exprStartOuter = /*Globals.getThreadNumber ()*/ m_data.getCodeGenerators ().getIndexCalculator ().calculateIndices (new Size (Integer.MAX_VALUE), cmpstmtOutput, options)[0];
		if (bHasNonUnitChunk)
			exprStartOuter = new BinaryExpression (loop.getChunkSize (), BinaryOperator.MULTIPLY, exprStartOuter);
		
 		RangeIterator loopOuter = new RangeIterator (
			idJ,
			exprStartOuter,
			new BinaryExpression (idNumBlocks.clone (), BinaryOperator.SUBTRACT, Globals.ONE.clone ()),
			Symbolic.simplify (
				ExpressionUtil.product (
					loop.getChunkSize (),
					m_data.getCodeGenerators ().getLoopCodeGenerator ().getNumberOfThreads (loop, options), /*loop.getNumberOfThreads (),*/
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
		Statement stmtBarrier = m_data.getCodeGenerators ().getBackendCodeGenerator ().getBarrier ();
		if (stmtBarrier != null)
			cmpstmtOutput.addStatement (stmtBarrier);

//		// bind the loop
////		m_data.getBoxInstanceManager ().bindNewBoxInstance (loop);
//	(loopInner == null ? loopOuter : loopInner).setOriginalSubdomainIterator (loop);
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
		// get the domain box
		Box boxDomain = loop.getDomainSubdomain ().getBox ();
		byte nDim = boxDomain.getDimensionality ();

		// determine whether there is a chunksize != 1
		boolean bHasNondefaultChunkSize = Symbolic.isTrue (new BinaryExpression (loop.getChunkSize (), BinaryOperator.COMPARE_EQ, new IntegerLiteral (1)), null) != Symbolic.ELogicalValue.TRUE;

		// has data transfers?
		boolean bHasDatatransfers = m_data.getCodeGenerators ().getStrategyAnalyzer ().isDataLoadedInIterator (loop, m_data.getArchitectureDescription ());
		DatatransferCodeGenerator dtcg = m_data.getCodeGenerators ().getDatatransferCodeGenerator ();
		MemoryObjectManager mgr = m_data.getData ().getMemoryObjectManager ();

		// add data transfer code if required
		if (bHasDatatransfers)
		{
			// allocate memory objects
			dtcg.allocateLocalMemoryObjects (loop, options);

			// load and wait
			StatementListBundle slbLoad = new StatementListBundle (new ArrayList<Statement> ());
			StencilNodeSet setInputNodes = mgr.getInputStencilNodes (loop.getIterator ());

			dtcg.loadData (setInputNodes, loop, slbLoad, options);
			dtcg.waitFor (setInputNodes, loop, slbLoad, options);

			CodeGeneratorUtil.addStatementsAtTop (cmpstmtLoopBody, slbLoad.getDefaultList ().getStatementsAsList ());
		}

		// get the size of the blocks
		Size sizeIterator = loop.getIteratorSubdomain ().getBox ().getSize ();
		if (bHasNondefaultChunkSize)
			sizeIterator.setCoord (0, new BinaryExpression (sizeIterator.getCoord (0), BinaryOperator.MULTIPLY, loop.getChunkSize ()));

//		// calculate the number of blocks in each direction
//		// ---
//		Size sizeNumBlocks = new Size (nDim);
//
//		// get a reference output memory object to determine the size of a block
//		MemoryObject moRef = mgr.getMemoryObject (loop.getIterator (), m_data.getStencilCalculation ().getStencilBundle ().getFusedStencil ().getOutputNodes ().iterator ().next (), true);
//		Size sizeMemObj = moRef.getSize (/*Globals.ZERO.clone (), Globals.ONE.clone ()*/);
//
//		for (int i = 0; i < nDim; i++)
//		{
//			/*
//			 * sizeNumBlocks.setCoord (i, ExpressionUtil.ceil
//			 * (loop.getNumberOfBlocksInDimension (i), sizeIterator.getCoord
//			 * (i)));
//			 */
//			// ThreadCodeGenerator.LOGGER.error
//			// ("FIX ME!!!! <size must be calculated from the reference memory object size that is attached to this parallelism level!!!!>");
//			// sizeNumBlocks.setCoord (i, m_data.getStencilCalculation
//			// ().getDomainMaxIdentifier (i));
//
//			sizeNumBlocks.setCoord (i, ExpressionUtil.ceil (loop.getDomainSubdomain ().getBox ().getSize ().getCoord (i), sizeMemObj.getCoord (i)));
//		}
//		// ---

		// calculate the block indices from the thread indices, add auxiliary
		// calculations to the initialization block (=> null)
		Expression[] rgBlockIndices = m_data.getCodeGenerators ().getIndexCalculator ().calculateIndices (boxDomain.getSize (), null, options);

		// create bound variables
		Point ptDomainBase = loop.getDomainSubdomain ().getBaseGridCoordinates ();
		Expression[] rgExprBounds = new Expression[2 * nDim];

		// expression checking whether v_*_min is within the parent iterator's bounds
		Expression exprInBounds = null;

		SubdomainGeneratedIdentifiers ids = m_data.getData ().getGeneratedIdentifiers ();
		for (int i = 0; i < nDim; i++)
		{
			// v_*_min = u_*_min + idx_* * v_*_size
			Identifier idMin = ids.getDimensionIndexIdentifier (loop.getIterator (), i);
			
			Expression exprInBoundsLocal = new BinaryExpression (idMin.clone (), BinaryOperator.COMPARE_LE, boxDomain.getMax ().getCoord (i).clone ());
			exprInBounds = exprInBounds == null ? exprInBoundsLocal : new BinaryExpression (exprInBounds, BinaryOperator.LOGICAL_AND, exprInBoundsLocal);

			Expression exprMin = new BinaryExpression (
				ptDomainBase.getCoord (i).clone (),
				BinaryOperator.ADD,
				new BinaryExpression (rgBlockIndices[i].clone (), BinaryOperator.MULTIPLY, sizeIterator.getCoord (i).clone ()));
			rgExprBounds[2 * i] = new AssignmentExpression (idMin.clone (), AssignmentOperator.NORMAL, Symbolic.optimizeExpression (exprMin));

			// v_*_max = v_*_min + v_*_size
			Identifier idMax = ids.getDimensionMaxIdentifier (loop.getIterator (), i).clone ();
			Expression exprSize = new BinaryExpression (idMin.clone (), BinaryOperator.ADD, sizeIterator.getCoord (i).clone ());

			// account for SIMD
			if (bContainsStencilCall)
			{
				int nSIMDVectorLength = m_data.getCodeGenerators ().getStencilCalculationCodeGenerator ().getLcmSIMDVectorLengths ();
				if (nSIMDVectorLength > 1)
					exprSize = ExpressionUtil.ceil (exprSize, new IntegerLiteral (nSIMDVectorLength));
			}

			rgExprBounds[2 * i + 1] = new AssignmentExpression (
				idMax,
				AssignmentOperator.NORMAL,
				Symbolic.simplify (exprSize, Symbolic.ALL_VARIABLES_INTEGER)
			);
		}

		// add the created bound to the generated code
		// if there is a non-default chunk size (i.e. != 1), the first variable
		// is captured in a loop, hence only add the bound variables except the first
		Statement rgStmtBounds[] = new Statement[bHasNondefaultChunkSize ? rgExprBounds.length - 2 : rgExprBounds.length];
		for (int i = bHasNondefaultChunkSize ? 1 : 0; i < rgExprBounds.length; i++)
			rgStmtBounds[bHasNondefaultChunkSize ? i - 1 : i] = new ExpressionStatement (rgExprBounds[i]);

		// determine the location where the index calculations are to be inserted
		CompoundStatement cmpstmtIndexCalculation = getIndexCalculationLocation (loop, cmpstmtLoopBody);
		if (cmpstmtIndexCalculation == null)
			m_data.getData ().addInitializationStatements (rgStmtBounds);
		else
			CodeGeneratorUtil.addStatements (cmpstmtIndexCalculation, rgStmtBounds);

		// add datatransfer code if required: store
		if (bHasDatatransfers)
		{
			// store
			StatementListBundle slbStore = new StatementListBundle (new ArrayList<Statement> ());
			dtcg.storeData (mgr.getOutputStencilNodes (loop.getIterator ()), loop, slbStore, options);
			CodeGeneratorUtil.addStatements (cmpstmtLoopBody, slbStore.getDefaultList ().getStatementsAsList ());
		}
		
		// add guards to prevent execution if out of bounds
		CompoundStatement cmpstmtNewLoopBody = null;
		if (exprInBounds != null)
		{
			cmpstmtNewLoopBody = new CompoundStatement ();
			cmpstmtLoopBody.setParent (null);
			cmpstmtNewLoopBody.addStatement (new IfStatement (exprInBounds, cmpstmtLoopBody));
		}
		else
			cmpstmtNewLoopBody = cmpstmtLoopBody;
		

		// create the loop for handling chunks
		if (bHasNondefaultChunkSize)
		{
			Identifier idMin = ids.getDimensionIndexIdentifier (loop.getIterator (), 0);
			Expression exprMin = ((AssignmentExpression) rgExprBounds[0]).getRHS ();

			cmpstmtOutput.addStatement (new RangeIterator (
				idMin.clone (),
				exprMin.clone (),
				new BinaryExpression (exprMin.clone (),	BinaryOperator.ADD, ExpressionUtil.decrement (loop.getChunkSize ())),
				Globals.ONE.clone (),
				cmpstmtNewLoopBody,
				loop.getParallelismLevel ()
			));
		}
		else
			cmpstmtOutput.addStatement (cmpstmtNewLoopBody.clone ());
		
		// synchronize
		Statement stmtBarrier = m_data.getCodeGenerators ().getBackendCodeGenerator ().getBarrier ();
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
