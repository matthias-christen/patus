package ch.unibas.cs.hpwc.patus.codegen.iterator;

import cetus.hir.AssignmentExpression;
import cetus.hir.AssignmentOperator;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.CompoundStatement;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.FunctionCall;
import cetus.hir.Identifier;
import cetus.hir.IntegerLiteral;
import cetus.hir.NameID;
import cetus.hir.Statement;
import cetus.hir.VariableDeclarator;
import ch.unibas.cs.hpwc.patus.analysis.StrategyAnalyzer;
import ch.unibas.cs.hpwc.patus.ast.RangeIterator;
import ch.unibas.cs.hpwc.patus.ast.StatementListBundle;
import ch.unibas.cs.hpwc.patus.ast.SubdomainIterator;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;
import ch.unibas.cs.hpwc.patus.codegen.Globals;
import ch.unibas.cs.hpwc.patus.codegen.IndexCalculatorCodeGenerator;
import ch.unibas.cs.hpwc.patus.codegen.SubdomainGeneratedIdentifiers;
import ch.unibas.cs.hpwc.patus.codegen.backend.IIndexing.IIndexingLevel;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;
import ch.unibas.cs.hpwc.patus.geometry.Size;
import ch.unibas.cs.hpwc.patus.symbolic.Symbolic;
import ch.unibas.cs.hpwc.patus.util.CodeGeneratorUtil;
import ch.unibas.cs.hpwc.patus.util.ExpressionUtil;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

public class MulticoreSubdomainIteratorCodeGenerator extends AbstractIteratorCodeGenerator
{
	public MulticoreSubdomainIteratorCodeGenerator (CodeGeneratorSharedObjects data,
		SubdomainIterator loop, CompoundStatement cmpstmtLoopBody, CompoundStatement cmpstmtOutput,
		boolean bContainsStencilCall, CodeGeneratorRuntimeOptions options)
	{
		super (data, loop, cmpstmtLoopBody, cmpstmtOutput, bContainsStencilCall, options);
	}
	
	/**
	 * 
	 * @param loop
	 * @param cmpstmtLoopBody
	 * @param cmpstmtOutput
	 * @param bContainsStencilCall
	 * @param options
	 */
	public void generate ()
	{
		CodeGeneratorSharedObjects data = getData ();
		SubdomainIterator loop = getSubdomainIterator ();
		IndexCalculatorCodeGenerator iccg = data.getCodeGenerators ().getIndexCalculator ();
		
		// TODO: data transfers
		
		// get the start and end parallelism level of the subdomain iterator "loop":
		// if "loop" is the inner-most parallel subdomain iterator it spans all the "remaining"
		// parallelism levels
		int nParallelismLevelStart = loop.getParallelismLevel ();
		int nParallelismLevelEnd = nParallelismLevelStart;
		if (StrategyAnalyzer.isInnerMostParallelLoop (loop))
			nParallelismLevelEnd = data.getCodeGenerators ().getBackendCodeGenerator ().getIndexingLevelsCount ();
		
		// create an array which will contain all the loop indices used in nested iterators or the stencil calculation.
		// these loop indices identify the point for which the stencil is calculated or the starting points for nested
		// subdomain iterators.
		// the array is filled by "generateLoopsForDim"
		int nMaxIndexDimension = Math.min (data.getStencilCalculation ().getDimensionality (), getMaxIndexDimension ());
		Expression[] rgIndices = new Expression[nMaxIndexDimension];
		
		// generate the loop nest
		Statement stmtBody = getLoopBody ();
		int nEnd = data.getStencilCalculation ().getDimensionality () > nMaxIndexDimension ? nMaxIndexDimension - 1 : nMaxIndexDimension;
		
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
				getOutputStatement (),
				i,
				i,
				false,
				containsStencilCall (),
				getOptions ()
			);
		}
		
		// last indexing dimension, if it has to emulate multiple dimensions
		if (data.getStencilCalculation ().getDimensionality () > nMaxIndexDimension)
		{
			// calculate the end expression
			Expression exprEnd = MulticoreSubdomainIteratorCodeGenerator.calculateNumberOfBlocksInMissingDims (loop, nMaxIndexDimension - 1);
				
			// create the loops for dimension i
			stmtBody = generateLoopsForDim (
				loop,
				iccg.calculateHardwareIndicesToOne (nMaxIndexDimension - 1, nParallelismLevelStart, nParallelismLevelEnd),
				ExpressionUtil.decrement (exprEnd),	// decrement because "end" is expected to be inclusive in generateLoopsForDim
				Globals.ONE.clone (),
				iccg.calculateTotalHardwareSize (nMaxIndexDimension - 1, nParallelismLevelStart, nParallelismLevelEnd),
				stmtBody, rgIndices, getOutputStatement (),
				nMaxIndexDimension - 1, loop.getDomainIdentifier ().getDimensionality () - 1,
				true, containsStencilCall (), getOptions ());			
		}
				
		// add the loop to the output statement
		getOutputStatement ().addStatement (stmtBody);
		
		// TODO: transfer data back

		// synchronize
//		Statement stmtBarrier = data.getCodeGenerators ().getBackendCodeGenerator ().getBarrier (loop.getParallelismLevel ());
		Statement stmtBarrier = data.getCodeGenerators ().getBackendCodeGenerator ().getBarrier (loop.getParallelismLevel () - 1); // TODO: check this!
		if (stmtBarrier != null)
			getOutputStatement ().addStatement (stmtBarrier);
	}

	/**
	 * Returns the maximum indexing dimension in the chosen hardware architecture.
	 * @return The maximum indexing dimension
	 */
	private int getMaxIndexDimension ()
	{
		CodeGeneratorSharedObjects data = getData ();

		int nMaxIndexDimension = 0;
		for (int i = 0; i < data.getCodeGenerators ().getBackendCodeGenerator ().getIndexingLevelsCount (); i++)
		{
			IIndexingLevel level = data.getCodeGenerators ().getBackendCodeGenerator ().getIndexingLevel (i);
			nMaxIndexDimension = Math.max (nMaxIndexDimension, level.getDimensionality ());
		}

		return nMaxIndexDimension;
	}

	/**
	 * Multiplies the numbers of blocks in dimensions <code>nStartDim</code>, <code>nStartDim+1</code>, ..., dimensionality.
	 * @param loop The loop for which the calculate the number of blocks
	 * @param nStartDim The start dimension
	 * @return The total number of blocks in dimensions <code>nStartDim</code>, <code>nStartDim+1</code>, ..., dimensionality
	 */
	private static Expression calculateNumberOfBlocksInMissingDims (SubdomainIterator loop, int nStartDim)
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
		CodeGeneratorSharedObjects data = getData ();
		SubdomainGeneratedIdentifiers ids = data.getData ().getGeneratedIdentifiers ();
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
				data.getCodeGenerators ().getStencilCalculationCodeGenerator ().getLcmSIMDVectorLengths () : 1;

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
	 * Adds the bounds variables to the original loop body.
	 * @param loop
	 * @param stmtOrignalBody
	 * @param nDimStart
	 * @param nDimEnd
	 * @return
	 */
	private CompoundStatement createLoopBody (SubdomainIterator loop, Statement stmtOrignalBody, int nDimStart, int nDimEnd, boolean bUseBlockIndices)
	{
		CodeGeneratorSharedObjects data = getData ();
		CompoundStatement cmpstmtBody = new CompoundStatement ();

		// add an assignment to the loop body that calculates the box bounds
		// 		subdomain = domain + i * iterator
		// and the actual code to calculate the lower and upper bounds

		Size sizeIterator = loop.getIteratorSubdomain ().getSize ();
		SubdomainGeneratedIdentifiers ids = data.getData ().getGeneratedIdentifiers ();

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
		data.getCodeGenerators ().getIndexCalculator ().calculateOneToMulti (
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
					new BinaryExpression (data.getStencilCalculation ().getDomainSize ().getMax ().getCoord (i).clone (), BinaryOperator.ADD, Globals.ONE.clone ())
				)
			)));
		}
		
		cmpstmtBody.addStatement (stmtOrignalBody.clone ());
		return cmpstmtBody;
	}	
}
