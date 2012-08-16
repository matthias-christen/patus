package ch.unibas.cs.hpwc.patus.codegen.iterator;

import cetus.hir.AssignmentExpression;
import cetus.hir.AssignmentOperator;
import cetus.hir.CompoundStatement;
import cetus.hir.ExpressionStatement;
import cetus.hir.Identifier;
import cetus.hir.IntegerLiteral;
import cetus.hir.Statement;
import cetus.hir.SymbolTools;
import ch.unibas.cs.hpwc.patus.ast.RangeIterator;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;
import ch.unibas.cs.hpwc.patus.geometry.Size;
import ch.unibas.cs.hpwc.patus.util.ExpressionUtil;

public class MulticoreRangeIteratorCodeGenerator extends AbstractIteratorCodeGenerator
{

	public MulticoreRangeIteratorCodeGenerator (CodeGeneratorSharedObjects data,
		RangeIterator loop, CompoundStatement cmpstmtLoopBody, CompoundStatement cmpstmtOutput,
		CodeGeneratorRuntimeOptions options)
	{
		super (data, loop, cmpstmtLoopBody, cmpstmtOutput, false, options);
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
	public void generate ()
	{
		CodeGeneratorSharedObjects data = getData ();
		RangeIterator loop = getRangeIterator ();
		CompoundStatement cmpstmtOutput = getOutputStatement ();

		// loop index for the outer loop
		Identifier idJ = SymbolTools.getTemp (cmpstmtOutput, loop.getLoopIndex (false).getSymbol ().getTypeSpecifiers (), loop.getLoopIndex ().getName ());

		// stride for the outer loop
		// strideJ = stride * chunksize * #thds
		Identifier idStrideJ = SymbolTools.getTemp (cmpstmtOutput, loop.getLoopIndex (false).getSymbol ().getTypeSpecifiers (), "stride");
		cmpstmtOutput.addStatement (new ExpressionStatement (new AssignmentExpression (
			idStrideJ,
			AssignmentOperator.NORMAL,
			ExpressionUtil.product (loop.getStep (), loop.getChunkSize (0), data.getCodeGenerators ().getLoopCodeGenerator ().getNumberOfThreadsInDimension (loop, 0, getOptions ()))
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
		cmpstmtLoopOuterBody.addStatement (new RangeIterator (loop.getLoopIndex (), idJ, idEndI, loop.getStep (), getLoopBody ().clone (), loop.getParallelismLevel ()));

		// outer loop
		cmpstmtOutput.addStatement (new RangeIterator (
			idJ,
			ExpressionUtil.sum (
				loop.getStart (),
				ExpressionUtil.product (
					/* Globals.getThreadNumber () */data.getCodeGenerators ().getIndexCalculator ().calculateIndicesFromHardwareIndices (new Size (Integer.MAX_VALUE), cmpstmtLoopOuterBody, getOptions ())[0],
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
//		Statement stmtBarrier = data.getCodeGenerators ().getBackendCodeGenerator ().getBarrier (loop.getParallelismLevel ());
		Statement stmtBarrier = data.getCodeGenerators ().getBackendCodeGenerator ().getBarrier (loop.getParallelismLevel () - 1);	// TODO: check this!
		if (stmtBarrier != null)
			cmpstmtOutput.addStatement (stmtBarrier);
	}
}
