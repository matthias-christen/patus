package ch.unibas.cs.hpwc.patus.codegen.iterator;

import java.util.List;

import cetus.hir.AnnotationStatement;
import cetus.hir.AssignmentExpression;
import cetus.hir.AssignmentOperator;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.CompoundStatement;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.Identifier;
import cetus.hir.IntegerLiteral;
import cetus.hir.NameID;
import cetus.hir.Statement;
import cetus.hir.Traversable;
import cetus.hir.VariableDeclarator;
import ch.unibas.cs.hpwc.patus.ast.IndexBoundsCalculationInsertionAnnotation;
import ch.unibas.cs.hpwc.patus.ast.Loop;
import ch.unibas.cs.hpwc.patus.ast.RangeIterator;
import ch.unibas.cs.hpwc.patus.ast.SubdomainIterator;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorData;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;
import ch.unibas.cs.hpwc.patus.codegen.Globals;
import ch.unibas.cs.hpwc.patus.codegen.SubdomainGeneratedIdentifiers;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;
import ch.unibas.cs.hpwc.patus.geometry.Size;
import ch.unibas.cs.hpwc.patus.symbolic.ExpressionOptimizer;
import ch.unibas.cs.hpwc.patus.symbolic.NotConvertableException;
import ch.unibas.cs.hpwc.patus.symbolic.Symbolic;
import ch.unibas.cs.hpwc.patus.util.CodeGeneratorUtil;
import ch.unibas.cs.hpwc.patus.util.ExpressionUtil;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

@Deprecated
public class MulticoreSubdomainIteratorCodeGenerator0 extends AbstractIteratorCodeGenerator
{

	public MulticoreSubdomainIteratorCodeGenerator0 (CodeGeneratorSharedObjects data,
		SubdomainIterator loop, CompoundStatement cmpstmtLoopBody, CompoundStatement cmpstmtOutput,
		boolean bContainsStencilCall, CodeGeneratorRuntimeOptions options)
	{
		super (data, loop, cmpstmtLoopBody, cmpstmtOutput, bContainsStencilCall, options);
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
	@Override
	public void generate ()
	{
		CodeGeneratorSharedObjects data = getData ();
		SubdomainIterator loop = getSubdomainIterator ();

		// compute the number of blocks
		// int xxx_blocks_cnt = prod_i (ceil (iterator_domain.sx(i) / iterator.sx(i)))
		Identifier idNumBlocks = data.getData ().getGeneratedIdentifiers ().getNumBlocksIdentifier (loop.getIterator ());

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
				data.getData ().getGeneratedIdentifiers ().getIndexIdentifier (getSubdomainIterator ().getIterator ()).clone (),
				idJ,
				new BinaryExpression (idJ, BinaryOperator.ADD, ExpressionUtil.decrement (loop.getChunkSize (0))),
				new IntegerLiteral (containsStencilCall () ? data.getCodeGenerators ().getStencilCalculationCodeGenerator ().getLcmSIMDVectorLengths () : 1),
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
			idJ = data.getData ().getGeneratedIdentifiers ().getIndexIdentifier (loop.getIterator ()).clone ();


		/////////////////////////////////
		// add bound variables

		// add an assignment to the loop body that calculates the box bounds
		// 		subdomain = domain + i * iterator
		// and the actual code to calculate the lower and upper bounds

		Size sizeIterator = loop.getIteratorSubdomain ().getSize ();
		byte nDim = loop.getDomainSubdomain ().getBox ().getDimensionality ();
		SubdomainGeneratedIdentifiers ids = data.getData ().getGeneratedIdentifiers ();

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
		data.getCodeGenerators ().getIndexCalculator ().calculateOneToMulti (
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
					new BinaryExpression (data.getStencilCalculation ().getDomainSize ().getMax ().getCoord (i).clone (), BinaryOperator.ADD, Globals.ONE.clone ())
				)
			)));
		}

		// determine the location where the index calculation is inserted
		// (the number-of-blocks calculation doesn't depend on the loop index (v_idx), but the index bounds (v_x_min, v_x_max, ...) do
		CompoundStatement cmpstmtIndexCalculation = MulticoreSubdomainIteratorCodeGenerator0.getIndexCalculationLocation (loop, getLoopBody ());

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
			data.getData ().addInitializationStatement (stmtInit);
		else
			CodeGeneratorUtil.addStatements (cmpstmtIndexCalculation, stmtInit);

		CodeGeneratorUtil.addStatementsAtTop (getLoopBody (), cmpstmtBoundsCalculations);

		// add the body to the inner loop after the last modification
		if (loopInner != null)
			loopInner.setLoopBody (getLoopBody ());


		/////////////////////////////////
		// create the outer loop

		Expression exprStartOuter = /*Globals.getThreadNumber ()*/ data.getCodeGenerators ().getIndexCalculator ().calculateIndicesFromHardwareIndices (new Size (Integer.MAX_VALUE), getOutputStatement (), getOptions ())[0];
		if (bHasNonUnitChunk)
			exprStartOuter = new BinaryExpression (loop.getChunkSize (0), BinaryOperator.MULTIPLY, exprStartOuter);
		
 		RangeIterator loopOuter = new RangeIterator (
			idJ,
			exprStartOuter,
			new BinaryExpression (idNumBlocks.clone (), BinaryOperator.SUBTRACT, Globals.ONE.clone ()),
			Symbolic.simplify (
				ExpressionUtil.product (
					loop.getChunkSize (0),
					data.getCodeGenerators ().getLoopCodeGenerator ().getNumberOfThreadsInDimension (loop, 0, getOptions ()), /*loop.getNumberOfThreads (),*/
					// account for SIMD if there is no inner loop and the loop contains a stencil call
					new IntegerLiteral (loopInner == null && containsStencilCall () ?
						data.getCodeGenerators ().getStencilCalculationCodeGenerator ().getLcmSIMDVectorLengths () : 1)
				),
				Symbolic.ALL_VARIABLES_INTEGER
			),
			loopInner == null ? getLoopBody () : loopInner,
			loop.getParallelismLevel ());

		// add the loop to the output statement
		getOutputStatement ().addStatement (loopOuter);

		// synchronize
//		Statement stmtBarrier = data.getCodeGenerators ().getBackendCodeGenerator ().getBarrier (loop.getParallelismLevel ());
		Statement stmtBarrier = data.getCodeGenerators ().getBackendCodeGenerator ().getBarrier (loop.getParallelismLevel () - 1); // TODO: check this!
		if (stmtBarrier != null)
			getOutputStatement ().addStatement (stmtBarrier);
	}
			
	
	/**
	 * 
	 * @param loop
	 * @param nStartDim
	 * @param nEndDim
	 * @return
	 */
	private static Expression calculateDomainSize (SubdomainIterator loop, int nStartDim, int nEndDim)
	{
		Expression exprSize = loop.getTotalDomainSubdomain ().getSize ().getCoord (nStartDim).clone ();
		for (int i = nStartDim + 1; i <= nEndDim; i++)
			exprSize = new BinaryExpression (exprSize, BinaryOperator.MULTIPLY, loop.getTotalDomainSubdomain ().getSize ().getCoord (i).clone ());
		return exprSize;
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
	private static CompoundStatement getIndexCalculationLocation (SubdomainIterator loop, CompoundStatement cmpstmtDefaultLocation)
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
