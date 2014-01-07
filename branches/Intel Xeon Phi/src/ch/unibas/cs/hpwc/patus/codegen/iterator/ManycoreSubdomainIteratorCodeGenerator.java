package ch.unibas.cs.hpwc.patus.codegen.iterator;

import java.util.ArrayList;

import cetus.hir.AssignmentExpression;
import cetus.hir.AssignmentOperator;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.CompoundStatement;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.Identifier;
import cetus.hir.IfStatement;
import cetus.hir.IntegerLiteral;
import cetus.hir.Statement;
import ch.unibas.cs.hpwc.patus.analysis.StrategyAnalyzer;
import ch.unibas.cs.hpwc.patus.ast.Loop;
import ch.unibas.cs.hpwc.patus.ast.RangeIterator;
import ch.unibas.cs.hpwc.patus.ast.StatementListBundle;
import ch.unibas.cs.hpwc.patus.ast.SubdomainIterator;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;
import ch.unibas.cs.hpwc.patus.codegen.DatatransferCodeGenerator;
import ch.unibas.cs.hpwc.patus.codegen.MemoryObjectManager;
import ch.unibas.cs.hpwc.patus.codegen.StencilNodeSet;
import ch.unibas.cs.hpwc.patus.codegen.SubdomainGeneratedIdentifiers;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;
import ch.unibas.cs.hpwc.patus.geometry.Point;
import ch.unibas.cs.hpwc.patus.symbolic.Symbolic;
import ch.unibas.cs.hpwc.patus.util.CodeGeneratorUtil;
import ch.unibas.cs.hpwc.patus.util.ExpressionUtil;

public class ManycoreSubdomainIteratorCodeGenerator extends	AbstractIteratorCodeGenerator
{

	public ManycoreSubdomainIteratorCodeGenerator (CodeGeneratorSharedObjects data,
		SubdomainIterator loop, CompoundStatement cmpstmtLoopBody, CompoundStatement cmpstmtOutput,
		boolean bContainsStencilCall, CodeGeneratorRuntimeOptions options)
	{
		super (data, loop, cmpstmtLoopBody, cmpstmtOutput, bContainsStencilCall, options);
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
	public void generate ()
	{
		CodeGeneratorSharedObjects data = getData ();
		SubdomainIterator loop = getSubdomainIterator ();

		loadData ();
		
		int nDim = loop.getDomainIdentifier ().getDimensionality ();

		// adjust the size of the blocks if chunking is used
//		Size sizeIterator = loop.getIteratorSubdomain ().getBox ().getSize ();
//		for (int i = 0; i < sizeIterator.getDimensionality (); i++)
//			if (!ExpressionUtil.isValue (loop.getChunkSize (i), 1))
//				sizeIterator.setCoord (i, new BinaryExpression (sizeIterator.getCoord (0), BinaryOperator.MULTIPLY, loop.getChunkSize (i)));
		
		// expression checking whether v_*_min is within the parent iterator's bounds
		SubdomainGeneratedIdentifiers ids = data.getData ().getGeneratedIdentifiers ();
		Expression exprInBounds = null;
		for (int i = 0; i < nDim; i++)
		{
			Expression exprInBoundsLocal = new BinaryExpression (ids.getDimensionIndexIdentifier (loop.getIterator (), i).clone (), BinaryOperator.COMPARE_LE, loop.getDomainSubdomain ().getBox ().getMax ().getCoord (i).clone ());
			exprInBounds = exprInBounds == null ? exprInBoundsLocal : new BinaryExpression (exprInBounds, BinaryOperator.LOGICAL_AND, exprInBoundsLocal);
		}
		Statement stmtLoopBody = new IfStatement (exprInBounds, getLoopBody ().clone ());
		
		// calculate the block indices from the thread indices, add auxiliary
		// calculations to the initialization block (=> null)
		Expression[] rgBlockIndices = data.getCodeGenerators ().getIndexCalculator ().calculateIndicesFromHardwareIndices (loop.getNumberOfBlocksPerDimension (), null, getOptions ());
		boolean bHasNestedLoops = StrategyAnalyzer.hasNestedLoops (loop);

		for (int i = 0; i < nDim; i++)
			stmtLoopBody = generateManyCoreLoopForDim (loop, stmtLoopBody, rgBlockIndices, i, bHasNestedLoops, containsStencilCall (), getOptions ());
		getOutputStatement ().addStatement (stmtLoopBody);

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
//		Statement stmtBarrier = data.getCodeGenerators ().getBackendCodeGenerator ().getBarrier (loop.getParallelismLevel ());
		Statement stmtBarrier = data.getCodeGenerators ().getBackendCodeGenerator ().getBarrier (loop.getParallelismLevel () - 1);	// TODO: check this!
		if (stmtBarrier != null)
			getOutputStatement ().addStatement (stmtBarrier);
	}

	private Statement generateManyCoreLoopForDim (SubdomainIterator loop,
		Statement stmtLoopBody,
		Expression[] rgBlockIndices,
		int nDim, boolean bHasNestedLoops, boolean bContainsStencilCall,
		CodeGeneratorRuntimeOptions options)
	{
		CodeGeneratorSharedObjects data = getData ();

		SubdomainGeneratedIdentifiers ids = data.getData ().getGeneratedIdentifiers ();
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
				int nSIMDVectorLength = data.getCodeGenerators ().getStencilCalculationCodeGenerator ().getLcmSIMDVectorLengths ();
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

	private void loadData ()
	{
		CodeGeneratorSharedObjects data = getData ();
		SubdomainIterator loop = getSubdomainIterator ();
		
		// has data transfers?
		boolean bHasDatatransfers = data.getCodeGenerators ().getStrategyAnalyzer ().isDataLoadedInIterator (loop, data.getArchitectureDescription ());

		// add data transfer code if required
		if (bHasDatatransfers)
		{
			if (true)
				throw new RuntimeException ("Not implemented");

			DatatransferCodeGenerator dtcg = data.getCodeGenerators ().getDatatransferCodeGenerator ();
			MemoryObjectManager mgr = data.getData ().getMemoryObjectManager ();

			// allocate memory objects
			dtcg.allocateLocalMemoryObjects (loop, getOptions ());

			// load and wait
			StatementListBundle slbLoad = new StatementListBundle (new ArrayList<Statement> ());
			StencilNodeSet setInputNodes = mgr.getInputStencilNodes (loop.getIterator ());

			dtcg.loadData (setInputNodes, loop, slbLoad, getOptions ());
			dtcg.waitFor (setInputNodes, loop, slbLoad, getOptions ());

			CodeGeneratorUtil.addStatementsAtTop (getLoopBody (), slbLoad.getDefaultList ().getStatementsAsList ());
		}		
	}
	
	private void storeData (SubdomainIterator loop, CompoundStatement cmpstmtLoopBody, CodeGeneratorRuntimeOptions options)
	{
		CodeGeneratorSharedObjects data = getData ();

		// has data transfers?
		boolean bHasDatatransfers = data.getCodeGenerators ().getStrategyAnalyzer ().isDataLoadedInIterator (loop, data.getArchitectureDescription ());

		// add datatransfer code if required: store
		if (bHasDatatransfers)
		{
			if (true)
				throw new RuntimeException ("Not implemented");

			DatatransferCodeGenerator dtcg = data.getCodeGenerators ().getDatatransferCodeGenerator ();
			MemoryObjectManager mgr = data.getData ().getMemoryObjectManager ();

			// store
			StatementListBundle slbStore = new StatementListBundle (new ArrayList<Statement> ());
			dtcg.storeData (mgr.getOutputStencilNodes (loop.getIterator ()), loop, slbStore, options);
			CodeGeneratorUtil.addStatements (cmpstmtLoopBody, slbStore.getDefaultList ().getStatementsAsList ());
		}
	}

}
