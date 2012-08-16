package ch.unibas.cs.hpwc.patus.codegen.iterator;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import cetus.hir.AssignmentExpression;
import cetus.hir.AssignmentOperator;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.CompoundStatement;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.Identifier;
import cetus.hir.IfStatement;
import cetus.hir.Statement;
import cetus.hir.UnaryExpression;
import cetus.hir.UnaryOperator;
import ch.unibas.cs.hpwc.patus.analysis.StrategyAnalyzer;
import ch.unibas.cs.hpwc.patus.ast.RangeIterator;
import ch.unibas.cs.hpwc.patus.ast.StatementListBundle;
import ch.unibas.cs.hpwc.patus.ast.SubdomainIterator;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;
import ch.unibas.cs.hpwc.patus.codegen.Globals;
import ch.unibas.cs.hpwc.patus.codegen.IndexCalculatorCodeGenerator;
import ch.unibas.cs.hpwc.patus.codegen.SubdomainGeneratedIdentifiers;
import ch.unibas.cs.hpwc.patus.codegen.ThreadCodeGenerator;
import ch.unibas.cs.hpwc.patus.codegen.backend.IIndexing;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;
import ch.unibas.cs.hpwc.patus.geometry.Box;
import ch.unibas.cs.hpwc.patus.util.CodeGeneratorUtil;
import ch.unibas.cs.hpwc.patus.util.ExpressionUtil;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 * This code generator converts Stragey {@link SubdomainIterator}s into C
 * <code>for</code> loop nests.
 * The code generator is less general than
 * {@link MulticoreSubdomainIteratorCodeGenerator}, but creates loop nests with
 * less overhead, and thus produces faster code.
 * Use this code generator if possible. {@link ThreadCodeGenerator} falls back
 * to {@link MulticoreSubdomainIteratorCodeGenerator} if usage of this code
 * generator is not possible.
 * 
 * @author Matthias-M. Christen
 */
public class MulticoreSubdomainIteratorCodeGenerator2 extends AbstractIteratorCodeGenerator
{
	//private final static Logger LOGGER = Logger.getLogger (MulticoreSubdomainIteratorCodeGenerator2.class);
	
	StatementListBundle m_slbInit;
	
	private Expression m_exprNumThreads;
	private Expression m_exprThreadIdx;
	private Identifier m_idBlockIdx;
	private Identifier m_idBlockIdxByChunk;
	
	
	public MulticoreSubdomainIteratorCodeGenerator2 (CodeGeneratorSharedObjects data,
		SubdomainIterator loop,
		CompoundStatement cmpstmtLoopBody, CompoundStatement cmpstmtOutput, boolean bContainsStencilCall,
		CodeGeneratorRuntimeOptions options)
	{
		super (data, loop, cmpstmtLoopBody, cmpstmtOutput, bContainsStencilCall, options);
		
		m_slbInit = new StatementListBundle (getOutputStatement ());
		
		// initialize helper variables 
		
		calculateThreadIdxAndNum ();
		
		String strIteratorName = getSubdomainIterator ().getIterator ().getName ();
		m_idBlockIdx = createIdentifier (StringUtil.concat (strIteratorName, "_blkidx"), Globals.ZERO.clone ());
		m_idBlockIdxByChunk = createIdentifier (StringUtil.concat (strIteratorName, "_blkidx_by_chunk"), Globals.ZERO.clone ());
		
			/*new Identifier (new VariableDeclarator (
			Globals.SPECIFIER_INDEX,
			new NameID (StringUtil.concat (strIteratorName, "_blkidx"))
		));*/
		
//		m_idBlockIdxByChunk = new Identifier (new VariableDeclarator (
//			Globals.SPECIFIER_INDEX,
//			new NameID (StringUtil.concat (strIteratorName, "_blkidx_by_chunk"))
//		));
	}
	
	/**
	 * Generates an implementation of the Strategy subdomain iterator
	 * <code>loop</code>.
	 * 
	 * @param loop
	 *            The input subdomain iterator
	 * @param cmpstmtLoopBody
	 *            The body to be placed in the generated loop nest
	 * @param cmpstmtOutput
	 *            The statement to which the generated construct will be added
	 * @param bContainsStencilCall
	 *            Flag indicating whether the subdomain iterator
	 *            <code>loop</code> contains a stencil computation
	 * @param options
	 *            Code generation options
	 */
	public void generate ()
	{
		CodeGeneratorSharedObjects data = getData ();
		SubdomainIterator loop = getSubdomainIterator ();
		SubdomainGeneratedIdentifiers ids = data.getData ().getGeneratedIdentifiers ();
		
		Box boxDomain = loop.getTotalDomainSubdomain ().getBox ();

		// we only support chunk sizes > 1 in the unit stride direction right now
		// emit a warning saying that chunk sizes in other dimension will be ignored (if > 1)
		checkChunkSizes ();
		
		// create the body of the loop nest
		CompoundStatement cmpstmtLoopNestBody = generateLoopNestBody ();
		
		// TODO: data transfers

		// grow the loop nest from inside out
		Statement stmtLoopNest = cmpstmtLoopNestBody;
		for (int i = 0; i < data.getStencilCalculation ().getDimensionality (); i++)
		{
			Identifier idLoopVariable = ids.getDimensionIndexIdentifier (loop.getIterator (), i);
			
			// create a compound statement if the loop body isn't a compound statement already
			if (stmtLoopNest instanceof RangeIterator)
			{
				CompoundStatement cmpstmtTmp = new CompoundStatement ();
				cmpstmtTmp.addStatement (stmtLoopNest);
				stmtLoopNest = cmpstmtTmp;
			}
			
			// add the maximum loop index computation
			CodeGeneratorUtil.addStatementAtTop ((CompoundStatement) stmtLoopNest, getMaxAssignment (idLoopVariable, i));

			// create the "for" loop for the current dimension
			stmtLoopNest = new RangeIterator (
				idLoopVariable.clone (),
				boxDomain.getMin ().getCoord (i).clone (),
				boxDomain.getMax ().getCoord (i).clone (),
				loop.getIterator ().getSubdomain ().getSize ().getCoord (i).clone (),
				stmtLoopNest,
				loop.getParallelismLevel ()
			);
		}
		
		// add the loop to the output statement
		getOutputStatement ().addStatement (stmtLoopNest);

		// TODO: transfer data back

		// synchronize
//		Statement stmtBarrier = data.getCodeGenerators ().getBackendCodeGenerator ().getBarrier (loop.getParallelismLevel ());
		Statement stmtBarrier = data.getCodeGenerators ().getBackendCodeGenerator ().getBarrier (loop.getParallelismLevel () - 1); // TODO: check this!
		if (stmtBarrier != null)
			getOutputStatement ().addStatement (stmtBarrier);
	}
	
	private void calculateThreadIdxAndNum ()
	{
		SubdomainIterator loop = getSubdomainIterator ();
		
		int nParallelismLevelStart = loop.getParallelismLevel ();
		int nParallelismLevelEnd = nParallelismLevelStart;
		
		IIndexing indexing = getData ().getCodeGenerators ().getBackendCodeGenerator ();
		
		if (nParallelismLevelStart > indexing.getIndexingLevelsCount ())
		{
			// no more parallelism levels => create a sequential implementation
			loop.setNumberOfThreads (1);
			m_exprNumThreads = Globals.ONE.clone ();
			m_exprThreadIdx = Globals.ZERO.clone ();
		}
		else
		{
			if (StrategyAnalyzer.isInnerMostParallelLoop (loop))
				nParallelismLevelEnd = indexing.getIndexingLevelsCount ();
			
			int nMaxDim = 0;
			for (int i = nParallelismLevelStart; i <= nParallelismLevelEnd; i++)
				nMaxDim = Math.max (nMaxDim, indexing.getIndexingLevelFromParallelismLevel (i).getDimensionality ());
			
			IndexCalculatorCodeGenerator iccg = getData ().getCodeGenerators ().getIndexCalculator ();
			Expression[] rgIdxInDim = new Expression[nMaxDim];
			Expression[] rgSizeInDim = new Expression[nMaxDim];
			for (int i = 0; i < nMaxDim; i++)
			{
				rgIdxInDim[i] = iccg.calculateHardwareIndicesToOne (i, nParallelismLevelStart, nParallelismLevelEnd);
				rgSizeInDim[i] = iccg.calculateTotalHardwareSize (i, nParallelismLevelStart, nParallelismLevelEnd);
			}
			
			m_exprNumThreads = getIdentifier (ExpressionUtil.product (rgSizeInDim), "numthds", m_slbInit, getOptions ());
			m_exprThreadIdx = getIdentifier (IndexCalculatorCodeGenerator.calculateMultiToOne (rgIdxInDim, rgSizeInDim), "thdidx", m_slbInit, getOptions ());
		}
	}
		
	/**
	 * Checks whether the subdomain iterator <code>loop</code> has non-unit
	 * chunks in one of the dimensions > 0.
	 * This is currently not supported and will be ignored.
	 */
	private void checkChunkSizes ()
	{
		int i = 0;
		for (Expression exprChunk : getSubdomainIterator ().getChunkSizes ())
		{
			if (i > 0 && !ExpressionUtil.isValue (exprChunk, 1))
				throw new NotImplementedException ();
			
			i++;
		}
	}
	
	/**
	 * Creates the body of the loop nest, building control structures around the
	 * original loop body, <code>m_cmpstmtLoopBody</code>.
	 * 
	 * @return The generated loop body for the loop nest to be generated
	 */
	private CompoundStatement generateLoopNestBody ()
	{
		if (getSubdomainIterator ().isSequential ())
			return getLoopBody ();
		
		CompoundStatement cmpstmtBody = new CompoundStatement ();
		
		Expression exprChunk = getSubdomainIterator ().getChunkSize (0);
		boolean bIsChunk1 = ExpressionUtil.isValue (exprChunk, 1);

		// if (blkidx[_by_chunk] == thdidx)
		//     <inner>
		Expression exprCond = new BinaryExpression (
			bIsChunk1 ? m_idBlockIdx.clone () : m_idBlockIdxByChunk.clone (),
			BinaryOperator.COMPARE_EQ,
			m_exprThreadIdx.clone ()
		);
		cmpstmtBody.addStatement (new IfStatement (exprCond, getLoopBody ().clone ()));

		// blkidx++
		cmpstmtBody.addStatement (new ExpressionStatement (new UnaryExpression (UnaryOperator.PRE_INCREMENT, m_idBlockIdx.clone ())));
		
		if (bIsChunk1)
		{
			// if (blkidx == numthds)
			//     blkidx = 0;
			
			cmpstmtBody.addStatement (new IfStatement (
				new BinaryExpression (m_idBlockIdx.clone (), BinaryOperator.COMPARE_EQ, m_exprNumThreads.clone ()),
				new ExpressionStatement (new AssignmentExpression (m_idBlockIdx.clone (), AssignmentOperator.NORMAL, Globals.ZERO.clone ()))
			));
		}
		else
		{
			// if (blkidx == chunk)
			// {
			//     blkidx = 0;
			//     blkidx_by_chunk++;
			//     if (blkidx_by_chunk == numthds)
			//         blkidx_by_chunk = 0;
			// }
			
			CompoundStatement cmpstmtIfBody = new CompoundStatement ();
			cmpstmtIfBody.addStatement (new ExpressionStatement (new AssignmentExpression (m_idBlockIdx.clone (), AssignmentOperator.NORMAL, Globals.ZERO.clone ())));
			cmpstmtIfBody.addStatement (new ExpressionStatement (new UnaryExpression (UnaryOperator.PRE_INCREMENT, m_idBlockIdxByChunk.clone ())));
			cmpstmtIfBody.addStatement (new IfStatement (
				new BinaryExpression (m_idBlockIdxByChunk.clone (), BinaryOperator.COMPARE_EQ, m_exprNumThreads.clone ()),
				new ExpressionStatement (new AssignmentExpression (m_idBlockIdxByChunk.clone (), AssignmentOperator.NORMAL, Globals.ZERO.clone ())) 
			));
			
			cmpstmtBody.addStatement (new IfStatement (
				new BinaryExpression (m_idBlockIdx.clone (), BinaryOperator.COMPARE_EQ, exprChunk.clone ()),
				cmpstmtIfBody
			));
		}
		
		return cmpstmtBody;
	}
		
	/**
	 * Creates a statement that computes the upper bound for an iterator nested
	 * in this one and assigns the value to a variable.
	 * 
	 * @param nDim
	 *            The dimension for which to compute the upper bound
	 * @return The assignment statement
	 */
	private Statement getMaxAssignment (Identifier idLoopVariable, int nDim)
	{
		CodeGeneratorSharedObjects data = getData ();
		SubdomainIterator loop = getSubdomainIterator ();

		return new ExpressionStatement (new AssignmentExpression (
			data.getData ().getGeneratedIdentifiers ().getDimensionMaxIdentifier (loop.getIterator (), nDim).clone (),
			AssignmentOperator.NORMAL,
			ExpressionUtil.min (
				// calculated maximum
				new BinaryExpression (idLoopVariable.clone (), BinaryOperator.ADD, loop.getIterator ().getSubdomain ().getSize ().getCoord (nDim).clone ()),
				// maximum grid index
				ExpressionUtil.increment (data.getStencilCalculation ().getDomainSize ().getMax ().getCoord (nDim).clone ())
			)
		));
	}	
}
