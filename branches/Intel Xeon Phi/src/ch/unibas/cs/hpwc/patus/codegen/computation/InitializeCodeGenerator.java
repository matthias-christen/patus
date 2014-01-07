package ch.unibas.cs.hpwc.patus.codegen.computation;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import cetus.hir.AssignmentExpression;
import cetus.hir.AssignmentOperator;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.CompoundStatement;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.FloatLiteral;
import cetus.hir.ForLoop;
import cetus.hir.IfStatement;
import cetus.hir.IntegerLiteral;
import cetus.hir.Specifier;
import cetus.hir.Statement;
import cetus.hir.Traversable;
import cetus.hir.UnaryExpression;
import cetus.hir.UnaryOperator;
import ch.unibas.cs.hpwc.patus.ast.IStatementList;
import ch.unibas.cs.hpwc.patus.ast.ParameterAssignment;
import ch.unibas.cs.hpwc.patus.ast.StatementList;
import ch.unibas.cs.hpwc.patus.ast.StatementListBundle;
import ch.unibas.cs.hpwc.patus.ast.SubdomainIdentifier;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;
import ch.unibas.cs.hpwc.patus.codegen.MemoryObjectManager;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;
import ch.unibas.cs.hpwc.patus.geometry.Box;
import ch.unibas.cs.hpwc.patus.geometry.Point;
import ch.unibas.cs.hpwc.patus.representation.Stencil;
import ch.unibas.cs.hpwc.patus.representation.StencilCalculation;
import ch.unibas.cs.hpwc.patus.representation.StencilNode;
import ch.unibas.cs.hpwc.patus.util.CodeGeneratorUtil;
import ch.unibas.cs.hpwc.patus.util.ExpressionUtil;
import ch.unibas.cs.hpwc.patus.util.StatementListBundleUtil;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 * Generates initialization code.
 */
class InitializeCodeGenerator extends AbstractStencilCalculationCodeGenerator
{
	///////////////////////////////////////////////////////////////////
	// Inner Types
	
	private static class StencilNodeEx
	{
		private Stencil m_stencil;
		private StencilNode m_node;
		
		public StencilNodeEx (Stencil stencil, StencilNode node)
		{
			m_stencil = stencil;
			m_node = node;
		}

		public Stencil getStencil ()
		{
			return m_stencil;
		}

		public StencilNode getNode ()
		{
			return m_node;
		}
	}

	
	///////////////////////////////////////////////////////////////////
	// Implementation

	public InitializeCodeGenerator (CodeGeneratorSharedObjects data, Expression exprStrategy, int nLcmSIMDVectorLengths, StatementListBundle slbGenerated, CodeGeneratorRuntimeOptions options)
	{
		super (data, exprStrategy, nLcmSIMDVectorLengths, slbGenerated, options);
	}
	
	@Override
	public void generate ()
	{
		if (m_data.getStencilCalculation ().getInitialization () == null)
		{
			// default initialization
			super.generate ();
		}
		else
		{
			// custom initialization as defined in the stencil specification
			generateInitialization ();
		}
	}
	
	/**
	 * Generate the initialization code as specified in the stencil specification.
	 */
	protected void generateInitialization ()
	{
		// collect sets of nodes for which the corresponding base grid dimensions are the same
		// (so that they can be grouped in "for" statements for the boundary initialization)
		Map<Box, List<StencilNodeEx>> mapNodes = new HashMap<> ();
		for (Stencil stencil : m_data.getStencilCalculation ().getInitialization ())
		{
			Expression exprStencil = stencil.getExpression ();			
			if (exprStencil != null)
			{
				for (StencilNode nodeOutput : stencil.getOutputNodes ())
				{
					String strArgName = MemoryObjectManager.createMemoryObjectName (null, nodeOutput, null, true);
					Box boxGridDimensions = ((StencilCalculation.GridType) m_data.getStencilCalculation ().getArgumentType (strArgName)).getBoxDimension ();

					List<StencilNodeEx> listNodes = mapNodes.get (boxGridDimensions);
					if (listNodes == null)
						mapNodes.put (boxGridDimensions, listNodes = new ArrayList<> ());
					listNodes.add (new StencilNodeEx (stencil, nodeOutput));
				}
			}
		}
		
		
		// generate the initialization for each node group
		
		// first check whether we're at the boundary of any of the grid coordinates
		
		Point ptStart = m_data.getStencilCalculation ().getDomainSize ().getMin ();
		Point ptEnd = m_data.getStencilCalculation ().getDomainSize ().getMax ();

		// temporarily replace m_sdidStencilArg: replace the current coordinate (in dimension nDim) by the local one
		SubdomainIdentifier sdidSave = m_sdidStencilArg;
		SubdomainIdentifier sdidTmp = new SubdomainIdentifier (StringUtil.concat (m_sdidStencilArg.getName (), "_loc"), m_sdidStencilArg.getSubdomain ());

		for (int nDim = 0; nDim < m_data.getStencilCalculation ().getDimensionality (); nDim++)
		{
			// are we at the start?
			generateBoundaryInitialization (
				nDim, sdidSave, sdidTmp, mapNodes,
				ptStart.getCoord (nDim),
				null,
				ExpressionUtil.decrement (ptStart.getCoord (nDim).clone ())
			);
			
			// are we at the end?
			generateBoundaryInitialization (
				nDim, sdidSave, sdidTmp, mapNodes,
				ptEnd.getCoord (nDim),
				ExpressionUtil.increment (ptEnd.getCoord (nDim).clone ()),
				null
			);
		}

		// restore the subdomain identifier
		m_sdidStencilArg = sdidSave;

		
		// then create the "usual" initialization
		for (Stencil stencil : m_data.getStencilCalculation ().getInitialization ())
		{
			Expression exprStencil = stencil.getExpression ();
			Specifier specDatatype = getDatatype (exprStencil);
			
			if (exprStencil != null)
			{
				for (StencilNode nodeOutput : stencil.getOutputNodes ())
					generateInitializationExpressions (m_slbGenerated, nodeOutput, exprStencil, specDatatype);
			}
		}
	}
	
	/**
	 * Creates the boundary check <code>if</code> statement and invokes the code generation for the <code>for</code> loops iterating over the boundary
	 * @param nDim
	 * @param sdidOrig
	 * @param mapNodes
	 * @param exprDomainLimit
	 * @param exprStart
	 * @param exprEnd
	 */
	protected void generateBoundaryInitialization (int nDim, SubdomainIdentifier sdidOrig, SubdomainIdentifier sdidReplacement,
		Map<Box, List<StencilNodeEx>> mapNodes, Expression exprDomainLimit, Expression exprStart, Expression exprEnd)
	{
		StatementListBundle slbBody = new StatementListBundle ();
		
		m_sdidStencilArg = sdidReplacement;
		
		// assign current values to the temporary subdomain identifier
		for (int i = 0; i < m_data.getStencilCalculation ().getDimensionality (); i++)
		{
			if (i != nDim)
			{
				slbBody.addStatement (new ExpressionStatement (new AssignmentExpression (
					getDimensionIndexIdentifier (i), AssignmentOperator.NORMAL, getDimensionIndexIdentifier (sdidOrig, i)
				)));
			}
		}

		// create the for loops (one for each node set)
		for (Box box : mapNodes.keySet ())
		{
			slbBody.addStatements (generateInitializationPerNodeSet (
				nDim,
				mapNodes.get (box),
				exprStart == null ? box.getMin ().getCoord (nDim).clone () : exprStart,
				exprEnd == null ? box.getMax ().getCoord (nDim).clone () : exprEnd
			));
		}
				
		// create the if statement
		m_sdidStencilArg = sdidOrig;	// we need the original identifier for the "if" control expression
		Expression exprControlStart = new BinaryExpression (getDimensionIndexIdentifier (nDim), BinaryOperator.COMPARE_EQ, exprDomainLimit.clone ());
		m_slbGenerated.addStatements (StatementListBundleUtil.createIfStatement (
			replaceStencilNodes (exprControlStart, Specifier.FLOAT /* dummy */, getDefaultOffset (), m_slbGenerated),
			slbBody,
			null
		));
	}
	
	/**
	 * Creates a <code>for</code> loop per node set (in <code>listNodes</code>)
	 * to initialize the boundary parts of the grids corresponding to the
	 * stencil nodes.
	 * 
	 * @param nDim
	 *            The dimension for which to create the special initialization
	 *            code
	 * @param listNodes
	 *            The list of nodes which to initialize
	 * @param exprStart
	 *            The start expression of the <code>for</code> loop
	 * @param exprEnd
	 *            The end expression of the <code>for</code> loop
	 * @return The statement list bundle containing the generated
	 *         <code>for</code> loop
	 */
	protected StatementListBundle generateInitializationPerNodeSet (int nDim, List<StencilNodeEx> listNodes, Expression exprStart, Expression exprEnd)
	{
		StatementListBundle slbFor = new StatementListBundle ();
		
		// create the for loop
		Expression exprLoopIdx = getDimensionIndexIdentifier (nDim);
		slbFor.addStatement (new ForLoop (
			new ExpressionStatement (new AssignmentExpression (exprLoopIdx, AssignmentOperator.NORMAL, exprStart.clone ())),
			new BinaryExpression (exprLoopIdx.clone (), BinaryOperator.COMPARE_LE, exprEnd.clone ()),
			new UnaryExpression (UnaryOperator.PRE_INCREMENT, exprLoopIdx.clone ()),
			new CompoundStatement ()
		));

		// generate the initialization statements
		StatementListBundle slbBody = new StatementListBundle ();
		for (StencilNodeEx n2 : listNodes)
		{
			Expression exprStencil = n2.getStencil ().getExpression ();
			generateInitializationExpressions (slbBody, n2.getNode (), exprStencil.clone (), getDatatype (exprStencil));
		}
		
		StatementListBundleUtil.addToLoopBody (slbFor, slbBody);

		// reset indices for next for loop
		m_data.getData ().getMemoryObjectManager ().resetIndices ();
		m_data.getCodeGenerators ().getUnrollGeneratedIdentifiers ().reset ();

		return slbFor;
	}
		
	protected void generateInitializationExpressions (StatementListBundle slbGenerated, StencilNode nodeOutput, Expression exprStencil, Specifier specDatatype)
	{
		for (ParameterAssignment pa : slbGenerated)
		{
			StatementList slGenerated = slbGenerated.getStatementList (pa);
			
			// replace the stencil nodes in the expression with the indexed memory object instances
			Expression exprMOStencil = replaceStencilNodes (exprStencil, specDatatype, getDefaultOffset (), slGenerated);
			Expression exprLHS = replaceStencilNodes (nodeOutput, specDatatype, getDefaultOffset (), slGenerated);							

			// create the calculation statement
			Statement stmtStencil = new ExpressionStatement (new AssignmentExpression (exprLHS, AssignmentOperator.NORMAL, exprMOStencil));

			// check if there are any constraints
			if (nodeOutput.getConstraint () != null)
				stmtStencil = new IfStatement (getConstraintCondition (nodeOutput), stmtStencil);

			if (StencilCalculationCodeGenerator.LOGGER.isDebugEnabled ())
				StencilCalculationCodeGenerator.LOGGER.debug (StringUtil.concat ("Adding stencil ", stmtStencil.toString ()));

			slGenerated.addStatement (stmtStencil);
		}
	}
	
	/**
	 * Default initialization.
	 */
	@Override
	protected void generateSingleCalculation (Stencil stencil, Specifier specDatatype, int[] rgOffsetIndex, StatementList slStencil, StatementList slAuxiliary)
	{
		if (stencil.getExpression () == null)
			return;

		boolean bVectorize =
			m_data.getOptions ().useNativeSIMDDatatypes () ||
			!m_options.getBooleanValue (CodeGeneratorRuntimeOptions.OPTION_NOVECTORIZE, false);

		// collect the stencil nodes that need to be initialized
		List<StencilNode> listNodesToInitialize = new LinkedList<> ();
		for (StencilNode nodeInput : stencil)
			if (!bVectorize || (bVectorize && !needsShuffling (nodeInput, specDatatype)))
				listNodesToInitialize.add (nodeInput);
		for (StencilNode nodeOutput : stencil.getOutputNodes ())
			listNodesToInitialize.add (nodeOutput);

		byte nDim = stencil.getDimensionality ();

		// add the initialization statements
		for (StencilNode node : listNodesToInitialize)
		{
			Expression exprLHS = bVectorize ?
				createSIMDStencilNode (node, specDatatype, rgOffsetIndex, slAuxiliary) :
				replaceStencilNodes (node, specDatatype, rgOffsetIndex, slAuxiliary);

			// initialize with some arbitrary value
			Expression exprValue = new FloatLiteral (10 * node.getIndex ().getTimeIndex () + (node.getIndex ().getVectorIndex () + 1));

			// make the values depend on the point in space
			if (node.getIndex ().getSpaceIndexEx ().length >= nDim)
			{
				// TODO: remove this restriction
				if (m_data.getOptions ().useNativeSIMDDatatypes ())
					StencilCalculationCodeGenerator.LOGGER.error ("Initialization to non-constant fields not implemented for native SIMD datatypes");
				else
				{
					double fCoeff = 0.1;
					for (int i = 0; i < nDim; i++)
					{
						Expression exprCoord = m_data.getData ().getGeneratedIdentifiers ().getDimensionIndexIdentifier (m_sdidStencilArg, i).clone ();
						int nNodeCoord = ExpressionUtil.getIntegerValue (node.getIndex ().getSpaceIndex (i));
						if (nNodeCoord + rgOffsetIndex[i] != 0)
							exprCoord = new BinaryExpression (exprCoord, BinaryOperator.ADD, new IntegerLiteral (nNodeCoord + rgOffsetIndex[i]));

						exprValue = new BinaryExpression (
							exprValue,
							BinaryOperator.ADD,
							new BinaryExpression (exprCoord, BinaryOperator.MULTIPLY, new FloatLiteral (fCoeff)));
						fCoeff /= 10;
					}
				}
			}

			Expression exprRHS = bVectorize ?
				m_data.getCodeGenerators ().getSIMDScalarGeneratedIdentifiers ().createVectorizedScalar (exprValue, specDatatype, m_slbGenerated, m_options) :
				exprValue;

			slStencil.addStatement (new ExpressionStatement (new AssignmentExpression (exprLHS, AssignmentOperator.NORMAL, exprRHS)));
		}
	}
	
	/**
	 * Alternate implementation of
	 * {@link AbstractStencilCalculationCodeGenerator#replaceStencilNodes(Expression, Specifier, int[], IStatementList)}
	 * which also replaces dimension identifiers by the appropriate iterator
	 * variables.
	 */
	@Override
	protected Expression recursiveReplaceStencilNodes (Expression exprMain, Traversable trv, Specifier specDatatype, int[] rgOffsetIndex, IStatementList slGenerated)
	{
		// check whether trv is an identifier corresponding to one of the dimension identifiers
		int nDim = CodeGeneratorUtil.getDimensionFromIdentifier (trv);
		if (nDim >= 0)
			return getDimensionIndexIdentifier (nDim);
		
		// default
		return super.recursiveReplaceStencilNodes (exprMain, trv, specDatatype, rgOffsetIndex, slGenerated);
	}
}