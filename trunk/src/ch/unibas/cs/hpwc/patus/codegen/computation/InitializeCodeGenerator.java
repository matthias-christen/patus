package ch.unibas.cs.hpwc.patus.codegen.computation;

import java.util.LinkedList;
import java.util.List;

import cetus.hir.AssignmentExpression;
import cetus.hir.AssignmentOperator;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.FloatLiteral;
import cetus.hir.IfStatement;
import cetus.hir.IntegerLiteral;
import cetus.hir.Specifier;
import cetus.hir.Statement;
import cetus.hir.Traversable;
import ch.unibas.cs.hpwc.patus.ast.IStatementList;
import ch.unibas.cs.hpwc.patus.ast.ParameterAssignment;
import ch.unibas.cs.hpwc.patus.ast.StatementList;
import ch.unibas.cs.hpwc.patus.ast.StatementListBundle;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;
import ch.unibas.cs.hpwc.patus.representation.Stencil;
import ch.unibas.cs.hpwc.patus.representation.StencilNode;
import ch.unibas.cs.hpwc.patus.util.CodeGeneratorUtil;
import ch.unibas.cs.hpwc.patus.util.ExpressionUtil;
import ch.unibas.cs.hpwc.patus.util.IntArray;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 * Generates initialization code.
 */
class InitializeCodeGenerator extends AbstractStencilCalculationCodeGenerator
{
	public InitializeCodeGenerator (CodeGeneratorSharedObjects data, Expression exprStrategy, int nLcmSIMDVectorLengths, StatementListBundle slGenerated, CodeGeneratorRuntimeOptions options)
	{
		super (data, exprStrategy, nLcmSIMDVectorLengths, slGenerated, options);
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
			generateInitialization ();
	}
	
	protected void generateInitialization ()
	{
		int[] rgOffsetIndex = IntArray.getArray (m_data.getStencilCalculation ().getDimensionality (), 0);
		
		for (Stencil stencil : m_data.getStencilCalculation ().getInitialization ())
		{
			Expression exprStencil = stencil.getExpression ();
			
			if (exprStencil != null)
			{
				Specifier specDatatype = getDatatype (exprStencil);

				// replace identifiers in the stencil expression
				exprStencil = exprStencil.clone ();

				// add the stencil computation to the generated code
				for (StencilNode nodeOutput : stencil.getOutputNodes ())
				{
					for (ParameterAssignment pa : m_slbGenerated)
					{
						StatementList slGenerated = m_slbGenerated.getStatementList (pa);
						
						// replace the stencil nodes in the expression with the indexed memory object instances
						Expression exprMOStencil = replaceStencilNodes (exprStencil, specDatatype, rgOffsetIndex, slGenerated);
						Expression exprLHS = replaceStencilNodes (nodeOutput, specDatatype, rgOffsetIndex, slGenerated);							
		
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
			}
		}
	}
	
	/**
	 * Default initialization.
	 */
	@Override
	protected void generateSingleCalculation (Stencil stencil, Specifier specDatatype, int[] rgOffsetIndex, StatementList slGenerated)
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
				createSIMDStencilNode (node, specDatatype, rgOffsetIndex, slGenerated) :
				replaceStencilNodes (node, specDatatype, rgOffsetIndex, slGenerated);

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

			slGenerated.addStatement (new ExpressionStatement (new AssignmentExpression (exprLHS, AssignmentOperator.NORMAL, exprRHS)));
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