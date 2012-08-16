package ch.unibas.cs.hpwc.patus.codegen.computation;

import cetus.hir.ArrayAccess;
import cetus.hir.AssignmentExpression;
import cetus.hir.AssignmentOperator;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.DepthFirstIterator;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.FunctionCall;
import cetus.hir.IntegerLiteral;
import cetus.hir.NameID;
import cetus.hir.Specifier;
import cetus.hir.Statement;
import cetus.hir.StringLiteral;
import ch.unibas.cs.hpwc.patus.ast.StatementList;
import ch.unibas.cs.hpwc.patus.ast.StatementListBundle;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;
import ch.unibas.cs.hpwc.patus.representation.Stencil;
import ch.unibas.cs.hpwc.patus.representation.StencilNode;
import ch.unibas.cs.hpwc.patus.util.CodeGeneratorUtil;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

class StencilCodeGenerator extends AbstractStencilCalculationCodeGenerator
{
	public StencilCodeGenerator (CodeGeneratorSharedObjects data, Expression exprStrategy, int nLcmSIMDVectorLengths, StatementListBundle slGenerated, CodeGeneratorRuntimeOptions options)
	{
		super (data, exprStrategy, nLcmSIMDVectorLengths, slGenerated, options);
	}

	/**
	 * Generates the code for a single stencil calculation
	 * @param stencil
	 * @param specDatatype
	 * @param nSIMDOffsetIndex The offsets of the indices
	 * @param slGenerated
	 */
	@Override
	public void generateSingleCalculation (Stencil stencil, Specifier specDatatype, int[] rgOffsetIndex, StatementList slGenerated)
	{
		Expression exprStencil = stencil.getExpression ();
		if (exprStencil != null)
		{
			// replace identifiers in the stencil expression
			exprStencil = exprStencil.clone ();

			// check whether the hardware / programming model supports explicit FMAs
			// TODO: support FMA when not vectorizing? (need to distinguish between non-vectorizing and vectorizing in HardwareDescription#getIntrinsicName)
//				if (m_hw.getIntrinsicName (Globals.FNX_FMA.getName (), specDatatype) != null)
//					exprStencil = m_data.getCodeGenerators ().getFMACodeGenerator ().applyFMAs (exprStencil, specDatatype);

			boolean bSuppressVectorization = m_options.getBooleanValue (CodeGeneratorRuntimeOptions.OPTION_NOVECTORIZE, false);
			boolean bFirst = true;

			if (!bSuppressVectorization)
				exprStencil = m_data.getCodeGenerators ().getFMACodeGenerator ().applyFMAs (exprStencil, specDatatype, true);

			// add the stencil computation to the generated code
			for (StencilNode nodeOutput : stencil.getOutputNodes ())
			{
				// SIMDize the stencil expression
				Expression exprVectorizedStencil = m_data.getCodeGenerators ().getBackendCodeGenerator ().createExpression (
					exprStencil.clone (), specDatatype, !bSuppressVectorization);

				// replace the stencil nodes in the expression with the indexed memory object instances
				Expression exprMOStencil = replaceStencilNodes (exprVectorizedStencil, specDatatype, rgOffsetIndex, slGenerated);
				Expression exprLHS = replaceStencilNodes (nodeOutput, specDatatype, rgOffsetIndex, slGenerated);

				// create a printf statement for debugging purposes
				if (bFirst && m_data.getOptions ().isDebugPrintStencilIndices ())
					createDebugPrint (exprLHS, specDatatype, slGenerated, bSuppressVectorization);
				bFirst = false;

				// create the calculation statement
				Statement stmtStencil = new ExpressionStatement (
					new AssignmentExpression (
						exprLHS,
						/*
						m_data.getMemoryObjectManager ().getMemoryObjectExpression (
							m_sdidStencilArg, nodeOutput, true, true, slGenerated),*/
						AssignmentOperator.NORMAL,
						exprMOStencil));

				if (StencilCalculationCodeGenerator.LOGGER.isDebugEnabled ())
					StencilCalculationCodeGenerator.LOGGER.debug (StringUtil.concat ("Adding stencil ", stmtStencil.toString ()));

				slGenerated.addStatement (stmtStencil);
			}
		}
	}

	/**
	 * Insert a <code>printf</code> statement for debugging purposes
	 * printing the index into the grid array at
	 * which the result is written.
	 * 
	 * @param exprLHS
	 *            The LHS stencil expression
	 * @param slGenerated
	 *            The statement list to which the generated
	 *            <code>printf</code> statement is added
	 */
	private void createDebugPrint (Expression exprLHS, Specifier specDatatype, StatementList slGenerated, boolean bNoVectorize)
	{
		Expression exprIdx = null;
		for (DepthFirstIterator it = new DepthFirstIterator (exprLHS); it.hasNext (); )
		{
			Object obj = it.next ();
			if (obj instanceof ArrayAccess)
			{
				exprIdx = ((ArrayAccess) obj).getIndex (0);
				break;
			}
		}

		if (exprIdx == null)
			exprIdx = new IntegerLiteral (0x1D000DEF);	// IDx_unDEF

		int nSIMDVectorLengthToPrint =
			!bNoVectorize && !m_data.getOptions ().useNativeSIMDDatatypes () && m_data.getArchitectureDescription ().useSIMD () ?
			m_data.getArchitectureDescription ().getSIMDVectorLength (specDatatype) : 1;

		for (int i = 0; i < nSIMDVectorLengthToPrint; i++)
		{
			slGenerated.addStatement (new ExpressionStatement (new FunctionCall (
				new NameID ("printf"),
				CodeGeneratorUtil.expressions (
					new StringLiteral (bNoVectorize ? "%d\\n" : "%d!\\n"),
					new BinaryExpression (exprIdx.clone (), BinaryOperator.ADD, new IntegerLiteral (i))
				)
			)));
		}
	}
}