package ch.unibas.cs.hpwc.patus.codegen.computation;

import java.util.ArrayList;

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
import ch.unibas.cs.hpwc.patus.ast.ParameterAssignment;
import ch.unibas.cs.hpwc.patus.ast.StatementList;
import ch.unibas.cs.hpwc.patus.ast.StatementListBundle;
import ch.unibas.cs.hpwc.patus.codegen.BoundaryCheckCodeGenerator;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;
import ch.unibas.cs.hpwc.patus.representation.Stencil;
import ch.unibas.cs.hpwc.patus.representation.StencilNode;
import ch.unibas.cs.hpwc.patus.util.CodeGeneratorUtil;
import ch.unibas.cs.hpwc.patus.util.StatementListBundleUtil;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

class StencilCodeGenerator extends AbstractStencilCalculationCodeGenerator
{
	///////////////////////////////////////////////////////////////////
	// Member Variables

	protected StatementListBundle m_slbGeneratedStencilComputation;

	
	///////////////////////////////////////////////////////////////////
	// Implementation

	public StencilCodeGenerator (CodeGeneratorSharedObjects data, Expression exprStrategy, int nLcmSIMDVectorLengths, StatementListBundle slbGenerated, CodeGeneratorRuntimeOptions options)
	{
		super (data, exprStrategy, nLcmSIMDVectorLengths, slbGenerated, options);
		m_slbGeneratedStencilComputation = new StatementListBundle (new ArrayList<Statement> ());
	}
	
	@Override
	public void generate ()
	{
		boolean bDoBoundaryChecks = m_options.getBooleanValue (CodeGeneratorRuntimeOptions.OPTION_DOBOUNDARYCHECKS, false) &&
			m_data.getStencilCalculation ().getBoundaries () != null;

		// don't vectorize if there are boundary checks
		boolean bNoVectorize = m_options.getBooleanValue (CodeGeneratorRuntimeOptions.OPTION_NOVECTORIZE, false);
		if (bDoBoundaryChecks)
			m_options.setOption (CodeGeneratorRuntimeOptions.OPTION_NOVECTORIZE, true);
			
		// generate code
		super.generate ();
		if (bDoBoundaryChecks)
			generateBoundaryChecks ();
		else
			m_slbGenerated.addStatements (m_slbGeneratedStencilComputation);
		
		// restore the option
		m_options.setOption (CodeGeneratorRuntimeOptions.OPTION_NOVECTORIZE, bNoVectorize);
	}

	/**
	 * Generates the code for a single stencil calculation
	 * @param stencil
	 * @param specDatatype
	 * @param nSIMDOffsetIndex The offsets of the indices
	 * @param slGenerated
	 */
	@Override
	public void generateSingleCalculation (Stencil stencil, Specifier specDatatype, int[] rgOffsetIndex, StatementList slStencilComputation, StatementList slAuxiliaryCalculations)
	{
		Expression exprStencil = stencil.getExpression ();
		if (exprStencil != null)
		{
			// replace identifiers in the stencil expression
			exprStencil = exprStencil.clone ();

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
				Expression exprMOStencil = replaceStencilNodes (exprVectorizedStencil, specDatatype, rgOffsetIndex, slAuxiliaryCalculations);
				Expression exprLHS = replaceStencilNodes (nodeOutput, specDatatype, rgOffsetIndex, slAuxiliaryCalculations);

				// create a printf statement for debugging purposes
				if (bFirst && m_data.getOptions ().isDebugPrintStencilIndices ())
					createDebugPrint (exprLHS, specDatatype, slAuxiliaryCalculations, bSuppressVectorization);
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

				slStencilComputation.addStatement (stmtStencil);
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
			exprIdx = new IntegerLiteral ("0x1D000DEF");	// IDx_unDEF

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
	
	protected void generateBoundaryChecks ()
	{
		// generate the code for the boundary stencils (create them in reverse order, since we
		// create the "else" branch first and append it to the next (previous in the list of boundaries) "if"

		// the last "else" is the stencil computation
		StatementListBundle slbElse = m_slbGeneratedStencilComputation;
		
		Stencil[] rgBoundaries = new Stencil[m_data.getStencilCalculation ().getBoundaries ().getStencilsCount ()];
		int i = 0;
		for (Stencil stcBoundary : m_data.getStencilCalculation ().getBoundaries ())
			rgBoundaries[i++] = stcBoundary;
			
		for (i = rgBoundaries.length - 1; i >= 0; i--)
		{
			Expression exprConstraint = null;
			for (StencilNode nodeOutput : rgBoundaries[i].getOutputNodes ())
				if (nodeOutput.getConstraint () != null)
					exprConstraint = BoundaryCheckCodeGenerator.addExpression (exprConstraint, getConstraintCondition (nodeOutput));
			
			if (exprConstraint != null)
			{
				StatementListBundle slbThen = new StatementListBundle ();
				for (ParameterAssignment pa : m_slbGenerated)
				{
					StatementList sl = slbThen.getStatementList (pa);
					generateSingleCalculation (rgBoundaries[i], getDatatype (rgBoundaries[i].getExpression ()), getDefaultOffset (), sl, sl);
				}
				
				slbElse = StatementListBundleUtil.createIfStatement (exprConstraint, slbThen, slbElse);
			}
			else
				throw new RuntimeException (StringUtil.concat (rgBoundaries[i].toString (), " is not a boundary."));	// TODO: handle this somewhere else...
		}
		
		m_slbGenerated.addStatements (slbElse);
	}
	
	@Override
	protected StatementList getStencilComputationStatementList (ParameterAssignment pa)
	{
		return m_slbGeneratedStencilComputation.getStatementList (pa);
	}
}