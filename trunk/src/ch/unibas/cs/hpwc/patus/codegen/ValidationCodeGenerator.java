/*******************************************************************************
 * Copyright (c) 2011 Matthias-M. Christen, University of Basel, Switzerland.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the GNU Lesser Public License v2.1
 * which accompanies this distribution, and is available at
 * http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
 *
 * Contributors:
 *     Matthias-M. Christen, University of Basel, Switzerland - initial API and implementation
 ******************************************************************************/
package ch.unibas.cs.hpwc.patus.codegen;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import cetus.hir.ArrayAccess;
import cetus.hir.AssignmentExpression;
import cetus.hir.AssignmentOperator;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.BreakStatement;
import cetus.hir.CompoundStatement;
import cetus.hir.Declaration;
import cetus.hir.DepthFirstIterator;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.FloatLiteral;
import cetus.hir.FunctionCall;
import cetus.hir.IDExpression;
import cetus.hir.Identifier;
import cetus.hir.IfStatement;
import cetus.hir.IntegerLiteral;
import cetus.hir.NameID;
import cetus.hir.Specifier;
import cetus.hir.Statement;
import cetus.hir.StringLiteral;
import cetus.hir.ValueInitializer;
import cetus.hir.VariableDeclaration;
import cetus.hir.VariableDeclarator;
import ch.unibas.cs.hpwc.patus.analysis.HIRAnalyzer;
import ch.unibas.cs.hpwc.patus.ast.RangeIterator;
import ch.unibas.cs.hpwc.patus.ast.StatementList;
import ch.unibas.cs.hpwc.patus.ast.StatementListBundle;
import ch.unibas.cs.hpwc.patus.ast.SubdomainIdentifier;
import ch.unibas.cs.hpwc.patus.ast.SubdomainIterator;
import ch.unibas.cs.hpwc.patus.codegen.CodeGenerationOptions.EDebugOption;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;
import ch.unibas.cs.hpwc.patus.geometry.Border;
import ch.unibas.cs.hpwc.patus.geometry.Size;
import ch.unibas.cs.hpwc.patus.geometry.Subdomain;
import ch.unibas.cs.hpwc.patus.geometry.Vector;
import ch.unibas.cs.hpwc.patus.representation.StencilNode;
import ch.unibas.cs.hpwc.patus.util.ASTUtil;
import ch.unibas.cs.hpwc.patus.util.CodeGeneratorUtil;

/**
 *
 * @author Matthias-M. Christen
 */
public class ValidationCodeGenerator
{
	///////////////////////////////////////////////////////////////////
	// Constants

	public final static String SUFFIX_REFERENCE = "_ref";


	///////////////////////////////////////////////////////////////////
	// Member Variables

	private CodeGeneratorSharedObjects m_data;

	/**
	 * The list of generated statements
	 */
	private StatementList m_sl;

	/**
	 * The code generator responsible for generating the actual code
	 */
	private SingleThreadCodeGenerator m_cg;

	/**
	 * Identifier holding a boolean variable that indicates whether there were validation errors
	 */
	private Identifier m_idHasValidationErrors;

	/**
	 * The time iterator variable
	 */
	private Identifier m_idTimeIdx;

	/**
	 * The identifier representing the base grid
	 */
	private SubdomainIdentifier m_sdidBase;

	/**
	 * The identifier representing the index of the iterator iterating point-wise over the base grid
	 */
	private SubdomainIdentifier m_sdidIterator;

	/**
	 * Code generation options
	 */
	private CodeGeneratorRuntimeOptions m_options;

	/**
	 * Set of grids
	 */
	private Set<IDExpression> m_setGrids;


	///////////////////////////////////////////////////////////////////
	// Implementation

	public ValidationCodeGenerator (CodeGeneratorSharedObjects data)
	{
		m_data = data;
		m_sl = new StatementList ();

		m_options = new CodeGeneratorRuntimeOptions ();
		m_options.setOption (CodeGeneratorRuntimeOptions.OPTION_NOVECTORIZE, true);
		m_options.setOption (CodeGeneratorRuntimeOptions.OPTION_STENCILCALCULATION, CodeGeneratorRuntimeOptions.VALUE_STENCILCALCULATION_VALIDATE);
		m_options.setOption (CodeGeneratorRuntimeOptions.OPTION_DOBOUNDARYCHECKS, true);
	}

	/**
	 * Generates the validation block.
	 * @param setGrids The set of all grids
	 * @return
	 */
	public StatementList generate (Set<IDExpression> setGrids)
	{
		if (!m_data.getOptions ().getCreateValidationCode ())
			return m_sl;

		m_setGrids = setGrids;
		m_cg = new SingleThreadCodeGenerator (m_data);

		createIdentifiers ();
		createReferenceSolutionComputation ();
		createValidation ();

		addVariableDeclarations ();

		return m_sl;
	}

	/**
	 * Create iterator identifiers.
	 */
	private void createIdentifiers ()
	{
		m_idTimeIdx = new Identifier (new VariableDeclarator (new NameID ("t_ref")));

		m_sdidBase = m_data.getStrategy ().getBaseDomain ().clone ();

		m_sdidIterator = new SubdomainIdentifier ("pt_ref", new Subdomain (
			m_sdidBase.getSubdomain (),
			Subdomain.ESubdomainType.POINT,
			new Size (Vector.getOnesVector (m_sdidBase.getDimensionality ()))));

		IDExpression idTimeIndex = m_data.getCodeGenerators ().getStrategyAnalyzer ().getTimeIndexVariable ();

		m_sdidBase.setTemporalIndex (idTimeIndex.clone ());
		m_sdidIterator.setTemporalIndex (idTimeIndex.clone ());
	}

	/**
	 * Creates the code computing the reference solution.
	 */
	private void createReferenceSolutionComputation ()
	{
		SubdomainIterator sgitRef = new SubdomainIterator (
			m_sdidIterator.clone (), m_sdidBase.clone (), new Border (m_sdidBase.getDimensionality ()), 1,	null,
			new ExpressionStatement (new AssignmentExpression (
				m_sdidIterator.clone (),
				AssignmentOperator.NORMAL,
				CodeGeneratorUtil.createStencilFunctionCall (m_sdidIterator.clone ()))),
			0);

		RangeIterator itTime = new RangeIterator (
			m_idTimeIdx, Globals.ONE.clone (), m_data.getStencilCalculation ().getMaxIterations ().clone (), null, 1, null, sgitRef, 0);
		itTime.setMainTemporalIterator (true);

		StatementListBundle slbRefCalculation = m_cg.generate (itTime, m_options);
		m_sl.addStatement ((Statement) ASTUtil.addSuffixToIdentifiers (slbRefCalculation.getDefault (), SUFFIX_REFERENCE, m_setGrids));
	}

	private void addPrintErrorStatement (CompoundStatement cmpstmtSetErrorInst, Expression exprGridAccessRef, Expression exprGridAccess)
	{
		byte nDim = m_sdidBase.getDimensionality ();

		String strArrayName = "";
		Expression exprIndex = null;
		for (DepthFirstIterator it = new DepthFirstIterator (exprGridAccessRef); it.hasNext (); )
		{
			Object obj = it.next ();
			if (obj instanceof ArrayAccess)
			{
				strArrayName = ((ArrayAccess) obj).getArrayName ().toString ();
				exprIndex = ((ArrayAccess) obj).getIndex (0);
				break;
			}
		}

		StringBuilder sb = new StringBuilder ("Validation failed for ");
		sb.append (strArrayName);
		sb.append ('[');

		for (int i = 0; i < nDim; i++)
		{
			if (i > 0)
				sb.append (",");
			sb.append ("%d");
		}
		sb.append ("] (index %d). Expected: %e, was: %e\\n");

		List<Expression> listPrintArgs = new ArrayList<> (4 + nDim);
		listPrintArgs.add (new StringLiteral (sb.toString ()));
		for (int i = 0; i < nDim; i++)
			listPrintArgs.add (m_data.getData ().getGeneratedIdentifiers ().getDimensionIndexIdentifier (m_sdidIterator, i).clone ());
		listPrintArgs.add (exprIndex == null ? new IntegerLiteral (-1) : exprIndex.clone ());
		listPrintArgs.add (exprGridAccessRef.clone ());
		listPrintArgs.add (exprGridAccess.clone ());
		cmpstmtSetErrorInst.addStatement (new ExpressionStatement (new FunctionCall (new NameID ("printf"), listPrintArgs)));
	}

	/**
	 * Creates the code doing the validation.
	 */
	private void createValidation ()
	{
		m_data.getData ().getMemoryObjectManager ().resetIndices ();
		m_data.getCodeGenerators ().getUnrollGeneratedIdentifiers ().reset ();

		// TODO: SIMD!

		// declare the validation variable
		VariableDeclarator decl = new VariableDeclarator (new NameID ("bHasErrors"));
		decl.setInitializer (new ValueInitializer (Globals.ZERO.clone ()));
		m_idHasValidationErrors = new Identifier (decl);
		m_sl.addDeclaration (new VariableDeclaration (Specifier.INT, decl));

		// add the validation computation code
		CompoundStatement cmpstmtValidation = new CompoundStatement ();
		StatementList slValidationCalculation = new StatementList (cmpstmtValidation);

		// create the compound statement for the "if" branch (that is executed when an error is found)
		CompoundStatement cmpstmtSetError = new CompoundStatement ();
		cmpstmtSetError.addStatement (new ExpressionStatement (new AssignmentExpression (
			m_idHasValidationErrors.clone (), AssignmentOperator.NORMAL, Globals.ONE.clone ())));

		if (!m_data.getOptions ().isDebugOptionSet (EDebugOption.PRINT_VALIDATION_ERRORS))
			cmpstmtSetError.addStatement (new BreakStatement ());

		// validate results on all grids
		for (StencilNode n : m_data.getStencilCalculation ().getStencilBundle ().getFusedStencil ().getOutputNodes ())
		{
			Expression exprGridAccess = m_data.getData ().getMemoryObjectManager ().getMemoryObjectExpression (
				m_sdidIterator, n, null, true, true, false, slValidationCalculation, m_options);
			exprGridAccess = (Expression) ASTUtil.addSuffixToIdentifiers (exprGridAccess, MemoryObjectManager.SUFFIX_OUTPUTGRID, m_setGrids);

			Expression exprGridAccessRef = exprGridAccess.clone ();
			exprGridAccessRef = (Expression) ASTUtil.addSuffixToIdentifiers (exprGridAccessRef, SUFFIX_REFERENCE, m_setGrids);

			// add the "print error" statement
			CompoundStatement cmpstmtSetErrorInst = cmpstmtSetError.clone ();
			if (m_data.getOptions ().isDebugOptionSet (EDebugOption.PRINT_VALIDATION_ERRORS))
				addPrintErrorStatement (cmpstmtSetErrorInst, exprGridAccessRef, exprGridAccess);

			// create the if statement and compute the relative error
			cmpstmtValidation.addStatement (new IfStatement (
				new BinaryExpression (
					new FunctionCall (
						new NameID ("fabs"),
						CodeGeneratorUtil.expressions (
							// compute the relative error
							new BinaryExpression (
								new BinaryExpression (exprGridAccess, BinaryOperator.SUBTRACT, exprGridAccessRef),
								BinaryOperator.DIVIDE,
								exprGridAccessRef.clone ()
							)
						)
					),
					BinaryOperator.COMPARE_GT,
					new FloatLiteral (m_data.getOptions ().getValidationTolerance ())
				),
				cmpstmtSetErrorInst
			));
		}

		// generate the code
		SubdomainIterator sgitValidate = new SubdomainIterator (
			m_sdidIterator.clone (),
			m_sdidBase.clone (),
			new Border (m_sdidBase.getDimensionality ()),
			1,
			null,
			cmpstmtValidation,
			0);

		m_sl.addStatements (m_cg.generate (sgitValidate, m_options).getDefaultList ());
	}

	/**
	 * Adds the required variable declarations to the generated list of statements.
	 */
	private void addVariableDeclarations ()
	{
		for (Declaration d : m_data.getData ().getDeclarationsToAdd ())
		{
			for (Object o : d.getDeclaredIDs ())
			{
				if (HIRAnalyzer.isReferenced ((IDExpression) o, m_sl))
					m_sl.addDeclaration (d.clone ());
			}
		}
	}

	public Expression getHasValidationErrors ()
	{
		return m_idHasValidationErrors;
	}
}
