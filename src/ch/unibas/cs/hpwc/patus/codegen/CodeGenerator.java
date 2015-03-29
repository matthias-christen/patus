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

import java.io.File;
import java.util.Iterator;
import java.util.List;

import org.apache.log4j.Logger;

import cetus.hir.AssignmentExpression;
import cetus.hir.AssignmentOperator;
import cetus.hir.BinaryExpression;
import cetus.hir.CompoundStatement;
import cetus.hir.DeclarationStatement;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.Identifier;
import cetus.hir.NameID;
import cetus.hir.Specifier;
import cetus.hir.Statement;
import cetus.hir.Traversable;
import cetus.hir.ValueInitializer;
import cetus.hir.VariableDeclaration;
import cetus.hir.VariableDeclarator;
import ch.unibas.cs.hpwc.patus.AbstractBaseCodeGenerator;
import ch.unibas.cs.hpwc.patus.analysis.HIRAnalyzer;
import ch.unibas.cs.hpwc.patus.ast.ParameterAssignment;
import ch.unibas.cs.hpwc.patus.ast.StatementList;
import ch.unibas.cs.hpwc.patus.ast.StatementListBundle;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 *
 * @author Matthias-M. Christen
 */
public class CodeGenerator extends AbstractBaseCodeGenerator
{
	///////////////////////////////////////////////////////////////////
	// Constants

	private final static boolean SINGLE_ASSIGNMENT = false;

	private final static Logger LOGGER = Logger.getLogger (CodeGenerator.class);


	///////////////////////////////////////////////////////////////////
	// Member Variables

	/**
	 * The code generator that generates the code one thread executes from the strategy code
	 */
	private ThreadCodeGenerator m_cgThreadCode;


	///////////////////////////////////////////////////////////////////
	// Implementation

	public CodeGenerator (CodeGeneratorSharedObjects data)
	{
		super ();
		
		m_data = data;
		m_cgThreadCode = new ThreadCodeGenerator (m_data);
	}

	/**
	 * Generates the code.
	 * @param unit The translation unit in which to place the kernels
	 * @param fileOutputDirectory The directory into which the generated code is written
	 * @param bIncludeAutotuneParameters Flag specifying whether to include the autotuning parameters
	 * 	in the function signatures
	 */
	public void generate (List<KernelSourceFile> listOutputs, File fileOutputDirectory, boolean bIncludeAutotuneParameters)
	{
		createFunctionParameterList (true, bIncludeAutotuneParameters);

		// create the stencil calculation code (the code that one thread executes)
		StatementListBundle slbThreadBody = createComputeCode ();

		// create the initialization code
		StatementListBundle slbInitializationBody = createInitializationCode (listOutputs);
		
		// add global declarations
		for (KernelSourceFile out : listOutputs)
			addAdditionalGlobalDeclarations (out, slbThreadBody.getDefault ());

		// add internal autotune parameters to the parameter list
		createFunctionInternalAutotuneParameterList (slbThreadBody);

		// do post-generation optimizations
		optimizeCode (slbThreadBody);

		// package the code into functions, add them to the translation unit, and write the code files
		for (KernelSourceFile out : listOutputs)
		{
			packageKernelSourceFile (out, slbThreadBody, slbInitializationBody, bIncludeAutotuneParameters);
			out.writeCode (this, m_data, fileOutputDirectory);
		}
	}
	
	private StatementListBundle createComputeCode ()
	{
		m_data.getData ().setCreatingInitialization (false);
		CodeGeneratorRuntimeOptions optionsStencil = new CodeGeneratorRuntimeOptions ();
		optionsStencil.setOption (CodeGeneratorRuntimeOptions.OPTION_STENCILCALCULATION, CodeGeneratorRuntimeOptions.VALUE_STENCILCALCULATION_STENCIL);
		optionsStencil.setOption (CodeGeneratorRuntimeOptions.OPTION_DOBOUNDARYCHECKS, m_data.getStencilCalculation ().getBoundaries () != null);

		CompoundStatement cmpstmtStrategyKernelThreadBody = m_cgThreadCode.generate (m_data.getStrategy ().getBody (), optionsStencil);
		m_data.getData ().capture ();

		StatementListBundle slbThreadBody = new SingleThreadCodeGenerator (m_data).generate (cmpstmtStrategyKernelThreadBody, optionsStencil);
		addAdditionalDeclarationsAndAssignments (slbThreadBody, optionsStencil);
		
		return slbThreadBody;
	}
	
	private StatementListBundle createInitializationCode (List<KernelSourceFile> listOutputs)
	{
		StatementListBundle slbInitializationBody = null;
		
		boolean bCreateInitialization = false;
		for (KernelSourceFile out : listOutputs)
			if (out.getCreateInitialization ())
			{
				bCreateInitialization = true;
				break;
			}
		
		if (bCreateInitialization)
		{
			m_data.getData ().setCreatingInitialization (true);
			m_data.getCodeGenerators ().reset ();
			m_data.getData ().reset ();
			CodeGeneratorRuntimeOptions optionsInitialize = new CodeGeneratorRuntimeOptions ();
			optionsInitialize.setOption (CodeGeneratorRuntimeOptions.OPTION_STENCILCALCULATION, CodeGeneratorRuntimeOptions.VALUE_STENCILCALCULATION_INITIALIZE);
			if (m_data.getArchitectureDescription ().useSIMD ())
				optionsInitialize.setOption (CodeGeneratorRuntimeOptions.OPTION_NOVECTORIZE, !m_data.getOptions ().useNativeSIMDDatatypes ());

			CompoundStatement cmpstmtStrategyInitThreadBody = m_cgThreadCode.generate (m_data.getStrategy ().getBody (), optionsInitialize);
			slbInitializationBody = new SingleThreadCodeGenerator (m_data).generate (cmpstmtStrategyInitThreadBody, optionsInitialize);
			addAdditionalDeclarationsAndAssignments (slbInitializationBody, optionsInitialize);
		}
		
		return slbInitializationBody;
	}


	////////////////

	private static int m_nTempCount = 0;
	private Expression substituteBinaryExpressionRecursive (List<Specifier> listSpecs, Expression expr, CompoundStatement cmpstmt)
	{
		if (expr instanceof BinaryExpression)
		{
			Expression exprLHS = substituteBinaryExpressionRecursive (listSpecs, ((BinaryExpression) expr).getLHS (), cmpstmt);
			Expression exprRHS = substituteBinaryExpressionRecursive (listSpecs, ((BinaryExpression) expr).getRHS (), cmpstmt);

			VariableDeclarator decl = new VariableDeclarator (new NameID (StringUtil.concat ("__tmp", CodeGenerator.m_nTempCount++)));
			decl.setInitializer (new ValueInitializer (new BinaryExpression (exprLHS, ((BinaryExpression) expr).getOperator (), exprRHS)));
			cmpstmt.addDeclaration (new VariableDeclaration (listSpecs, decl));
			return new Identifier (decl);
		}

		return expr.clone ();
	}

	@SuppressWarnings("unchecked")
	private CompoundStatement substituteBinaryExpression (Identifier idLHS, AssignmentOperator op, BinaryExpression expr)
	{
		CompoundStatement cmpstmt = new CompoundStatement ();
		cmpstmt.addStatement (new ExpressionStatement (new AssignmentExpression (
			idLHS.clone (),
			op,
			substituteBinaryExpressionRecursive (idLHS.getSymbol ().getTypeSpecifiers (), expr, cmpstmt))));
		return cmpstmt;
	}

	private void substituteBinaryExpressions (Traversable trv)
	{
		if (trv instanceof ExpressionStatement)
		{
			Expression expr = ((ExpressionStatement) trv).getExpression ();
			if (expr instanceof AssignmentExpression)
			{
				AssignmentExpression aexpr = (AssignmentExpression) expr;
				if (aexpr.getLHS () instanceof Identifier && aexpr.getRHS () instanceof BinaryExpression)
					((Statement) trv).swapWith (substituteBinaryExpression ((Identifier) aexpr.getLHS (), aexpr.getOperator (), (BinaryExpression) ((AssignmentExpression) expr).getRHS ()));
			}
		}
		else
		{
			for (Traversable trvChild : trv.getChildren ())
				substituteBinaryExpressions (trvChild);
		}
	}

	////////////////

	/**
	 * Do post-code generation optimizations (loop unrolling, ...).
	 * @param cmpstmtBody
	 * @return
	 */
	protected void optimizeCode (StatementListBundle slbInput)
	{
		// create one assignment for each subexpression
		if (CodeGenerator.SINGLE_ASSIGNMENT)
		{
			for (ParameterAssignment pa : slbInput)
			{
				StatementList sl = slbInput.getStatementList (pa);
				for (Statement stmt : sl.getStatementsAsList ())
					substituteBinaryExpressions (stmt);
			}
		}

		// remove declarations of unused variables
		for (ParameterAssignment pa : slbInput)
		{
			LOGGER.debug (StringUtil.concat ("Removing unused variables from ", pa.toString ()));

			StatementList sl = slbInput.getStatementList (pa);
			List<Statement> list = sl.getStatementsAsList ();
			boolean bModified = false;

			for (Iterator<Statement> it = list.iterator (); it.hasNext (); )
			{
				Statement stmt = it.next ();
				if (stmt instanceof DeclarationStatement && ((DeclarationStatement) stmt).getDeclaration () instanceof VariableDeclaration)
				{
					VariableDeclaration vdecl = (VariableDeclaration) ((DeclarationStatement) stmt).getDeclaration ();
					if (vdecl.getNumDeclarators () == 1)
					{
						if (!HIRAnalyzer.isReferenced (vdecl.getDeclarator (0).getID (), sl))
						{
							it.remove ();
							bModified = true;
						}
					}
				}
			}

			if (bModified)
				slbInput.replaceStatementList (pa, new StatementList (list));
		}

		// remove
	}

	/**
	 *
	 * @param bIncludeAutotuneParameters
	 * @return
	 */
	public String getIncludesAndDefines (boolean bIncludeAutotuneParameters)
	{
		/*
		return StringUtil.concat (
			"#include <stdio.h>\n#include <stdlib.h>\n\n",
			bIncludeAutotuneParameters ? "#include \"kerneltest.h\"\n\n" : null);
		*/

		//return "#define t_max 1\n#define THREAD_NUMBER 0\n#define NUMBER_OF_THREADS 1\n\n";
		//return "#define t_max 1";

		StringBuilder sb = new StringBuilder ();

		if (m_data.getOptions ().isDebugPrintStencilIndices ())
			sb.append ("#include <stdio.h>\n");

		// include files
		for (String strFile : m_data.getArchitectureDescription ().getIncludeFiles ())
		{
			sb.append ("#include \"");
			sb.append (strFile);
			sb.append ("\"\n");
		}

		sb.append ("#include <stdint.h>\n");
		sb.append ("#include \"patusrt.h\"\n");
		
		sb.append ("#include \"");
		sb.append (CodeGenerationOptions.DEFAULT_TUNEDPARAMS_FILENAME);
		sb.append ("\"\n");

		////////
		//sb.append ("#define t_max 1");
		////////

		return sb.toString ();
	}
}
