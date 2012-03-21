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

import cetus.hir.ArrayAccess;
import cetus.hir.AssignmentExpression;
import cetus.hir.AssignmentOperator;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.CompoundStatement;
import cetus.hir.Declaration;
import cetus.hir.DeclarationStatement;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.ForLoop;
import cetus.hir.FunctionCall;
import cetus.hir.IntegerLiteral;
import cetus.hir.NameID;
import cetus.hir.PointerSpecifier;
import cetus.hir.Procedure;
import cetus.hir.ProcedureDeclarator;
import cetus.hir.Specifier;
import cetus.hir.TranslationUnit;
import cetus.hir.UnaryExpression;
import cetus.hir.UnaryOperator;
import cetus.hir.ValueInitializer;
import cetus.hir.VariableDeclaration;
import cetus.hir.VariableDeclarator;
import ch.unibas.cs.hpwc.patus.representation.StencilCalculation;
import ch.unibas.cs.hpwc.patus.util.CodeGeneratorUtil;

/**
 *
 * @author Matthias-M. Christen
 * 
 * @deprecated
 */
public class MainFunctionCodeGenerator
{
	///////////////////////////////////////////////////////////////////
	// Constants

	private static final NameID VAR_CMDLINEARGS = new NameID ("g_rgCmdLineArgs");

	private static final NameID FNX_TIC = new NameID ("tic");
	private static final NameID FNX_TOC = new NameID ("toc");


	///////////////////////////////////////////////////////////////////
	// Member Variables

	/**
	 * The strategy
	 */
	private Strategy m_strategy;

	/**
	 * The stencil calculation
	 */
	private StencilCalculation m_stencil;

	/**
	 * Data shared among all the code generator classes
	 */
	private CodeGeneratorSharedObjects m_data;


	///////////////////////////////////////////////////////////////////
	// Implementation

	/**
	 *
	 * @param strategy
	 * @param stencil
	 */
	public MainFunctionCodeGenerator (Strategy strategy, StencilCalculation stencil, CodeGeneratorSharedObjects data)
	{
		m_strategy = strategy;
		m_stencil = stencil;

		m_data = data;
	}

	/**
	 *
	 * @param unit
	 */
	private static void generateRuntimeForwardDeclarations (TranslationUnit unit)
	{
		// generate forward declarations
		unit.addDeclarationFirst (CodeGeneratorUtil.createForwardDeclaration (Specifier.VOID, FNX_TIC));
		unit.addDeclarationFirst (CodeGeneratorUtil.createForwardDeclaration (Specifier.VOID, FNX_TOC));
	}

	/**
	 * Generates global variables.
	 * @param tu The translation unit to which the symbols will be added
	 */
	private void generateGlobals (TranslationUnit unit)
	{
		// global for the program arguments
		unit.addDeclarationFirst (CodeGeneratorUtil.createArrayDeclaration (VAR_CMDLINEARGS, new IntegerLiteral (getArgumentsCount ())));
	}

	/**
	 *
	 * @return
	 */
	private int getArgumentsCount ()
	{
		// internal autotuning parameters
		int nArgsCount = m_data.getData ().getInternalAutotuningParametersCount ();

		// count the "auto" params of the strategy
		for (Declaration decl : m_strategy.getParameters ())
		{
			if (decl instanceof VariableDeclaration)
			{
				List<Specifier> listSpecifiers = ((VariableDeclaration) decl).getSpecifiers ();
				if (listSpecifiers != null)
					if (listSpecifiers.contains (Specifier.AUTO))
						nArgsCount++;
			}
		}

		return nArgsCount;
	}

	/**
	 * Generates the <code>main</code> function
	 * <pre>
	 * 	int main (int argc, char** argv) { ... }
	 * </pre>
	 * @param unit The translation unit in which the function will be placed.
	 */
	public void generateMain (TranslationUnit unit)
	{
		MainFunctionCodeGenerator.generateRuntimeForwardDeclarations (unit);
		generateGlobals (unit);


		// generates int main (int argc, char** argv)

		// function parameters for main
		// ++++++++++++++++++++++++++++++++++++++++++
		// int main (int argc, char** argv) { ... }
		// ++++++++++++++++++++++++++++++++++++++++++

		NameID idArgc = new NameID ("argc");
		NameID idArgv = new NameID ("argv");

		List<Specifier> listArgvSpecifiers = new ArrayList<> ();
		listArgvSpecifiers.add (Specifier.CHAR);
		listArgvSpecifiers.add (PointerSpecifier.UNQUALIFIED);
		listArgvSpecifiers.add (PointerSpecifier.UNQUALIFIED);

		List<Declaration> listMainParams = new ArrayList<> ();
		listMainParams.add (new VariableDeclaration (Specifier.INT, new VariableDeclarator (idArgc.clone ())));
		listMainParams.add (new VariableDeclaration (
			CodeGeneratorUtil.specifiers (Specifier.CHAR, PointerSpecifier.UNQUALIFIED, PointerSpecifier.UNQUALIFIED),
			new VariableDeclarator (idArgv.clone ())));

		// create the function body
		CompoundStatement cmpstmtBody = new CompoundStatement ();


		// get program arguments:
		// ++++++++++++++++++++++++++++++++++++++++++
		// for (int i = 0; i < argc; i++)
		//     g_rgCmdLineArgs[i] = atoi (argv[i]);
		// ++++++++++++++++++++++++++++++++++++++++++

		NameID idI = new NameID ("i");
		VariableDeclarator declI = new VariableDeclarator (Specifier.INT, idI);
		declI.setInitializer (new ValueInitializer (new IntegerLiteral (0)));

		List<Expression> listAtoiArgs = new ArrayList<> (1);
		listAtoiArgs.add (new ArrayAccess (idArgv, idI));

		cmpstmtBody.addStatement (new ForLoop (
			new DeclarationStatement (new VariableDeclaration (declI)),
			new BinaryExpression (idI, BinaryOperator.COMPARE_LT, idArgc),
			new UnaryExpression (UnaryOperator.PRE_INCREMENT, idI),
			new ExpressionStatement (new AssignmentExpression (
				new ArrayAccess (VAR_CMDLINEARGS, idI),
				AssignmentOperator.NORMAL,
				new FunctionCall (new NameID ("atoi"), listAtoiArgs)))));


		// add the timing code
		// ++++++++++++++++++++++++++++++++++++++++++
		// tic ();
		// stencil ();
		// toc ();
		// ++++++++++++++++++++++++++++++++++++++++++

		cmpstmtBody.addStatement (new ExpressionStatement (new FunctionCall (FNX_TIC)));
		cmpstmtBody.addStatement (new ExpressionStatement (new FunctionCall (new NameID (m_stencil.getName ()))));
		cmpstmtBody.addStatement (new ExpressionStatement (new FunctionCall (FNX_TOC)));

		// add the code to the translation unit
		unit.addDeclaration (new Procedure (
			Specifier.INT,
			new ProcedureDeclarator (new NameID ("main"), listMainParams),
			cmpstmtBody));
	}
}
