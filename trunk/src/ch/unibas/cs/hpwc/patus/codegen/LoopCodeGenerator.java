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

import java.util.LinkedList;

import org.apache.log4j.Logger;

import cetus.hir.AssignmentExpression;
import cetus.hir.AssignmentOperator;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.CompoundStatement;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.ForLoop;
import cetus.hir.NameID;
import cetus.hir.Statement;
import cetus.hir.VariableDeclaration;
import cetus.hir.VariableDeclarator;
import ch.unibas.cs.hpwc.patus.ast.Loop;
import ch.unibas.cs.hpwc.patus.ast.ParameterAssignment;
import ch.unibas.cs.hpwc.patus.ast.RangeIterator;
import ch.unibas.cs.hpwc.patus.ast.StatementList;
import ch.unibas.cs.hpwc.patus.ast.StatementListBundle;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;
import ch.unibas.cs.hpwc.patus.util.AnalyzeTools;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 *
 * @author Matthias-M. Christen
 */
public class LoopCodeGenerator
{
	///////////////////////////////////////////////////////////////////
	// Constants

	private final static Logger LOGGER = Logger.getLogger (LoopCodeGenerator.class);


	///////////////////////////////////////////////////////////////////
	// Member Variables

	/**
	 * The data objects that are shared across the code generators.
	 */
	private CodeGeneratorSharedObjects m_data;


	///////////////////////////////////////////////////////////////////
	// Implementation

	public LoopCodeGenerator (CodeGeneratorSharedObjects data)
	{
		m_data = data;
	}

	/**
	 *
	 * @param loop
	 * @param exprLoopIndexStartValue
	 * @param cmpstmtLoopBody
	 * @param cmpstmtOut
	 */
	public StatementListBundle generate (RangeIterator loop, Expression exprLoopIndexStartValue, StatementListBundle slbLoopBody, StatementListBundle slbGeneratedCode, CodeGeneratorRuntimeOptions options)
	{
		if (LOGGER.isDebugEnabled ())
			LOGGER.debug (StringUtil.concat ("Generating code with options ", options.toString ()));

		if (slbGeneratedCode == null)
			slbGeneratedCode = new StatementListBundle (new LinkedList<Statement> ());

		// check whether there are actual statements in the loop body
		for (ParameterAssignment pa : slbLoopBody)
		{
			StatementList sl = slbLoopBody.getStatementList (pa);
			CompoundStatement cmpstmtLoopBody = sl.getCompoundStatement ();

			if (AnalyzeTools.containsEffectiveStatement (cmpstmtLoopBody))
			{
				// generate the loop code
				m_data.getData ().addDeclaration (new VariableDeclaration (Globals.SPECIFIER_INDEX, new VariableDeclarator (new NameID (loop.getLoopIndex ().getName ()))));
				sl.clear ();
				sl.addStatement (
					new ForLoop (
						new ExpressionStatement (new AssignmentExpression (loop.getLoopIndex (), AssignmentOperator.NORMAL, exprLoopIndexStartValue.clone ())),
						new BinaryExpression (loop.getLoopIndex (), BinaryOperator.COMPARE_LE, loop.getEnd ()),
						new AssignmentExpression (loop.getLoopIndex (), AssignmentOperator.ADD, loop.getStep ()),
						cmpstmtLoopBody));
			}
		}

		slbGeneratedCode.addStatements (slbLoopBody);

		return slbGeneratedCode;
	}

	/**
	 * Returns the &quot;real&quot; number of threads that execute the loop
	 * <code>loop</code>,
	 * i.e. replaces the {@link Globals#NUMBER_OF_THREADS} placeholder by the
	 * number provided
	 * by the hardware/programming model-specific backend.
	 * 
	 * @param loop
	 *            The loop
	 * @return An expression that evaluates the the number of threads that
	 *         execute <code>loop</code>
	 */
	public Expression getNumberOfThreadsInDimension (Loop loop, int nDim, CodeGeneratorRuntimeOptions options)
	{
		if (Globals.NUMBER_OF_THREADS.equals (loop.getNumberOfThreads ()))
		{
			return m_data.getCodeGenerators ().getConstantGeneratedIdentifiers ().getConstantIdentifier (
				m_data.getCodeGenerators ().getBackendCodeGenerator ().getIndexingLevelFromParallelismLevel (loop.getParallelismLevel ()).getSizeForDimension (nDim),
				"dimsize", Globals.SPECIFIER_SIZE, null, null, options);
		}

		return loop.getNumberOfThreads ();
	}
}
