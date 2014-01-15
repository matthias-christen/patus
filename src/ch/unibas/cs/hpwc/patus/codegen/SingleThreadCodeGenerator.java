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

import org.apache.log4j.Logger;

import cetus.hir.AnnotationStatement;
import cetus.hir.CommentAnnotation;
import cetus.hir.CompoundStatement;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.IfStatement;
import cetus.hir.Statement;
import cetus.hir.Traversable;
import ch.unibas.cs.hpwc.patus.analysis.StrategyAnalyzer;
import ch.unibas.cs.hpwc.patus.ast.BoundaryCheck;
import ch.unibas.cs.hpwc.patus.ast.RangeIterator;
import ch.unibas.cs.hpwc.patus.ast.StatementListBundle;
import ch.unibas.cs.hpwc.patus.ast.SubdomainIterator;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;
import ch.unibas.cs.hpwc.patus.util.StatementListBundleUtil;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 *
 * @author Matthias-M. Christen
 */
public class SingleThreadCodeGenerator implements ICodeGenerator
{
	///////////////////////////////////////////////////////////////////
	// Constants

	private final static Logger LOGGER = Logger.getLogger (SingleThreadCodeGenerator.class);


	///////////////////////////////////////////////////////////////////
	// Member Variables

	/**
	 * The shared data
	 */
	private CodeGeneratorSharedObjects m_data;

	/**
	 * The code generator that generates C code for subdomain iterators
	 */
	private SubdomainIteratorCodeGenerator m_cgSubdomain;
	
	private BoundaryCheckCodeGenerator m_cgBoundaryCheck;


	///////////////////////////////////////////////////////////////////
	// Implementation

	/**
	 * Creates a new code generator instance.
	 * @param data The shared code generator data
	 */
	public SingleThreadCodeGenerator (CodeGeneratorSharedObjects data)
	{
		m_data = data;
		
		m_cgSubdomain = new SubdomainIteratorCodeGenerator (this, m_data);
		m_cgBoundaryCheck = new BoundaryCheckCodeGenerator (m_data);
	}

	/**
	 * Generates C code from the input representation generated by
	 * {@link ThreadCodeGenerator}.
	 * @param trvInput The input structure
	 * @return
	 */
	@Override
	public StatementListBundle generate (Traversable trvInput, CodeGeneratorRuntimeOptions options)
	{
		if (SingleThreadCodeGenerator.LOGGER.isDebugEnabled ())
			SingleThreadCodeGenerator.LOGGER.debug (StringUtil.concat ("Generating code (options: ", options.toString (), ") for ", trvInput.toString ()));

		// process the range iterator
		if (trvInput instanceof RangeIterator)
			return generateRangeIterator ((RangeIterator) trvInput, options);
		
		// process the subdomain iterator
		if (trvInput instanceof SubdomainIterator)
			return generateSubdomainIterator ((SubdomainIterator) trvInput, options);
		
		if (trvInput instanceof BoundaryCheck)
			return generateBoundaryCheck ((BoundaryCheck) trvInput, options);

		if (trvInput instanceof CompoundStatement)
			return generateCompoundStatement ((CompoundStatement) trvInput, options);
		
		// process if statements
		if (trvInput instanceof IfStatement)
			return generateIfStatement ((IfStatement) trvInput, options);

		// process an expression statement
		if (trvInput instanceof ExpressionStatement)
			return generateExpression (((ExpressionStatement) trvInput).getExpression (), options);

		// Other statement...
		if (trvInput instanceof Statement)
			return new StatementListBundle ((Statement) trvInput);
		
		return new StatementListBundle ();
	}
	
	private boolean isOuterMostTemporalLoop (RangeIterator it)
	{
		return it.getLoopIndex ().equals (m_data.getCodeGenerators ().getStrategyAnalyzer ().getTimeIndexVariable ());		
	}

	/**
	 * Generates code for a range iterator.
	 * @param it
	 */
	protected StatementListBundle generateRangeIterator (RangeIterator it, CodeGeneratorRuntimeOptions options)
	{
		boolean bIsIteratorNeeded = !options.hasValue (CodeGeneratorRuntimeOptions.OPTION_STENCILCALCULATION, CodeGeneratorRuntimeOptions.VALUE_STENCILCALCULATION_INITIALIZE) ||
			!isOuterMostTemporalLoop (it);
		
		if (!it.getNumberOfThreads ().equals (Globals.ONE))
			throw new RuntimeException ("Parallel loops are not supported");

		// generate the code for the loop body
		StatementListBundle slbLoopBody = generate (it.getLoopBody ().clone (), options);
		if (!bIsIteratorNeeded)
			return slbLoopBody;

		StatementListBundle slbGenerated = new StatementListBundle (new ArrayList<Statement> ());
		slbGenerated.addStatement (new AnnotationStatement (new CommentAnnotation (it.getLoopHeadAnnotation ())));

		// generate the pointer swapping code if this is the inner most temporal loop
		if (m_data.getCodeGenerators ().getStrategyAnalyzer ().isInnerMostTemporalLoop (it) ||
			options.hasValue (CodeGeneratorRuntimeOptions.OPTION_STENCILCALCULATION, CodeGeneratorRuntimeOptions.VALUE_STENCILCALCULATION_VALIDATE))
		{
			m_data.getData ().getMemoryObjectManager ().swapMemoryObjectPointers (it, slbLoopBody, options);
		}

		return m_data.getCodeGenerators ().getLoopCodeGenerator ().generate (it, it.getStart (), slbLoopBody, slbGenerated, options);
	}

	/**
	 * Generates code for a subdomain iterator.
	 * @param it
	 */
	protected StatementListBundle generateSubdomainIterator (SubdomainIterator it, CodeGeneratorRuntimeOptions options)
	{
		if (!it.getNumberOfThreads ().equals (Globals.ONE))
			throw new RuntimeException ("Parallel loops are not supported");

		StatementListBundle slbGenerated = new StatementListBundle (new ArrayList<Statement> ());
		slbGenerated.addStatement (new AnnotationStatement (new CommentAnnotation (it.getLoopHeadAnnotation ())));
		slbGenerated.addStatements (m_cgSubdomain.generate (it, options));

		return slbGenerated;
	}
	
	protected StatementListBundle generateBoundaryCheck (BoundaryCheck bc, CodeGeneratorRuntimeOptions options)
	{
		// generate the control expression to check whether we're in a boundary region
		Expression exprCheck = m_cgBoundaryCheck.generate (bc, options);
		
		// create the calculations with the boundary checks
		StatementListBundle slbWithChecks = new StatementListBundle (new ArrayList<Statement> ());
		CodeGeneratorRuntimeOptions optionsWithChecks = options.clone ();
		optionsWithChecks.setOption (CodeGeneratorRuntimeOptions.OPTION_DOBOUNDARYCHECKS, true);
		slbWithChecks.addStatements (generate (bc.getWithChecks (), optionsWithChecks));

		// exprCheck is null if no checks are required, but since we're in a boundary check, we need to return the "with checks" statement
		if (exprCheck == null)
			return slbWithChecks;

		StatementListBundle slbWithoutChecks = new StatementListBundle (new ArrayList<Statement> ());
		CodeGeneratorRuntimeOptions optionsWithoutChecks = options.clone ();
		optionsWithoutChecks.setOption (CodeGeneratorRuntimeOptions.OPTION_DOBOUNDARYCHECKS, false);
		slbWithoutChecks.addStatements (generate (bc.getWithoutChecks (), optionsWithoutChecks));

		return StatementListBundleUtil.createIfStatement (exprCheck, slbWithChecks, slbWithoutChecks);
	}
	
	protected StatementListBundle generateCompoundStatement (CompoundStatement cmpstmt, CodeGeneratorRuntimeOptions options)
	{
		StatementListBundle slbGenerated = new StatementListBundle (new ArrayList<Statement> ());

		// process all the children of the compound statement
		for (Traversable trvChild : cmpstmt.getChildren ())
			if (trvChild instanceof Statement)
				slbGenerated.addStatements (generate (trvChild, options));

		return slbGenerated;
	}
	
	protected StatementListBundle generateIfStatement (IfStatement stmtIf, CodeGeneratorRuntimeOptions options)
	{
		StatementListBundle slbThen = new StatementListBundle (new ArrayList<Statement> ());
		slbThen.addStatements (generate (stmtIf.getThenStatement (), options));
		
		StatementListBundle slbElse = null;
		if (stmtIf.getElseStatement () != null)
		{
			slbElse = new StatementListBundle (new ArrayList<Statement> ());
			slbThen.addStatements (generate (stmtIf.getElseStatement (), options));
		}
		
		return StatementListBundleUtil.createIfStatement (stmtIf.getControlExpression (), slbThen, slbElse);
	}

	/**
	 * Executes the expression <code>expr</code>.
	 * @param expr The expression to execute
	 */
	protected StatementListBundle generateExpression (Expression expr, CodeGeneratorRuntimeOptions options)
	{
		if (StrategyAnalyzer.isStencilCall (expr))
			return m_data.getCodeGenerators ().getStencilCalculationCodeGenerator ().generate (expr, options);

		StatementListBundle slGenerated = new StatementListBundle (new ArrayList<Statement> ());
		if (options.getIntValue (CodeGeneratorRuntimeOptions.OPTION_LOOPUNROLLINGFACTOR, 1) > 1)
			throw new RuntimeException ("Loop unrolling for expressions not yet implemented");
		else
			slGenerated.addStatement (new ExpressionStatement (expr.clone ()));

		return slGenerated;
	}
}
