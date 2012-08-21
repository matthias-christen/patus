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

import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import cetus.hir.AnnotationStatement;
import cetus.hir.CompoundStatement;
import cetus.hir.DeclarationStatement;
import cetus.hir.Expression;
import cetus.hir.Statement;
import cetus.hir.Traversable;
import ch.unibas.cs.hpwc.patus.analysis.LoopAnalyzer;
import ch.unibas.cs.hpwc.patus.analysis.StrategyAnalyzer;
import ch.unibas.cs.hpwc.patus.ast.BoundaryCheck;
import ch.unibas.cs.hpwc.patus.ast.IndexBoundsCalculationInsertionAnnotation;
import ch.unibas.cs.hpwc.patus.ast.Loop;
import ch.unibas.cs.hpwc.patus.ast.RangeIterator;
import ch.unibas.cs.hpwc.patus.ast.StatementListBundle;
import ch.unibas.cs.hpwc.patus.ast.SubdomainIterator;
import ch.unibas.cs.hpwc.patus.codegen.iterator.ManycoreSubdomainIteratorCodeGenerator;
import ch.unibas.cs.hpwc.patus.codegen.iterator.MulticoreRangeIteratorCodeGenerator;
import ch.unibas.cs.hpwc.patus.codegen.iterator.MulticoreSubdomainIteratorCodeGenerator;
import ch.unibas.cs.hpwc.patus.codegen.iterator.MulticoreSubdomainIteratorCodeGenerator2;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;
import ch.unibas.cs.hpwc.patus.util.CodeGeneratorUtil;
import ch.unibas.cs.hpwc.patus.util.ExpressionUtil;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 * Generates the function executed by a single thread.
 *
 * @author Matthias-M. Christen
 */
public class ThreadCodeGenerator
{
	///////////////////////////////////////////////////////////////////
	// Constants

	//private final static Logger LOGGER = Logger.getLogger (ThreadCodeGenerator.class);

	
	///////////////////////////////////////////////////////////////////
	// Member Variables

	/**
	 * The shared data containing all the information for the code generator
	 * (including the stencil calculation and the strategy)
	 */
	private CodeGeneratorSharedObjects m_data;

	
	///////////////////////////////////////////////////////////////////
	// Implementation

	/**
	 *
	 * @param strategy
	 * @param stencil
	 * @param cgThreadFunction
	 */
	public ThreadCodeGenerator (CodeGeneratorSharedObjects data)
	{
		m_data = data;
	}

	/**
	 * Generates the code executed per thread.
	 */
	public CompoundStatement generate (Statement stmtInput, CodeGeneratorRuntimeOptions options)
	{
		// create the function body
		CompoundStatement cmpstmtFunctionBody = new CompoundStatement ();
		generateLoop (stmtInput, cmpstmtFunctionBody, options);

		return cmpstmtFunctionBody;
	}

	/**
	 * Generates the C loop nest.
	 *
	 * @param stmtInput
	 *            The statement to process
	 * @param cmpstmtOutput
	 *            The output structure to which the generated code will be
	 *            appended
	 * @param nLoopNestDepth
	 *            The current loop nesting depth
	 */
	private void generateLoop (Statement stmtInput, CompoundStatement cmpstmtOutput, CodeGeneratorRuntimeOptions options)
	{
		for (Traversable t : stmtInput.getChildren ())
		{
			// add the strategy code as a comment
			if (!(t instanceof DeclarationStatement))
				CodeGeneratorUtil.addComment (cmpstmtOutput, t.toString ());

			// add the code
			if (t instanceof RangeIterator)
				generateRangeIterator ((RangeIterator) t, stmtInput, cmpstmtOutput, options);
			else if (t instanceof SubdomainIterator)
				generateSubdomainIterator ((SubdomainIterator) t, stmtInput, cmpstmtOutput, options);
			else if (t instanceof Statement && !(t instanceof DeclarationStatement))
				cmpstmtOutput.addStatement (((Statement) t).clone ());
		}
	}

	/**
	 * Creates the C code for a range iterator.
	 *
	 * @param loop
	 * @param stmtInput
	 * @param cmpstmtOutput
	 */
	protected void generateRangeIterator (RangeIterator loop, Statement stmtInput, CompoundStatement cmpstmtOutput, CodeGeneratorRuntimeOptions options)
	{
		// create the code for the children
		CompoundStatement cmpstmtLoopBody = new CompoundStatement ();
		cmpstmtLoopBody.setParent (cmpstmtOutput);
		generateLoop (loop, cmpstmtLoopBody, options);

		Expression exprTripCount = LoopAnalyzer.getConstantTripCount (loop.getStart (), loop.getEnd (), loop.getStep ());
		if (exprTripCount != null && ExpressionUtil.isValue (exprTripCount, 1))
		{
			// trip count is 1: omit the loop
			StatementListBundle slb = new StatementListBundle (cmpstmtOutput);

			cmpstmtLoopBody.setParent (null);
			slb.addStatement (cmpstmtLoopBody);
			m_data.getData ().getMemoryObjectManager ().swapMemoryObjectPointers (null, slb, options);
		}
		else
		{
			if (!loop.isSequential ())
			{
				// replace a parallel loop by the part a thread executes
				switch (m_data.getCodeGenerators ().getBackendCodeGenerator ().getThreading ())
				{
				case MULTI:
					// calculate linearized block indices from the linearized
					// thread index
					new MulticoreRangeIteratorCodeGenerator (m_data, loop, cmpstmtLoopBody, cmpstmtOutput, options).generate ();
					break;

				case MANY:
					// calculate ND block indices from an ND thread index
					new MulticoreRangeIteratorCodeGenerator (m_data, loop, cmpstmtLoopBody, cmpstmtOutput, options).generate ();
					break;

				default:
					throw new RuntimeException (StringUtil.concat ("Code generation for ", m_data.getCodeGenerators ().getBackendCodeGenerator ().getThreading (), " not implemented"));
				}
			}
			else
			{
				// sequential loop: leave the loop untouched
				RangeIterator loopGenerated = (RangeIterator) loop.clone ();
				loopGenerated.setLoopBody (cmpstmtLoopBody);
				cmpstmtOutput.addStatement (loopGenerated);
			}

			// generate the pointer swapping code if this is the inner most
			// temporal loop
			if (m_data.getCodeGenerators ().getStrategyAnalyzer ().isInnerMostTemporalLoop (loop))
			{
				StatementListBundle slbLoopBody = new StatementListBundle (cmpstmtLoopBody);
				m_data.getData ().getMemoryObjectManager ().swapMemoryObjectPointers (loop, slbLoopBody, options);
			}
		}
	}

	protected void generateSubdomainIterator (
		SubdomainIterator loop, Statement stmtInput, CompoundStatement cmpstmtOutput, CodeGeneratorRuntimeOptions options)
	{
		boolean bDoBoundaryChecks = options.getBooleanValue (CodeGeneratorRuntimeOptions.OPTION_DOBOUNDARYCHECKS, false);
		
		// create the code for the children
		CompoundStatement cmpstmtLoopBody = new CompoundStatement ();
		cmpstmtLoopBody.setParent (cmpstmtOutput);
		if (bDoBoundaryChecks)
		{
			CompoundStatement cmpstmtWithBndChecks = new CompoundStatement ();
			generateLoop (loop, cmpstmtWithBndChecks, options);
			
			CompoundStatement cmpstmtWithoutBndChecks = new CompoundStatement ();
			CodeGeneratorRuntimeOptions optionsNoBndChecks = options.clone ();
			optionsNoBndChecks.setOption (CodeGeneratorRuntimeOptions.OPTION_DOBOUNDARYCHECKS, false);
			generateLoop (loop, cmpstmtWithoutBndChecks, optionsNoBndChecks);
			
			cmpstmtLoopBody.addStatement (new BoundaryCheck (loop, cmpstmtWithBndChecks, cmpstmtWithoutBndChecks));
		}
		else
			generateLoop (loop, cmpstmtLoopBody, options);

		// add the annotation telling where to add index bounds calculations
		CodeGeneratorUtil.addStatementAtTop (cmpstmtLoopBody, new AnnotationStatement (new IndexBoundsCalculationInsertionAnnotation (loop)));
		
		// determine whether the loop contains a stencil call
		boolean bContainsStencilCall = StrategyAnalyzer.directlyContainsStencilCall (loop);

		// // if the loop entails data transfers, let the
		// SubdomainIteratorCodeGenerator do the parallelization
		if (!loop.isSequential () /* && !m_data.getCodeGenerators ().getStrategyAnalyzer ().isDataLoadedInIterator (loop, m_data.getArchitectureDescription ()) */)
		{
			// replace a parallel loop by the part a thread executes
			switch (m_data.getCodeGenerators ().getBackendCodeGenerator ().getThreading ())
			{
			case MULTI:
				// calculate linearized block indices from the linearized thread index
				try
				{
					new MulticoreSubdomainIteratorCodeGenerator2 (m_data, loop, cmpstmtLoopBody, cmpstmtOutput, bContainsStencilCall, options).generate ();
				}
				catch (NotImplementedException e)
				{
					new MulticoreSubdomainIteratorCodeGenerator (m_data, loop, cmpstmtLoopBody, cmpstmtOutput, bContainsStencilCall, options).generate ();
				}
				break;

			case MANY:
				// calculate ND block indices from an ND thread index
				new ManycoreSubdomainIteratorCodeGenerator (m_data, loop, cmpstmtLoopBody, cmpstmtOutput, bContainsStencilCall, options).generate ();
				break;

			default:
				throw new RuntimeException (StringUtil.concat ("Code generation for ", m_data.getCodeGenerators ().getBackendCodeGenerator ().getThreading (), " not implemented"));
			}
			
//			new MulticoreSubdomainIteratorCodeGenerator (m_data).generate (loop, cmpstmtLoopBody, cmpstmtOutput, bContainsStencilCall, options);
		}
		else
		{
			// leave the loop untouched
			Loop loopGenerated = loop.clone ();
			loopGenerated.setLoopBody (cmpstmtLoopBody);
			cmpstmtOutput.addStatement (loopGenerated);
		}
	}	
}
