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
package ch.unibas.cs.hpwc.patus.codegen.backend;

import java.util.List;

import cetus.hir.Expression;
import cetus.hir.Specifier;
import cetus.hir.Statement;
import cetus.hir.Traversable;
import ch.unibas.cs.hpwc.patus.ast.StatementList;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;
import ch.unibas.cs.hpwc.patus.codegen.GlobalGeneratedIdentifiers;
import ch.unibas.cs.hpwc.patus.codegen.GlobalGeneratedIdentifiers.Variable;
import ch.unibas.cs.hpwc.patus.codegen.KernelSourceFile;
import ch.unibas.cs.hpwc.patus.codegen.backend.AbstractNonKernelFunctionsImpl.EOutputGridType;

/**
 * Provides default implementations for certain methods of the {@link IArithmetic}
 * and the {@link INonKernelFunctions} interfaces.
 *
 * @author Matthias-M. Christen
 */
public abstract class AbstractBackend implements IBackend
{
	///////////////////////////////////////////////////////////////////
	// Member Variables

	protected CodeGeneratorSharedObjects m_data;

	
	// Mixin classes

	protected AbstractArithmeticImpl m_mixinArithmetic;

	protected AbstractNonKernelFunctionsImpl m_mixinNonKernelFunctions;
	

	///////////////////////////////////////////////////////////////////
	// Implementation

	public AbstractBackend (CodeGeneratorSharedObjects data)
	{
		m_data = data;

		m_mixinArithmetic = new AbstractArithmeticImpl (m_data)
		{
			@Override
			public Traversable splat (Expression expr, Specifier specDatatype)
			{
				return null;
			}

			@Override
			public Expression shuffle(Expression expr1, Expression expr2, Specifier specDatatype, int nOffset)
			{
				return null;
			}
		};

		m_mixinNonKernelFunctions = new AbstractNonKernelFunctionsImpl (m_data)
		{
		};
	}

	@Override
	public void setKernelSourceFile (KernelSourceFile ksf)
	{
		m_mixinNonKernelFunctions.setKernelSourceFile (ksf);
	}
	
	@Override
	public IIndexingLevel getIndexingLevelFromParallelismLevel (int nParallelismLevel)
	{
		return IndexingLevelUtil.getIndexingLevelFromParallelismLevel (this, nParallelismLevel);
	}

	/**
	 * @see AbstractNonKernelFunctionsImpl#getExpressionForVariable(Variable)
	 */
	public Expression getExpressionForVariable (GlobalGeneratedIdentifiers.Variable variable)
	{
		return m_mixinNonKernelFunctions.getExpressionForVariable (variable);
	}

	/**
	 * @see AbstractNonKernelFunctionsImpl#getExpressionForVariable(Variable, EOutputGridType)
	 */
	public Expression getExpressionForVariable (GlobalGeneratedIdentifiers.Variable variable, EOutputGridType typeOutputGrid)
	{
		return m_mixinNonKernelFunctions.getExpressionForVariable (variable, typeOutputGrid);
	}

	/**
	 * @see AbstractNonKernelFunctionsImpl#getExpressionsForVariables(List)
	 */
	public List<Expression> getExpressionsForVariables (List<Variable> listVariables)
	{
		return m_mixinNonKernelFunctions.getExpressionsForVariables (listVariables);
	}

	/**
	 * @see AbstractNonKernelFunctionsImpl#getExpressionForVariable(Variable, EOutputGridType)
	 */
	public List<Expression> getExpressionsForVariables (List<Variable> listVariables, EOutputGridType typeOutputGrid)
	{
		return m_mixinNonKernelFunctions.getExpressionsForVariables (listVariables, typeOutputGrid);
	}


	///////////////////////////////////////////////////////////////////
	// IParallel Implementation

	@Override
	public Statement getBarrier (int nParallelismLevel)
	{
		return m_data.getArchitectureDescription ().getBarrier (nParallelismLevel);
	}


	///////////////////////////////////////////////////////////////////
	// IIArithmetic Implementation

	@Override
	public Expression createExpression (Expression exprIn, Specifier specDatatype, boolean bVectorize)
	{
		return m_mixinArithmetic.createExpression (exprIn, specDatatype, bVectorize);
	}

	@Override
	public Expression unary_plus (Expression expr, Specifier specDatatype, boolean bVectorize)
	{
		return m_mixinArithmetic.unary_plus (expr, specDatatype, bVectorize);
	}

	@Override
	public Expression unary_minus (Expression expr, Specifier specDatatype, boolean bVectorize)
	{
		return m_mixinArithmetic.unary_minus (expr, specDatatype, bVectorize);
	}

	@Override
	public Expression plus (Expression expr1, Expression expr2, Specifier specDatatype, boolean bVectorize)
	{
		return m_mixinArithmetic.plus (expr1, expr2, specDatatype, bVectorize);
	}

	@Override
	public Expression minus (Expression expr1, Expression expr2, Specifier specDatatype, boolean bVectorize)
	{
		return m_mixinArithmetic.minus (expr1, expr2, specDatatype, bVectorize);
	}

	@Override
	public Expression multiply (Expression expr1, Expression expr2, Specifier specDatatype, boolean bVectorize)
	{
		return m_mixinArithmetic.multiply (expr1, expr2, specDatatype, bVectorize);
	}

	@Override
	public Expression divide (Expression expr1, Expression expr2, Specifier specDatatype, boolean bVectorize)
	{
		return m_mixinArithmetic.divide (expr1, expr2, specDatatype, bVectorize);
	}

	@Override
	public Expression fma (Expression exprSummand, Expression exprFactor1, Expression exprFactor2, Specifier specDatatype, boolean bVectorize)
	{
		return m_mixinArithmetic.fma (exprSummand, exprFactor1, exprFactor2, specDatatype, bVectorize);
	}

	@Override
	public Expression fms (Expression exprSummand, Expression exprFactor1, Expression exprFactor2, Specifier specDatatype, boolean bVectorize)
	{
		return m_mixinArithmetic.fms (exprSummand, exprFactor1, exprFactor2, specDatatype, bVectorize);
	}

	@Override
	public Expression sqrt (Expression expr, Specifier specDatatype, boolean bVectorize)
	{
		return m_mixinArithmetic.sqrt (expr, specDatatype, bVectorize);
	}
	
	@Override
	public Expression vector_reduce_sum (Expression expr, Specifier specDatatype)
	{
		return m_mixinArithmetic.vector_reduce_sum (expr, specDatatype);
	}

	@Override
	public Expression vector_reduce_product (Expression expr, Specifier specDatatype)
	{
		return m_mixinArithmetic.vector_reduce_product (expr, specDatatype);
	}

	@Override
	public Expression vector_reduce_min (Expression expr, Specifier specDatatype)
	{
		return m_mixinArithmetic.vector_reduce_min (expr, specDatatype);
	}

	@Override
	public Expression vector_reduce_max (Expression expr, Specifier specDatatype)
	{
		return m_mixinArithmetic.vector_reduce_max (expr, specDatatype);
	}

	
	///////////////////////////////////////////////////////////////////
	// IAdditionalKernelSpecific Implementation

	@Override
	public String getAdditionalKernelSpecificCode ()
	{
		return null;
	}


	///////////////////////////////////////////////////////////////////
	// INonKernelFunctions Implementation

	@Override
	public void initializeNonKernelFunctionCG ()
	{
		m_mixinNonKernelFunctions.initializeNonKernelFunctionCG ();
	}

	@Override
	public StatementList forwardDecls ()
	{
		return m_mixinNonKernelFunctions.forwardDecls ();
	}

	@Override
	public StatementList declareGrids ()
	{
		return m_mixinNonKernelFunctions.declareGrids ();
	}

	@Override
	public StatementList allocateGrids ()
	{
		return m_mixinNonKernelFunctions.allocateGrids ();
	}

	@Override
	public StatementList initializeGrids ()
	{
		return m_mixinNonKernelFunctions.initializeGrids ();
	}

	@Override
	public StatementList sendData ()
	{
		return m_mixinNonKernelFunctions.sendData ();
	}

	@Override
	public StatementList receiveData ()
	{
		return m_mixinNonKernelFunctions.receiveData ();
	}

	@Override
	public StatementList computeStencil ()
	{
		return m_mixinNonKernelFunctions.computeStencil ();
	}

	@Override
	public StatementList validateComputation ()
	{
		return m_mixinNonKernelFunctions.validateComputation ();
	}
	
	@Override
	public StatementList writeGrids (String strFilenameFormat, String strType)
	{
		return m_mixinNonKernelFunctions.writeGrids (strFilenameFormat, strType);
	}

	@Override
	public StatementList deallocateGrids()
	{
		return m_mixinNonKernelFunctions.deallocateGrids ();
	}
	
	@Override
	public Expression getFlopsPerStencil ()
	{
		return m_mixinNonKernelFunctions.getFlopsPerStencil ();
	}

	@Override
	public Expression getGridPointsCount ()
	{
		return m_mixinNonKernelFunctions.getGridPointsCount ();
	}

	@Override
	public Expression getBytesTransferred ()
	{
		return m_mixinNonKernelFunctions.getBytesTransferred ();
	}

	@Override
	public Expression getDoValidation ()
	{
		return m_mixinNonKernelFunctions.getDoValidation ();
	}

	@Override
	public Expression getValidates ()
	{
		return m_mixinNonKernelFunctions.getValidates ();
	}
			
	@Override
	public String getTestNonautotuneExeParams ()
	{
		return m_mixinNonKernelFunctions.getTestNonautotuneExeParams ();
	}
	
	@Override
	public String getAutotuner ()
	{
		return m_mixinNonKernelFunctions.getAutotuner ();
	}
	
	@Override
	public String getExeParams ()
	{
		return m_mixinNonKernelFunctions.getExeParams ();
	}
}
