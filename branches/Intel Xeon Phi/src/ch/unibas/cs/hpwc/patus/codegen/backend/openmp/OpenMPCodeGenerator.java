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
package ch.unibas.cs.hpwc.patus.codegen.backend.openmp;

import java.util.ArrayList;
import java.util.List;

import cetus.hir.AssignmentExpression;
import cetus.hir.AssignmentOperator;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.FunctionCall;
import cetus.hir.IDExpression;
import cetus.hir.Identifier;
import cetus.hir.Initializer;
import cetus.hir.IntegerLiteral;
import cetus.hir.Literal;
import cetus.hir.NameID;
import cetus.hir.PointerSpecifier;
import cetus.hir.SizeofExpression;
import cetus.hir.Specifier;
import cetus.hir.Traversable;
import cetus.hir.Typecast;
import cetus.hir.UnaryExpression;
import cetus.hir.UnaryOperator;
import cetus.hir.ValueInitializer;
import cetus.hir.VariableDeclaration;
import cetus.hir.VariableDeclarator;
import ch.unibas.cs.hpwc.patus.ast.StatementListBundle;
import ch.unibas.cs.hpwc.patus.ast.SubdomainIdentifier;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;
import ch.unibas.cs.hpwc.patus.codegen.MemoryObject;
import ch.unibas.cs.hpwc.patus.codegen.MemoryObjectManager;
import ch.unibas.cs.hpwc.patus.codegen.backend.AbstractBackend;
import ch.unibas.cs.hpwc.patus.codegen.backend.IIndexing;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;
import ch.unibas.cs.hpwc.patus.representation.StencilNode;
import ch.unibas.cs.hpwc.patus.util.CodeGeneratorUtil;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 *
 * @author Matthias-M. Christen
 */
public class OpenMPCodeGenerator extends AbstractBackend
{
	///////////////////////////////////////////////////////////////////
	// Constants

	private static final boolean USE_STRUCT_INITIALIZER = true;

	private static final IIndexing.IIndexingLevel INDEXING_LEVEL = new IIndexingLevel ()
	{
		@Override
		public int getDimensionality ()
		{
			return 1;
		}

		@Override
		public boolean isVariable ()
		{
			return false;
		}

		@Override
		public Expression getIndexForDimension (int nDimension)
		{
			return new FunctionCall (new NameID ("omp_get_thread_num"));
		}

		@Override
		public Expression getSizeForDimension (int nDimension)
		{
			return new FunctionCall (new NameID ("omp_get_num_threads"));
		}

		@Override
		public int getDefaultBlockSize(int nDimension)
		{
			// not needed
			return 0;
		};
	};


	///////////////////////////////////////////////////////////////////
	// Member Variables

	private int m_nConstSuffix = 0;


	///////////////////////////////////////////////////////////////////
	// Implementation

	public OpenMPCodeGenerator (CodeGeneratorSharedObjects data)
	{
		super (data);
	}


	///////////////////////////////////////////////////////////////////
	// IArithmetic Implementation

	@Override
	public Expression shuffle (Expression expr1, Expression expr2, Specifier specDatatype, int nOffset)
	{
		int nSIMDVectorLength = m_data.getArchitectureDescription ().getSIMDVectorLength (specDatatype);

		switch (nSIMDVectorLength)
		{
		case 2:
			switch (nOffset)
			{
			case 0:
				return expr1.clone ();

			case 1:
				return new FunctionCall (
					new NameID ("_mm_shuffle_pd"),
					CodeGeneratorUtil.expressions (expr1.clone (), expr2.clone (), new IntegerLiteral (1)));

			default:
				throw new RuntimeException ("offset must be 0 or 1.");
			}

		case 4:
			switch (nOffset)
			{
			case 0:
				return expr1.clone ();

			case 1:
				//
				//  +---------+---------+
				//  | 0 1 2 3 | 4 5 6 7 |
				//  +---------+---------+
				//    0 1 2 3   0 1 2 3
				//
				//     \      \   /
				//     |       \ /
				//     |
				//     |      3 3 0 0    <==
				//     |    +---------+
				//     |    | 3 3 4 4 |
				//     |    +---------+
				//     |      0 1 2 3
				//     \
				//      +----\/
				//
				//         1 2 0 2   <==
				//       +---------+
				//       | 1 2 3 4 |
				//       +---------+

				return new FunctionCall (
					new NameID ("_mm_shuffle_ps"),
					CodeGeneratorUtil.expressions (
						expr1.clone (),
						new FunctionCall (
							new NameID ("_mm_shuffle_ps"),
							CodeGeneratorUtil.expressions (expr1.clone (), expr2.clone (), new IntegerLiteral (0x0f) /* _MM_SHUFFLE (0, 0, 3, 3) */)),
						new IntegerLiteral (0x89) /* _MM_SHUFFLE (2, 0, 2, 1) */
					)
				);

			case 2:
				//
				//  +---------+---------+
				//  | 0 1 2 3 | 4 5 6 7 |
				//  +---------+---------+
				//    0 1 2 3   0 1 2 3
				//
				//          \   /
				//           \ /
				//
				//         2 3 0 1   <==
				//       +---------+
				//       | 2 3 4 5 |
				//       +---------+
				//

				return new FunctionCall (
					new NameID ("_mm_shuffle_ps"),
					CodeGeneratorUtil.expressions (expr1.clone (), expr2.clone (), new IntegerLiteral (0x4e) /* _MM_SHUFFLE (1, 0, 3, 2) */));

			case 3:
				//
				//    0 1 2 3   0 1 2 3
				//  +---------+---------+
				//  | 0 1 2 3 | 4 5 6 7 |
				//  +---------+---------+
				//
				//       \   /       /
				//        \ /        |
				//                   |
				//       3 3 0 0 <== |
				//     +---------+   |
				//     | 3 3 4 4 |   |
				//     +---------+   |
				//       0 1 2 3     |
				//                   /
				//             \/----+
				//
				//         0 2 1 2  <==
				//       +---------+
				//       | 3 4 5 6 |
				//       +---------+

				return new FunctionCall (
					new NameID ("_mm_shuffle_ps"),
					CodeGeneratorUtil.expressions (
						new FunctionCall (
							new NameID ("_mm_shuffle_ps"),
							CodeGeneratorUtil.expressions (expr1.clone (), expr2.clone (), new IntegerLiteral (0x0f) /* _MM_SHUFFLE (0, 0, 3, 3) */)),
						expr2.clone (),
						new IntegerLiteral (0x98) /* _MM_SHUFFLE (2, 1, 2, 0) */
					)
				);

			default:
				throw new RuntimeException ("offset must be within the range 0..3");
			}

		default:
			throw new RuntimeException (StringUtil.concat ("shuffle has not been implemented for SIMD vector length ", nSIMDVectorLength));
		}
	}

	@Override
	public Expression unary_minus (Expression expr, Specifier specDatatype, boolean bVectorize)
	{
		/*
		if (bVectorize)
		{
			if (Specifier.FLOAT.equals (specDatatype))
				return subtract (new FunctionCall (new NameID ("_mm_setzero_ps")), expr.clone (), specDatatype, true);
			else if (Specifier.DOUBLE.equals (specDatatype))
				return subtract (new FunctionCall (new NameID ("_mm_setzero_pd")), expr.clone (), specDatatype, true);
		}*/

		// return default implementation
		return super.unary_minus (expr, specDatatype, bVectorize);
	}
	
	@SuppressWarnings("static-method")
	protected String getVecLoadFunctionName (Specifier specDatatype)
	{
		String strFunction = null;
		if (Specifier.FLOAT.equals (specDatatype))
			strFunction = "_mm_load1_ps";
		else if (Specifier.DOUBLE.equals (specDatatype))
			strFunction = "_mm_load1_pd";
		
		return strFunction;
	}
	
	@SuppressWarnings("static-method")
	protected boolean hasVecLoadFunctionPointerArg ()
	{
		return true;
	}
	
	@SuppressWarnings("static-method")
	protected Initializer createExpressionInitializer (Expression expr, int nSIMDVectorLength)
	{
		List<Expression> listValues = new ArrayList<> (nSIMDVectorLength);
		for (int i = 0; i < nSIMDVectorLength; i++)
			listValues.add (expr.clone ());
		return new Initializer (listValues);		
	}
	
	protected Expression createLoadInitializer (Specifier specDatatype, Expression expr)
	{
		String strFunction = getVecLoadFunctionName (specDatatype);
		if (strFunction == null)
			throw new RuntimeException ("Unknown data type");

		Expression exprArg = expr.clone ();
		if (hasVecLoadFunctionPointerArg ())
		{
			if (!(exprArg instanceof IDExpression))
			{
				// create a temporary variable
				exprArg = new Identifier (createTemporary (specDatatype, exprArg));
			}
			
			exprArg = new UnaryExpression (UnaryOperator.ADDRESS_OF, exprArg);
		}
		
		return new FunctionCall (new NameID (strFunction), CodeGeneratorUtil.expressions (exprArg));
	}
	
	private VariableDeclarator createTemporary (Specifier specDatatype, Expression expr)
	{
		VariableDeclarator decl = new VariableDeclarator (CodeGeneratorUtil.createNameID ("const", m_nConstSuffix++));
		m_data.getData ().addDeclaration (new VariableDeclaration (specDatatype, decl));
		decl.setInitializer (new ValueInitializer (expr.clone ()));
		
		return decl;
	}

	@Override
	public Traversable splat (Expression expr, Specifier specDatatype)
	{
		int nSIMDVectorLength = m_data.getArchitectureDescription ().getSIMDVectorLength (specDatatype);
		if (nSIMDVectorLength == 1)
			return expr;

		if (expr instanceof Literal)
		{
			if (USE_STRUCT_INITIALIZER)
				return createExpressionInitializer (expr, nSIMDVectorLength);
			
			return createLoadInitializer (specDatatype, expr);
		}
		else if (expr instanceof IDExpression)
		{
			// initialize stencil parameter constants as we would initialize literals
			if (m_data.getStencilCalculation ().isArgument (((IDExpression) expr).getName ()) && USE_STRUCT_INITIALIZER)
				return createExpressionInitializer (expr, nSIMDVectorLength);
			
			// _mm_load1_p{s|d} (*p)
			return createLoadInitializer (specDatatype, expr);
		}
		else
		{
			VariableDeclarator decl = createTemporary (specDatatype, expr);			
			return USE_STRUCT_INITIALIZER ?
				createExpressionInitializer (new Identifier (decl), nSIMDVectorLength) :
				createLoadInitializer (specDatatype, new Identifier (decl));
		}
	}
	

	///////////////////////////////////////////////////////////////////
	// IIndexing Implementation

	@Override
	public int getIndexingLevelsCount ()
	{
		return 1;
	}

	@Override
	public IIndexingLevel getIndexingLevel (int nIndexingLevel)
	{
		return OpenMPCodeGenerator.INDEXING_LEVEL;
	}

	@Override
	public EThreading getThreading ()
	{
		return IIndexing.EThreading.MULTI;
	}


	///////////////////////////////////////////////////////////////////
	// IDataTransfer Implementation

	@Override
	public void allocateData (
		StencilNode node, MemoryObject mo, Expression exprMemoryObject, int nParallelismLevel, StatementListBundle slbCode)
	{
		slbCode.addStatement (new ExpressionStatement (
			new AssignmentExpression (
				exprMemoryObject.clone (),
				AssignmentOperator.NORMAL,
				new Typecast (
					CodeGeneratorUtil.specifiers (mo.getDatatype (), PointerSpecifier.UNQUALIFIED),
					new FunctionCall (
						new NameID ("malloc"),
						CodeGeneratorUtil.expressions (new BinaryExpression (
							mo.getSize ().getVolume (),
							BinaryOperator.MULTIPLY,
							new SizeofExpression (CodeGeneratorUtil.specifiers (mo.getDatatype ()))))
					)
				)
			)
		));
	}

	@Override
	public void doAllocateData (int nParallelismLevel, StatementListBundle slbCode, CodeGeneratorRuntimeOptions options)
	{
	}

	@Override
	public void loadData (StencilNode node, SubdomainIdentifier sdidSourceIterator,
		MemoryObject moDestination, MemoryObject moSource, int nParallelismLevel,
		StatementListBundle slbCode, CodeGeneratorRuntimeOptions options)
	{
		MemoryObjectManager mgr = m_data.getData ().getMemoryObjectManager ();

		// use intrinsics to preload data
		slbCode.addStatement (new ExpressionStatement (
			new FunctionCall (
				new Typecast (
					CodeGeneratorUtil.specifiers (Specifier.CHAR, PointerSpecifier.UNQUALIFIED),
					mgr.getMemoryObjectExpression (sdidSourceIterator, node, null, true, true, true, slbCode, options)
				),
				CodeGeneratorUtil.expressions (new NameID ("_MM_HINT_T0"))
			)
		));
	}

	@Override
	public void doLoadData (int nParallelismLevel, StatementListBundle slbCode, CodeGeneratorRuntimeOptions options)
	{
	}

	@Override
	public void storeData (StencilNode node, SubdomainIdentifier sdidSourceIterator,
		MemoryObject moDestination, MemoryObject moSource, int nParallelismLevel,
		StatementListBundle slbCode, CodeGeneratorRuntimeOptions options)
	{
		// no need to store data
		/*
		slbCode.addStatement (new ExpressionStatement (
			new AssignmentExpression (
				exprDestinationMemoryObject.clone (),
				AssignmentOperator.NORMAL,
				new FunctionCall (
					new NameID ("__pseudo_store__"),
					CodeGeneratorUtil.expressions (exprSourceMemoryObject.clone ())))
			));
		*/
	}

	@Override
	public void doStoreData (int nParallelismLevel, StatementListBundle slbCode, CodeGeneratorRuntimeOptions options)
	{
	}

	@Override
	public void waitFor (StencilNode node, MemoryObject mo, int nParallelismLevel, StatementListBundle slbCode)
	{
		// no data synchronization needed
	}

	@Override
	public void doWaitFor (int nParallelismLevel, StatementListBundle slbCode, CodeGeneratorRuntimeOptions options)
	{
	}


	///////////////////////////////////////////////////////////////////
	// INonKernelFunctions Implementation

	// use default implementation
}
