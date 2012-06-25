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
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import cetus.hir.ArrayAccess;
import cetus.hir.ArraySpecifier;
import cetus.hir.AssignmentExpression;
import cetus.hir.AssignmentOperator;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.FunctionCall;
import cetus.hir.IDExpression;
import cetus.hir.Identifier;
import cetus.hir.Initializer;
import cetus.hir.IntegerLiteral;
import cetus.hir.Literal;
import cetus.hir.NameID;
import cetus.hir.Specifier;
import cetus.hir.Symbol;
import cetus.hir.Traversable;
import cetus.hir.ValueInitializer;
import cetus.hir.VariableDeclaration;
import cetus.hir.VariableDeclarator;
import ch.unibas.cs.hpwc.patus.ast.StatementListBundle;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;
import ch.unibas.cs.hpwc.patus.util.ExpressionUtil;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 *
 * @author Matthias-M. Christen
 */
public class ConstantGeneratedIdentifiers
{
	///////////////////////////////////////////////////////////////////
	// Inner Types
	
	private static class ExpressionArray
	{
		private Expression[] m_rgExpressions;
		
		public ExpressionArray (Expression[] rgExpressions)
		{
			m_rgExpressions = rgExpressions;
		}

		@Override
		public int hashCode ()
		{
			return Arrays.hashCode (m_rgExpressions);
		}

		@Override
		public boolean equals (Object obj)
		{
			if (this == obj)
				return true;
			if (obj == null)
				return false;
			if (getClass () != obj.getClass ())
				return false;
			ExpressionArray other = (ExpressionArray) obj;
			if (!Arrays.equals (m_rgExpressions, other.m_rgExpressions))
				return false;
			return true;
		}		
	}

	
	///////////////////////////////////////////////////////////////////
	// Member Variables

	private CodeGeneratorSharedObjects m_data;

	/**
	 * The map of current integer suffices per identifier name
	 */
	private Map<String, Integer> m_mapTempIdentifierSuffices;

	/**
	 * The map caching the generated constant identifiers
	 */
	private Map<Integer, Map<String, Map<ExpressionArray, Expression>>> m_mapConstantCache;


	///////////////////////////////////////////////////////////////////
	// Implementation

	/**
	 * Creates the code generator.
	 * @param data
	 */
	public ConstantGeneratedIdentifiers (CodeGeneratorSharedObjects data)
	{
		m_data = data;

		m_mapTempIdentifierSuffices = new HashMap<> ();
		m_mapConstantCache = new HashMap<> ();
	}

	private int getTempIdentifierSuffix (String strIdentifier)
	{
		Integer nSuffix = m_mapTempIdentifierSuffices.get (strIdentifier);
		if (nSuffix == null)
			nSuffix = 0;

		m_mapTempIdentifierSuffices.put (strIdentifier, nSuffix + 1);
		return nSuffix;
	}
	
	/**
	 * Create a new declarator for an identifier named
	 * <code>strIdentifierName</code> of type <code>specDatatype</code>.
	 * 
	 * @param strIdentifierName
	 *            The name of the identifier to create. The name will be given a
	 *            numerical suffix to avoid name collisions.
	 * @param specDatatype
	 *            The datatype of the new identifier
	 * @param bIsArray
	 *            Specifies whether the identifier to create is an array
	 * @return A variable declarator for a new identifier named
	 *         <code>strIdentifierName</code>
	 */
	public VariableDeclarator createDeclarator (String strIdentifierName, Specifier specDatatype, boolean bIsArray, Integer nArrayLength)
	{
		NameID nid = new NameID (StringUtil.concat (strIdentifierName, getTempIdentifierSuffix (strIdentifierName)));
		
		VariableDeclarator decl = null;
		if (bIsArray)
		{
			ArraySpecifier spec = null;
			if (nArrayLength == null)
				spec = new ArraySpecifier ();
			else
				spec = new ArraySpecifier (new IntegerLiteral (nArrayLength));
			
			decl = new VariableDeclarator (nid, spec);
		}
		else
			decl = new VariableDeclarator (nid);
			
		VariableDeclaration declaration = new VariableDeclaration (m_data.getArchitectureDescription ().getType (specDatatype), decl);
		m_data.getData ().addDeclaration (declaration);
		
		return decl;
	}
	
	/**
	 * Determines whether a vectorized constant should be created.
	 * 
	 * @param options
	 *            The runtime code generation options
	 * @return <code>true</code> iff a vectorized constant is to be created
	 */
	private boolean isVectorizationCreated (CodeGeneratorRuntimeOptions options)
	{
		// always vectorize if native SIMD datatypes are used
		if (m_data.getOptions ().useNativeSIMDDatatypes ())
			return true;
		
		// vectorize if OPTION_NOVECTORIZE==false and SIMD is used
		return !options.getBooleanValue (CodeGeneratorRuntimeOptions.OPTION_NOVECTORIZE, false) && m_data.getArchitectureDescription ().useSIMD ();
	}
	
	/**
	 * Returns the length of a SIMD vector of type <code>specDatatype</code> or
	 * <code>1</code> if the constant is not to be vectorized.
	 * 
	 * @param specDatatype
	 *            The datatype of the constant
	 * @param options
	 *            The runtime code generation options
	 * @return The length of a SIMD vector for the type
	 *         <code>specDatatype</code>
	 */
	private int getSIMDVectorLength (Specifier specDatatype, CodeGeneratorRuntimeOptions options)
	{
		return isVectorizationCreated (options) ?
			m_data.getArchitectureDescription ().getSIMDVectorLength (specDatatype) :
			1;
	}
	
	private Traversable[] createConstantCalculationsArray (
		IConstantExpressionCalculator calc, Expression[] rgConstants, Specifier specDatatype, CodeGeneratorRuntimeOptions options)
	{
		if (calc != null)
			return createConstantCalculationsArrayWithCalculator (calc, rgConstants, specDatatype, options);
		
		if (rgConstants.length == 1)
		{
			if (rgConstants[0] instanceof Literal || rgConstants[0] instanceof IDExpression)
				return null;
			return new Traversable[] { rgConstants[0] };
		}

		///
		Traversable[] rgResult = new Traversable[rgConstants.length];
		for (int i = 0; i < rgConstants.length; i++)
			rgResult[i] = rgConstants[i];
		return rgResult;
		//throw new RuntimeException ("If no constant calculator is provided, getConstantIdentifier only works for size-1 constant arrays.");
	}

	private Traversable[] createConstantCalculationsArrayWithCalculator (
		IConstantExpressionCalculator calc, Expression[] rgConstants, Specifier specDatatype, CodeGeneratorRuntimeOptions options)
	{
		final boolean bVectorize = isVectorizationCreated (options);
		Traversable[] rgConstantCalculations = null;
		
		for (int i = 0; i < rgConstants.length; i++)
		{
			Traversable trvConstantCalculation = calc.calculateConstantExpression (rgConstants[i], specDatatype, bVectorize);
			if (trvConstantCalculation == null || (rgConstants.length == 1 && rgConstants[0].equals (trvConstantCalculation)))
				return null;
			
			if (rgConstantCalculations == null)
				rgConstantCalculations = new Traversable[rgConstants.length];
			
			rgConstantCalculations[i] = trvConstantCalculation;
		}
		
		return rgConstantCalculations;
	}
	
	/**
	 * Determines whether the expression <code>expr</code> is a number literal
	 * or a stencil parameter.
	 * 
	 * @param expr
	 *            The expression to test
	 * @return <code>true</code> iff the expression in
	 *         <code>expr<code> is either a number literals or a stencil parameters
	 */
	private boolean isLiteralOrParam (Expression expr)
	{
		// is expr a stencil parameter?
		if ((expr instanceof NameID) && m_data.getStencilCalculation ().isParameter (((NameID) expr).getName ()))
			return true;
		
		// is it a number literal?
		return ExpressionUtil.isNumberLiteral (expr);
	}

	/**
	 * Deterimes whether the expression <code>expr</code> contains an
	 * {@link IDExpression} which isn't a stencil parameter.
	 * 
	 * @param expr
	 *            The expression to check
	 * @return <code>true</code> iff the expression contains an identifier which
	 *         isn't a stencil parameter
	 */
	private boolean containsNonParamIdentifier (Expression expr)
	{
//		for (DepthFirstIterator it = new DepthFirstIterator (expr); it.hasNext (); )
//		{
//			Object obj = it.next ();
//			if (obj instanceof IDExpression)
//				if (!isLiteralOrParam ((IDExpression) obj))
//					return true;
//		}
//		
//		return false;
		
		return containsNonParamIdentifierRecursive (expr);
	}
	
	private boolean containsNonParamIdentifierRecursive (Traversable trvParent)
	{
		if (trvParent instanceof FunctionCall)
		{
			for (Object objArg : ((FunctionCall) trvParent).getArguments ())
				if (containsNonParamIdentifierRecursive ((Traversable) objArg))
					return true;
		}
		else if (trvParent instanceof IDExpression)
		{
			if (!isLiteralOrParam ((IDExpression) trvParent))
			{
				if (trvParent instanceof Identifier)
				{
					Symbol decl = ((Identifier) trvParent).getSymbol ();
					if (decl != null && (decl instanceof VariableDeclarator))
					{
						if (containsNonParamIdentifierRecursive (((VariableDeclarator) decl).getInitializer ()))
							return true;
					}
					else
						return true;
				}
				else
					return true;
			}
		}
		else
		{
			for (Traversable trv : trvParent.getChildren ())
				if (containsNonParamIdentifierRecursive (trv))
					return true;
		}
		
		return false;
	}

	private Initializer createInitializer (IConstantExpressionCalculator calc,
		Traversable[] rgConstantCalculations, VariableDeclarator declGeneratedVar, Specifier specDatatype,
		StatementListBundle slbGeneratedCode, CodeGeneratorRuntimeOptions options)
	{
		// special case if there is only one constant to initialize
		if (rgConstantCalculations.length == 1)
		{
			if (rgConstantCalculations[0] instanceof Initializer)
				return (Initializer) rgConstantCalculations[0];
			else if (rgConstantCalculations[0] instanceof Expression)
			{
//				if (containsNonParamIdentifier ((Expression) rgConstantCalculations[0]))
//					return null;
				return new ValueInitializer ((Expression) rgConstantCalculations[0]);
			}
			else
				throw new RuntimeException ("Unknown initializer");
		}
		
		// more than one constants...
		
		// TODO: nicer: some other way to find out whether to vectorize
		int nSIMDVecLen = calc instanceof SIMDScalarGeneratedIdentifiers ? getSIMDVectorLength (specDatatype, options) : 1;
		
		// create a values array holding the initializer values
//		List<Traversable> listValues = new ArrayList<> (rgConstantCalculations.length * nSIMDVecLen);
		List<Traversable> listValues = new ArrayList<> (rgConstantCalculations.length);
		
		for (int i = 0; i < rgConstantCalculations.length; i++)
		{
			if (rgConstantCalculations[i] instanceof Expression)
			{
				if (isLiteralOrParam ((Expression) rgConstantCalculations[i]))
					listValues.add (rgConstantCalculations[i]);
				else
				{
					// add dummy values to the values array
//					for (int j = 0; j < nSIMDVecLen; j++)
//						listValues.add (ExpressionUtil.createFloatLiteral (0, specDatatype));
					
					///
					List<Expression> listDummyValues = new ArrayList<> (nSIMDVecLen);
					for (int j = 0; j < nSIMDVecLen; j++)
						listDummyValues.add (ExpressionUtil.createFloatLiteral (0, specDatatype));
					listValues.add (new Initializer (listDummyValues));
					///
	
					// initialize the array element in the code
					slbGeneratedCode.addStatement (new ExpressionStatement (new AssignmentExpression (
						new ArrayAccess (declGeneratedVar.getID ().clone (), new IntegerLiteral (i)),
						AssignmentOperator.NORMAL,
						((Expression) rgConstantCalculations[i]).clone ()
					)));
				}
			}
			else if (rgConstantCalculations[i] instanceof Initializer)
			{
				// if the entry is an initializer, extract the values and add them to the values array
//				for (Traversable trvVal : rgConstantCalculations[i].getChildren ())
//					listValues.add ((Expression) trvVal);

				///
				listValues.add (rgConstantCalculations[i]);
				///
			}
			else
				throw new RuntimeException ("Unknown initializer");
		}
		
		// create a new initializer from the values array
		return new Initializer (listValues);
	}
	
	private static void addAssignmentStatement (Traversable trvConstantCalculation, Expression exprGeneratedId, StatementListBundle slbGeneratedCode)
	{
		Expression exprValue = null;
		if (trvConstantCalculation instanceof Expression)
			exprValue = (Expression) trvConstantCalculation;
		else if (trvConstantCalculation instanceof Initializer)
			exprValue = (Expression) ((Initializer) trvConstantCalculation).getChildren ().get (0);
		else
			throw new RuntimeException ("Unknown initializer");

		slbGeneratedCode.addStatement (new ExpressionStatement (
			new AssignmentExpression (exprGeneratedId.clone (), AssignmentOperator.NORMAL, exprValue)));
	}
	
	/**
	 * Creates an identifier for a constant value. The identifier is declared
	 * and the constant value is assigned to the identifier.
	 * 
	 * @param exprConstant
	 *            A constants for which possibly vectorized arrays
	 *            holding this constant is created
	 * @param strIdentifierName
	 *            The name of the new identifier holding the constant value
	 * @param specDatatype
	 *            The datatype of the constant
	 * @param slbGeneratedCode
	 *            The statement list bundle to which the code is added.
	 *            Can be <code>null</code>, in which case the code always will
	 *            be added to the function's declaration section
	 * @param calc
	 *            An object that calculates an expression for
	 *            <code>exprConstant</code> that is taken to be the value that
	 *            is replaces <code>exprConstant</code> in future calls of this
	 *            method. If <code>null</code>, no substitution for
	 *            <code>exprConstant</code> is done, i.e. the expression itself
	 *            is saved in an identifier
	 * @return The new identifier which is to be used in lieu of the constant,
	 *         or the original constant if no new identifier has to be created
	 */
	public Expression getConstantIdentifier (
		Expression exprConstant, String strIdentifierName, Specifier specDatatype, StatementListBundle slbGeneratedCode,
		IConstantExpressionCalculator calc, CodeGeneratorRuntimeOptions options)
	{
		return getConstantIdentifier (new Expression[] { exprConstant }, strIdentifierName, specDatatype, slbGeneratedCode, calc, options);
	}
	
	/**
	 * Creates an identifier for an array of constant values.
	 * The identifier is declared and the constant value is assigned to the identifier.
	 * 
	 * @param rgConstants
	 *            An array of constants for which possibly vectorized arrays
	 *            holding these constants are created
	 * @param strIdentifierName
	 *            The name of the new identifier holding the constant values
	 * @param specDatatype
	 *            The datatype of the constants
	 * @param slbGeneratedCode
	 *            The statement list bundle to which the code is added.
	 *            Can be <code>null</code>, in which case the code always will
	 *            be added to the function's declaration section
	 * @param calc
	 *            An object that calculates an expression for
	 *            <code>exprConstant</code> that is taken to be the value that
	 *            is replaces <code>exprConstant</code> in future calls of this
	 *            method. If <code>null</code>, no substitution for
	 *            <code>exprConstant</code> is done, i.e. the expression itself
	 *            is saved in an identifier
	 * @return The new identifier which is to be used in lieu of the constant,
	 *         or the original constant if no new identifier has to be created
	 */
	public Expression getConstantIdentifier (
		Expression[] rgConstants, String strIdentifierName, Specifier specDatatype, StatementListBundle slbGeneratedCode,
		IConstantExpressionCalculator calc, CodeGeneratorRuntimeOptions options)
	{
		int nSIMDVectorLength =	getSIMDVectorLength (specDatatype, options);

		// try to retrieve the expression from the cache
		Map<String, Map<ExpressionArray, Expression>> mapConstantCache = m_mapConstantCache.get (nSIMDVectorLength);
		if (mapConstantCache == null)
			m_mapConstantCache.put (nSIMDVectorLength, mapConstantCache = new HashMap<> ());
		Map<ExpressionArray, Expression> mapExpressions = mapConstantCache.get (strIdentifierName);
		if (mapExpressions == null)
			mapConstantCache.put (strIdentifierName, mapExpressions = new HashMap<> ());

		ExpressionArray eaConstants = new ExpressionArray (rgConstants);
		Expression exprResult = mapExpressions.get (eaConstants);

		// return it if it has been found
		if (exprResult != null)
			return exprResult.clone ();

		// no expression found, calculate it and add it to the cache
		
		// create the expressions calculating the constants
		Traversable[] rgConstantCalculations = createConstantCalculationsArray (calc, rgConstants, specDatatype, options);
		if (rgConstantCalculations == null)
			return rgConstants[0];

		// create the variable declaration
		VariableDeclarator decl = createDeclarator (strIdentifierName, specDatatype, rgConstants.length > 1, null);
		exprResult = new Identifier (decl);

		// initialize the newly created identifier if possible
		Initializer initializer = createInitializer (calc, rgConstantCalculations, decl, specDatatype, slbGeneratedCode, options);
		if (initializer != null)
			decl.setInitializer (initializer);
		else
		{
			for (Traversable trvConstantCalc : rgConstantCalculations)
				ConstantGeneratedIdentifiers.addAssignmentStatement (trvConstantCalc, exprResult, slbGeneratedCode);
		}
		
		mapExpressions.put (eaConstants, exprResult);
		return exprResult;
	}
	
	/**
	 * Removes all the identifiers named <code>strIdentifierName</code> from the
	 * cache.
	 * 
	 * @param strIdentifierName
	 *            The name of the identifier to remove
	 */
	public void clearIdentifier (String strIdentifierName)
	{
		for (int nSIMDVecLen : m_mapConstantCache.keySet ())
		{
			Map<String, Map<ExpressionArray, Expression>> map = m_mapConstantCache.get (nSIMDVecLen);
			map.remove (strIdentifierName);
		}
	}

	public void reset ()
	{
		m_mapConstantCache.clear ();
		m_mapTempIdentifierSuffices.clear ();
	}
}
