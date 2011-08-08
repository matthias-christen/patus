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

import java.util.HashMap;
import java.util.Map;

import cetus.hir.AssignmentExpression;
import cetus.hir.AssignmentOperator;
import cetus.hir.DeclarationStatement;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.IDExpression;
import cetus.hir.Identifier;
import cetus.hir.Initializer;
import cetus.hir.Literal;
import cetus.hir.NameID;
import cetus.hir.Specifier;
import cetus.hir.Traversable;
import cetus.hir.ValueInitializer;
import cetus.hir.VariableDeclaration;
import cetus.hir.VariableDeclarator;
import ch.unibas.cs.hpwc.patus.ast.StatementListBundle;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 *
 * @author Matthias-M. Christen
 */
public class ConstantGeneratedIdentifiers
{
	///////////////////////////////////////////////////////////////////
	// Member Variables

	private CodeGeneratorSharedObjects m_data;

	/**
	 *
	 */
	private Map<String, Integer> m_mapTempIdentifierSuffices;

	/**
	 *
	 */
	private Map<Integer, Map<Expression, Expression>> m_mapConstantCache;


	///////////////////////////////////////////////////////////////////
	// Implementation

	/**
	 * Creates the code generator.
	 * @param data
	 */
	public ConstantGeneratedIdentifiers (CodeGeneratorSharedObjects data)
	{
		m_data = data;

		m_mapTempIdentifierSuffices = new HashMap<String, Integer> ();
		m_mapConstantCache = new HashMap<Integer, Map<Expression, Expression>> ();
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
	 *
	 * @param exprConstant
	 * @param strIdentifierName
	 * @param specDatatype
	 * @param slbGeneratedCode The statement list bundle to which the code is added.
	 * 	Can be <code>null</code>, in which case the code always will be added to the
	 * 	function's declaration section
	 * @param calc An object that calculates an expression for <code>exprConstant</code>
	 * 	that is taken to be the value that is replaces <code>exprConstant</code> in
	 * 	future calls of this method. If <code>null</code>, no substitution for
	 * 	<code>exprConstant</code> is done, i.e. the expression itself is saved in an
	 * 	identifier
	 * @return
	 */
	public Expression getConstantIdentifier (
		Expression exprConstant, String strIdentifierName, Specifier specDatatype, StatementListBundle slbGeneratedCode,
		IConstantExpressionCalculator calc, CodeGeneratorRuntimeOptions options)
	{
		boolean bVectorize =
			// always vectorize if native SIMD datatypes are used
			m_data.getOptions ().useNativeSIMDDatatypes () ||
			// vectorize if OPTION_NOVECTORIZE==false and SIMD is used
			(!options.getBooleanValue (CodeGeneratorRuntimeOptions.OPTION_NOVECTORIZE, false) && m_data.getArchitectureDescription ().useSIMD ());

		int nSIMDVectorLength =	bVectorize ? m_data.getArchitectureDescription ().getSIMDVectorLength (specDatatype) : 1;

		Map<Expression, Expression> mapConstantCache = m_mapConstantCache.get (nSIMDVectorLength);
		if (mapConstantCache == null)
			m_mapConstantCache.put (nSIMDVectorLength, mapConstantCache = new HashMap<Expression, Expression> ());

		// try to retrieve the expression from the cache
		Expression exprResult = mapConstantCache.get (exprConstant);

		// return it if it has been found
		if (exprResult != null)
			return exprResult.clone ();

		// no expression found, calculate it and add it to the cache
		// create the variable declaration
		Traversable trvConstantCalculation = null;
		if (calc != null)
		{
			trvConstantCalculation = calc.calculateConstantExpression (exprConstant, specDatatype, bVectorize);
			if (trvConstantCalculation == null || exprConstant.equals (trvConstantCalculation))
				return exprConstant;
		}
		else
		{
			trvConstantCalculation = exprConstant;
			if (exprConstant instanceof Literal || exprConstant instanceof IDExpression)
				return exprConstant;
		}

		VariableDeclarator decl = new VariableDeclarator (new NameID (StringUtil.concat (strIdentifierName, getTempIdentifierSuffix (strIdentifierName))));
		VariableDeclaration declaration = new VariableDeclaration (m_data.getArchitectureDescription ().getType (specDatatype), decl);
		m_data.getData ().addDeclaration (declaration);

		exprResult = new Identifier (decl);

		// determine where to declare the variable (local or global)
		if (exprConstant instanceof Literal || (exprConstant instanceof NameID && m_data.getStencilCalculation ().isArgument (((NameID) exprConstant).getName ())) || slbGeneratedCode == null)
			decl.setInitializer (trvConstantCalculation instanceof Initializer ? (Initializer) trvConstantCalculation : new ValueInitializer ((Expression) trvConstantCalculation));
		else
		{
			Expression exprValue = null;
			if (trvConstantCalculation instanceof Expression)
				exprValue = (Expression) trvConstantCalculation;
			else if (trvConstantCalculation instanceof Initializer)
				exprValue = (Expression) ((Initializer) trvConstantCalculation).getChildren ().get (0);
			else
				throw new RuntimeException ("Unknown initializer");

			slbGeneratedCode.addStatement (new ExpressionStatement (new AssignmentExpression (exprResult.clone (), AssignmentOperator.NORMAL, exprValue)));
		}

		mapConstantCache.put (exprConstant, exprResult);
		return exprResult;
	}
}
