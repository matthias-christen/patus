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

import cetus.hir.DepthFirstIterator;
import cetus.hir.Expression;
import cetus.hir.FloatLiteral;
import cetus.hir.Initializer;
import cetus.hir.Specifier;
import cetus.hir.Traversable;
import ch.unibas.cs.hpwc.patus.ast.StatementListBundle;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;

/**
 *
 * @author Matthias-M. Christen
 */
public class SIMDScalarGeneratedIdentifiers implements IConstantExpressionCalculator
{
	///////////////////////////////////////////////////////////////////
	// Constants

	public final static String SPLATNAME_SCALAR = "splat";
	public final static String SPLATNAME_ARRAY = "constarr";

	
	///////////////////////////////////////////////////////////////////
	// Member Variables

	private CodeGeneratorSharedObjects m_data;


	///////////////////////////////////////////////////////////////////
	// Implementation

	public SIMDScalarGeneratedIdentifiers (CodeGeneratorSharedObjects data)
	{
		m_data = data;
	}

	/**
	 * Creates a SIMD substitution expression for the scalar expression
	 * <code>exprScalar</code>.
	 * 
	 * @param exprScalar
	 *            The scalar to SIMDize
	 * @param specDatatype
	 *            The data type of the expression
	 * @param slbGeneratedCode
	 *            The statement list bundle to which additionally generated code
	 *            is added
	 * @param options
	 *            Code generation options
	 * @return The SIMD expression replacing the scalar expression
	 *         <code>exprScalar</code>
	 */
	public Expression createVectorizedScalar (
		Expression exprScalar, Specifier specDatatype, StatementListBundle slbGeneratedCode, CodeGeneratorRuntimeOptions options)
	{
		return createVectorizedScalars (new Expression[] { exprScalar }, specDatatype, slbGeneratedCode, options);
	}
	
	/**
	 * Creates a SIMD substitution expression for the array of scalar expression
	 * <code>rgScalars</code>.
	 * 
	 * @param rgScalars
	 *            The array of scalars to SIMDize
	 * @param specDatatype
	 *            The data type of the expression
	 * @param slbGeneratedCode
	 *            The statement list bundle to which additionally generated code
	 *            is added
	 * @param options
	 *            Code generation options
	 * @return The SIMD expression replacing the scalar expression
	 *         <code>rgScalars</code>
	 */
	public Expression createVectorizedScalars (
		 Expression[] rgScalars, Specifier specDatatype, StatementListBundle slbGeneratedCode, CodeGeneratorRuntimeOptions options)
	{
		// handle degenerate cases
		if (rgScalars.length == 0)
			return null;
		
		Expression[] rgScalarsNormalized = new Expression[rgScalars.length];
		for (int i = 0; i < rgScalars.length; i++)
			rgScalarsNormalized[i] = SIMDScalarGeneratedIdentifiers.replaceScalar (rgScalars[i], specDatatype);
		
		return m_data.getCodeGenerators ().getConstantGeneratedIdentifiers ().getConstantIdentifier (
			rgScalarsNormalized,
			rgScalars.length == 1 ? SPLATNAME_SCALAR : SPLATNAME_ARRAY,
			specDatatype,
			slbGeneratedCode,
			this,
			options
		);
	}

	/**
	 * Replace literals with the literal with the suffix corresponding to the
	 * datatype <code>specDatatype</code>.
	 * 
	 * @param exprScalar
	 *            An expression in which any literals will be replaced by
	 *            literals with the suffix corresponding to the datatype
	 * @param specDatatype
	 *            The datatype
	 * @return An expression equivalent to <code>exprScalar</code>, but with
	 *         literals replaced to reflect the datatype
	 *         <code>specDatatype</code>
	 */
	private static Expression replaceScalar (Expression exprScalar, Specifier specDatatype)
	{
		if (exprScalar instanceof FloatLiteral)
			return SIMDScalarGeneratedIdentifiers.replaceFloatLiteral ((FloatLiteral) exprScalar, specDatatype);

		Expression exprScalarNew = exprScalar.clone ();
		for (DepthFirstIterator it = new DepthFirstIterator (exprScalarNew); it.hasNext (); )
		{
			Object obj = it.next ();
			if (obj instanceof FloatLiteral)
			{
				FloatLiteral litNew = replaceFloatLiteral ((FloatLiteral) obj, specDatatype);
				if (litNew != obj)
					((Expression) obj).swapWith (litNew);
			}
		}
		return exprScalarNew;
	}

	private static FloatLiteral replaceFloatLiteral (FloatLiteral literal, Specifier specDatatype)
	{
		if (specDatatype.equals (Specifier.FLOAT))
			return new FloatLiteral (literal.getValue (), "f");
		return literal;
	}
	
	@Override
	public Traversable calculateConstantExpression (Expression exprScalar, Specifier specDatatype, boolean bVectorize)
	{
		// nothing to do if no SIMD
		if (m_data.getArchitectureDescription ().getSIMDVectorLength (specDatatype) == 1 || !bVectorize)
			return exprScalar;

		// get the splatted value from the backend generator
		// we either expect an expression or an initializer
		Traversable trvSplat = m_data.getCodeGenerators ().getBackendCodeGenerator ().splat (exprScalar, specDatatype);

		if (trvSplat != exprScalar && trvSplat != null)
		{
			// only do something if the backend code generator did something (trvSplat != exprScalar)
			// and the operation is supported by the backend generator

			// check that the return type is an expression or an initializer
			if (!(trvSplat instanceof Initializer) && !(trvSplat instanceof Expression))
				throw new RuntimeException ("splat: unrecognized return type");

			// create an intermediate identifier if the scalar is a literal or a initializer has been returned
			if ((trvSplat instanceof Expression) || (trvSplat instanceof Initializer))
				return trvSplat;
		}

		return null;
	}
}
