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
	// Member Variables

	private CodeGeneratorSharedObjects m_data;


	///////////////////////////////////////////////////////////////////
	// Implementation

	public SIMDScalarGeneratedIdentifiers (CodeGeneratorSharedObjects data)
	{
		m_data = data;
	}

	/**
	 * Creates an SIMD substitution expression for the scalar expression <code>exprScalar</code>.
	 * @param exprScalar
	 * @param specDatatype
	 * @return The SIMD expression replacing the scalar expression <code>exprScalar</code>
	 */
	public Expression createVectorizedScalar (Expression exprScalar, Specifier specDatatype, StatementListBundle slbGeneratedCode, CodeGeneratorRuntimeOptions options)
	{
		exprScalar = replaceScalar (exprScalar, specDatatype);
		return m_data.getCodeGenerators ().getConstantGeneratedIdentifiers ().getConstantIdentifier (
			exprScalar, "splat", specDatatype, slbGeneratedCode, this, options);
	}

	/**
	 * Replace literals with the literal with the suffix corresponding to the datatype.
	 * @param exprScalar
	 * @param specDatatype
	 * @return
	 */
	private Expression replaceScalar (Expression exprScalar, Specifier specDatatype)
	{
		if (exprScalar instanceof FloatLiteral)
			return replaceFloatLiteral ((FloatLiteral) exprScalar, specDatatype);

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

	private FloatLiteral replaceFloatLiteral (FloatLiteral literal, Specifier specDatatype)
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
