/*******************************************************************************
 * Copyright (c) 2011 Matthias-M. Christen, University of Basel, Switzerland.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the GNU Lesser Public License v2.1
 * which accompanies this distribution, and is available at
 * http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
 * 
 * Contributors:
 * Matthias-M. Christen, University of Basel, Switzerland - initial API and
 * implementation
 ******************************************************************************/
package ch.unibas.cs.hpwc.patus.codegen.backend;

import cetus.hir.Expression;
import cetus.hir.Specifier;
import cetus.hir.Traversable;

/**
 * Interface for arithmetic expressions.
 * 
 * @author Matthias-M. Christen
 */
public interface IArithmetic
{
	/**
	 * 
	 * @param exprIn
	 * @param specDatatype
	 * @return
	 */
	public abstract Expression createExpression (Expression exprIn,	Specifier specDatatype, boolean bVectorize);

	/**
	 * Unary plus.
	 * 
	 * @param expr
	 *            The expression to which to apply the operator
	 * @param specDatatype
	 *            The basic datatype of the expression (<code>float</code>,
	 *            <code>double</code>, ...)
	 * @return The expression after applying the operator to <code>expr</code>
	 */
	public abstract Expression unary_plus (Expression expr,	Specifier specDatatype, boolean bVectorize);

	/**
	 * Unary minus.
	 * 
	 * @param expr
	 *            The expression to which to apply the operator
	 * @param specDatatype
	 *            The basic datatype of the expression (<code>float</code>,
	 *            <code>double</code>, ...)
	 * @return The expression after applying the operator to <code>expr</code>
	 */
	public abstract Expression unary_minus (Expression expr, Specifier specDatatype, boolean bVectorize);

	/**
	 * Adds <code>expr1</code> and <code>expr2</code>.
	 * 
	 * @param expr1
	 *            The first summand
	 * @param expr2
	 *            The second summand
	 * @param specDatatype
	 *            The basic datatype of the expression (<code>float</code>,
	 *            <code>double</code>, ...)
	 * @return An expression for <code>expr1</code> + <code>expr2</code>
	 */
	public abstract Expression plus (Expression expr1, Expression expr2, Specifier specDatatype, boolean bVectorize);

	/**
	 * Subtracts <code>expr2</code> from <code>expr1</code>.
	 * 
	 * @param expr1
	 *            The minuend
	 * @param expr2
	 *            The subtrahend
	 * @param specDatatype
	 *            The basic datatype of the expression (<code>float</code>,
	 *            <code>double</code>, ...)
	 * @return An expression for <code>expr1</code> - <code>expr2</code>
	 */
	public abstract Expression minus (Expression expr1, Expression expr2, Specifier specDatatype, boolean bVectorize);

	/**
	 * Multiplies <code>expr1</code> by <code>expr2</code>.
	 * 
	 * @param expr1
	 *            The first factor
	 * @param expr2
	 *            The second factor
	 * @param specDatatype
	 *            The basic datatype of the expression (<code>float</code>,
	 *            <code>double</code>, ...)
	 * @return An expression for <code>expr1</code> * <code>expr2</code>
	 */
	public abstract Expression multiply (Expression expr1, Expression expr2, Specifier specDatatype, boolean bVectorize);

	/**
	 * Divides <code>expr1</code> by <code>expr2</code>.
	 * 
	 * @param expr1
	 *            The dividend
	 * @param expr2
	 *            The divisor
	 * @param specDatatype
	 *            The basic datatype of the expression (<code>float</code>,
	 *            <code>double</code>, ...)
	 * @return An expression for <code>expr1</code> / <code>expr2</code>
	 */
	public abstract Expression divide (Expression expr1, Expression expr2, Specifier specDatatype, boolean bVectorize);

	/**
	 * Creates a fused multiply-add (FMA) expression: Returns an expression for
	 * <code>exprFactor1</code> * <code>exprFactor2</code> +
	 * <code>exprSummand</code>.
	 * 
	 * @param exprSummand
	 *            The summand
	 * @param exprFactor1
	 *            The first factor
	 * @param exprFactor2
	 *            The second factor
	 * @param specDatatype
	 *            The basic datatype of the expression (<code>float</code>,
	 *            <code>double</code>, ...)
	 * @return An expression for <code>exprFactor1</code> *
	 *         <code>exprFactor2</code> + <code>exprSummand</code>
	 */
	public abstract Expression fma (Expression exprSummand,	Expression exprFactor1, Expression exprFactor2, Specifier specDatatype,	boolean bVectorize);

	/**
	 * Creates a fused multiply-subtract (FMS) expression: Returns an expression
	 * for <code>exprFactor1</code> * <code>exprFactor2</code> -
	 * <code>exprSummand</code>.
	 * 
	 * @param exprSummand
	 *            The summand
	 * @param exprFactor1
	 *            The first factor
	 * @param exprFactor2
	 *            The second factor
	 * @param specDatatype
	 *            The basic datatype of the expression (<code>float</code>,
	 *            <code>double</code>, ...)
	 * @return An expression for <code>exprFactor1</code> *
	 *         <code>exprFactor2</code> - <code>exprSummand</code>
	 */
	public abstract Expression fms (Expression exprSummand,	Expression exprFactor1, Expression exprFactor2, Specifier specDatatype,	boolean bVectorize);

	/**
	 * Calculates the square root of <code>expr</code>.
	 * 
	 * @param expr
	 *            The expression of which to calculate the square root
	 * @param specDatatype
	 *            The basic datatype of the expression (<code>float</code>,
	 *            <code>double</code>, ...)
	 * @return The square root of <code>expr</code>
	 */
	public abstract Expression sqrt (Expression expr, Specifier specDatatype, boolean bVectorize);

	/**
	 * Extracts a misaligned SIMD vector from [<code>expr1</code>,
	 * <code>expr2</code>], offset by <code>nOffset</code> with respect to
	 * <code>expr1</code>.
	 * 
	 * @param expr1
	 *            Expression for the first SIMD vector
	 * @param expr2
	 *            Expression for the second SIMD vector, pasted to the right of
	 *            <code>expr1</code>
	 * @param specDatatype
	 *            The basic datatype of the expression The datatype of the
	 *            vectors
	 *            (<code>float</code>, <code>double</code>, ...)
	 * @param nOffset
	 *            The offset for the extracted vector with respect to
	 *            <code>expr1</code>
	 * @return A SIMD vector as a subvector of [<code>expr1</code>,
	 *         <code>expr2</code>], offset by <code>nOffset</code>
	 */
	public abstract Expression shuffle (Expression expr1, Expression expr2,	Specifier specDatatype, int nOffset);

	/**
	 * Creates a SIMD vector from the scalar expression <code>expr</code> by
	 * duplicating the
	 * scalar expression across the SIMD vector. E.g., 0.5 &rarr; { 0.5, 0.5,
	 * 0.5, 0.5 }.
	 * 
	 * @param expr
	 *            The scalar expression to duplicate
	 * @param specDatatype
	 *            The basic datatype of the expression (<code>float</code>,
	 *            <code>double</code>, ...)
	 * @return A SIMD vector with entries <code>expr</code>
	 */
	public abstract Traversable splat (Expression expr, Specifier specDatatype);
	
	public abstract Expression vector_reduce_sum (Expression expr, Specifier specDatatype);
	public abstract Expression vector_reduce_product (Expression expr, Specifier specDatatype);
	public abstract Expression vector_reduce_min (Expression expr, Specifier specDatatype);
	public abstract Expression vector_reduce_max (Expression expr, Specifier specDatatype);
}
