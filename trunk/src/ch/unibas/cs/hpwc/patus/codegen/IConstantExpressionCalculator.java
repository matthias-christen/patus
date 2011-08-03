package ch.unibas.cs.hpwc.patus.codegen;

import cetus.hir.Expression;
import cetus.hir.Specifier;
import cetus.hir.Traversable;

/**
 *
 * @author Matthias-M. Christen
 */
public interface IConstantExpressionCalculator
{
	/**
	 *
	 * @param expr
	 * @param specDatatype
	 * @return
	 */
	public abstract Traversable calculateConstantExpression (Expression expr, Specifier specDatatype, boolean bVectorize);
}
