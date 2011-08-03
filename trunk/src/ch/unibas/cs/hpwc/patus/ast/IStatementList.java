package ch.unibas.cs.hpwc.patus.ast;

import cetus.hir.Declaration;
import cetus.hir.Statement;

public interface IStatementList
{
	/**
	 *
	 * @param stmt
	 */
	public abstract void addStatement (Statement stmt);

	/**
	 *
	 * @param declaration
	 */
	public abstract void addDeclaration (Declaration declaration);

	public void addStatementAtTop (Statement stmt);


}
