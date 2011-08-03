package ch.unibas.cs.hpwc.patus.codegen;

import cetus.hir.Traversable;
import ch.unibas.cs.hpwc.patus.ast.StatementListBundle;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;

public interface ICodeGenerator
{
	/**
	 * Generates code for a particular type of input.
	 * @param stmtInput
	 * @return
	 */
	public abstract StatementListBundle generate (Traversable trvInput, CodeGeneratorRuntimeOptions options);
}
