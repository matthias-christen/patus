package ch.unibas.cs.hpwc.patus.codegen;

import cetus.hir.Traversable;
import ch.unibas.cs.hpwc.patus.ast.StatementListBundle;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;


public interface IInnermostLoopCodeGenerator extends ICodeGenerator
{
	/**
	 * Generates code for an innermost loop containing a stencil call.
	 * 
	 * @param sdit
	 *            The subdomain iterator containing the innermost loop with a
	 *            stencil computation for which to generate the code
	 * @param options
	 *            Code generation-time options
	 */
	@Override
	public StatementListBundle generate (Traversable sdit, CodeGeneratorRuntimeOptions options);
	
	/**
	 * Tells whether for the instance of the inner most loop code generator the
	 * assembly section in the architecture description is required.
	 * 
	 * @return <code>true</code> iff the instance of the code generator requires
	 *         the assembly specification
	 */
	public abstract boolean requiresAssemblySection ();
}
