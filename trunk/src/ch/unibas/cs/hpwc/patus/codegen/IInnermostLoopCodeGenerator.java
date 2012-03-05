package ch.unibas.cs.hpwc.patus.codegen;


public interface IInnermostLoopCodeGenerator extends ICodeGenerator
{
	/**
	 * Tells whether for the instance of the inner most loop code generator the assembly
	 * section in the architecture description is required.
	 * @return <code>true</code> iff the instance of the code generator requires the
	 * 	assembly specification
	 */
	public abstract boolean requiresAssemblySection ();
}
