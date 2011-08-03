package ch.unibas.cs.hpwc.patus.codegen.backend;

public interface IAdditionalKernelSpecific
{
	/**
	 * Returns a string with additional code to be added after the includes and before the actual kernel code.
	 * @return Additional code
	 */
	public abstract String getAdditionalKernelSpecificCode ();
}
