package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import cetus.hir.Specifier;
import ch.unibas.cs.hpwc.patus.arch.IArchitectureDescription;

/**
 * 
 * @author Matthias-M. Christen
 */
public interface IInstruction
{
	/**
	 * Issues the instruction, i.e., adds the assembly code to <code>sbResult</code>.
	 * @param specDatatype The data type to which to specialize the instruction when creating the assembly
	 * @param arch The architecture description
	 * @param sbResult The string builder to which the result is added
	 */
	public abstract void issue (Specifier specDatatype, IArchitectureDescription arch, StringBuilder sbResult);
}
