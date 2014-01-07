package ch.unibas.cs.hpwc.patus.codegen.backend.assembly.optimize;

import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.InstructionList;

/**
 * Interface definition for instruction list optimizers
 * @author Matthias-M. Christen
 */
public interface IInstructionListOptimizer
{
	/**
	 * Does an optimization pass on the input instruction list <code>il</code>.
	 * @param il The instruction list to optimize
	 * @return The optimized instruction list
	 */
	public abstract InstructionList optimize (InstructionList il);
}
