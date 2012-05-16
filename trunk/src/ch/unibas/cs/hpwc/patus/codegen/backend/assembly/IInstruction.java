package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import java.util.Map;


/**
 * 
 * @author Matthias-M. Christen
 */
public interface IInstruction
{
	/**
	 * Issues the instruction, i.e., adds the assembly code to
	 * <code>sbResult</code>.
	 * 
	 * @param sbResult
	 *            The string builder to which the result is added
	 */
	public abstract void issue (StringBuilder sbResult);
	
	/**
	 * Returns the Java code to create this instruction.
	 * @return
	 */
	public abstract String toJavaCode (Map<IOperand, String> mapOperands);
}
