package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import java.util.Map;

import ch.unibas.cs.hpwc.patus.arch.TypeBaseIntrinsicEnum;
import ch.unibas.cs.hpwc.patus.ast.ParameterAssignment;


/**
 * Interface for assembly instructions.
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
	 * Returns the intrinsic corresponding to this instruction or
	 * <code>null</code> if the instruction doesn't correspond to an intrinsic
	 * defined in the architecture description.
	 * 
	 * @return The intrinsic corresponding to this instruction
	 */
	public abstract TypeBaseIntrinsicEnum getIntrinsic ();
	
	/**
	 * Returns the Java code to create this instruction.
	 * @return
	 */
	public abstract String toJavaCode (Map<IOperand, String> mapOperands);
	
	public abstract void setParameterAssignment (ParameterAssignment pa);
	public abstract ParameterAssignment getParameterAssignment ();
}
