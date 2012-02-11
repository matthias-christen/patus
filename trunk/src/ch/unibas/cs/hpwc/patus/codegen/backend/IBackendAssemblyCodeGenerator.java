package ch.unibas.cs.hpwc.patus.codegen.backend;

import cetus.hir.Statement;
import ch.unibas.cs.hpwc.patus.arch.TypeBaseIntrinsicEnum;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;

/**
 * 
 * @author Matthias-M. Christen
 */
public interface IBackendAssemblyCodeGenerator
{
	///////////////////////////////////////////////////////////////////
	// Constants

	public final static String OPTION_ASSEMBLY_UNROLLFACTOR = "AsmUnroll";
	
	public final static String INSTR_ADD = TypeBaseIntrinsicEnum.PLUS.value ();
	public final static String INSTR_SUB = TypeBaseIntrinsicEnum.MINUS.value ();
	public final static String INSTR_MUL = TypeBaseIntrinsicEnum.MULTIPLY.value ();
	public final static String INSTR_DIV = TypeBaseIntrinsicEnum.DIVIDE.value ();
	public final static String INSTR_NEG = TypeBaseIntrinsicEnum.UNARY_MINUS.value ();
	public final static String INSTR_FMA = TypeBaseIntrinsicEnum.FMA.value ();
	public final static String INSTR_FMS = TypeBaseIntrinsicEnum.FMS.value ();
	public final static String INSTR_MOV_GPR = TypeBaseIntrinsicEnum.MOVE_GPR.value ();
	public final static String INSTR_MOV_FPR = TypeBaseIntrinsicEnum.MOVE_FPR.value ();
	public final static String INSTR_MOV_FPR_UNALIGNED = TypeBaseIntrinsicEnum.MOVE_FPR_UNALIGNED.value ();


	///////////////////////////////////////////////////////////////////
	// Methods
	

	/**
	 * Returns a statement containing the inline assembly.
	 * @param options The code generation options
	 * @return A statement encapsulating the inline assembly
	 */
	public abstract Statement generate (CodeGeneratorRuntimeOptions options);
}
