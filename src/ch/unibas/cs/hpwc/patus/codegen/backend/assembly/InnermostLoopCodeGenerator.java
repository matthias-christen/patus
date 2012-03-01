package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import java.util.HashMap;
import java.util.Map;

import cetus.hir.Specifier;
import ch.unibas.cs.hpwc.patus.analysis.ReuseNodesCollector;
import ch.unibas.cs.hpwc.patus.arch.TypeBaseIntrinsicEnum;
import ch.unibas.cs.hpwc.patus.arch.TypeRegisterType;
import ch.unibas.cs.hpwc.patus.ast.SubdomainIterator;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;
import ch.unibas.cs.hpwc.patus.representation.Stencil;
import ch.unibas.cs.hpwc.patus.representation.StencilNode;

/**
 * 
 * @author Matthias-M. Christen
 */
public abstract class InnermostLoopCodeGenerator
{
	///////////////////////////////////////////////////////////////////
	// Constants
	
	protected final static String INPUT_LOOPMAX = "loop_max";
	
	protected final static String OPTION_INLINEASM_UNROLLFACTOR = "iasm_unroll";
	
	/**
	 * The minimum size of a set of stencil nodes of the same grid which have the
	 * same non-unit stride coordinates, so that registers corresponding to stencil nodes
	 * get reused in the innermost loop 
	 */
	public final static int MIN_REUSE_SET_SIZE = 3;
	
	public final static int MAX_REGISTERS_FOR_CONSTANTS = 3; 

	
	///////////////////////////////////////////////////////////////////
	// Member Variables

	private CodeGeneratorSharedObjects m_data;
	
	private CodeGeneratorRuntimeOptions m_options;
	
	/**
	 * The subdomain iterator encapsulating the innermost loop
	 */
	private SubdomainIterator m_sdit;
	
	/**
	 * The unroll factor within the innermost loop
	 */
	private int m_nUnrollFactor;
	
	/**
	 * The generated inline assembly section
	 */
	private StencilAssemblySection m_assemblySection;
	
	/**
	 * Flag indicating whether the architecture supports SIMD intrinsics
	 */
	private boolean m_bArchSupportsSIMD;
	
	/**
	 * Flag indicating whether the architecture supports unaligned data movement
	 */
	private boolean m_bArchSupportsUnalignedMoves;
	
	/**
	 * The register used as loop counter for this inner-most loop
	 */
	private IOperand.IRegisterOperand m_regCounter;
	
	private Map<String, Map<StencilNode, IOperand.IRegisterOperand>> m_mapReuseRegisters;
	

	///////////////////////////////////////////////////////////////////
	// Implementation

	public InnermostLoopCodeGenerator (CodeGeneratorSharedObjects data, SubdomainIterator sdit, CodeGeneratorRuntimeOptions options)
	{
		m_data = data;
		m_sdit = sdit;
		m_options = options;
		
		m_assemblySection = new StencilAssemblySection (m_data, m_sdit.getIterator (), m_options);
		m_nUnrollFactor = options.getIntValue (OPTION_INLINEASM_UNROLLFACTOR, 1);
	
		m_bArchSupportsSIMD = m_data.getArchitectureDescription ().getSIMDVectorLength (Specifier.FLOAT) > 1;
		m_bArchSupportsUnalignedMoves = true;
		if (m_bArchSupportsSIMD)
			m_bArchSupportsUnalignedMoves = m_data.getArchitectureDescription ().getIntrinsic (TypeBaseIntrinsicEnum.MOVE_FPR_UNALIGNED.value (), Specifier.FLOAT) != null;
				
		// add the length of the inner most loop as an input
		m_assemblySection.addInput (INPUT_LOOPMAX, m_sdit.getDomainSubdomain ().getSize ().getCoord (0));
		
		// request a register to be used as loop counter
		m_regCounter = m_assemblySection.getFreeRegister (TypeRegisterType.GPR);
		
		m_mapReuseRegisters = new HashMap<String, Map<StencilNode, IOperand.IRegisterOperand>> ();
		findReuseRegisters ();
	}
		
	public void generate ()
	{
		
	}
	
	/**
	 * 
	 */
	private void findReuseRegisters ()
	{
		// estimate the registers used
		int nAvailableRegisters = m_data.getArchitectureDescription ().getRegistersCount (TypeRegisterType.SIMD);
		
		// subtract the registers used to store constants
		if (m_assemblySection.getConstantsCount () <= MAX_REGISTERS_FOR_CONSTANTS)
			nAvailableRegisters -= m_assemblySection.getConstantsCount ();
		
		// subtract the registers used for the calculations
		// TODO: unrolling?
		RegisterAllocator regcnt = new RegisterAllocator (null);
		int nRegsForCalculations = 0;
		for (Stencil stencil : m_data.getStencilCalculation ().getStencilBundle ())
			nRegsForCalculations = Math.max (nRegsForCalculations, regcnt.countRegistersNeeded (stencil.getExpression ()));
		nAvailableRegisters -= nRegsForCalculations;
		
		// find stencil node sets to reuse within the innermost loop
		ReuseNodesCollector reuse = new ReuseNodesCollector (m_data.getStencilCalculation ().getStencilBundle ().getFusedStencil ().getAllNodes (), 0);
		reuse.getSetsWithMaxNodesConstraint (nAvailableRegisters, m_nUnrollFactor);
	}
	
	protected CodeGeneratorRuntimeOptions getRuntimeOptions ()
	{
		return m_options;
	}
	
	public StencilAssemblySection getAssemblySection ()
	{
		return m_assemblySection;
	}
	
	public boolean isSIMDSupported ()
	{
		return m_bArchSupportsSIMD;
	}
	
	public int getSIMDVectorLength ()
	{
		return m_data.getArchitectureDescription ().getSIMDVectorLength (m_assemblySection.getDatatype ());
	}
	
	/**
	 * Returns the size (in Bytes) of the data type of the stencil computation. 
	 * @return The size (in Bytes) of the computation data type
	 */
	public int getBaseTypeSize ()
	{
		return AssemblySection.getTypeSize (m_assemblySection.getDatatype ());
	}
	
	public boolean isUnalignedMoveSupported ()
	{
		return m_bArchSupportsUnalignedMoves;
	}
	
	public IOperand.IRegisterOperand getCounterRegister ()
	{
		return m_regCounter;
	}
	
	abstract public InstructionList generatePrologHeader ();
	
	abstract public InstructionList generatePrologFooter ();
	
	abstract public InstructionList generateMainHeader ();
	
	abstract public InstructionList generateMainFooter ();
	
	abstract public InstructionList generateEpilogHeader ();
	
	abstract public InstructionList generateEpilogFooter ();
}
