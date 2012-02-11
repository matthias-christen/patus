package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import cetus.hir.Specifier;
import ch.unibas.cs.hpwc.patus.arch.TypeBaseIntrinsicEnum;
import ch.unibas.cs.hpwc.patus.arch.TypeRegister;
import ch.unibas.cs.hpwc.patus.arch.TypeRegisterType;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;

public abstract class InnermostLoopCodeGenerator
{
	private CodeGeneratorSharedObjects m_data;
	
	private AssemblySection m_assemblySection;
	
	private boolean m_bArchSupportsSIMD;
	
	/**
	 * Flag indicating whether the architecture supports unaligned data movement
	 */
	private boolean m_bArchSupportsUnalignedMoves;
	
	private TypeRegister m_regCounter;
	
	
	public InnermostLoopCodeGenerator (AssemblySection as, CodeGeneratorSharedObjects data)
	{
		m_assemblySection = as;
		m_data = data;
	
		m_bArchSupportsSIMD = m_data.getArchitectureDescription ().getSIMDVectorLength (Specifier.FLOAT) > 1;
		m_bArchSupportsUnalignedMoves = true;
		if (m_bArchSupportsSIMD)
			m_bArchSupportsUnalignedMoves = m_data.getArchitectureDescription ().getIntrinsic (TypeBaseIntrinsicEnum.MOVE_FPR_UNALIGNED.value (), Specifier.FLOAT) != null;
		
		m_regCounter = m_assemblySection.getFreeRegister (TypeRegisterType.GPR);
	}
	
	public void generate ()
	{
		
	}
	
	public AssemblySection getAssemblySection ()
	{
		return m_assemblySection;
	}
	
	public boolean isSIMDSupported ()
	{
		return m_bArchSupportsSIMD;
	}
	
	public boolean isUnalignedMoveSupported ()
	{
		return m_bArchSupportsUnalignedMoves;
	}
	
	public TypeRegister getCounterRegister ()
	{
		return m_regCounter;
	}
	
	abstract public void generatePrologHeader ();
	abstract public void generatePrologFooter ();
	abstract public void generateMainHeader ();
	abstract public void generateMainFooter ();
	abstract public void generateEpilogHeader ();
	abstract public void generateEpilogFooter ();
}
