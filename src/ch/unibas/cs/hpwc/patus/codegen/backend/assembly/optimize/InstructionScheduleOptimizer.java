package ch.unibas.cs.hpwc.patus.codegen.backend.assembly.optimize;

import cetus.hir.Specifier;
import ch.unibas.cs.hpwc.patus.arch.IArchitectureDescription;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.InstructionList;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.InstructionScheduler;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.analyze.DependenceAnalysis;

public class InstructionScheduleOptimizer implements IInstructionListOptimizer
{
	private IArchitectureDescription m_arch;
	private Specifier m_specDatatype;
	
	public InstructionScheduleOptimizer (IArchitectureDescription arch, Specifier specDatatype)
	{
		m_arch = arch;
		m_specDatatype = specDatatype;
	}
	
	@Override
	public InstructionList optimize (InstructionList il)
	{
		return new InstructionScheduler (new DependenceAnalysis (il, m_arch).run (m_specDatatype), m_arch).schedule ();
	}
}
