package ch.unibas.cs.hpwc.patus.codegen.backend.assembly.optimize;

import ch.unibas.cs.hpwc.patus.arch.IArchitectureDescription;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.InstructionList;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.InstructionScheduler;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.analyze.DependenceAnalysis;

public class InstructionScheduleOptimizer implements IInstructionListOptimizer
{
	private IArchitectureDescription m_arch;
	
	public InstructionScheduleOptimizer (IArchitectureDescription arch)
	{
		m_arch = arch;
	}
	
	@Override
	public InstructionList optimize (InstructionList il)
	{
		return new InstructionScheduler (new DependenceAnalysis (il).run (), m_arch).schedule ();
	}
}
