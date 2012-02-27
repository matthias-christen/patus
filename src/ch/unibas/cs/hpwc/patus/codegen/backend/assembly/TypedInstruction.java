package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import cetus.hir.Specifier;
import ch.unibas.cs.hpwc.patus.arch.IArchitectureDescription;

/**
 * 
 * @author Matthias-M. Christen
 */
public class TypedInstruction
{
	private IInstruction m_instruction;
	private Specifier m_specDatatype;
	
	public TypedInstruction (IInstruction instruction, Specifier specDatatype)
	{
		m_instruction = instruction;
		m_specDatatype = specDatatype;
	}
	
	public void issue (IArchitectureDescription arch, StringBuilder sbResult)
	{
		m_instruction.issue (m_specDatatype, arch, sbResult);
	}
}
