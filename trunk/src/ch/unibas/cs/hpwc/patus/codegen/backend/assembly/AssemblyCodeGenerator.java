package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import cetus.hir.Specifier;
import cetus.hir.Statement;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;
import ch.unibas.cs.hpwc.patus.codegen.backend.IBackendAssemblyCodeGenerator;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;

/**
 * <p>Creates inline assembly sections.</p>
 * <p>Instructions are added to the section by invoking the
 * {@link AssemblyCodeGenerator#issueInstruction(Instruction, Specifier, StringBuilder)}
 * method. After all instructions have been added, call the
 * {@link AssemblyCodeGenerator#generate(CodeGeneratorRuntimeOptions)}
 * method.</p>
 * <p>Note that this class is not thread-safe.</p>
 * 
 * @author Matthias-M. Christen
 */
public class AssemblyCodeGenerator implements IBackendAssemblyCodeGenerator
{
	///////////////////////////////////////////////////////////////////
	// Member Variables

	private CodeGeneratorSharedObjects m_data;
	
	
	///////////////////////////////////////////////////////////////////
	// Implementation

	public AssemblyCodeGenerator (CodeGeneratorSharedObjects data)
	{
		m_data = data;
		
	}
	
	@Override
	public Statement generate (CodeGeneratorRuntimeOptions options)
	{
		// TODO Auto-generated method stub
		return null;
	}
}
