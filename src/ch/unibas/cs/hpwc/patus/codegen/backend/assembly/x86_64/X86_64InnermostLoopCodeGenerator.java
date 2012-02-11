package ch.unibas.cs.hpwc.patus.codegen.backend.assembly.x86_64;

import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.AssemblySection;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.IOperand;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.InnermostLoopCodeGenerator;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.Instruction;

public class X86_64InnermostLoopCodeGenerator extends InnermostLoopCodeGenerator
{

	public X86_64InnermostLoopCodeGenerator (AssemblySection as, CodeGeneratorSharedObjects data)
	{
		super (as, data);
	}

	@Override
	public void generatePrologHeader ()
	{
		AssemblySection as = getAssemblySection ();
		
		as.addInstruction (new Instruction ("mov", new IOperand[] { new IOperand.InputRef (0) }), null);
	}

	@Override
	public void generatePrologFooter ()
	{
		// TODO Auto-generated method stub
		
	}

	@Override
	public void generateMainHeader ()
	{
		// TODO Auto-generated method stub
		
	}

	@Override
	public void generateMainFooter ()
	{
		// TODO Auto-generated method stub
		
	}

	@Override
	public void generateEpilogHeader ()
	{
		// TODO Auto-generated method stub
		
	}

	@Override
	public void generateEpilogFooter ()
	{
		// TODO Auto-generated method stub
		
	}
}
