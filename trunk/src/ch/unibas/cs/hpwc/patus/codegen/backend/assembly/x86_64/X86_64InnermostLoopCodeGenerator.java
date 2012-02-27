package ch.unibas.cs.hpwc.patus.codegen.backend.assembly.x86_64;

import ch.unibas.cs.hpwc.patus.arch.TypeRegisterType;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.AssemblySection;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.IOperand;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.InnermostLoopCodeGenerator;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.Instruction;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.InstructionList;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.Label;
import ch.unibas.cs.hpwc.patus.util.MathUtil;

public class X86_64InnermostLoopCodeGenerator extends InnermostLoopCodeGenerator
{
	///////////////////////////////////////////////////////////////////
	// Member Variables

	private IOperand m_regSaveCounter;
	
	
	///////////////////////////////////////////////////////////////////
	// Implementation

	public X86_64InnermostLoopCodeGenerator (AssemblySection as, CodeGeneratorSharedObjects data)
	{
		super (as, data);
		
		// find a free register to save the initial value of the counter
		m_regSaveCounter = getAssemblySection ().getFreeRegister (TypeRegisterType.GPR);
	}

	@Override
	public void generatePrologHeader (int nSIMDVectorLength)
	{
		AssemblySection as = getAssemblySection ();
		InstructionList l = new InstructionList ();
		
		IOperand.IRegisterOperand regCnt = getCounterRegister ();
		
		l.addInstruction (new Instruction ("mov", new IOperand[] { as.getInput ("output grid address"), regCnt }));
		l.addInstruction (new Instruction ("add", new IOperand[] { new IOperand.Immediate (nSIMDVectorLength - 1), regCnt }));
		l.addInstruction (new Instruction ("and", new IOperand[] { new IOperand.Immediate (nSIMDVectorLength - 1), regCnt }));
		l.addInstruction (new Instruction ("sub", new IOperand[] { new IOperand.Immediate (nSIMDVectorLength), regCnt }));
		l.addInstruction (new Instruction ("neg", new IOperand[] { regCnt }));
		l.addInstruction (new Instruction ("shr", new IOperand[] { new IOperand.Immediate (MathUtil.log2 (nBaseTypeSize)), regCnt }));
		l.addInstruction (new Instruction ("cmp", new IOperand[] { regCnt, as.getInput ("loop length") }));
		l.addInstruction (new Instruction ("jg", new IOperand[] { new IOperand.LabelOperand ("hdr1") }));
		l.addInstruction (new Instruction ("mov", new IOperand[] { as.getInput ("loop length"), regCnt }));
		l.addInstruction (new Label ("hdr1"));
		l.addInstruction (new Instruction ("mov", new IOperand[] { regCnt, m_regSaveCounter }));
		l.addInstruction (new Instruction ("or", new IOperand[] { regCnt, regCnt }));
		l.addInstruction (new Instruction ("jz", new IOperand[] { new IOperand.LabelOperand ("") }));
	}

	@Override
	public void generatePrologFooter ()
	{
		// increment addresses and decrement counter
	}

	@Override
	public void generateMainHeader (int nSIMDVectorLength)
	{
		AssemblySection as = getAssemblySection ();
		InstructionList l = new InstructionList ();
		
		IOperand.IRegisterOperand regCnt = getCounterRegister ();
		
		// restore the loop counter
		l.addInstruction (new Label ());
		l.addInstruction (new Instruction ("mov", new IOperand[] { m_regSaveCounter, regCnt }));
		l.addInstruction (new Label ());
		l.addInstruction (new Instruction ("sub", new IOperand[] { as.getInput ("loop length"), regCnt }));
		l.addInstruction (new Instruction ("neg", new IOperand[] { regCnt }));
		l.addInstruction (new Instruction ("shr", new IOperand[] { MathUtil.log2 (nSIMDVectorLength), regCnt }));
		l.addInstruction (new Instruction ("mov", new IOperand[] { regCnt, m_regTmp }));
		l.addInstruction (new Instruction ("or", new IOperand[] { regCnt, regCnt }));
		l.addInstruction (new Instruction ("jz", new IOperand[] { new IOperand.LabelOperand ("") }));
	}

	@Override
	public void generateMainFooter ()
	{
		// increment addresses and decrement counter
	}

	@Override
	public void generateEpilogHeader ()
	{
		// TODO Auto-generated method stub
		
	}

	@Override
	public void generateEpilogFooter ()
	{
		// increment addresses and decrement counter		
	}
}
