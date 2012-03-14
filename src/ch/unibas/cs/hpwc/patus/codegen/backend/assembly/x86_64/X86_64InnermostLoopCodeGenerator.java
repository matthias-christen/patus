package ch.unibas.cs.hpwc.patus.codegen.backend.assembly.x86_64;

import ch.unibas.cs.hpwc.patus.arch.TypeRegisterType;
import ch.unibas.cs.hpwc.patus.ast.SubdomainIterator;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.IOperand;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.InnermostLoopCodeGenerator;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.Instruction;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.InstructionList;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.Label;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.StencilAssemblySection;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;
import ch.unibas.cs.hpwc.patus.representation.StencilNode;
import ch.unibas.cs.hpwc.patus.util.MathUtil;

/**
 * Generates inline assembly code for the inner most loop for x86_64 architectures.
 * 
 * TODO: scalar code when the number of iterates in the inner most loop is < vector length
 * 
 * @author Matthias-M. Christen
 */
public class X86_64InnermostLoopCodeGenerator extends InnermostLoopCodeGenerator
{
	///////////////////////////////////////////////////////////////////
	// Constants
	
	private final static String LABEL_PROLOGHDR_LESSTHANMAX = "phdr_ltmax";
	private final static String LABEL_MAINHDR = "mhdr";
	private final static String LABEL_MAINHDR_STARTCOMPUTATION = "mhdr_startcomp";
	private final static String LABEL_EPILOGHDR = "ehdr";
	private final static String LABEL_EPILOGHDR_ENDCOMPUTATION = "ehdr_endcomp";

	
	protected class CodeGenerator extends InnermostLoopCodeGenerator.CodeGenerator
	{
		///////////////////////////////////////////////////////////////////
		// Member Variables

		private IOperand m_regSaveCounter;
		private IOperand m_regTmp;
		
		
		///////////////////////////////////////////////////////////////////
		// Implementation

		public CodeGenerator (SubdomainIterator sdit, CodeGeneratorRuntimeOptions options)
		{
			super (sdit, options);
			
			// find a free register to save the initial value of the counter
			m_regSaveCounter = getAssemblySection ().getFreeRegister (TypeRegisterType.GPR);
			m_regTmp = getAssemblySection ().getFreeRegister (TypeRegisterType.GPR);
		}
		
		private StencilNode getOutputStencilNode ()
		{
			return m_data.getStencilCalculation ().getStencilBundle ().getFusedStencil ().getOutputNodes ().iterator ().next ();
		}
		
		@Override
		public InstructionList generatePrologHeader ()
		{
			StencilAssemblySection as = getAssemblySection ();
			InstructionList l = new InstructionList ();
			
			IOperand.IRegisterOperand regCounter = getCounterRegister ();
			int nSIMDVectorLengthInBytes = getSIMDVectorLength () * getBaseTypeSize ();
			
			l.addInstruction (new Instruction ("mov", as.getGrid (getOutputStencilNode (), 0), regCounter));
			l.addInstruction (new Instruction ("add", new IOperand.Immediate (nSIMDVectorLengthInBytes - 1), regCounter));
			l.addInstruction (new Instruction ("and", new IOperand.Immediate (nSIMDVectorLengthInBytes - 1), regCounter));
			l.addInstruction (new Instruction ("sub", new IOperand.Immediate (nSIMDVectorLengthInBytes), regCounter));
			l.addInstruction (new Instruction ("neg", regCounter));
			l.addInstruction (new Instruction ("shr", new IOperand.Immediate (MathUtil.log2 (getBaseTypeSize ())), regCounter));
			
			l.addInstruction (new Instruction ("cmp", regCounter, as.getInput (INPUT_LOOPMAX)));
			l.addInstruction (new Instruction ("jg",  Label.getLabelOperand (LABEL_PROLOGHDR_LESSTHANMAX)));
			l.addInstruction (new Instruction ("mov", as.getInput (INPUT_LOOPMAX), regCounter));
			
			l.addInstruction (Label.getLabel (LABEL_PROLOGHDR_LESSTHANMAX));
			l.addInstruction (new Instruction ("mov", regCounter, m_regSaveCounter));
			l.addInstruction (new Instruction ("or",  regCounter, regCounter));
			l.addInstruction (new Instruction ("jz",  Label.getLabelOperand (LABEL_MAINHDR)));
			
			return l;
		}

		@Override
		public InstructionList generatePrologFooter ()
		{
			// increment addresses
			
			StencilAssemblySection as = getAssemblySection ();
			InstructionList l = new InstructionList ();
			
			IOperand.IRegisterOperand regCounter = getCounterRegister ();
			
			l.addInstruction (new Instruction ("shl", new IOperand.Immediate (MathUtil.log2 (getBaseTypeSize ())), regCounter));
			for (IOperand opGridAddrRegister : as.getGrids ())
				l.addInstruction (new Instruction ("add", regCounter, opGridAddrRegister));
			
			return l;
		}

		@Override
		public InstructionList generateMainHeader ()
		{
			StencilAssemblySection as = getAssemblySection ();
			InstructionList l = new InstructionList ();
			
			IOperand.IRegisterOperand regCounter = getCounterRegister ();
			int nSIMDVectorLength = getSIMDVectorLength ();
			int nLoopUnrollingFactor = getRuntimeOptions ().getIntValue (OPTION_INLINEASM_UNROLLFACTOR, 1);
			
			// restore the loop counter
			l.addInstruction (new Instruction ("mov", m_regSaveCounter, regCounter));
			l.addInstruction (Label.getLabel (LABEL_MAINHDR));
			l.addInstruction (new Instruction ("sub", as.getInput (INPUT_LOOPMAX), regCounter));
			l.addInstruction (new Instruction ("neg", regCounter));
			l.addInstruction (new Instruction ("shr", new IOperand.Immediate (MathUtil.log2 (nSIMDVectorLength * nLoopUnrollingFactor)), regCounter));
			l.addInstruction (new Instruction ("mov", regCounter, m_regTmp));
			l.addInstruction (new Instruction ("or",  regCounter, regCounter));
			l.addInstruction (new Instruction ("jz",  Label.getLabelOperand (LABEL_EPILOGHDR)));
			l.addInstruction (Label.getLabel (LABEL_MAINHDR_STARTCOMPUTATION));		
			
			return l;
		}

		@Override
		public InstructionList generateMainFooter ()
		{
			// increment addresses and decrement counter

			StencilAssemblySection as = getAssemblySection ();
			InstructionList l = new InstructionList ();
			
			int nSIMDVectorLengthInBytes = getSIMDVectorLength () * getBaseTypeSize ();
			
			// increment pointers
			for (IOperand opGridAddrRegister : as.getGrids ())
				l.addInstruction (new Instruction ("add", new IOperand.Immediate (nSIMDVectorLengthInBytes), opGridAddrRegister));
			
			// loop
			l.addInstruction (new Instruction ("dec", getCounterRegister ()));
			l.addInstruction (new Instruction ("jnz", Label.getLabelOperand (LABEL_MAINHDR_STARTCOMPUTATION)));

			return l;
		}

		@Override
		public InstructionList generateEpilogHeader ()
		{
			// TODO Auto-generated method stub
			
			StencilAssemblySection as = getAssemblySection ();
			InstructionList l = new InstructionList ();
			
			IOperand.IRegisterOperand regCounter = getCounterRegister ();
			int nSIMDVectorLength = getSIMDVectorLength ();
			int nLoopUnrollingFactor = getRuntimeOptions ().getIntValue (OPTION_INLINEASM_UNROLLFACTOR, 1);
			
			l.addInstruction (Label.getLabel (LABEL_EPILOGHDR));
			l.addInstruction (new Instruction ("mov", new IOperand.Immediate (nSIMDVectorLength), regCounter));
			l.addInstruction (new Instruction ("sub", as.getInput (INPUT_LOOPMAX), regCounter));
			l.addInstruction (new Instruction ("shl", new IOperand.Immediate (MathUtil.log2 (nSIMDVectorLength * nLoopUnrollingFactor)), m_regTmp));
			l.addInstruction (new Instruction ("add", m_regTmp, regCounter));
			l.addInstruction (new Instruction ("add", m_regSaveCounter, regCounter));
			l.addInstruction (new Instruction ("cmp", new IOperand.Immediate (nSIMDVectorLength), regCounter));
			l.addInstruction (new Instruction ("je",  Label.getLabelOperand (LABEL_EPILOGHDR_ENDCOMPUTATION)));
			
			// adjust the pointers
			l.addInstruction (new Instruction ("shl", new IOperand.Immediate (MathUtil.log2 (getBaseTypeSize ())), regCounter));
			for (IOperand opGridAddrRegister : as.getGrids ())
				l.addInstruction (new Instruction ("sub", regCounter, opGridAddrRegister));		

			return l;
		}

		@Override
		public InstructionList generateEpilogFooter ()
		{
			InstructionList l = new InstructionList ();

			// end of computation label
			l.addInstruction (Label.getLabel (LABEL_EPILOGHDR_ENDCOMPUTATION));
			
			return l;		
		}		
	}


	public X86_64InnermostLoopCodeGenerator (CodeGeneratorSharedObjects data)
	{
		super (data);
	}
	
	@Override
	protected InnermostLoopCodeGenerator.CodeGenerator newCodeGenerator (SubdomainIterator sdit, CodeGeneratorRuntimeOptions options)
	{
		return new X86_64InnermostLoopCodeGenerator.CodeGenerator (sdit, options);
	}
}
