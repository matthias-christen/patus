package ch.unibas.cs.hpwc.patus.codegen.backend.assembly.x86_64;

import java.util.HashSet;
import java.util.Set;

import ch.unibas.cs.hpwc.patus.arch.TypeRegisterType;
import ch.unibas.cs.hpwc.patus.ast.Parameter;
import ch.unibas.cs.hpwc.patus.ast.ParameterAssignment;
import ch.unibas.cs.hpwc.patus.ast.SubdomainIterator;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.AssemblySection;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.IOperand;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.InnermostLoopCodeGenerator;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.Instruction;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.InstructionList;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.Label;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.StencilAssemblySection;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.StencilAssemblySection.OperandWithInstructions;
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
		
	/**
	 *  Generate a CMOV instruction
	 */
	private final static boolean USE_CMOV = true;
	
	
	private final static String LABEL_PROLOGHDR_LESSTHANMAX = "phdr_ltmax";
	private final static String LABEL_UNROLLEDMAINHDR = "umhdr";
	private final static String LABEL_UNROLLEDMAINHDR_STARTCOMPUTATION = "umhdr_startcomp";
	private final static String LABEL_SIMPLEMAINHDR = "smhdr";
	private final static String LABEL_SIMPLEMAINHDR_STARTCOMPUTATION = "smhdr_startcomp";
	private final static String LABEL_EPILOGHDR = "ehdr";
	private final static String LABEL_EPILOGHDR_ENDCOMPUTATION = "ehdr_endcomp";

	
	protected abstract class AbstractCodeGenerator extends InnermostLoopCodeGenerator.CodeGenerator
	{
		///////////////////////////////////////////////////////////////////
		// Member Variables
		
		private X86_64PrefetchingCodeGenerator m_cgPrefetch;

		
		///////////////////////////////////////////////////////////////////
		// Implementation

		public AbstractCodeGenerator (SubdomainIterator sdit, CodeGeneratorRuntimeOptions options)
		{
			super (sdit, options);
			initialize ();
			m_cgPrefetch = m_data.getOptions ().getCreatePrefetching () ? new X86_64PrefetchingCodeGenerator (m_data, getAssemblySection ()) : null;
		}

		/**
		 * Increments the addresses and decrements the loop counter.
		 */
		protected InstructionList generateMainFooter (String strHeadLabel, int nLoopUnrollingFactor)
		{
			StencilAssemblySection as = getAssemblySection ();
			InstructionList l = new InstructionList ();
			
			int nSIMDVectorLengthInBytes = getSIMDVectorLength () * getBaseTypeSize ();
			IOperand.Immediate opIncrement = new IOperand.Immediate (nSIMDVectorLengthInBytes * nLoopUnrollingFactor);
			
			// increment pointers
			for (IOperand opGridAddrRegister : as.getGrids ())
				l.addInstruction (new Instruction ("addq", opIncrement, opGridAddrRegister));
			
			// rotate reuse registers
			rotateReuseRegisters (l);

			decrementMainLoopCounterAndJump (l, strHeadLabel, nLoopUnrollingFactor);

			return l;
		}

		/**
		 * Decrement the loop counter and jump to the loop head if there are
		 * more iterations to be performed.
		 * 
		 * @param il
		 *            The instruction list to which to add the generated
		 *            instructions
		 */
		protected abstract void decrementMainLoopCounterAndJump (InstructionList il, String strHeadLabel, int nLoopUnrollingFactor);
		
		protected InstructionList generatePrefetching ()
		{
			if (m_cgPrefetch == null)
				return null;
			
			InstructionList il = new InstructionList ();
			Parameter param = new Parameter ("_prefetch");
			
			// keep track of codes to prevent duplicates
			Set<String> setPrefetchingCodes = new HashSet<> ();
			
			for (PrefetchConfig config : PrefetchConfig.getAllConfigs ())
			{
				InstructionList ilPrefetching = m_cgPrefetch.generate (config);

				String strCode = ilPrefetching.toStringWithoutComments ();
				if (setPrefetchingCodes.contains (strCode))
					continue;
				
				il.addParameter (param, config.toInteger ());
				il.addInstructions (new ParameterAssignment (param, config.toInteger ()), ilPrefetching);
				setPrefetchingCodes.add (strCode);
			}
			
			return il;
		}
	}
	
	
	/**
	 * Code generator for aligned vector accesses.
	 */
	protected class CodeGeneratorAligned extends AbstractCodeGenerator
	{
		///////////////////////////////////////////////////////////////////
		// Member Variables

		protected IOperand m_regMainItersCount;
		private IOperand m_regPrologLength;
		
		
		///////////////////////////////////////////////////////////////////
		// Implementation

		public CodeGeneratorAligned (SubdomainIterator sdit, CodeGeneratorRuntimeOptions options)
		{
			super (sdit, options);
			
			// find a free register to save the initial value of the counter
			m_regPrologLength = getAssemblySection ().getFreeRegister (TypeRegisterType.GPR);

			m_regMainItersCount = getAssemblySection ().getFreeRegister (TypeRegisterType.GPR);
		}
		
		private StencilNode getOutputStencilNode ()
		{
			return m_data.getStencilCalculation ().getStencilBundle ().getFusedStencil ().getOutputNodes ().iterator ().next ();
		}
		
		@Override
		public InstructionList generatePrologHeader ()
		{
			StencilAssemblySection as = getAssemblySection ();
			as.reset ();
			
			InstructionList l = new InstructionList ();
			
			IOperand.IRegisterOperand regCounter = getCounterRegister ();
			int nSIMDVectorLengthInBytes = getSIMDVectorLength () * getBaseTypeSize ();
			
			OperandWithInstructions opGrid = as.getGrid (getOutputStencilNode (), 0);
			l.addInstructions (opGrid.getInstrPre ());
			IOperand opGridAddress = opGrid.getOp ();
			if (opGrid.getOp () instanceof IOperand.Address)
				opGridAddress = ((IOperand.Address) opGrid.getOp ()).getRegBase ();
						
			// mask the last log2(nSIMDVectorLengthInBytes) bits of the address opGridAddress
			l.addInstruction (new Instruction ("mov", opGridAddress, regCounter));
			l.addInstructions (opGrid.getInstrPost ());
			l.addInstruction (new Instruction ("add", new IOperand.Immediate (nSIMDVectorLengthInBytes - 1), regCounter));
			l.addInstruction (new Instruction ("and", new IOperand.Immediate (nSIMDVectorLengthInBytes - 1), regCounter));
			
			// compute (nSIMDVectorLengthInBytes - bits) / sizeof (datatype)
			// this is the number of elements we need to compute in the prologue
			l.addInstruction (new Instruction ("sub", new IOperand.Immediate (nSIMDVectorLengthInBytes), regCounter));
			l.addInstruction (new Instruction ("neg", regCounter));
			l.addInstruction (new Instruction ("shr", new IOperand.Immediate (MathUtil.log2 (getBaseTypeSize ())), regCounter));
			
			// make sure that this computed number of elements doesn't exceed the actual number of elements
			// (compute min(#elts, #actual num elts))
			l.addInstruction (new Instruction ("cmp", regCounter, as.getInput (InnermostLoopCodeGenerator.INPUT_LOOPTRIPCOUNT)));
			if (USE_CMOV)
			{
				// http://www.masm32.com/board/index.php?PHPSESSID=b5aabf9f9b1249cce5f3610ddaff70cc&topic=18246.0
				l.addInstruction (new Instruction ("cmovng", as.getInput (InnermostLoopCodeGenerator.INPUT_LOOPTRIPCOUNT), regCounter));
			}
			else
			{
				l.addInstruction (new Instruction ("jg",  Label.getLabelOperand (LABEL_PROLOGHDR_LESSTHANMAX)));
				l.addInstruction (new Instruction ("mov", as.getInput (InnermostLoopCodeGenerator.INPUT_LOOPTRIPCOUNT), regCounter));
			}
			
			// check whether there is work to do, otherwise jump to the main loop
			if (!USE_CMOV)
				l.addInstruction (Label.getLabel (LABEL_PROLOGHDR_LESSTHANMAX));
			
			l.addInstruction (new Instruction ("mov", regCounter, m_regPrologLength));
			l.addInstruction (new Instruction ("or",  regCounter, regCounter));
			l.addInstruction (new Instruction ("jz",  Label.getLabelOperand (LABEL_UNROLLEDMAINHDR)));
			
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
				l.addInstruction (new Instruction ("addq", regCounter, opGridAddrRegister));
			
			return l;
		}

		@Override
		public InstructionList generateUnrolledMainHeader ()
		{
			AssemblySection as = getAssemblySection ();
			InstructionList l = new InstructionList ();
			
			IOperand.IRegisterOperand regCounter = getCounterRegister ();
			int nSIMDVectorLength = getSIMDVectorLength ();
			int nLoopUnrollingFactor = getRuntimeOptions ().getIntValue (OPTION_INLINEASM_UNROLLFACTOR, 1);
			
			// compute the number of main loop iterations

			// restore the loop counter
			l.addInstruction (new Instruction ("mov", m_regPrologLength, regCounter));
			l.addInstruction (Label.getLabel (LABEL_UNROLLEDMAINHDR));
			
			// compute (loop_trip_count - #prolog_elts) / (simd_vec_len * unrolling_factor)
			// (this is the number of iterations)
			l.addInstruction (new Instruction ("sub", as.getInput (InnermostLoopCodeGenerator.INPUT_LOOPTRIPCOUNT), regCounter));
			l.addInstruction (new Instruction ("neg", regCounter));
			l.addInstruction (new Instruction ("shr", new IOperand.Immediate (MathUtil.log2 (nSIMDVectorLength * nLoopUnrollingFactor)), regCounter));
			
			// save this value
			l.addInstruction (new Instruction ("mov", regCounter, m_regMainItersCount));
			
			// check whether there is work to do; if not, jump to the cleanup
			l.addInstruction (new Instruction ("or",  regCounter, regCounter));
			l.addInstruction (new Instruction ("jz",  Label.getLabelOperand (nLoopUnrollingFactor > 1 ? LABEL_SIMPLEMAINHDR : LABEL_EPILOGHDR)));
			l.addInstruction (Label.getLabel (LABEL_UNROLLEDMAINHDR_STARTCOMPUTATION));
			l.addInstruction (new Instruction (".align 4"));
			
			l.addInstructions (generatePrefetching ());
			
			return l;
		}
		
		@Override
		public InstructionList generateUnrolledMainFooter ()
		{
			return generateMainFooter (
				LABEL_UNROLLEDMAINHDR_STARTCOMPUTATION,
				getRuntimeOptions ().getIntValue (OPTION_INLINEASM_UNROLLFACTOR, 1)
			);
		}
		
		@Override
		public InstructionList generateSimpleMainHeader ()
		{
			AssemblySection as = getAssemblySection ();
			InstructionList l = new InstructionList ();
			
			IOperand.IRegisterOperand regCounter = getCounterRegister ();
			int nSIMDVectorLength = getSIMDVectorLength ();
			int nLoopUnrollingFactor = getRuntimeOptions ().getIntValue (OPTION_INLINEASM_UNROLLFACTOR, 1);
			
			l.addInstruction (Label.getLabel (LABEL_SIMPLEMAINHDR));
			
			// compute the number of elements computed in the main loop (veclen * unroll * #iters)
			l.addInstruction (new Instruction ("mov", m_regMainItersCount, regCounter));
			l.addInstruction (new Instruction ("shl", new IOperand.Immediate (MathUtil.log2 (nSIMDVectorLength * nLoopUnrollingFactor)), regCounter));
			
			// add the number of elements computed in the prologue
			l.addInstruction (new Instruction ("add", m_regPrologLength, regCounter));
			
			// subtract the loop trip count and take the negative to get the number of remaining elements
			l.addInstruction (new Instruction ("sub", as.getInput (InnermostLoopCodeGenerator.INPUT_LOOPTRIPCOUNT), regCounter));
			l.addInstruction (new Instruction ("neg", regCounter));
			
			// process them in vectorized fashion: divide by the vector length
			l.addInstruction (new Instruction ("shr", new IOperand.Immediate (MathUtil.log2 (nSIMDVectorLength)), regCounter));

			// check whether there is work to do; if not, jump to the epilogue
			l.addInstruction (new Instruction ("or",  regCounter, regCounter));
			l.addInstruction (new Instruction ("jz",  Label.getLabelOperand (LABEL_EPILOGHDR)));
			l.addInstruction (Label.getLabel (LABEL_SIMPLEMAINHDR_STARTCOMPUTATION));		
			l.addInstruction (new Instruction (".align 4"));
			
			l.addInstructions (generatePrefetching ());

			return l;
		}
		
		@Override
		public InstructionList generateSimpleMainFooter ()
		{
			return generateMainFooter (LABEL_SIMPLEMAINHDR_STARTCOMPUTATION, 1);
		}
		
		@Override
		protected void decrementMainLoopCounterAndJump (InstructionList l, String strHeadLabel, int nLoopUnrollingFactor)
		{
			// loop: decrement the loop counter and jump to the loop head if not zero
			//l.addInstruction (new Instruction ("dec", getCounterRegister ()));
			l.addInstruction (new Instruction ("sub", new IOperand.Immediate (1), getCounterRegister ()));
			l.addInstruction (new Instruction ("jnz", Label.getLabelOperand (strHeadLabel)));
		}

		@Override
		public InstructionList generateEpilogHeader ()
		{
			StencilAssemblySection as = getAssemblySection ();
			InstructionList l = new InstructionList ();
			
			l.addInstruction (Label.getLabel (LABEL_EPILOGHDR));

			// epi_length = (INPUT_LOOPTRIPCOUNT - pro_length) (mod veclen)
			// since
			// 		INPUT_LOOPTRIPCOUNT = pro_length + k*main_length + epi_length,
			//		main_lenght = 0 (mod veclen))
			
			// we reuse the prolog length register to compute the epilog length
			IOperand regEpilogLength = m_regPrologLength;
			
			// we compute -(INPUT_LOOPTRIPCOUNT - pro_length)
			l.addInstruction (new Instruction ("sub", as.getInput (InnermostLoopCodeGenerator.INPUT_LOOPTRIPCOUNT), regEpilogLength));
			l.addInstruction (new Instruction ("and", new IOperand.Immediate (getSIMDVectorLength () - 1), regEpilogLength));

			// nothing to do if the epilogue length was 0
			//l.addInstruction (new Instruction ("or", m_regPrologLength, regEpilogLength));	// this is not needed; "and" does the job already
			l.addInstruction (new Instruction ("jz",  Label.getLabelOperand (LABEL_EPILOGHDR_ENDCOMPUTATION)));

			// adjust the pointers
			
			// we need to subtract (veclen - epi_length % veclen), but
			// veclen - epi_length % veclen = -epi_length % veclen (if the % operator maps to 0 .. veclen-1; if we use
			// 		a % b := a & (b-1)
			// where b is a power of 2, this is the case)
			
			l.addInstruction (new Instruction ("shl", new IOperand.Immediate (MathUtil.log2 (getBaseTypeSize ())), regEpilogLength));			
			for (IOperand opGridAddrRegister : as.getGrids ())
				l.addInstruction (new Instruction ("sub", regEpilogLength, opGridAddrRegister));		

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


	/**
	 * Code generator used for unaligned vector accesses (all accesses in the main loop are
	 * unaligned; no prolog loop needed) 
	 */
	protected class CodeGeneratorUnaligned extends AbstractCodeGenerator
	{
		///////////////////////////////////////////////////////////////////
		// Implementation
		
		public CodeGeneratorUnaligned (SubdomainIterator sdit,	CodeGeneratorRuntimeOptions options)
		{
			super (sdit, options);
		}

		@Override
		public InstructionList generatePrologHeader ()
		{
			// no prolog needed
			return null;
		}

		@Override
		public InstructionList generatePrologFooter ()
		{
			// no prolog needed
			return null;
		}

		@Override
		public InstructionList generateUnrolledMainHeader ()
		{
			AssemblySection as = getAssemblySection ();
			InstructionList l = new InstructionList ();
			
			IOperand.IRegisterOperand regCounter = getCounterRegister ();
			int nSIMDVectorLength = getSIMDVectorLength ();
			int nLoopUnrollingFactor = getRuntimeOptions ().getIntValue (OPTION_INLINEASM_UNROLLFACTOR, 1);
			
			// compute the number of main loop iterations

			// restore the loop counter
			l.addInstruction (new Instruction ("mov", as.getInput (InnermostLoopCodeGenerator.INPUT_LOOPTRIPCOUNT), regCounter));
			l.addInstruction (Label.getLabel (LABEL_UNROLLEDMAINHDR));
						
			// check whether there is work to do; if not, jump to the cleanup
			l.addInstruction (new Instruction ("cmp", new IOperand.Immediate (nSIMDVectorLength * nLoopUnrollingFactor), regCounter));
			l.addInstruction (new Instruction ("jl",  Label.getLabelOperand (nLoopUnrollingFactor > 1 ? LABEL_SIMPLEMAINHDR : LABEL_EPILOGHDR)));
			l.addInstruction (Label.getLabel (LABEL_UNROLLEDMAINHDR_STARTCOMPUTATION));
			l.addInstruction (new Instruction (".align 4"));
			
			return l;
		}

		@Override
		public InstructionList generateUnrolledMainFooter ()
		{
			return generateMainFooter (
				LABEL_UNROLLEDMAINHDR_STARTCOMPUTATION,
				getRuntimeOptions ().getIntValue (OPTION_INLINEASM_UNROLLFACTOR, 1)
			);
		}
		
		@Override
		public InstructionList generateSimpleMainHeader ()
		{
			InstructionList l = new InstructionList ();
			
			IOperand.IRegisterOperand regCounter = getCounterRegister ();
			
			l.addInstruction (Label.getLabel (LABEL_SIMPLEMAINHDR));
			
			// check whether there is any work to do
			l.addInstruction (new Instruction ("or", regCounter, regCounter));
			l.addInstruction (new Instruction ("jz", Label.getLabelOperand (LABEL_EPILOGHDR)));
						
			l.addInstruction (Label.getLabel (LABEL_SIMPLEMAINHDR_STARTCOMPUTATION));
			//l.addInstruction (new Instruction (".align 4"));

			return l;
		}
		
		@Override
		public InstructionList generateSimpleMainFooter ()
		{
			return generateMainFooter (LABEL_SIMPLEMAINHDR_STARTCOMPUTATION, 1);
		}

		@Override
		protected void decrementMainLoopCounterAndJump (InstructionList l, String strHeadLabel, int nLoopUnrollingFactor)
		{
			IOperand.IRegisterOperand regCounter = getCounterRegister ();
			int nSIMDVectorLength = getSIMDVectorLength ();

			l.addInstruction (new Instruction ("sub", new IOperand.Immediate (nSIMDVectorLength * nLoopUnrollingFactor), regCounter));
			l.addInstruction (new Instruction ("cmp", new IOperand.Immediate (nSIMDVectorLength * nLoopUnrollingFactor), regCounter));
			l.addInstruction (new Instruction ("jg", Label.getLabelOperand (strHeadLabel)));
		}
		
		@Override
		public InstructionList generateEpilogHeader ()
		{
			StencilAssemblySection as = getAssemblySection ();
			IOperand.IRegisterOperand regCounter = getCounterRegister ();

			InstructionList l = new InstructionList ();
			
			l.addInstruction (Label.getLabel (LABEL_EPILOGHDR));

			// check whether there is any work to do
			// if there is, adjust the pointers such that the vectors are aligned at the end of the compute domain
			// in the unit stride direction:
			// regCounter contains the number of vector elements that are left to compute =>
			// shift the pointers by -(vec_len - regCounter) = regCounter - vec_len
			l.addInstruction (new Instruction ("or", regCounter, regCounter));
			l.addInstruction (new Instruction ("jle", Label.getLabelOperand (LABEL_EPILOGHDR_ENDCOMPUTATION)));
			
			// adjust the pointers
			l.addInstruction (new Instruction ("sub", new IOperand.Immediate (getSIMDVectorLength ()), regCounter));
			
			for (IOperand opGridAddrRegister : as.getGrids ())
				l.addInstruction (new Instruction ("add", regCounter, opGridAddrRegister));			

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
		return hasAlignmentRestrictions () ?
			new X86_64InnermostLoopCodeGenerator.CodeGeneratorAligned (sdit, options) :
			new X86_64InnermostLoopCodeGenerator.CodeGeneratorUnaligned (sdit, options);
	}
}
