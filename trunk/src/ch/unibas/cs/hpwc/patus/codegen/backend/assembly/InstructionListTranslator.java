package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import org.apache.log4j.Logger;

import cetus.hir.Specifier;
import ch.unibas.cs.hpwc.patus.arch.IArchitectureDescription;
import ch.unibas.cs.hpwc.patus.arch.TypeBaseIntrinsicEnum;
import ch.unibas.cs.hpwc.patus.arch.TypeArchitectureType.Intrinsics.Intrinsic;
import ch.unibas.cs.hpwc.patus.codegen.Globals;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.optimize.UnneededPseudoRegistersRemover;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 * Translates an instruction list in the generic format (using base intrinsic names
 * and three-operand instructions) into an instruction list specific to an architecture.
 * 
 * @author Matthias-M. Christen
 */
public class InstructionListTranslator
{
	///////////////////////////////////////////////////////////////////
	// Constants
	
	private final static Logger LOGGER = Logger.getLogger (InstructionListTranslator.class);

	
	///////////////////////////////////////////////////////////////////
	// Inner Types
	
	private static class ArgSets
	{
		private String[] m_rgArgsGeneric;
		private Argument[] m_rgArgsSpecific;
		
		public ArgSets (String[] rgArgsGeneric, Argument[] rgArgsSpecific)
		{
			m_rgArgsGeneric = rgArgsGeneric;
			m_rgArgsSpecific = rgArgsSpecific;
		}

		public String[] getArgsGeneric ()
		{
			return m_rgArgsGeneric;
		}

		public Argument[] getArgsSpecific ()
		{
			return m_rgArgsSpecific;
		}

		@Override
		public int hashCode ()
		{
			return 31 * (31 + Arrays.hashCode (m_rgArgsGeneric)) + Arrays.hashCode (m_rgArgsSpecific);
		}

		@Override
		public boolean equals (Object obj)
		{
			if (this == obj)
				return true;
			if (obj == null)
				return false;
			if (!(obj instanceof ArgSets))
				return false;
			
			if (!Arrays.equals (m_rgArgsGeneric, ((ArgSets) obj).getArgsGeneric ()))
				return false;
			if (!Arrays.equals (m_rgArgsSpecific, ((ArgSets) obj).getArgsSpecific ()))
				return false;
			return true;
		}
	}
	
	private class ArgumentPermutationCalculator
	{
		private int[] m_rgPermutation;
		
		public ArgumentPermutationCalculator (Intrinsic intrinsic)
		{
			// check whether the argument permutation has been saved for the intrinsic
			m_rgPermutation = m_mapIntrinsicArgsPermutation.get (intrinsic.getBaseName ());
			
			if (m_rgPermutation == null)
			{			
				// if not, check whether permutation for the same generic-specific argument pair has already been computed 
				Argument[] rgArgs = Arguments.parseArguments (intrinsic.getArguments ());
				String[] rgArgsGeneric = Globals.getIntrinsicArguments (intrinsic.getBaseName ());
				ArgSets argsets = new ArgSets (rgArgsGeneric, rgArgs);
				m_rgPermutation = m_mapArgsPermutation.get (argsets);
				
				if (m_rgPermutation == null)
				{
					// if not, compute the permutation
					m_rgPermutation = new int[rgArgsGeneric.length + 1];
					
					for (int i = 0; i < rgArgsGeneric.length; i++)
						m_rgPermutation[i] = getArgNum (rgArgs, rgArgsGeneric[i]);
				}
			
				m_mapIntrinsicArgsPermutation.put (intrinsic.getBaseName (), m_rgPermutation);
			}
		}

		/**
		 * Finds the argument named <code>strArgName</code> in the specific intrinsic argument list
		 * and returns its number.
		 * @param rgArgs
		 * @param strArgName
		 * @return
		 */
		private int getArgNum (Argument[] rgArgs, String strArgName)
		{
			Argument arg = null;
			
			if (Globals.ARGNAME_LHS.equals (strArgName))
				arg = Arguments.getLHS (rgArgs);
			else if (Globals.ARGNAME_RHS.equals (strArgName))
				arg = Arguments.getRHS (rgArgs);
			else
				arg = Arguments.getNamedArgument (rgArgs, strArgName);
			
			return arg == null ? -1 : arg.getNumber ();
		}
		
		public IOperand[] permuteArguments (IOperand[] rgGenericArgs)
		{
			return null;
		}
	}

	
	///////////////////////////////////////////////////////////////////
	// Member Variables

	private InstructionList m_ilIn;
	private InstructionList m_ilOut;
	
	private IArchitectureDescription m_architecture;
	private Specifier m_specDatatype;
	private Map<String, int[]> m_mapIntrinsicArgsPermutation;
	private Map<ArgSets, int[]> m_mapArgsPermutation;
	
	
	///////////////////////////////////////////////////////////////////
	// Implementation

	private InstructionListTranslator (IArchitectureDescription arch, InstructionList ilIn, Specifier specDatatype)
	{
		m_architecture = arch;
		m_ilIn = ilIn;
		m_ilOut = new InstructionList ();
		
		m_specDatatype = specDatatype;
		m_mapIntrinsicArgsPermutation = new HashMap<String, int[]> ();
		m_mapArgsPermutation = new HashMap<InstructionListTranslator.ArgSets, int[]> ();
	}
		
	/**
	 * arg1   arg2   result   commutative  accepts_arg1  accepts_arg2  accepts_result      instruction(s)                                             Pattern No.
	 * 
	 * reg    reg    reg/mem               reg[&mem]     reg[&mem]     y reg&mem           op arg1, arg2, result                                      1
	 * mem    reg    reg/mem               reg&mem       reg[&mem]     y reg&mem           op arg1, arg2, result                                      1
	 *               reg/mem  no           reg           reg[&mem]     y reg&mem           mov arg1, tmp; op tmp, arg2, result                        2
	 *               reg/mem  yes          reg           reg&mem       y reg&mem           op arg2, arg1, result                                      3
	 *               reg/mem               reg           reg           y reg&mem           mov arg1, tmp; op tmp, arg2, result                        2
	 * reg    mem    reg/mem               reg[&mem]     reg&mem       y reg&mem           op arg1, arg2, result                                      1
	 *               reg/mem  no           reg[&mem]     reg           y reg&mem           mov arg2, tmp; op arg1, tmp, result                        4
	 *               reg/mem  yes          reg&mem       reg           y reg&mem           op arg2, arg1, result                                      3
	 *               reg/mem               reg           reg           y reg&mem           mov arg2, tmp; op arg1, tmp, result                        4
	 * mem    mem    reg/mem               reg&mem       reg&mem       y reg&mem           op arg1, arg2, result                                      1
	 *               reg/mem               reg&mem       reg           y reg&mem           mov arg2, tmp; op arg1, tmp, result                        4
	 *               reg/mem               reg           reg&mem       y reg&mem           mov arg1, tmp; op tmp, arg2, result                        2
	 *               reg/mem               reg           reg           y reg&mem           mov arg1, tmp1; mov arg2, tmp2; op tmp1, tmp2, result      5
	 * 
	 * reg    reg    mem                   reg[&mem]     reg[&mem]     y reg               op arg1, arg2, tmp; mov tmp, result                        6
	 * mem    reg    mem                   reg&mem       reg[&mem]     y reg               op arg1, arg2, tmp; mov tmp, result                        6
	 *               mem      no           reg           reg[&mem]     y reg               mov arg1, tmp1; op tmp1, arg2, tmp2; mov tmp2, result      7
	 *               mem      yes          reg           reg&mem       y reg               op arg2, arg1, tmp; mov tmp, result                        8
	 *               mem                   reg           reg           y reg               mov arg1, tmp1; op tmp1, arg2, tmp2; mov tmp2, result      7
	 * reg    mem    mem                   reg[&mem]     reg&mem       y reg               op arg1, arg2, tmp; mov tmp, result                        6
	 *               mem      no           reg[&mem]     reg           y reg               mov arg2, tmp1; op arg1, tmp1, tmp2; mov tmp2, result      9
	 *               mem      yes          reg&mem       reg           y reg               op arg2, arg1, tmp; mov tmp, result                        8
	 *               mem                   reg           reg           y reg               mov arg2, tmp1; op arg1, tmp1, tmp2; mov tmp2, result      9
	 * mem    mem    mem                   reg&mem       reg&mem       y reg               op arg1, arg2, tmp; mov tmp, result                        6
	 *               mem                   reg&mem       reg           y reg               mov arg2, tmp1; op arg1, tmp1, tmp2; mov tmp2, result      9
	 *               mem                   reg           reg&mem       y reg               mov arg1, tmp1; op tmp1, arg2, tmp2; mov tmp2, result      7
	 *               mem                   reg           reg           y reg               mov arg1, tmp1; mov arg2, tmp2; op tmp1, tmp2, tmp3; mov tmp3, result    10
	 * 
	 * [ case: arg2 == result ]
	 * reg    reg=Re reg                   reg[/mem]     reg[/mem]     n (res->arg2)       op arg1, arg2  { as above, remove "result" }
	 * 
	 * [ case: arg2 != result ]                          arg2 is written to!
	 * reg    reg!=R reg                   reg[&mem]     reg[&mem]     n (res->arg2)       mov arg2, result; op arg1, result                         11
	 * mem    reg    reg                   reg&mem       reg[&mem]     n (res->arg2)       mov arg2, result; op arg1, result                         11
	 *               reg  no               reg           reg[&mem]     n (res->arg2)       mov arg1, tmp; mov arg2, result; op tmp, result           12
	 *               reg  yes              reg           reg&mem       n (res->arg2)       mov arg1, result; op arg2, result                         13
	 *               reg                   reg           reg           n (res->arg2)       mov arg1, tmp; mov arg2, result; op tmp, result           12
	 * reg    mem    reg                   reg[&mem]     reg[&mem]     n (res->arg2)       mov arg2, result; op arg1, result                         11
	 * mem    mem    reg                   reg&mem       reg&mem       n (res->arg2)       mov arg2, result; op arg1, result                         11
	 *               reg                   reg&mem       reg           n (res->arg2)       mov arg2, result; op arg1, result                         11
	 *               reg                   reg           reg&mem       n (res->arg2)       mov arg1, tmp; mov arg2, result; op tmp, result           12
	 *               reg                   reg           reg           n (res->arg2)       mov arg1, tmp; mov arg2, result; op tmp, result           12
	 * 
	 * reg    reg!=R mem                   reg[&mem]     reg[&mem]     n (res->arg2)       mov arg2, tmp; op arg1, tmp; mov tmp, result                    14
	 * mem    reg    mem                   reg&mem       reg[&mem]     n (res->arg2)       mov arg2, tmp; op arg1, tmp; mov tmp, result                    14
	 *               mem                   reg           reg[&mem]     n (res->arg2)       mov arg1, tmp1; mov arg2, tmp2; op tmp1, tmp2; mov tmp2, result 15
	 * reg    mem    mem                   reg[&mem]     reg[&mem]     n (res->arg2)       mov arg2, tmp; op arg1, tmp; mov tmp, result                    14
	 * mem    mem    mem                   reg&mem       reg[&mem]     n (res->arg2)       mov arg2, tmp; op arg1, tmp; mov tmp, result                    14
	 *               mem                   reg           reg[&mem]     n (res->arg2)       mov arg1, tmp1; mov arg2, tmp2; op tmp1, tmp2; mov tmp2, result 15
	 * 
	 * 
	 * op a, b, c  =>  mov b, c; op a, c
	 * 
	 * op arg1, arg2, result  =>  op' arg1, arg2, result    if src(arg1)=args(op', 1) and src(arg2)=args(op', 2)
	 *                            op' arg2, arg1, result    if op commutative and src(arg2)=mem, src(arg1)=reg and args(op')="[reg/]mem,reg,=reg"
	 *                            mov arg1, tmp; op' tmp, arg2, result
	 *                            mov arg2, tmp; op' arg1, tmp, result
	 *                            mov arg1, tmp1; mov arg2, tmp2; op' tmp1, tmp2, result
	 *                            
	 * 
	 * @param intrinsic
	 * @return
	 */
	private void translateInstruction (IInstruction instruction)
	{
		// TODO: !!!!!
		// TODO: Requires additional instruction in case the instruction has no distinct output register!?

		
		// nothing to do if this is not an instance of Instruction
		if (!(instruction instanceof Instruction))
		{
			m_ilOut.addInstruction (instruction);
			return;
		}
		
		// get the original instruction
		Instruction instr = (Instruction) instruction;
		
		// try to find the intrinsic corresponding to m_strIntrinsicBaseName
		Intrinsic intrinsic = m_architecture.getIntrinsic (instr.getIntrinsicBaseName (), m_specDatatype);
		
		// if the base name doesn't correspond to an intrinsic defined in the architecture description,
		// use m_strIntrinsicBaseName as instruction mnemonic
		String strInstruction = intrinsic == null ? instr.getIntrinsicBaseName () : intrinsic.getName ();
		//boolean bIsVectorInstruction = arch.getSIMDVectorLength (specDatatype) > 1;
		
		if (intrinsic == null)
		{
			LOGGER.info (StringUtil.concat ("No intrinsic found for the instruction ", strInstruction));
			m_ilOut.addInstruction (new Instruction (strInstruction, instr.getOperands ()));
			return;
		}
		
		// get the intrinsic corresponding to the operator of the binary expression
		Argument[] rgArgs = Arguments.parseArguments (intrinsic.getArguments ());
		
		// is there an output argument?
		boolean bHasOutput = Arguments.hasOutput (rgArgs);
		
		IOperand[] rgRealOperands = new IOperand[rgArgs.length];
		
		TypeBaseIntrinsicEnum type = Globals.getIntrinsicBase (intrinsic.getBaseName ());
		if (type == null)
			throw new RuntimeException (StringUtil.concat ("No TypeBaseIntrinsicEnum for ", intrinsic.getBaseName ()));

		// convert the base intrinsic arguments to the arguments of the actual intrinsic as defined in the
		// architecture description
		String[] rgIntrinsicArgNames = Globals.getIntrinsicArguments (type);
//		for (int i = 0; i < instr.getOperands ().length; i++)
//			rgRealOperands[getArgNum (rgArgs, rgIntrinsicArgNames[i])] = instr.getOperands ()[i];
			
		IOperand opResult = new IOperand.PseudoRegister (); //isReservedRegister (rgOpRHS[i]) ? new IOperand.PseudoRegister () : rgOpRHS[i];
			
		if (bHasOutput && rgArgs.length > instr.getOperands ().length)
		{
			// the instruction requires an output operand distinct from the input operands
			int nOutputArgNum = Arguments.getOutput (rgArgs).getNumber ();
			if (nOutputArgNum != -1)
				rgRealOperands[nOutputArgNum] = opResult;
		}
		
		m_ilOut.addInstruction (new Instruction (strInstruction, rgRealOperands));
	}

	/**
	 * 
	 * @return
	 */
	private InstructionList run ()
	{
		for (IInstruction instruction : new UnneededPseudoRegistersRemover ().optimize (m_ilIn))
			translateInstruction (instruction);		
		return m_ilOut;
	}
	
	/**
	 * 
	 * @param arch
	 * @param ilIn
	 * @param specDatatype
	 * @return
	 */
	public static InstructionList translate (IArchitectureDescription arch, InstructionList ilIn, Specifier specDatatype)
	{
		InstructionListTranslator translator = new InstructionListTranslator (arch, ilIn, specDatatype);
		return translator.run ();
	}
}
