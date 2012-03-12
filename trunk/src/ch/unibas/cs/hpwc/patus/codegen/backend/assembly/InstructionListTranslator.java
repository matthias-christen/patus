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
