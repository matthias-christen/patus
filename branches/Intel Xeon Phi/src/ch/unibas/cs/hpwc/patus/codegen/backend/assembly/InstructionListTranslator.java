package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import org.apache.log4j.Logger;

import cetus.hir.Specifier;
import ch.unibas.cs.hpwc.patus.arch.ArchitectureDescriptionManager;
import ch.unibas.cs.hpwc.patus.arch.IArchitectureDescription;
import ch.unibas.cs.hpwc.patus.arch.TypeArchitectureType.Intrinsics.Intrinsic;
import ch.unibas.cs.hpwc.patus.arch.TypeBaseIntrinsicEnum;
import ch.unibas.cs.hpwc.patus.arch.TypeRegisterType;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;
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
	
	private final static int UNDEFINED = -1;

	
	///////////////////////////////////////////////////////////////////
	// Inner Types
	
//	private static class ArgSets
//	{
//		private String[] m_rgArgsGeneric;
//		private Argument[] m_rgArgsSpecific;
//		
//		public ArgSets (String[] rgArgsGeneric, Argument[] rgArgsSpecific)
//		{
//			m_rgArgsGeneric = rgArgsGeneric;
//			m_rgArgsSpecific = rgArgsSpecific;
//		}
//
//		public String[] getArgsGeneric ()
//		{
//			return m_rgArgsGeneric;
//		}
//
//		public Argument[] getArgsSpecific ()
//		{
//			return m_rgArgsSpecific;
//		}
//
//		@Override
//		public int hashCode ()
//		{
//			return 31 * (31 + Arrays.hashCode (m_rgArgsGeneric)) + Arrays.hashCode (m_rgArgsSpecific);
//		}
//
//		@Override
//		public boolean equals (Object obj)
//		{
//			if (this == obj)
//				return true;
//			if (obj == null)
//				return false;
//			if (!(obj instanceof ArgSets))
//				return false;
//			
//			if (!Arrays.equals (m_rgArgsGeneric, ((ArgSets) obj).getArgsGeneric ()))
//				return false;
//			if (!Arrays.equals (m_rgArgsSpecific, ((ArgSets) obj).getArgsSpecific ()))
//				return false;
//			return true;
//		}
//	}
//	
//	private class ArgumentPermutationCalculator
//	{
//		private int[] m_rgPermutation;
//		
//		public ArgumentPermutationCalculator (Intrinsic intrinsic)
//		{
//			// check whether the argument permutation has been saved for the intrinsic
//			m_rgPermutation = m_mapIntrinsicArgsPermutation.get (intrinsic.getBaseName ());
//			
//			if (m_rgPermutation == null)
//			{			
//				// if not, check whether permutation for the same generic-specific argument pair has already been computed 
//				Argument[] rgArgs = Arguments.parseArguments (intrinsic.getArguments ());
//				String[] rgArgsGeneric = Globals.getIntrinsicArguments (intrinsic.getBaseName ());
//				ArgSets argsets = new ArgSets (rgArgsGeneric, rgArgs);
//				m_rgPermutation = m_mapArgsPermutation.get (argsets);
//				
//				if (m_rgPermutation == null)
//				{
//					// if not, compute the permutation
//					m_rgPermutation = new int[rgArgsGeneric.length + 1];
//					
//					for (int i = 0; i < rgArgsGeneric.length; i++)
//						m_rgPermutation[i] = getArgNum (rgArgs, rgArgsGeneric[i]);
//				}
//			
//				m_mapIntrinsicArgsPermutation.put (intrinsic.getBaseName (), m_rgPermutation);
//			}
//		}
//
//		
//		public IOperand[] permuteArguments (IOperand[] rgGenericArgs)
//		{
//			return null;
//		}
//	}

	
	///////////////////////////////////////////////////////////////////
	// Member Variables
	
	private CodeGeneratorSharedObjects m_data;

	private InstructionList m_ilIn;
	private InstructionList m_ilOut;
	
	private Specifier m_specDatatype;
//	private Map<String, int[]> m_mapIntrinsicArgsPermutation;
//	private Map<ArgSets, int[]> m_mapArgsPermutation;
	
	/**
	 * Contains the registers which got reused as output registers by the {@link UnneededPseudoRegistersRemover}.
	 * This is a pass-through data structure.
	 */
	private Set<IOperand.PseudoRegister> m_setReusedRegisters;
	
	
	///////////////////////////////////////////////////////////////////
	// Implementation

	private InstructionListTranslator (CodeGeneratorSharedObjects data, InstructionList ilIn, Specifier specDatatype, Set<IOperand.PseudoRegister> setReusedRegisters)
	{
		m_data = data;
		m_ilIn = ilIn;
		m_ilOut = new InstructionList ();
		
		m_specDatatype = specDatatype;
		m_setReusedRegisters = setReusedRegisters;
		
//		m_mapIntrinsicArgsPermutation = new HashMap<String, int[]> ();
//		m_mapArgsPermutation = new HashMap<InstructionListTranslator.ArgSets, int[]> ();
	}
	
	/**
	 * Finds the intrinsic for the instruction <code>instruction</code>.
	 * 
	 * @param instruction
	 *            The instruction for which to find the corresponding intrinsic.
	 * @return The instrinsic corresponding to the instruction <code>instruction</code>
	 */
	private Intrinsic getIntrinsicForInstruction (Instruction instruction)
	{
		return m_data.getArchitectureDescription ().getIntrinsic (instruction.getInstructionName (), m_specDatatype);
	}
	
	/**
	 * Finds the index of the argument in <code>rgArgs</code> which is the output argument.
	 * 
	 * @param rgArgs
	 *            The array of arguments to search
	 * @return The index of the argument in <code>rgArgs</code> which is the output argument
	 */
	private static int getOutputArgumentIndex (Argument[] rgArgs)
	{
		Argument argOut = Arguments.getOutput (rgArgs);
		return argOut == null ? rgArgs.length - 1 : argOut.getNumber ();
	}

	/**
	 * Finds the argument named <code>strArgName</code> in the specific intrinsic argument list
	 * and returns its number.
	 * 
	 * @param rgArgs
	 *            The array of arguments
	 * @param strArgName
	 *            The generic name of the argument (defined in {@link Globals}) for which to look
	 * @return The number of the argument (i.e., the index within the <code>rgArgs</code> array), or
	 * 	{@link InstructionListTranslator#UNDEFINED} if the argument couldn't be found
	 */
	private static int getArgNum (Argument[] rgArgs, String strArgName)
	{
		Argument arg = null;
		
		if (Globals.ARGNAME_LHS.equals (strArgName))
			arg = Arguments.getLHS (rgArgs);
		else if (Globals.ARGNAME_RHS.equals (strArgName))
			arg = Arguments.getRHS (rgArgs);
		else
			arg = Arguments.getNamedArgument (rgArgs, strArgName);
		
		return arg == null ? UNDEFINED : arg.getNumber ();
	}

	/**
	 * Calculates the permutations to convert between source and destination indices
	 * (where source is the generic instruction, destination is the target architecture specific instruction).
	 * 
	 * @param intrinsic
	 *            The intrinsic
	 * @param rgSourceOps
	 *            The array of source operands of the instruction
	 * @param rgDestArgs
	 *            The array of destination operands (the intrinsic arguments)
	 * @param rgPermSourceToDest
	 *            This array will contain the source &rarr; destination permutation on
	 *            exit of the method
	 * @param rgPermDestToSource
	 *            This array will contain the destination &rarr; source permutation on
	 *            exit of the method
	 */
	private static void getArgumentPermutations (Intrinsic intrinsic, IOperand[] rgSourceOps, Argument[] rgDestArgs, int[] rgPermSourceToDest, int[] rgPermDestToSource)
	{
		Arrays.fill (rgPermSourceToDest, UNDEFINED);
		Arrays.fill (rgPermDestToSource, UNDEFINED);
		
		String[] rgArgNamesGeneric = Globals.getIntrinsicArguments (intrinsic.getBaseName ());
		
		if (rgArgNamesGeneric != null)
		{
			for (int i = 0; i < rgSourceOps.length; i++)
			{
				rgPermSourceToDest[i] = i < rgSourceOps.length - 1 ?
					InstructionListTranslator.getArgNum (rgDestArgs, rgArgNamesGeneric[i]) :
					InstructionListTranslator.getOutputArgumentIndex (rgDestArgs);				// last argument is output
					
				if (0 <= rgPermSourceToDest[i] && rgPermSourceToDest[i] < rgPermDestToSource.length)
				{
					if (rgPermDestToSource[rgPermSourceToDest[i]] == UNDEFINED)
						rgPermDestToSource[rgPermSourceToDest[i]] = i;
				}
			}
		}
		else
		{
			// no intrinsic arguments defined for this intrinsic; just return the identity permutation
			for (int i = 0; i < rgPermSourceToDest.length; i++)
				rgPermSourceToDest[i] = i;
			for (int i = 0; i < rgPermDestToSource.length; i++)
				rgPermDestToSource[i] = i;
		}
	}
	
	/**
	 * Determines whether the source (generic instruction) and destination (architecture-specific intrinsic)
	 * operands are compatible, i.e., checks the operand types (register, memory address).
	 * 
	 * @param opSource
	 *            The source operand to test
	 * @param argDest
	 *            The destination operand to test
	 * @return <code>true</code> iff both operands have the same type
	 */
	private static boolean isCompatible (IOperand opSource, Argument argDest)
	{
		if (opSource instanceof IOperand.IRegisterOperand)
			return argDest.isRegister ();
		return argDest.isMemory ();
	}
	
	private TypeBaseIntrinsicEnum getMovFpr (boolean bIsLoad, IOperand... op)
	{
		return InstructionListTranslator.getMovFpr (m_data, bIsLoad, m_specDatatype, op);
	}
	
	/**
	 * Returns an aligned/unaligned move instruction depending on the operand
	 * and, if the operand is an address, on the displacement. The method
	 * assumes that any addresses (both base addresses and indices) are aligned
	 * at vector boundaries.
	 * 
	 * @param arch
	 *            The architecture description
	 * @param specDatatype
	 *            A datatype or <code>null</code> if a generic data type is to
	 *            be used (if all SIMD vectors have the same length in bits,
	 *            <code>null</code> is fine)
	 * @param rgOperands
	 *            The operands to examine and from which to determine what move
	 *            instruction to use
	 * @return The name of the move instruction to use to move the operands
	 *         <code>rgOperands</code>
	 */
	public static TypeBaseIntrinsicEnum getMovFpr (CodeGeneratorSharedObjects data, boolean bIsLoad, Specifier specDatatype, IOperand... rgOperands)
	{
		if (data.getOptions ().isAlwaysUseNonalignedMoves ())
			return bIsLoad ? TypeBaseIntrinsicEnum.LOAD_FPR_UNALIGNED : TypeBaseIntrinsicEnum.STORE_FPR_UNALIGNED;
		
		int nVectorLength = -1;
		
		for (IOperand op : rgOperands)
		{
			if (op instanceof IOperand.Address)
			{
				// assume that base addresses are aligned at vector boundaries
				// so if the displacement is not a multiple of the vector length, we need to do an
				// unaligned load
				
				if (nVectorLength == -1)
				{
					Specifier specType = specDatatype == null ? Globals.BASE_DATATYPES[0] : specDatatype;
					nVectorLength =	data.getArchitectureDescription ().getSIMDVectorLength (specType) * ArchitectureDescriptionManager.getTypeSize (specType);
				}
				
				if ((((IOperand.Address) op).getDisplacement () % nVectorLength) != 0)
					return bIsLoad ? TypeBaseIntrinsicEnum.LOAD_FPR_UNALIGNED : TypeBaseIntrinsicEnum.STORE_FPR_UNALIGNED;
			}
		}

		return bIsLoad ? TypeBaseIntrinsicEnum.LOAD_FPR_ALIGNED : TypeBaseIntrinsicEnum.STORE_FPR_ALIGNED;
	}
	
	/**
	 * If instruction arguments are not compatible (e.g., an argument is passed by address, but the
	 * intrinsic requires a register), this method tries to find operands which can be interchanged
	 * and which make the operands compatible after interchanging.
	 * 
	 * @param intrinsic
	 *            The intrinsic to issue
	 * @param rgSourceOps
	 *            The array of operands of the original, generic instruction
	 * @param rgDestArgs
	 *            The array of intrinsic arguments
	 * @param rgPermSourceToDest
	 *            The source-to-destination permutation
	 * @param rgPermDestToSource
	 *            The destination-to-source permutation
	 * @return
	 */
	private static IOperand[] compatibilizeCommutatives (Intrinsic intrinsic,
		IOperand[] rgSourceOps, Argument[] rgDestArgs, int[] rgPermSourceToDest, int[] rgPermDestToSource,
		int nOutputArgDestIndex, boolean bIntrinsicHasSharedResult)
	{
		if (bIntrinsicHasSharedResult)
		{
			// if there is an input argument that is the same as the output argument
			// try to permute it to the "shared" position
			
			IOperand opShared = rgSourceOps[rgPermDestToSource[nOutputArgDestIndex]];
			int nCommonInOutIdx = InstructionListTranslator.getIndexOfNonSharedInputArgsResult (rgSourceOps, opShared);
			if (nCommonInOutIdx != -1)
			{
				int nOldPos = nCommonInOutIdx;
				int nNewPos = rgPermDestToSource[nOutputArgDestIndex];
				
				if (Globals.canSwapIntrinsicArguments (intrinsic.getBaseName (), nOldPos, nNewPos) &&
					InstructionListTranslator.isCompatible (rgSourceOps[nOldPos], rgDestArgs[rgPermSourceToDest[nNewPos]]) &&
					InstructionListTranslator.isCompatible (rgSourceOps[nNewPos], rgDestArgs[rgPermSourceToDest[nOldPos]]))
				{
					// do the swap
					IOperand[] rgSourceOpsSwapped = new IOperand[rgSourceOps.length];
					for (int k = 0; k < rgSourceOpsSwapped.length; k++)
					{
						if (k == nOldPos)
							rgSourceOpsSwapped[k] = rgSourceOps[nNewPos];
						else if (k == nNewPos)
							rgSourceOpsSwapped[k] = rgSourceOps[nOldPos];
						else
							rgSourceOpsSwapped[k] = rgSourceOps[k];
					}
					
					return rgSourceOpsSwapped;
				}
			}
		}
		
		for (int i = 0; i < rgSourceOps.length; i++)
		{
			if (rgPermSourceToDest[i] != UNDEFINED)
			{
				if (!InstructionListTranslator.isCompatible (rgSourceOps[i], rgDestArgs[rgPermSourceToDest[i]]))
				{
					// try to find an argument which can be swapped with this non-compatible one
					// and check whether both operands are compatible after swapping
					
					for (int j = 0; j < rgSourceOps.length - 1; j++)
					{
						if (i == j)
							continue;
						
						if (Globals.canSwapIntrinsicArguments (intrinsic.getBaseName (), i, j))
						{
							if (rgPermSourceToDest[j] != UNDEFINED &&
								InstructionListTranslator.isCompatible (rgSourceOps[i], rgDestArgs[rgPermSourceToDest[j]]) &&
								InstructionListTranslator.isCompatible (rgSourceOps[j], rgDestArgs[rgPermSourceToDest[i]]))
							{
								// assume that only one swap is to be done, i.e., there are only
								// two arguments that can be interchanged
								
								// do the swap
								IOperand[] rgSourceOpsSwapped = new IOperand[rgSourceOps.length];
								for (int k = 0; k < rgSourceOpsSwapped.length; k++)
								{
									if (k == i)
										rgSourceOpsSwapped[k] = rgSourceOps[j];
									else if (k == j)
										rgSourceOpsSwapped[k] = rgSourceOps[i];
									else
										rgSourceOpsSwapped[k] = rgSourceOps[k];
								}
								
								return rgSourceOpsSwapped;
							}
						}
					}
				}
			}
		}
		
		return rgSourceOps;
	}
	
	/**
	 * Creates the architecture-specific instructions to implement the generic instruction
	 * <code>instruction</code> for the intrinsic <code>intrinsic</code>.
	 * 
	 * @param instruction
	 *            The generic instruction
	 * @param intrinsic
	 *            The intrinsic corresponding to <code>instruction</code>
	 * @param rgSourceOps
	 *            The array of operands of the generic instruction
	 * @param rgDestArgs
	 *            The array of intrinsic arguments
	 * @param rgPermSourceToDest
	 *            The argument permutation source &rarr; destination (where source is the generic instruction,
	 *            destination is the target architecture specific instruction)
	 * @param rgPermDestToSource
	 *            The argument permutation destination &rarr; source (where source is the generic instruction,
	 *            destination is the target architecture specific instruction)
	 * @param nOutputArgDestIndex
	 *            The index of the output argument in the array of intrinsic arguments, <code>rgDestArgs</code>
	 * @param bIntrinsicHasSharedResult
	 *            <code>true</code> iff the intrinsic requires that an argument is a shared in/out
	 */
	private void createInstructions (Instruction instruction, Intrinsic intrinsic,
		IOperand[] rgSourceOps, Argument[] rgDestArgs, int[] rgPermSourceToDest, int[] rgPermDestToSource,
		int nOutputArgDestIndex, boolean bIntrinsicHasSharedResult)
	{
		// maps operands to substitute operands within the actual generated computation instruction
		Map<IOperand, IOperand> mapSubstitutions = new HashMap<> ();
		
		IOperand[] rgDestOps = new IOperand[rgDestArgs.length];
		IOperand opSourceOutput = rgSourceOps[rgSourceOps.length - 1];
		
		boolean bHasNonCompatibleResultOperand = false;
		IOperand opTmpResultOperand = null;
		
		if (bIntrinsicHasSharedResult)
		{
			// find the operand which, in the intrinsic, is both input and output
			IOperand opShared = rgSourceOps[rgPermDestToSource[nOutputArgDestIndex]];
			
			// if the respective input and the output arguments are different, move the value of the input to the result
			// the result will then be overwritten by the intrinsic
			if (!opSourceOutput.equals (opShared))
			{
				IOperand opOut = opSourceOutput;
				boolean bIsOneOfNonSharedInputArgsResult = InstructionListTranslator.getIndexOfNonSharedInputArgsResult (rgSourceOps, opShared) != -1;
				if (!(opSourceOutput instanceof IOperand.IRegisterOperand) || bIsOneOfNonSharedInputArgsResult)
				{
					bHasNonCompatibleResultOperand = true;
					opTmpResultOperand = new IOperand.PseudoRegister (TypeRegisterType.SIMD);
					opOut = opTmpResultOperand;
				}

				// opOut can replace both opShared (the input operand, which in the architecture-specific intrinsic
				// is also an output argument) and opOut (the operand, to which the result is written)
				mapSubstitutions.put (opShared, opOut);
				if (!bIsOneOfNonSharedInputArgsResult)
					mapSubstitutions.put (opSourceOutput, opOut);
				
				Instruction instrNewMov = new Instruction (getMovFpr (opShared instanceof IOperand.Address, opShared), opShared, opOut);
				instrNewMov.setParameterAssignment (instruction.getParameterAssignment ());
				translateInstruction (instrNewMov);
			}
		}
		
		// gather operands and issue move instructions for non-compatible operands
		for (int i = 0; i < rgSourceOps.length; i++)
		{
			if (rgPermSourceToDest[i] != UNDEFINED)
			{
				boolean bIsResultOperand = i == rgSourceOps.length - 1;
				
				IOperand opSubstitute = mapSubstitutions.get (rgSourceOps[i]);
				if (opSubstitute != null)
				{
					// if already a non-compatible result operand has been found,
					// substitute the corresponding operand with the temporary one
					
					rgDestOps[rgPermSourceToDest[i]] = opSubstitute;
				}
				else
				{				
					boolean bIsCompatible = isCompatible (rgSourceOps[i], rgDestArgs[rgPermSourceToDest[i]]);
					rgDestOps[rgPermSourceToDest[i]] = bIsCompatible ? rgSourceOps[i] : new IOperand.PseudoRegister (TypeRegisterType.SIMD);
					
					if (!bIsCompatible)
					{
						if (bIsResultOperand)
						{
							// this is the result operand
							// move instruction will be generated after issuing the main instruction
							
							bHasNonCompatibleResultOperand = true;
							opTmpResultOperand = rgDestOps[rgPermSourceToDest[i]];
						}
						else
						{
							// mov arg_i, tmp
							mapSubstitutions.put (rgSourceOps[i], rgDestOps[rgPermSourceToDest[i]]);
							
							Instruction instrNewMov = new Instruction (
								getMovFpr (rgSourceOps[i] instanceof IOperand.Address, rgSourceOps[i]),
								rgSourceOps[i],
								rgDestOps[rgPermSourceToDest[i]]
							);
							instrNewMov.setParameterAssignment (instruction.getParameterAssignment ());
							
							translateInstruction (instrNewMov);
						}
					}
				}
			}
		}
		
		// add the main instruction
////
//if (instruction.getIntrinsicBaseName ().equals (TypeBaseIntrinsicEnum.DIVIDE.value ()))
//{
//	IOperand.PseudoRegister opTmp = new IOperand.PseudoRegister (TypeRegisterType.SIMD);
//	m_ilOut.addInstruction (new Instruction ("vrcpps", rgDestOps[0], opTmp));
//	m_ilOut.addInstruction (new Instruction ("vmulps", rgDestOps[1], opTmp, rgDestOps[2]));
//}
//else
////
		String strInstruction = intrinsic.getName ();
		
		boolean bIsLoad = intrinsic.getBaseName ().equals (TypeBaseIntrinsicEnum.LOAD_FPR_ALIGNED.value ());
		boolean bIsStore = intrinsic.getBaseName ().equals (TypeBaseIntrinsicEnum.STORE_FPR_ALIGNED.value ());
		if (bIsLoad || bIsStore)
		{
			Intrinsic i = m_data.getArchitectureDescription ().getIntrinsic (getMovFpr (bIsLoad, rgDestOps).value (), m_specDatatype);
			if (i != null)
				strInstruction = i.getName ();
		}
		
		Instruction instrNew = new Instruction (strInstruction, instruction.getIntrinsic (), rgDestOps);
		instrNew.setParameterAssignment (instruction.getParameterAssignment ());
		m_ilOut.addInstruction (instrNew);
		
		// add a move-result instruction if needed
		if (bHasNonCompatibleResultOperand)
		{
			// mov tmp, result
			Instruction instrNewMov = new Instruction (
				getMovFpr (opTmpResultOperand instanceof IOperand.Address, opTmpResultOperand),
				opTmpResultOperand,
				rgSourceOps[rgSourceOps.length - 1]
			);
			instrNewMov.setParameterAssignment (instruction.getParameterAssignment ());
			translateInstruction (instrNewMov);
		}
	}
	
	private static int getIndexOfNonSharedInputArgsResult (IOperand[] rgSourceOps, IOperand opShared)
	{
		if (rgSourceOps.length == 0)
			return -1;
		
		IOperand opOutput = rgSourceOps[rgSourceOps.length - 1];
		for (int i = 0; i < rgSourceOps.length - 1; i++)
		{
			if (rgSourceOps[i] == opShared)
				continue;
			if (rgSourceOps[i].equals (opOutput))
				return i;
		}

		return -1;
	}

	private static Argument[] createGenericArguments (int nArgsCount)
	{
		Argument[] rgRes = new Argument[nArgsCount];
		for (int i = 0; i < nArgsCount; i++)
			rgRes[i] = new Argument ("reg/mem", i);
		return rgRes;
	}

	/**
	 * Translates the generic instruction <code>instruction</code> into an architecture-specific one.
	 * The generated instruction(s) are added to the <code>m_ilOut</code> instruction list.
	 * 
	 * @param instruction
	 *            The instruction to translate
	 */
	private void translateInstruction (Instruction instruction)
	{
		Intrinsic intrinsic = getIntrinsicForInstruction (instruction);
		if (intrinsic == null)
		{
			LOGGER.debug (StringUtil.concat ("No intrinsic found for the instruction ", instruction.getInstructionName ()));
			m_ilOut.addInstruction (instruction);
			return;
		}
		
		// get the source operands, i.e., the instruction arguments
		IOperand[] rgSourceOps = instruction.getOperands ();
		IOperand opSourceOutput = rgSourceOps[rgSourceOps.length - 1];
	
		// get the destination operands, i.e., the arguments of the architecture-specific intrinsic
		Argument[] rgDestArgs = null;
		if (intrinsic.getArguments () != null)
			rgDestArgs = Arguments.parseArguments (intrinsic.getArguments ());
		else
		{
			LOGGER.debug (StringUtil.concat ("No arguments were defined for the intrinsic ", intrinsic.getBaseName (), ". Assuming generic arguments."));
			rgDestArgs = InstructionListTranslator.createGenericArguments (rgSourceOps.length);
		}

		// check whether the number of arguments of the instruction and the intrinsic to be generated match
		// the source instruction always has a result operand (the last instruction argument); in the intrinsic 
		// the result might be merged into one of the operands, so there might be one argument less
		if (rgSourceOps.length != rgDestArgs.length && rgSourceOps.length - 1 != rgDestArgs.length)
		{
			LOGGER.error (StringUtil.concat ("The arguments for the instruction ", instruction.toString (),
				" mapped to the intrinsic ", intrinsic.getBaseName (), " don't match"));
			return;
		}
		
		boolean bIntrinsicHasSharedResult = rgSourceOps.length - 1 == rgDestArgs.length;
		int nOutputArgDestIndex = InstructionListTranslator.getOutputArgumentIndex (rgDestArgs);
		
		int[] rgPermSourceToDest = new int[rgSourceOps.length];
		int[] rgPermDestToSource = new int[rgDestArgs.length];
		InstructionListTranslator.getArgumentPermutations (intrinsic, rgSourceOps, rgDestArgs, rgPermSourceToDest, rgPermDestToSource);
				
		if (!bIntrinsicHasSharedResult)
		{
			// the intrinsic to generate has a distinct output operand
			
			// if possible, swap commutative operands if that helps saving MOVs
			rgSourceOps = InstructionListTranslator.compatibilizeCommutatives (
				intrinsic, rgSourceOps, rgDestArgs, rgPermSourceToDest, rgPermDestToSource, nOutputArgDestIndex, bIntrinsicHasSharedResult);
		}
		else
		{
			// the intrinsic to generate has an operand which is used for shared input and output,
			// i.e., upon completion of the instruction, the shared operand is overwritten with the result
			
			if (opSourceOutput.equals (rgSourceOps[rgPermDestToSource[nOutputArgDestIndex]]))
			{
				// the argument of the instruction corresponding to the shared intrinsic operand
				// is the same as the instruction's result argument 
			}
			else
			{
				if (rgSourceOps[rgSourceOps.length - 1] instanceof IOperand.IRegisterOperand)
				{
					// if possible, swap commutative operands if that helps saving MOVs
					rgSourceOps = InstructionListTranslator.compatibilizeCommutatives (
						intrinsic, rgSourceOps, rgDestArgs, rgPermSourceToDest, rgPermDestToSource, nOutputArgDestIndex, bIntrinsicHasSharedResult);
				}
			}
		}

		createInstructions (instruction, intrinsic, rgSourceOps, rgDestArgs, rgPermSourceToDest, rgPermDestToSource, nOutputArgDestIndex, bIntrinsicHasSharedResult);
	}
			
	private void translateInstruction (IInstruction instruction)
	{
		// nothing to do if this is not an instance of Instruction
		if (!(instruction instanceof Instruction))
		{
			m_ilOut.addInstruction (instruction);
			return;
		}
		
		translateInstruction ((Instruction) instruction);
	}

	/**
	 * Runs the instruction list translator.
	 * 
	 * @return The translated instruction list
	 */
	private InstructionList run ()
	{
		IArchitectureDescription arch = m_data.getArchitectureDescription ();
		
		// don't use register reusing for now (screws up the instruction scheduler)
		final boolean bUseRegisterRemover = /*true*/ false && arch.hasNonDestructiveOperations ();
		
		InstructionList ilIn = bUseRegisterRemover ?
			new UnneededPseudoRegistersRemover (arch, InstructionList.EInstructionListType.GENERIC, m_setReusedRegisters).optimize (m_ilIn) :
			m_ilIn;
		
		for (IInstruction instruction : ilIn)
			translateInstruction (instruction);
		
		return m_ilOut;
	}
	
	/**
	 * Translates the input instruction list <code>ilIn</code>, which contains generic instructions,
	 * to an instruction list containing architecture-specific instructions. The translation is defined
	 * by the architecture description <code>arch</code>.
	 * 
	 * @param arch
	 *            The architecture description defining the translation
	 * @param ilIn
	 *            The input instruction list to translate, which contains generic instructions
	 * @param specDatatype
	 *            The target data type
	 * @return A new instruction list, which is the result of the translation of the generic instruction
	 *         list <code>ilIn</code> to the architecture-specific one, for which the translation units are
	 *         provided by the architecture description <code>arch</code>
	 */
	public static InstructionList translate (CodeGeneratorSharedObjects data, InstructionList ilIn, Specifier specDatatype, Set<IOperand.PseudoRegister> setReusedRegisters)
	{
		InstructionListTranslator translator = new InstructionListTranslator (data, ilIn, specDatatype, setReusedRegisters);
		return translator.run ();
	}
	
	public static InstructionList translate (CodeGeneratorSharedObjects data, IInstruction instruction, Specifier specDatatype)
	{
		InstructionList ilIn = new InstructionList ();
		ilIn.addInstruction (instruction);
		return InstructionListTranslator.translate (data, ilIn, specDatatype, null);
	}
}
