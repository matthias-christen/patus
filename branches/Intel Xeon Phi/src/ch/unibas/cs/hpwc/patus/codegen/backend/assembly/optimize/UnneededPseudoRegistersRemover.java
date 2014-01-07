package ch.unibas.cs.hpwc.patus.codegen.backend.assembly.optimize;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import ch.unibas.cs.hpwc.patus.arch.IArchitectureDescription;
import ch.unibas.cs.hpwc.patus.arch.TypeRegisterType;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.IInstruction;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.IOperand;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.IOperand.PseudoRegister;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.analyze.InstructionListAnalyzer;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.Instruction;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.InstructionList;

/**
 * Replaces instructions of the form
 * <pre>
 * 	op arg<sub>1</sub>, arg<sub>2</sub>, ..., arg<sub>n</sub>, {pseudoreg-j}
 * </pre>
 * where one of the arguments <code>arg<sub>i</sub></code> is a pseudo register,
 * say, <code>{pseudoreg-k}</code>, by
 * <pre>
 * 	op arg<sub>1</sub>, ..., arg<sub>i-1</sub>, {pseudoreg-k}, arg<sub>i+1</sub>, ..., arg<sub>n</sub>, {pseudoreg-k}
 * </pre>
 * if this is possible (i.e., if <code>{pseudoreg-j}</code> is never read anymore
 * in the instruction list.
 * This means, that, by replaceing <code>{pseudoreg-j}</code> by <code>{pseudoreg-k}</code>
 * one pseudo register is saved.
 * 
 * <p><b>Assumes that operations are non-destructive and that the result operand is the last operand</b></p>
 * 
 * @author Matthias-M. Christen
 */
public class UnneededPseudoRegistersRemover implements IInstructionListOptimizer
{
	///////////////////////////////////////////////////////////////////
	// Member Variables
	
	private IArchitectureDescription m_arch;
	
	private InstructionList.EInstructionListType m_iltype;
	
	private Set<IOperand.PseudoRegister> m_setReusedRegisters;

	/**	
	 * A map containing the pseudo registers (values) by which a particular
	 * pseudo register (key) is substituted
	 */
	private Map<IOperand.PseudoRegister, IOperand.PseudoRegister> m_mapSubstitute;
	
	/**
	 * A map containing the set of pseudo registers which are substituted by the
	 * key pseudo register
	 */
	private Map<IOperand.PseudoRegister, Set<IOperand.PseudoRegister>> m_mapSubsitutedRegisters;
	
	
	///////////////////////////////////////////////////////////////////
	// Implementation

	public UnneededPseudoRegistersRemover (IArchitectureDescription arch, InstructionList.EInstructionListType iltype,
		Set<IOperand.PseudoRegister> setReusedRegisters)
	{
		m_arch = arch;
		m_iltype = iltype;
		
		m_setReusedRegisters = setReusedRegisters;
		
		m_mapSubstitute = new HashMap<> ();
		m_mapSubsitutedRegisters = new HashMap<> ();
	}
	
	/**
	 * Substitutes any erased pseudo registers in the operands array by the pseudo register
	 * which was substituted for the erased one.
	 * 
	 * @param rgOps
	 *            The array of instruction arguments
	 * @return The array of instruction arguments with the erased pseudo registers substituted appropriately
	 */
	private IOperand[] substitutePseudoRegisters (IOperand[] rgOps)
	{
		// note that if registers are to be substituted, we always need to return a new array
		// (if not, we would modify the original operands array in "optimize," and so the substitution map won't work)
		
		IOperand[] rgOpsNew = new IOperand[rgOps.length];
		
		for (int i = 0; i < rgOps.length; i++)
		{
			if (IOperand.PseudoRegister.isPseudoRegisterOfType (rgOps[i], TypeRegisterType.SIMD))
			{
				rgOpsNew[i] = m_mapSubstitute.get (rgOps[i]);
				if (rgOpsNew[i] == null)
					rgOpsNew[i] = rgOps[i];
			}
			else
				rgOpsNew[i] = rgOps[i];
		}
		
		return rgOpsNew;
	}
	
	/**
	 * Determines whether the instruction has a pseudo register as an input.
	 * 
	 * @param rgOps
	 *            The array of instruction arguments
	 * @return <code>true</code> iff at least one of the input arguments in the
	 *         array <code>rgOps</code> is a {@link PseudoRegister}
	 */
	private static boolean hasInputPseudoRegisters (IOperand[] rgOps)
	{
		// omit the last (=result) operand
		for (int i = 0; i < rgOps.length - 1; i++)
			if (IOperand.PseudoRegister.isPseudoRegisterOfType (rgOps[i], TypeRegisterType.SIMD))
				return true;
		return false;
	}
	
	private void addSubstitute (IOperand.PseudoRegister regOld, IOperand.PseudoRegister regNew)
	{
		m_mapSubstitute.put (regOld, regNew);
		
		Set<IOperand.PseudoRegister> set = m_mapSubsitutedRegisters.get (regNew);
		if (set == null)
			m_mapSubsitutedRegisters.put (regNew, set = new HashSet<> ());
		set.add (regOld);
		
		if (m_setReusedRegisters != null)
			m_setReusedRegisters.add (regNew);
	}	
	
	@Override
	public InstructionList optimize (InstructionList il)
	{
		InstructionList ilResult = new InstructionList ();
		int nCurrentInstructionIdx = 0;
		
		for (IInstruction instruction : il)
		{
			if (instruction instanceof Instruction && ((Instruction) instruction).getOperands ().length > 0)
			{
				Instruction instr = (Instruction) instruction;
				IOperand[] rgOps = instr.getOperands ();
				
				boolean bHasPseudoRegisters = UnneededPseudoRegistersRemover.hasInputPseudoRegisters (rgOps);
				boolean bIsResultPseudoRegister = IOperand.PseudoRegister.isPseudoRegisterOfType (rgOps[rgOps.length - 1], TypeRegisterType.SIMD);
				
				// substitute any pseudo register with pseudo registers previously marked as substitutees
				IOperand[] rgOpsNew = bHasPseudoRegisters || bIsResultPseudoRegister ? substitutePseudoRegisters (rgOps) : rgOps;
				
				// if the result operand is a pseudo register, check if there are other
				// pseudo register operands, and if there are, and the result register
				// differs from all the other pseudo register operands, try to find
				// a pseudo register operand, which is never read anymore, and substitute
				// that for the result pseudo register
				
				if (bIsResultPseudoRegister && bHasPseudoRegisters)
				{
					IOperand.PseudoRegister regNewResult = null;
					for (int i = 0; i < rgOpsNew.length - 1; i++)
					{
						if (IOperand.PseudoRegister.isPseudoRegisterOfType (rgOpsNew[i], TypeRegisterType.SIMD))
						{
							// nothing to do if the register is already the same as the output register
							if (rgOpsNew[i].equals (rgOps[rgOps.length - 1]))
								break;
							
							if (InstructionListAnalyzer.isLastRead (
								m_arch, il, m_iltype, (IOperand.PseudoRegister) rgOpsNew[i], nCurrentInstructionIdx, m_mapSubsitutedRegisters))
							{
								regNewResult = (IOperand.PseudoRegister) rgOpsNew[i];
								break;
							}
						}
					}
					
					// substitute
					if (regNewResult != null)
					{
						rgOpsNew[rgOpsNew.length - 1] = regNewResult;
						addSubstitute ((IOperand.PseudoRegister) rgOps[rgOps.length - 1], regNewResult);
					}
				}
				
				ilResult.addInstruction (new Instruction (instr.getInstructionName (), instr.getIntrinsic (), rgOpsNew));
			}
			else
				ilResult.addInstruction (instruction);
			
			nCurrentInstructionIdx++;
		}
		
		return ilResult;
	}
}