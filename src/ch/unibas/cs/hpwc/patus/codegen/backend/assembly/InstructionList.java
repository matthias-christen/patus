package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.apache.log4j.Logger;

import cetus.hir.Specifier;
import ch.unibas.cs.hpwc.patus.arch.TypeArchitectureType.Intrinsics.Intrinsic;
import ch.unibas.cs.hpwc.patus.arch.TypeBaseIntrinsicEnum;
import ch.unibas.cs.hpwc.patus.arch.TypeRegisterType;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 * 
 * @author Matthias-M. Christen
 */
public class InstructionList implements Iterable<IInstruction>
{
	///////////////////////////////////////////////////////////////////
	// Constants
	
	private final static Logger LOGGER = Logger.getLogger (InstructionList.class);

	
	///////////////////////////////////////////////////////////////////
	// Member Variables

	/**
	 * The list of instructions within this portion of the inline assembly section
	 */
	private List<IInstruction> m_listInstructions;


	private int m_nSpillArrayIndex;
		
	
	///////////////////////////////////////////////////////////////////
	// Implementation

	public InstructionList ()
	{
		m_listInstructions = new ArrayList<> ();
	}

	public void addInstruction (IInstruction instruction)
	{
		if (instruction != null)
		{
			///
//			if (instruction instanceof Instruction && !m_listInstructions.isEmpty ())
//			{
//				if ("mov".equals (((Instruction) instruction).getIntrinsicBaseName ()))
//				{
//					if (m_listInstructions.get (m_listInstructions.size () - 1).toString ().equals (instruction.toString ()))
//						return;
//				}
//			}
			///
				
			m_listInstructions.add (instruction);
		}
	}
	
	public void addInstruction (IInstruction instruction, StencilAssemblySection.OperandWithInstructions op)
	{
		addInstructions (op.getInstrPre ());
		addInstruction (instruction);
		addInstructions (op.getInstrPost ());
	}
	
	public void addInstructions (InstructionList il)
	{
		if (il != null)
		{
			for (IInstruction instr : il)
				addInstruction (instr);
		}
	}
	
	public void addInstructions (IInstruction[] rgInstructions)
	{
		if (rgInstructions != null)
		{
			for (IInstruction instr : rgInstructions)
				addInstruction (instr);
		}
	}

	@Override
	public Iterator<IInstruction> iterator ()
	{
		return m_listInstructions.iterator ();
	}
	
	/**
	 * 
	 * @param as
	 * @return
	 */
	public InstructionList allocateRegisters (AssemblySection as)
	{
		LOGGER.info ("Performing live analysis and allocating registers...");
				
		// do a live analysis
		LiveAnalysis analysis = new LiveAnalysis (this);
		Map<TypeRegisterType, LAGraph> mapGraphs = analysis.run ();
		
		List<Integer> listOffsets = new ArrayList<> ();
		m_nSpillArrayIndex = as.getConstantsAndParamsCount ();

		// allocate registers
		Map<IOperand.PseudoRegister, IOperand.IRegisterOperand> map = null;
		boolean bAllRegistersAllocated = false;
		while (!bAllRegistersAllocated)
		{
			try
			{
				// try to allocate the registers
				map = RegisterAllocator.mapPseudoRegistersToRegisters (mapGraphs, as);
				bAllRegistersAllocated = true;
			}
			catch (TooFewRegistersException e)
			{
				if (e.getRegisterType ().equals (TypeRegisterType.SIMD))
				{
					for (int i = 0; i < e.getExcessRegisterRequirement (); i++)
						spillRegisters (as, analysis, listOffsets);
					analysis.createLAGraphEdges (mapGraphs);
				}
				else
					e.printStackTrace ();
			}
		}
		
		as.addSpillMemorySpace (
			m_nSpillArrayIndex - as.getConstantsAndParamsCount (),
			as instanceof StencilAssemblySection ? ((StencilAssemblySection) as).getDatatype () : Specifier.FLOAT
		);
		
		// replace the pseudo registers by allocated registers
		return replacePseudoRegisters (map);
	}
	
	private void spillRegisters (AssemblySection as, LiveAnalysis analysis, List<Integer> listIndexOffsets)
	{
		// find the point and the register at which the corresponding register
		// is not accessed for the largest amount of time
		int[][] rgData = analysis.getLivePseudoRegisters ();
		int nMaxNoAccess = 0;
		int nNoAccessInstrIdx = 0;
		int nNoAccessRegIdx = 0;
		for (int nInstrIdx = 0; nInstrIdx < rgData.length; nInstrIdx++)
			for (int nRegIdx = 0; nRegIdx < rgData[nInstrIdx].length; nRegIdx++)
				if (rgData[nInstrIdx][nRegIdx] > nMaxNoAccess)
				{
					nMaxNoAccess = rgData[nInstrIdx][nRegIdx];
					nNoAccessInstrIdx = nInstrIdx;
					nNoAccessRegIdx = nRegIdx;
				}
		
		LOGGER.info (StringUtil.concat ("Spilling ", analysis.getPseudoRegisters ()[nNoAccessRegIdx].toString (),
			" to memory at instruction ", nNoAccessInstrIdx, "; loading back at instruction ", nNoAccessInstrIdx + nMaxNoAccess,
			" (", nMaxNoAccess, " instructions)"));
		
		// add memory operations to save and restore the register
		addSpillInstruction (as, analysis, nNoAccessInstrIdx + 1, nNoAccessRegIdx, true, listIndexOffsets);
		addSpillInstruction (as, analysis, nNoAccessInstrIdx + nMaxNoAccess, nNoAccessRegIdx, false, listIndexOffsets);
		m_nSpillArrayIndex++;
		
		// modify the analysis matrix
		rgData[nNoAccessInstrIdx][nNoAccessRegIdx] = LiveAnalysis.STATE_LIVE;
		for (int nInstrIdx = nNoAccessInstrIdx + 1; nInstrIdx < nNoAccessInstrIdx + nMaxNoAccess; nInstrIdx++)
			rgData[nInstrIdx][nNoAccessRegIdx] = LiveAnalysis.STATE_DEAD;
	}
	
	private void addSpillInstruction (AssemblySection as, LiveAnalysis analysis, int nInstrIdx, int nNoAccessRegIdx, boolean bSpillToMemory, List<Integer> listIndexOffsets)
	{
		Intrinsic intrinsic = as.getArchitectureDescription ().getIntrinsic (TypeBaseIntrinsicEnum.MOVE_FPR.value (), Specifier.FLOAT);
		
		IOperand opReg = analysis.getPseudoRegisters ()[nNoAccessRegIdx];
		IOperand opMem = new IOperand.Address (
			as.getInput (AssemblySection.INPUT_CONSTANTS_ARRAYPTR),
			m_nSpillArrayIndex * as.getArchitectureDescription ().getSIMDVectorLengthInBytes ()
		);
		
		m_listInstructions.add (
			InstructionList.getOffsetIndex (nInstrIdx, listIndexOffsets),
			new Instruction (
				intrinsic.getName (),
				bSpillToMemory ? opReg : opMem,
				bSpillToMemory ? opMem : opReg
			)
		);
		
		listIndexOffsets.add (nInstrIdx);
	}
	
	private static int getOffsetIndex (int nIdxOld, List<Integer> listIndexOffsets)
	{
		int nOffset = 0;
		for (int nInsertIndex : listIndexOffsets)
			if (nInsertIndex < nIdxOld)
				nOffset++;
		return nIdxOld + nOffset;
	}
	
	/**
	 * 
	 * @param graph
	 * @return
	 */
	public InstructionList replacePseudoRegisters (Map<IOperand.PseudoRegister, IOperand.IRegisterOperand> mapPseudoRegsToRegs)
	{
		InstructionList il = new InstructionList ();
		for (IInstruction instr : this)
		{
			IInstruction instrNew = instr;
			if (instr instanceof Instruction)
			{
				IOperand[] rgOps = ((Instruction) instr).getOperands ();
				IOperand[] rgOpsNew = null;

				// search all operands of the instruction and replace pseudo register instances by actual registers
				for (int i = 0; i < rgOps.length; i++)
				{
					IOperand opNew = null;
					
					// translate pseudo registers
					if (rgOps[i] instanceof IOperand.PseudoRegister)
						opNew = mapPseudoRegsToRegs.get (rgOps[i]);
					else if (rgOps[i] instanceof IOperand.Address)
					{
						IOperand.Address opAddr = (IOperand.Address) rgOps[i];
						
						IOperand.IRegisterOperand regBase = null;
						IOperand.IRegisterOperand regIndex = null;
						if (opAddr.getRegBase () instanceof IOperand.PseudoRegister)
							regBase = mapPseudoRegsToRegs.get (opAddr.getRegBase ());
						if (opAddr.getRegIndex () != null && (opAddr.getRegIndex () instanceof IOperand.PseudoRegister))
							regIndex = mapPseudoRegsToRegs.get (opAddr.getRegIndex ());
						
						if (regBase != null || regIndex != null)
						{
							opNew = new IOperand.Address (
								regBase == null ? opAddr.getRegBase () : regBase,
								regIndex == null ? opAddr.getRegIndex () : regIndex,
								opAddr.getScale (),
								opAddr.getDisplacement ()
							);
						}
					}

					// set the new operand to the operands array
					if (opNew != null)
					{
						// create the new operands array if it hasn't been created yet
						if (rgOpsNew == null)
						{
							rgOpsNew = new IOperand[rgOps.length];
							for (int j = 0; j < rgOps.length; j++)
								rgOpsNew[j] = rgOps[j];
						}

						rgOpsNew[i] = opNew;
					}
				}
				
				// generate a new instruction if one of the operands has been modified
				if (rgOpsNew != null)
					instrNew = new Instruction (((Instruction) instrNew).getIntrinsicBaseName (), rgOpsNew);
			}
			
			il.addInstruction (instrNew);
		}
		
		return il;
	}
	
	/**
	 * Replaces the instructions in the key set of the map
	 * <code>mapInstructionReplacements</code> by the corresponding map values.
	 * 
	 * @param mapInstructionReplacements
	 *            The map defining the mapping between old and new instruction
	 *            names
	 * @return A new instruction list with instructions replaced as defined in
	 *         the map <code>mapInstructionReplacements</code>
	 */
	public InstructionList replaceInstructions (Map<String, String> mapInstructionReplacements)
	{
		InstructionList il = new InstructionList ();
		for (IInstruction instr : this)
		{
			IInstruction instrNew = instr;
			if (instr instanceof Instruction)
			{
				String strInstrRepl = mapInstructionReplacements.get (((Instruction) instr).getIntrinsicBaseName ());
				if (strInstrRepl != null)
					instrNew = new Instruction (strInstrRepl, ((Instruction) instr).getOperands ());
			}

			il.addInstruction (instrNew);
		}
		
		return il;
	}

	/**
	 * Returns number of instructions in this instruction list.
	 * 
	 * @return The number of instructions in the instruction list
	 */
	public int size ()
	{
		return m_listInstructions.size ();
	}
	
	public boolean isEmpty ()
	{
		return m_listInstructions.isEmpty ();
	}

	@Override
	public String toString ()
	{
		StringBuilder sb = new StringBuilder ();
		for (IInstruction i : this)
		{
			sb.append (i.toString ());
			sb.append ("\n");
		}
		
		return sb.toString ();
	}
}