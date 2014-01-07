package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.log4j.Logger;

import cetus.hir.Specifier;
import ch.unibas.cs.hpwc.patus.arch.TypeArchitectureType.Intrinsics.Intrinsic;
import ch.unibas.cs.hpwc.patus.arch.TypeBaseIntrinsicEnum;
import ch.unibas.cs.hpwc.patus.arch.TypeRegisterType;
import ch.unibas.cs.hpwc.patus.ast.Parameter;
import ch.unibas.cs.hpwc.patus.ast.ParameterAssignment;
import ch.unibas.cs.hpwc.patus.ast.StatementListBundle;
import ch.unibas.cs.hpwc.patus.codegen.Globals;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.analyze.LAGraph;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.analyze.LiveAnalysis;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 * 
 * @author Matthias-M. Christen
 */
public class InstructionList implements Iterable<IInstruction>
{
	///////////////////////////////////////////////////////////////////
	// Inner Types
	
	public enum EInstructionListType
	{
		GENERIC,
		SPECIFIC
	}
	
	
	///////////////////////////////////////////////////////////////////
	// Constants
	
	private final static Logger LOGGER = Logger.getLogger (InstructionList.class);

	
	///////////////////////////////////////////////////////////////////
	// Member Variables

	/**
	 * The list of instructions within this portion of the inline assembly section
	 */
	private List<IInstruction> m_listInstructions;
	
	private Set<Parameter> m_setParameters;

	/**
	 * The current array index of the array into which register values are spilled
	 */
	private int m_nSpillArrayIndex;
		
	
	///////////////////////////////////////////////////////////////////
	// Implementation

	public InstructionList ()
	{
		m_listInstructions = new ArrayList<> ();
		
		m_setParameters = new HashSet<> ();
		m_setParameters.add (StatementListBundle.DEFAULT_PARAM.clone ());
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
			if (instruction.getParameterAssignment () != null)
				addParameter (instruction);
		}
	}
	
	private void addParameter (IInstruction instruction)
	{
		for (Parameter param : instruction.getParameterAssignment ())
			addParameter (param, instruction.getParameterAssignment ().getParameterValue (param));
	}
	
	public void addParameter (Parameter param, int nParamValue)
	{
		m_setParameters.add (param);
		param.addValue (nParamValue);
	}
	
	public void addInstruction (IInstruction instruction, StencilAssemblySection.OperandWithInstructions op)
	{
		addInstructions (op.getInstrPre ());
		addInstruction (instruction);
		addInstructions (op.getInstrPost ());
	}
	
	public void addInstructionAtTop (IInstruction instruction)
	{
		if (instruction != null)
		{
			m_listInstructions.add (0, instruction);
			if (instruction.getParameterAssignment () != null)
				addParameter (instruction);
		}
	}
	
	public void addInstructions (InstructionList il)
	{
		if (il != null)
		{
			for (IInstruction instr : il)
				addInstruction (instr);
		}
	}
	
	public void addInstructions (ParameterAssignment pa, InstructionList il)
	{
		if (pa == null)
			addInstructions (il);
		else if (il != null)
		{
			for (IInstruction instr : il)
			{
				if (instr.getParameterAssignment () == null)
					instr.setParameterAssignment (pa);

				addInstruction (instr);
			}
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
	
	public void addInstructions (ParameterAssignment pa, IInstruction[] rgInstructions)
	{
		if (pa == null)
			addInstructions (rgInstructions);
		else if (rgInstructions != null)
		{
			for (IInstruction instr : rgInstructions)
			{
				if (instr.getParameterAssignment () == null)
					instr.setParameterAssignment (pa);

				addInstruction (instr);
			}
		}
	}

	@Override
	public Iterator<IInstruction> iterator ()
	{
		return m_listInstructions.iterator ();
	}
	
	/**
	 * Returns an iterator that iterates over all instructions that match (
	 * <code>param</code>, <code>nParamValue</code>)
	 * 
	 * @param param
	 *            The parameter to match
	 * @param nParamValue
	 *            The parameter value to match
	 * @return An iterator over all instructions matching (<code>param</code>,
	 *         <code>nParamValue</code>)
	 */
	public Iterator<IInstruction> iterator (final Parameter param, final int nParamValue)
	{
		return new Iterator<IInstruction> ()
		{
			private int m_nCurIdx = 0;
			
			@Override
			public boolean hasNext ()
			{
				if (m_nCurIdx >= m_listInstructions.size ())
					return false;
				
				for (int i = m_nCurIdx; i < m_listInstructions.size (); i++)
				{
					IInstruction instr = m_listInstructions.get (i);
					if (instr.getParameterAssignment () != null)
					{
						if (instr.getParameterAssignment ().matches (param, nParamValue))
							return true;
					}
					else
						return true;
				}
				
				return false;
			}

			@Override
			public IInstruction next ()
			{
				for ( ; m_nCurIdx < m_listInstructions.size (); )
				{
					IInstruction instr = m_listInstructions.get (m_nCurIdx);
					m_nCurIdx++;
					
					if (instr.getParameterAssignment () != null)
					{
						if (instr.getParameterAssignment ().matches (param, nParamValue))
							return instr;
					}
					else
						return instr;
				}
				
				return null;
			}

			@Override
			public void remove ()
			{
				throw new RuntimeException ("Not implemented");
			}
		};
	}
	
	public Iterable<IInstruction> getInstructions (final Parameter param, final int nParamValue)
	{
		return new Iterable<IInstruction> ()
		{
			@Override
			public Iterator<IInstruction> iterator ()
			{
				return InstructionList.this.iterator (param, nParamValue);
			}
		};
	}
	
	/**
	 * Returns an iterable over all the parameters (instruction tags) used in
	 * this instruction list
	 * 
	 * @return An iterable over used parameters
	 */
	public Iterable<Parameter> getParameters ()
	{
		return m_setParameters;
	}
		
	/**
	 * 
	 * @param as
	 * @return
	 */
	public InstructionList allocateRegisters (AssemblySection as, Set<IOperand.PseudoRegister> setReusedRegisters)
	{
		LOGGER.debug ("Performing live analysis and allocating registers...");
				
		// do a live analysis
		LiveAnalysis analysis = new LiveAnalysis (as.getArchitectureDescription (), this);
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
				if (e.getRegisterType ().equals (TypeRegisterType.SIMD) || e.getRegisterType ().equals (TypeRegisterType.GPR))
				{
					for (int i = 0; i < e.getExcessRegisterRequirement (); i++)
						spillRegisters (as, analysis, listOffsets, e.getRegisterType (), setReusedRegisters);
					analysis.createLAGraphEdges (mapGraphs);
				}
				else
					e.printStackTrace ();
			}
		}
		
		as.addSpillMemorySpace (
			m_nSpillArrayIndex - as.getConstantsAndParamsCount (),
			as instanceof StencilAssemblySection ? ((StencilAssemblySection) as).getDatatype () : Globals.BASE_DATATYPES[0]
		);
		
		// replace the pseudo registers by allocated registers
		return replacePseudoRegisters (map);
	}
	
	private void spillRegisters (AssemblySection as, LiveAnalysis analysis, List<Integer> listIndexOffsets, TypeRegisterType regtype, Set<IOperand.PseudoRegister> setReusedRegisters)
	{
		// find the point and the register at which the corresponding register
		// is not accessed for the largest amount of time
		int[][] rgData = analysis.getLivePseudoRegisters ();
		int nMaxNoAccess = 0;
		int nNoAccessInstrIdx = 0;
		int nNoAccessRegIdx = 0;
		for (int nInstrIdx = 0; nInstrIdx < rgData.length; nInstrIdx++)
			for (int nRegIdx = 0; nRegIdx < rgData[nInstrIdx].length; nRegIdx++)
				if (rgData[nInstrIdx][nRegIdx] > nMaxNoAccess && analysis.getPseudoRegisters ()[nRegIdx].getRegisterType ().equals (regtype))
				{
					nMaxNoAccess = rgData[nInstrIdx][nRegIdx];
					nNoAccessInstrIdx = nInstrIdx;
					nNoAccessRegIdx = nRegIdx;
				}
		
		LOGGER.info (StringUtil.concat ("Spilling ", analysis.getPseudoRegisters ()[nNoAccessRegIdx].toString (),
			" to memory at instruction ", nNoAccessInstrIdx, "; loading back at instruction ", nNoAccessInstrIdx + nMaxNoAccess,
			" (", nMaxNoAccess, " instructions)"));
		
		if (nMaxNoAccess == 0)
			throw new RuntimeException ("Register allocation failed.");
		
		// add memory operations to save and restore the register
		if (regtype.equals (TypeRegisterType.GPR))
		{
			// TODO: currently only implemented for reloading grid addresses
			addReloadFromAddressInstruction (as, analysis, nNoAccessInstrIdx + nMaxNoAccess, nNoAccessRegIdx, listIndexOffsets);
		}
		else if (regtype.equals (TypeRegisterType.SIMD))
		{
			// check whether the register holds a constant
			IOperand.PseudoRegister opReg = analysis.getPseudoRegisters ()[nNoAccessRegIdx];
			if ((as instanceof StencilAssemblySection) && ((StencilAssemblySection) as).isConstantOrParam (opReg) && (setReusedRegisters == null || !setReusedRegisters.contains (opReg)))
			{
				// if it holds a constant, we don't need to spill out to memory, just reload the next time we use it
				addReloadFromAddressInstruction (as, analysis, nNoAccessInstrIdx + nMaxNoAccess, nNoAccessRegIdx, listIndexOffsets);
			}
			else
			{
				addSIMDSpillInstruction (as, analysis, nNoAccessInstrIdx + 1, nNoAccessRegIdx, true, listIndexOffsets);
				addSIMDSpillInstruction (as, analysis, nNoAccessInstrIdx + nMaxNoAccess, nNoAccessRegIdx, false, listIndexOffsets);
				m_nSpillArrayIndex++;
			}
		}
		
		// modify the analysis matrix
		rgData[nNoAccessInstrIdx][nNoAccessRegIdx] = LiveAnalysis.STATE_LIVE;
		for (int nInstrIdx = nNoAccessInstrIdx + 1; nInstrIdx < nNoAccessInstrIdx + nMaxNoAccess; nInstrIdx++)
			rgData[nInstrIdx][nNoAccessRegIdx] = LiveAnalysis.STATE_DEAD;
	}
	
	private void addReloadFromAddressInstruction (AssemblySection as, LiveAnalysis analysis, int nInstrIdx, int nNoAccessRegIdx, List<Integer> listIndexOffsets)
	{
		// find the first load instruction
		Instruction instrLoad = null;
		for (int i = 0; i < nInstrIdx; i++)
			if (analysis.getLivePseudoRegisters ()[i][nNoAccessRegIdx] != LiveAnalysis.STATE_DEAD)
			{
				instrLoad = (Instruction) analysis.getInstruction (i);
				break;
			}
		
		if (instrLoad == null)
		{
			LOGGER.error (StringUtil.concat ("Looking for load instruction of register ",
				analysis.getPseudoRegisters ()[nNoAccessRegIdx].toString (), ", but none found:"));
			System.out.println (analysis.toString ());
			throw new RuntimeException ("No load instruction found");
		}
		
		m_listInstructions.add (
			InstructionList.getOffsetIndex (nInstrIdx, listIndexOffsets),
			new Instruction (instrLoad)
		);
		
		listIndexOffsets.add (nInstrIdx);
	}
	
	private void addSIMDSpillInstruction (AssemblySection as, LiveAnalysis analysis, int nInstrIdx, int nNoAccessRegIdx, boolean bSpillToMemory, List<Integer> listIndexOffsets)
	{
		TypeBaseIntrinsicEnum type = bSpillToMemory ? TypeBaseIntrinsicEnum.STORE_FPR_ALIGNED : TypeBaseIntrinsicEnum.LOAD_FPR_ALIGNED; 
		Intrinsic intrinsic = as.getArchitectureDescription ().getIntrinsic (type.value (),	Specifier.FLOAT);
		
		IOperand opReg = analysis.getPseudoRegisters ()[nNoAccessRegIdx];
		IOperand opMem = new IOperand.Address (
			as.getInput (AssemblySection.INPUT_CONSTANTS_ARRAYPTR),
			m_nSpillArrayIndex * as.getArchitectureDescription ().getSIMDVectorLengthInBytes ()
		);
		
		m_listInstructions.add (
			InstructionList.getOffsetIndex (nInstrIdx, listIndexOffsets),
			new Instruction (
				intrinsic.getName (),
				type,
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
					instrNew = new Instruction (((Instruction) instrNew).getInstructionName (), instrNew.getIntrinsic (), rgOpsNew);
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
				String strInstrRepl = mapInstructionReplacements.get (((Instruction) instr).getInstructionName ());
				if (strInstrRepl != null)
					instrNew = new Instruction (strInstrRepl, instr.getIntrinsic (), ((Instruction) instr).getOperands ());
				// TODO: intrinsic should also be replaced
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
	
	public void clear ()
	{
		m_listInstructions.clear ();
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
	
	public String toStringWithoutComments ()
	{
		StringBuilder sb = new StringBuilder ();
		for (IInstruction i : this)
		{
			if (!(i instanceof Comment))
			{
				sb.append (i.toString ());
				sb.append ("\n");
			}
		}
		
		return sb.toString ();		
	}
	
	public String toJavaCode ()
	{
		Map<IOperand, String> mapOperands = new HashMap<> ();
		StringBuilder sb = new StringBuilder ();
		
		for (IInstruction i : this)
		{
			sb.append (i.toJavaCode (mapOperands));
			sb.append ('\n');
		}
		
		return sb.toString ();
	}
}
