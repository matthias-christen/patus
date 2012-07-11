package ch.unibas.cs.hpwc.patus.codegen.backend.assembly.optimize;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.log4j.Logger;

import cetus.hir.Specifier;
import ch.unibas.cs.hpwc.patus.arch.TypeRegisterType;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.IInstruction;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.IOperand;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.Instruction;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.InstructionList;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.InstructionListTranslator;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 * @author Matthias-M. Christen
 */
public class MultipleMemoryLoadRemover implements IInstructionListOptimizer
{
	///////////////////////////////////////////////////////////////////
	// Constants
	
	private final static int NUM_READAHEAD_INSTRUCTIONS = 4;
	
	private final static Logger LOGGER = Logger.getLogger (MultipleMemoryLoadRemover.class);


	///////////////////////////////////////////////////////////////////
	// Inner Types
	
	private class Optimizer
	{
		///////////////////////////////////////////////////////////////////
		// Member Variables
		
		private InstructionList m_ilInput;
		
		private InstructionList m_ilOutput;
		
		private IInstruction[] m_rgReadAheadInstructions;
		
		private int m_nReadAheadInstructionIdx;
		
		private Map<IOperand.Address, IOperand.IRegisterOperand> m_mapReplacements;

		
		///////////////////////////////////////////////////////////////////
		// Implementation

		public Optimizer (InstructionList il)
		{
			m_ilInput = il;
			m_ilOutput = new InstructionList ();
			
			m_rgReadAheadInstructions = new IInstruction[NUM_READAHEAD_INSTRUCTIONS];
			m_nReadAheadInstructionIdx = 0;
			
			m_mapReplacements = new HashMap<> ();
		}
		
		/**
		 * Carry out the optimization.
		 * 
		 * @return The optimized instruction list
		 */
		public InstructionList run ()
		{
			// keep a list of rotating common address sets
			List<Set<IOperand.Address>> listCommonAddrSets = new ArrayList<> ();
			for (int i = 0; i < NUM_READAHEAD_INSTRUCTIONS; i++)
				listCommonAddrSets.add (null);

			for (IInstruction instr : m_ilInput)
			{
				// feed the instruction into the buffer
				addInstructionToBuffer (instr);
				
				// check for common memory access expressions
				Set<IOperand.Address> setCurrentCommonAddrs = getCommonMemoryReferences ();
				listCommonAddrSets.set (m_nReadAheadInstructionIdx, setCurrentCommonAddrs);
				
				for (IOperand.Address opNewAddr : getNewCommonMemoryReferences (
					listCommonAddrSets.get (normalizeIndex (m_nReadAheadInstructionIdx - 1)),
					setCurrentCommonAddrs))
				{
					IOperand.IRegisterOperand opReg = new IOperand.PseudoRegister (TypeRegisterType.SIMD);
					createMovInstruction (opNewAddr, opReg);
					m_mapReplacements.put (opNewAddr, opReg);
				}

				// add an instruction to the output list
				addInstruction ();

				// remove any unused replacements, i.e., the ones not contained in the saved sets
				removeUnusedReplacements (listCommonAddrSets);
			}
			
			// empty the buffer
			for (int i = 0; i < NUM_READAHEAD_INSTRUCTIONS; i++)
			{
				addInstruction ();
				m_nReadAheadInstructionIdx = normalizeIndex (m_nReadAheadInstructionIdx + 1);
			}
			
			return m_ilOutput;
		}
		
		private int normalizeIndex (int nIndex)
		{
			int nIdx = nIndex;
			
			while (nIdx < 0)
				nIdx += NUM_READAHEAD_INSTRUCTIONS;
			while (nIdx >= NUM_READAHEAD_INSTRUCTIONS)
				nIdx -= NUM_READAHEAD_INSTRUCTIONS;

			return nIdx;
		}
		
		/**
		 * Creates a move instruction and adds it to the output instruction list
		 * <code>m_ilOutput</code>.
		 * 
		 * @param opSrc
		 *            The source operand
		 * @param opDest
		 *            The destination operand
		 */
		private void createMovInstruction (IOperand opSrc, IOperand opDest)
		{
			Instruction instr = new Instruction (InstructionListTranslator.getMovFpr (m_data, opSrc instanceof IOperand.Address, null, opSrc), opSrc, opDest);
			
			if (LOGGER.isDebugEnabled ())
				LOGGER.debug (StringUtil.concat ("Created MOV: ", instr.toString ()));
			
			if (!m_bTranslateGenerated)
				m_ilOutput.addInstruction (instr);
			else
				m_ilOutput.addInstructions (InstructionListTranslator.translate (m_data, instr, Specifier.FLOAT));
		}
			
		/**
		 * Adds the instruction <code>instr</code> to the internal buffer. This
		 * buffer is checked for multiple occurrences of memory address operands.
		 * 
		 * @param instr
		 *            The instruction to add to the buffer
		 */
		private void addInstructionToBuffer (IInstruction instr)
		{		
			m_rgReadAheadInstructions[m_nReadAheadInstructionIdx] = instr;				
			m_nReadAheadInstructionIdx = normalizeIndex (m_nReadAheadInstructionIdx + 1);
		}
		
		/**
		 * Adds the current instruction at the end of the buffer (the oldest
		 * instruction) to the instruction output list and eventually replaces
		 * memory access operands by register operands into which the value has
		 * been loaded.
		 */
		private void addInstruction ()
		{
			// add the oldest instruction in the buffer
			// (the index was incremented previously in addInstructionToBuffer, hence the oldest
			// index (before incrementing) is now 
			// (m_nReadAheadInstructionIdx-1) - NUM_READAHEAD_INSTRUCTIONS + 1 == m_nReadAheadInstructionIdx
			// (since we use the index (mod NUM_READAHEAD_INSTRUCTIONS))
			
			if (m_rgReadAheadInstructions[m_nReadAheadInstructionIdx] == null)
				return;
			
			IInstruction instr = m_rgReadAheadInstructions[m_nReadAheadInstructionIdx];
			if (instr instanceof Instruction)
			{
				IOperand[] rgOpsOld = ((Instruction) instr).getOperands ();
				IOperand[] rgOpsNew = null;
				
				for (int i = 0; i < rgOpsOld.length - 1; i++)
				{
					IOperand opReplace = m_mapReplacements.get (rgOpsOld[i]);
					if (opReplace != null)
					{
						if (rgOpsNew == null)
							rgOpsNew = new IOperand[((Instruction) instr).getOperands ().length];
						
						rgOpsNew[i] = opReplace;
					}
				}
				
				if (rgOpsNew != null)
				{
					for (int i = 0; i < rgOpsOld.length; i++)
						if (rgOpsNew[i] == null)
							rgOpsNew[i] = rgOpsOld[i];
					
					Instruction instrNew = new Instruction (((Instruction) instr).getInstructionName (), instr.getIntrinsic (), rgOpsNew);
					
					if (LOGGER.isDebugEnabled ())
						LOGGER.debug (StringUtil.concat ("Replacing: ", instr.toString (), " --> ", instrNew.toString ()));
					
					instr = instrNew;
				}
			}
			
			m_ilOutput.addInstruction (instr);
			m_rgReadAheadInstructions[m_nReadAheadInstructionIdx] = null;
		}
		
		private Iterable<IOperand.Address> getNewCommonMemoryReferences (Set<IOperand.Address> setSaved, Set<IOperand.Address> setCurrent)
		{
			Set<IOperand.Address> setNew = new HashSet<> ();
			for (IOperand.Address op : setCurrent)
				if (setSaved == null || !setSaved.contains (op))
					setNew.add (op);
			
			return setNew;
		}

		/**
		 * Gathers the memory references in the buffer
		 * <code>m_rgReadAheadInstructions</code> which occur more than once and
		 * returns them as a set.
		 * 
		 * @return A set of memory references occurring more than once in the
		 *         current buffer contents
		 */
		private Set<IOperand.Address> getCommonMemoryReferences ()
		{
			Set<IOperand.Address> setAddresses = new HashSet<> ();
			Set<IOperand.Address> setCommonAddrs = new HashSet<> ();
			Set<IOperand.IRegisterOperand> setOutputRegs = new HashSet<> ();
			
			for (IInstruction instruction : m_rgReadAheadInstructions)
			{
				if (instruction == null)
					continue;
				
				if (instruction instanceof Instruction)
				{
					// find read addresses (i.e., read input operands, but not the output operand,
					// which is at the last position)
					
					///
					if (((Instruction) instruction).getInstructionName ().equals ("mov"))
						continue;
					///
					
					IOperand[] rgOps = ((Instruction) instruction).getOperands ();
					if (rgOps.length == 0)
						continue;
					
					for (int i = 0; i < rgOps.length - 1; i++)
					{
						if (rgOps[i] instanceof IOperand.Address)
						{
							if (setAddresses.contains (rgOps[i]))
								setCommonAddrs.add ((IOperand.Address) rgOps[i]);
							else
								setAddresses.add ((IOperand.Address) rgOps[i]);
						}
					}
					if (rgOps[rgOps.length - 1] instanceof IOperand.IRegisterOperand)
						setOutputRegs.add ((IOperand.IRegisterOperand) rgOps[rgOps.length - 1]);
				}
			}
			
			// check whether registers used to calculate the addresses from which data is read
			// are modified within the window (in m_rgReadAheadInstructions), i.e., if a register
			// occurs as output register in setOutputRegs
			
			for (Iterator<IOperand.Address> it = setCommonAddrs.iterator (); it.hasNext (); )
			{
				IOperand.Address opAddr = it.next ();
				if ((opAddr.getRegBase () != null && setOutputRegs.contains (opAddr.getRegBase ())) ||
					(opAddr.getRegIndex () != null && setOutputRegs.contains (opAddr.getRegIndex ())))
				{
					it.remove ();
				}
			}
			
			return setCommonAddrs;
		}
		
		private void removeUnusedReplacements (List<Set<IOperand.Address>> listCurrentSets)
		{
			List<IOperand.Address> listToRemove = new ArrayList<> (m_mapReplacements.size ());
			
			// find the keys to remove
			for (IOperand.Address opAddr : m_mapReplacements.keySet ())
			{
				boolean bContains = false;
				for (Set<IOperand.Address> set : listCurrentSets)
				{
					if (set == null)
						continue;
					if (set.contains (opAddr))
					{
						bContains = true;
						break;
					}
				}
				
				if (!bContains)
					listToRemove.add (opAddr);
			}
			
			for (IOperand.Address op : listToRemove)
				m_mapReplacements.remove (op);
		}
	}

	
	///////////////////////////////////////////////////////////////////
	// Member Variables

	private CodeGeneratorSharedObjects m_data;
	
	/**
	 * Flag specifying whether instructions generated in this optimizer have to be translated
	 */
	private boolean m_bTranslateGenerated;

	
	///////////////////////////////////////////////////////////////////
	// Implementation

	public MultipleMemoryLoadRemover (CodeGeneratorSharedObjects data, boolean bTranslateGenerated)
	{
		m_data = data;
		m_bTranslateGenerated = bTranslateGenerated;
	}
	
	@Override
	public InstructionList optimize (InstructionList il)
	{
		return new Optimizer (il).run ();
	}
}
