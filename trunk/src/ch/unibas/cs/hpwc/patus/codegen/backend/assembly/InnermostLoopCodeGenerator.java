package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import cetus.hir.Specifier;
import ch.unibas.cs.hpwc.patus.analysis.ReuseNodesCollector;
import ch.unibas.cs.hpwc.patus.arch.TypeBaseIntrinsicEnum;
import ch.unibas.cs.hpwc.patus.arch.TypeRegisterType;
import ch.unibas.cs.hpwc.patus.ast.SubdomainIterator;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;
import ch.unibas.cs.hpwc.patus.codegen.StencilNodeSet;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;
import ch.unibas.cs.hpwc.patus.representation.Stencil;
import ch.unibas.cs.hpwc.patus.representation.StencilNode;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 * 
 * @author Matthias-M. Christen
 */
public abstract class InnermostLoopCodeGenerator
{
	///////////////////////////////////////////////////////////////////
	// Constants
	
	protected final static String INPUT_LOOPMAX = "loop_max";
	
	protected final static String OPTION_INLINEASM_UNROLLFACTOR = "iasm_unroll";
	
	/**
	 * The minimum size of a set of stencil nodes of the same grid which have the
	 * same non-unit stride coordinates, so that registers corresponding to stencil nodes
	 * get reused in the innermost loop 
	 */
	public final static int MIN_REUSE_SET_SIZE = 3;
	
	public final static int MAX_REGISTERS_FOR_CONSTANTS = 3; 

	
	///////////////////////////////////////////////////////////////////
	// Member Variables

	private CodeGeneratorSharedObjects m_data;
	
	private CodeGeneratorRuntimeOptions m_options;
	
	/**
	 * The subdomain iterator encapsulating the innermost loop
	 */
	private SubdomainIterator m_sdit;
	
	/**
	 * The unroll factor within the innermost loop
	 */
	private int m_nUnrollFactor;
	
	/**
	 * The generated inline assembly section
	 */
	private StencilAssemblySection m_assemblySection;
	
	/**
	 * Flag indicating whether the architecture supports SIMD intrinsics
	 */
	private boolean m_bArchSupportsSIMD;
	
	/**
	 * Flag indicating whether the architecture supports unaligned data movement
	 */
	private boolean m_bArchSupportsUnalignedMoves;
	
	/**
	 * The register used as loop counter for this inner-most loop
	 */
	private IOperand.IRegisterOperand m_regCounter;
	
	private Map<String, Map<StencilNode, IOperand.IRegisterOperand>> m_mapReuseNodesToRegisters;
	private List<List<IOperand.IRegisterOperand>> m_listReuseRegisterSets;
	

	///////////////////////////////////////////////////////////////////
	// Implementation

	public InnermostLoopCodeGenerator (CodeGeneratorSharedObjects data, SubdomainIterator sdit, CodeGeneratorRuntimeOptions options)
	{
		m_data = data;
		m_sdit = sdit;
		m_options = options;
		
		m_assemblySection = new StencilAssemblySection (m_data, m_sdit.getIterator (), m_options);
		m_nUnrollFactor = options.getIntValue (OPTION_INLINEASM_UNROLLFACTOR, 1);
	
		m_bArchSupportsSIMD = m_data.getArchitectureDescription ().getSIMDVectorLength (Specifier.FLOAT) > 1;
		m_bArchSupportsUnalignedMoves = true;
		if (m_bArchSupportsSIMD)
			m_bArchSupportsUnalignedMoves = m_data.getArchitectureDescription ().getIntrinsic (TypeBaseIntrinsicEnum.MOVE_FPR_UNALIGNED.value (), Specifier.FLOAT) != null;
				
		// add the length of the inner most loop as an input
		m_assemblySection.addInput (INPUT_LOOPMAX, m_sdit.getDomainSubdomain ().getSize ().getCoord (0));
		
		// request a register to be used as loop counter
		m_regCounter = m_assemblySection.getFreeRegister (TypeRegisterType.GPR);
		
		// assign registers to the stencil nodes, which are to be reused in unit stride direction
		assignReuseRegisters ();
	}
		
	public void generate ()
	{
		
	}
	
	/**
	 * Assign registers to the stencil nodes, which are to be reused in unit stride
	 * direction within the innermost loop.
	 */
	private void assignReuseRegisters ()
	{
		m_mapReuseNodesToRegisters = new HashMap<String, Map<StencilNode, IOperand.IRegisterOperand>> ();
		m_listReuseRegisterSets = new LinkedList<List<IOperand.IRegisterOperand>> ();
		
		for (StencilNodeSet set : findReuseStencilNodeSets ())
		{
			// sort nodes by unit stride direction coordinate
			List<StencilNode> listNodes = new ArrayList<StencilNode> (set.size ());
			for (StencilNode n : set)
				listNodes.add (n);
			Collections.sort (listNodes, new Comparator<StencilNode> ()
			{
				@Override
				public int compare (StencilNode n1, StencilNode n2)
				{
					return n1.getSpaceIndex ()[0] - n2.getSpaceIndex ()[0];
				}
			});
			
			// request registers for reuse stencil nodes
			List<IOperand.IRegisterOperand> listSet = new LinkedList<IOperand.IRegisterOperand> ();
			m_listReuseRegisterSets.add (listSet);
			
			int nPrevCoord = Integer.MIN_VALUE;
			for (StencilNode node : listNodes)
			{
				// add missing intermediates
				if (nPrevCoord != Integer.MIN_VALUE)
				{
					for (int i = nPrevCoord; i < node.getSpaceIndex ()[0]; i++)
						listSet.add (m_assemblySection.getFreeRegister (TypeRegisterType.SIMD));
				}
				
				Map<StencilNode, IOperand.IRegisterOperand> map = m_mapReuseNodesToRegisters.get (node.getName ());
				if (map == null)
					m_mapReuseNodesToRegisters.put (node.getName (), map = new HashMap<StencilNode, IOperand.IRegisterOperand> ());
				
				// request a new register for the
				IOperand.IRegisterOperand opReg = m_assemblySection.getFreeRegister (TypeRegisterType.SIMD);
				listSet.add (opReg);
				map.put (node, opReg);
			}
		}
	}
	
	/**
	 * Finds sets of stencil nodes which can be reused cyclically during iterating over the
	 * unit stride dimension.
	 * @return An iterable over reuse stencil node sets
	 */
	private Iterable<StencilNodeSet> findReuseStencilNodeSets ()
	{
		// estimate the registers used
		int nAvailableRegisters = m_data.getArchitectureDescription ().getRegistersCount (TypeRegisterType.SIMD);
		
		// subtract the registers used to store constants
		if (m_assemblySection.getConstantsCount () <= MAX_REGISTERS_FOR_CONSTANTS)
			nAvailableRegisters -= m_assemblySection.getConstantsCount ();
		
		// subtract the registers used for the calculations
		RegisterAllocator regcnt = new RegisterAllocator (null);
		int nRegsForCalculations = 0;
		for (Stencil stencil : m_data.getStencilCalculation ().getStencilBundle ())
			nRegsForCalculations = Math.max (nRegsForCalculations, regcnt.countRegistersNeeded (stencil.getExpression ()));
		nAvailableRegisters -= nRegsForCalculations * m_nUnrollFactor;
		
		// error if there are too few registers
		if (nAvailableRegisters < 0)
		{
			if (m_nUnrollFactor > 1)
				throw new RuntimeException (StringUtil.concat ("Not enough registers available for the loop unrolling factor ", m_nUnrollFactor));
			else
				throw new RuntimeException ("Not enough registers available for the inline assembly code generation.");
		}
		
		// find stencil node sets to reuse within the innermost loop
		ReuseNodesCollector reuse = new ReuseNodesCollector (m_data.getStencilCalculation ().getStencilBundle ().getFusedStencil ().getAllNodes (), 0);
		return reuse.getSetsWithMaxNodesConstraint (nAvailableRegisters, m_nUnrollFactor);
	}
	
	protected CodeGeneratorRuntimeOptions getRuntimeOptions ()
	{
		return m_options;
	}
	
	public StencilAssemblySection getAssemblySection ()
	{
		return m_assemblySection;
	}
	
	public boolean isSIMDSupported ()
	{
		return m_bArchSupportsSIMD;
	}
	
	public int getSIMDVectorLength ()
	{
		return m_data.getArchitectureDescription ().getSIMDVectorLength (m_assemblySection.getDatatype ());
	}
	
	/**
	 * Returns the size (in Bytes) of the data type of the stencil computation. 
	 * @return The size (in Bytes) of the computation data type
	 */
	public int getBaseTypeSize ()
	{
		return AssemblySection.getTypeSize (m_assemblySection.getDatatype ());
	}
	
	public boolean isUnalignedMoveSupported ()
	{
		return m_bArchSupportsUnalignedMoves;
	}
	
	public IOperand.IRegisterOperand getCounterRegister ()
	{
		return m_regCounter;
	}
	
	abstract public InstructionList generatePrologHeader ();
	
	abstract public InstructionList generatePrologFooter ();
	
	abstract public InstructionList generateMainHeader ();
	
	abstract public InstructionList generateMainFooter ();
	
	abstract public InstructionList generateEpilogHeader ();
	
	abstract public InstructionList generateEpilogFooter ();
}
