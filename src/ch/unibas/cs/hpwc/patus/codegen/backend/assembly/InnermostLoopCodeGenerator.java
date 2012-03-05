package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import cetus.hir.Specifier;
import cetus.hir.Statement;
import cetus.hir.Traversable;
import ch.unibas.cs.hpwc.patus.analysis.ReuseNodesCollector;
import ch.unibas.cs.hpwc.patus.arch.TypeBaseIntrinsicEnum;
import ch.unibas.cs.hpwc.patus.arch.TypeRegisterType;
import ch.unibas.cs.hpwc.patus.ast.StatementListBundle;
import ch.unibas.cs.hpwc.patus.ast.SubdomainIterator;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;
import ch.unibas.cs.hpwc.patus.codegen.IInnermostLoopCodeGenerator;
import ch.unibas.cs.hpwc.patus.codegen.StencilNodeSet;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;
import ch.unibas.cs.hpwc.patus.representation.Stencil;
import ch.unibas.cs.hpwc.patus.representation.StencilNode;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 * 
 * @author Matthias-M. Christen
 */
public abstract class InnermostLoopCodeGenerator implements IInnermostLoopCodeGenerator
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
	
	/**
	 * The maximum number of registers to be reserved for constants
	 */
	public final static int MAX_REGISTERS_FOR_CONSTANTS = 3; 


	///////////////////////////////////////////////////////////////////
	// Inner Types
	
	protected abstract class CodeGenerator
	{
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
		 * The register used as loop counter for this inner-most loop
		 */
		private IOperand.IRegisterOperand m_regCounter;
		
		private Map<String, Map<StencilNode, IOperand.IRegisterOperand>> m_mapReuseNodesToRegisters;
		private List<List<IOperand.IRegisterOperand>> m_listReuseRegisterSets;
		
		/**
		 * Maps constant values to registers, in which the constants are stored during the computation
		 */
		private Map<Double, IOperand.IRegisterOperand> m_mapConstants;

		
		public CodeGenerator (SubdomainIterator sdit, CodeGeneratorRuntimeOptions options)
		{
			m_sdit = sdit;
			m_options = options;
			
			m_assemblySection = new StencilAssemblySection (m_data, m_sdit.getIterator (), m_options);
			m_nUnrollFactor = options.getIntValue (OPTION_INLINEASM_UNROLLFACTOR, 1);
			
			// add the length of the inner most loop as an input
			m_assemblySection.addInput (INPUT_LOOPMAX, m_sdit.getDomainSubdomain ().getSize ().getCoord (0));
			
			// request a register to be used as loop counter
			m_regCounter = m_assemblySection.getFreeRegister (TypeRegisterType.GPR);		
		}
		
		/**
		 * Generates the inline assembly code for an innermost loop.
		 * @return The generated statement list bundle
		 */
		public StatementListBundle generate ()
		{
			// assign registers to the stencil nodes, which are to be reused in unit stride direction
			assignReuseRegisters ();
			assignConstantRegisters ();
			
			StatementListBundle slb = new StatementListBundle ();
			
			// generate the instruction list doing the computation
			Map<StencilNode, IOperand.IRegisterOperand> mapReuse = new HashMap<StencilNode, IOperand.IRegisterOperand> ();
			for (String strGrid : m_mapReuseNodesToRegisters.keySet ())
			{
				Map<StencilNode, IOperand.IRegisterOperand> map = m_mapReuseNodesToRegisters.get (strGrid);
				for (StencilNode node : map.keySet ())
					mapReuse.put (node, map.get (node));
			}
			AssemblyExpressionCodeGenerator cgExpr = new AssemblyExpressionCodeGenerator (m_assemblySection, m_data, mapReuse, m_mapConstants);
			
			InstructionList listInstrComputation = new InstructionList ();
			for (Stencil stencil : m_data.getStencilCalculation ().getStencilBundle ())
				listInstrComputation.addInstructions (cgExpr.generate (stencil.getExpression (), m_options));
			
			// generate the loop
			Map<String, String> mapUnalignedMoves = new HashMap<String, String> ();
			mapUnalignedMoves.put (TypeBaseIntrinsicEnum.MOVE_FPR.value (), TypeBaseIntrinsicEnum.MOVE_FPR_UNALIGNED.value ());
			InstructionList listInstr = new InstructionList ();
			
			listInstr.addInstructions (generatePrologHeader ());
			listInstr.addInstructions (listInstrComputation.replaceInstructions (mapUnalignedMoves));
			listInstr.addInstructions (generatePrologFooter ());
			
			listInstr.addInstructions (generateMainHeader ());
			listInstr.addInstructions (listInstrComputation);
			listInstr.addInstructions (generateMainFooter ());
			
			listInstr.addInstructions (generateEpilogHeader ());
			listInstr.addInstructions (listInstrComputation.replaceInstructions (mapUnalignedMoves));
			listInstr.addInstructions (generateEpilogFooter ());
			
			// create the inline assembly statement
			Statement stmt = m_assemblySection.generate (m_options);
			
			slb.addStatements (m_assemblySection.getAuxiliaryStatements ());
			slb.addStatement (stmt);
			
			return slb;
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
			RegisterAllocator regcnt = new RegisterAllocator (m_data, m_assemblySection, null);
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
		
		/**
		 * Assigns constants to SIMD registers.
		 */
		private void assignConstantRegisters ()
		{
			m_mapConstants = new HashMap<Double, IOperand.IRegisterOperand> ();
			if (m_assemblySection.getConstantsCount () < MAX_REGISTERS_FOR_CONSTANTS)
			{
				for (double fConstant : m_assemblySection.getConstants ())
					m_mapConstants.put (fConstant, m_assemblySection.getFreeRegister (TypeRegisterType.SIMD));
			}
		}
		
		protected CodeGeneratorRuntimeOptions getRuntimeOptions ()
		{
			return m_options;
		}
		
		public StencilAssemblySection getAssemblySection ()
		{
			return m_assemblySection;
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
		
		public IOperand.IRegisterOperand getCounterRegister ()
		{
			return m_regCounter;
		}

		/**
		 * Returns the instruction list implementing the header of the prolog loop.
		 * The prolog loop header precedes the prolog computation, which deals with non-aligned
		 * first grid points.
		 * @return The prolog loop header instruction list
		 */
		abstract public InstructionList generatePrologHeader ();
		
		/**
		 * 
		 * @return
		 */
		abstract public InstructionList generatePrologFooter ();
		
		/**
		 * 
		 * @return
		 */
		abstract public InstructionList generateMainHeader ();
		
		/**
		 * 
		 * @return
		 */
		abstract public InstructionList generateMainFooter ();
		
		/**
		 * 
		 * @return
		 */
		abstract public InstructionList generateEpilogHeader ();
		
		/**
		 * 
		 * @return
		 */
		abstract public InstructionList generateEpilogFooter ();
	}

	
	///////////////////////////////////////////////////////////////////
	// Member Variables

	private CodeGeneratorSharedObjects m_data;
	
	/**
	 * Flag indicating whether the architecture supports SIMD intrinsics
	 */
	private boolean m_bArchSupportsSIMD;
	
	/**
	 * Flag indicating whether the architecture supports unaligned data movement
	 */
	private boolean m_bArchSupportsUnalignedMoves;
	
	

	///////////////////////////////////////////////////////////////////
	// Implementation

	public InnermostLoopCodeGenerator (CodeGeneratorSharedObjects data)
	{
		m_data = data;
	
		m_bArchSupportsSIMD = m_data.getArchitectureDescription ().getSIMDVectorLength (Specifier.FLOAT) > 1;
		m_bArchSupportsUnalignedMoves = true;
		if (m_bArchSupportsSIMD)
			m_bArchSupportsUnalignedMoves = m_data.getArchitectureDescription ().getIntrinsic (TypeBaseIntrinsicEnum.MOVE_FPR_UNALIGNED.value (), Specifier.FLOAT) != null;				
	}
	
	@Override
	public StatementListBundle generate (Traversable trvInput, CodeGeneratorRuntimeOptions options)
	{
		return newCodeGenerator ((SubdomainIterator) trvInput, options).generate ();
	}
	
	/**
	 * Creates a new instance of the actual code generator.
	 * @param sdit
	 * @param options
	 * @return
	 */
	protected abstract InnermostLoopCodeGenerator.CodeGenerator newCodeGenerator (SubdomainIterator sdit, CodeGeneratorRuntimeOptions options);

	/**
	 * Determines whether the architecture supports unaligned moves of SIMD vectors.
	 * @return <code>true</code> iff unaligned moves of SIMD vectors are supported
	 */
	public boolean isUnalignedMoveSupported ()
	{
		return m_bArchSupportsUnalignedMoves;
	}	

	/**
	 * Determines whether the architecture supports SIMD.
	 * @return <code>true</code> iff the architecture supports SIMD
	 */
	public boolean isSIMDSupported ()
	{
		return m_bArchSupportsSIMD;
	}
	
	@Override
	public boolean requiresAssemblySection ()
	{
		return true;
	}
}
