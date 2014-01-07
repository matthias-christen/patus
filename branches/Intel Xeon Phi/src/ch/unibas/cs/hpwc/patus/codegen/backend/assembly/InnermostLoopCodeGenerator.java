package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import cetus.hir.Expression;
import cetus.hir.IntegerLiteral;
import cetus.hir.NameID;
import cetus.hir.Specifier;
import cetus.hir.Traversable;
import ch.unibas.cs.hpwc.patus.analysis.ReuseNodesCollector;
import ch.unibas.cs.hpwc.patus.analysis.StencilAnalyzer;
import ch.unibas.cs.hpwc.patus.arch.ArchitectureDescriptionManager;
import ch.unibas.cs.hpwc.patus.arch.TypeArchitectureType.Intrinsics.Intrinsic;
import ch.unibas.cs.hpwc.patus.arch.TypeBaseIntrinsicEnum;
import ch.unibas.cs.hpwc.patus.arch.TypeRegisterType;
import ch.unibas.cs.hpwc.patus.ast.StatementListBundle;
import ch.unibas.cs.hpwc.patus.ast.SubdomainIterator;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;
import ch.unibas.cs.hpwc.patus.codegen.Globals;
import ch.unibas.cs.hpwc.patus.codegen.IInnermostLoopCodeGenerator;
import ch.unibas.cs.hpwc.patus.codegen.StencilNodeSet;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.AssemblySection.EAssemblySectionInputType;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.IOperand.IRegisterOperand;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.optimize.IInstructionListOptimizer;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.optimize.InstructionScheduleOptimizer;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.optimize.LoadStoreMover;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.optimize.MultipleMemoryLoadRemover;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.optimize.SimpleUnneededAddressLoadRemover;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;
import ch.unibas.cs.hpwc.patus.representation.Stencil;
import ch.unibas.cs.hpwc.patus.representation.StencilNode;
import ch.unibas.cs.hpwc.patus.util.ExpressionUtil;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 * 
 * @author Matthias-M. Christen
 */
public abstract class InnermostLoopCodeGenerator implements IInnermostLoopCodeGenerator
{
	///////////////////////////////////////////////////////////////////
	// Constants
	
	public final static String INPUT_LOOPTRIPCOUNT = "loop_max";
	
	public final static String OPTION_INLINEASM_UNROLLFACTOR = "iasm_unroll";
	
	/**
	 * The minimum size of a set of stencil nodes of the same grid which have the
	 * same non-unit stride coordinates, so that registers corresponding to stencil nodes
	 * get reused in the innermost loop 
	 */
	public final static int MIN_REUSE_SET_SIZE = 3;
	

	///////////////////////////////////////////////////////////////////
	// Inner Types
	
	private static class InstructionListWithAssemblySectionState
	{
		private InstructionList m_il;
		private AssemblySection.AssemblySectionState m_as;
		
		public InstructionListWithAssemblySectionState (InstructionList il, AssemblySection as)
		{
			m_il = il;
			m_as = as.getAssemblySectionState ();
		}
		
		public InstructionList getInstructionList ()
		{
			return m_il;
		}
		
		public AssemblySection.AssemblySectionState getAssemblySectionState ()
		{
			return m_as;
		}
	}
	
	protected abstract class CodeGenerator
	{
		///////////////////////////////////////////////////////////////////
		// Member Variables

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
		private Map<IOperand.IRegisterOperand, StencilNode> m_mapRegistersToReuseNodes;
		private List<List<IOperand.IRegisterOperand>> m_listReuseRegisterSets;
				
		private Map<NameID, IOperand.IRegisterOperand[]> m_mapTemporaries;
		
		private IInstructionListOptimizer[] m_rgPreTranslateOptimizers;
		private IInstructionListOptimizer[] m_rgPreRegAllocOptimizers;
		private IInstructionListOptimizer[] m_rgPostTranslateOptimizers;

		
		///////////////////////////////////////////////////////////////////
		// Implementation

		public CodeGenerator (SubdomainIterator sdit, CodeGeneratorRuntimeOptions options)
		{
			m_sdit = sdit;
			m_options = options;
			
			m_mapTemporaries = new HashMap<> ();
			
			m_assemblySection = new StencilAssemblySection (m_data, m_sdit.getIterator (), m_options);
			m_nUnrollFactor = options.getIntValue (OPTION_INLINEASM_UNROLLFACTOR, 1);
						
			// the memory will be clobbered after the stencil calculation
			m_assemblySection.setMemoryClobbered (true);
			m_assemblySection.setConditionCodesClobbered (true);
			
			// request a register to be used as loop counter
			m_regCounter = m_assemblySection.getFreeRegister (TypeRegisterType.GPR);	
			
			initializeOptimizers (true);
		}
		
		/**
		 * initialize the optimizers to execute before/after the translation of the
		 * computation instruction list.
		 */
		protected void initializeOptimizers (boolean bPerformOptimizations)
		{
			if (bPerformOptimizations)
			{
				m_rgPreTranslateOptimizers = new IInstructionListOptimizer[] {
					new MultipleMemoryLoadRemover (m_data, false),
				};
				
				m_rgPreRegAllocOptimizers = new IInstructionListOptimizer[] {
					m_data.getOptions ().getUseOptimalInstructionScheduling () ?
						new InstructionScheduleOptimizer (m_data.getArchitectureDescription (), m_assemblySection.getDatatype ()) :
						new LoadStoreMover (m_data.getArchitectureDescription ())
				};
				
				m_rgPostTranslateOptimizers = new IInstructionListOptimizer[] {
					new SimpleUnneededAddressLoadRemover ()		//new UnneededAddressLoadRemover (m_data.getArchitectureDescription ())
				};
			}
			else
			{
				m_rgPreTranslateOptimizers = new IInstructionListOptimizer[] { };
				m_rgPreRegAllocOptimizers = new IInstructionListOptimizer[] { };
				m_rgPostTranslateOptimizers = new IInstructionListOptimizer[] { };
			}
		}
		
		protected void initialize ()
		{
			// create the assembly section inputs, i.e., assign registers with grid pointers,
			// strides, and the constant array pointer
			m_assemblySection.createInputs ();

			// add the length of the inner most loop as an input
			m_assemblySection.addInput (
				INPUT_LOOPTRIPCOUNT,
				ExpressionUtil.min (
					m_sdit.getDomainSubdomain ().getSize ().getCoord (0),
					m_data.getStencilCalculation ().getDomainSize ().getSize ().getCoord (0)
				),
				EAssemblySectionInputType.CONSTANT
			);
		}
		
		/**
		 * Generates the inline assembly code for an innermost loop.
		 * 
		 * @return The generated statement list bundle
		 */
		public StatementListBundle generate ()
		{
			InstructionList il = new InstructionList ();
			
			Specifier specType = m_assemblySection.getDatatype ();

			// assign registers to the stencil nodes, which are to be reused in unit stride direction
			assignReuseRegisters (il);
			il = m_assemblySection.translate (il, specType);
			
			StatementListBundle slb = new StatementListBundle ();
			
			// generate the instruction list doing the computation
			Map<StencilNode, IOperand.IRegisterOperand> mapReuse = createReuseMap ();
										
			// generate the instructions for the computation
			InstructionList ilComputationUnrolled = generateComputation (mapReuse, m_options);
			
			// create a non-unrolled version for prologue and epilogue loops
			InstructionList ilComputationNotUnrolled = ilComputationUnrolled;
			int nUnrollingFactor = m_options.getIntValue (InnermostLoopCodeGenerator.OPTION_INLINEASM_UNROLLFACTOR);
			if (nUnrollingFactor != 1)
			{
				CodeGeneratorRuntimeOptions optNoUnroll = m_options.clone ();
				optNoUnroll.setOption (InnermostLoopCodeGenerator.OPTION_INLINEASM_UNROLLFACTOR, 1);
				ilComputationNotUnrolled = generateComputation (mapReuse, optNoUnroll);
			}
							
			// generate the loop
			Map<String, String> mapUnalignedMoves = new HashMap<> ();
			Intrinsic intrLoadFpr = m_data.getArchitectureDescription ().getIntrinsic (TypeBaseIntrinsicEnum.LOAD_FPR_ALIGNED.value (), specType);
			Intrinsic intrLoadFprUnaligned = m_data.getArchitectureDescription ().getIntrinsic (TypeBaseIntrinsicEnum.LOAD_FPR_UNALIGNED.value (), specType);
			Intrinsic intrStoreFpr = m_data.getArchitectureDescription ().getIntrinsic (TypeBaseIntrinsicEnum.STORE_FPR_ALIGNED.value (), specType);
			Intrinsic intrStoreFprUnaligned = m_data.getArchitectureDescription ().getIntrinsic (TypeBaseIntrinsicEnum.STORE_FPR_UNALIGNED.value (), specType);
			mapUnalignedMoves.put (intrLoadFpr.getName (), intrLoadFprUnaligned.getName ());
			mapUnalignedMoves.put (intrStoreFpr.getName (), intrStoreFprUnaligned.getName ());
			
			InstructionList ilComputationNotUnrolledUnaligned = ilComputationNotUnrolled.replaceInstructions (mapUnalignedMoves);
						
			// unaligned prologue
			if (hasAlignmentRestrictions ())
			{
				il.addInstruction (new Comment ("unaligned prolog"));
				il.addInstructions (m_assemblySection.translate (generatePrologHeader (), specType));
				il.addInstructions (ilComputationNotUnrolledUnaligned);
				il.addInstructions (m_assemblySection.translate (generatePrologFooter (), specType));
			}
			
			// unrolled main loop
			il.addInstruction (new Comment (hasAlignmentRestrictions () ? "unrolled aligned main loop" : "unrolled main loop"));
			il.addInstructions (m_assemblySection.translate (generateUnrolledMainHeader (), specType));
			
			if (hasAlignmentRestrictions ())
				il.addInstructions (ilComputationUnrolled);
			else
				il.addInstructions (ilComputationUnrolled.replaceInstructions (mapUnalignedMoves));
			
			il.addInstructions (m_assemblySection.translate (generateUnrolledMainFooter (), specType));
			
			// non-unrolled, aligned main computation (clean up unrolling)
			if (nUnrollingFactor > 1)
			{
				InstructionList ilSimpleMainHeader = generateSimpleMainHeader ();
				if (ilSimpleMainHeader != null)
				{
					il.addInstruction (new Comment ("aligned unrolling cleanup loop"));
					il.addInstructions (m_assemblySection.translate (ilSimpleMainHeader, specType));
					il.addInstructions (hasAlignmentRestrictions () ? ilComputationNotUnrolled : ilComputationNotUnrolledUnaligned);
					il.addInstructions (m_assemblySection.translate (generateSimpleMainFooter (), specType));
				}
			}
			
			// unaligned epilogue
			il.addInstruction (new Comment ("unaligned epilog"));
			il.addInstructions (m_assemblySection.translate (generateEpilogHeader (), specType));
			il.addInstructions (ilComputationNotUnrolledUnaligned);
			il.addInstructions (m_assemblySection.translate (generateEpilogFooter (), specType));
			
			// create the inline assembly statement
			StatementListBundle slbGenerated = m_assemblySection.generate (il, m_options);
			
			slb.addStatements (m_assemblySection.getAuxiliaryStatements ());
			slb.addStatements (slbGenerated);
			
			return slb;
		}
		
		private Map<StencilNode, IOperand.IRegisterOperand> createReuseMap ()
		{
			Map<StencilNode, IOperand.IRegisterOperand> mapReuse = new HashMap<> ();
			for (String strGrid : m_mapReuseNodesToRegisters.keySet ())
			{
				Map<StencilNode, IOperand.IRegisterOperand> map = m_mapReuseNodesToRegisters.get (strGrid);
				for (StencilNode node : map.keySet ())
					mapReuse.put (node, map.get (node));
			}
			
			return mapReuse;
		}
		
		/**
		 * Generates the instruction list doing the actual computation and
		 * immediately translates it to
		 * the architecture specific format.
		 * 
		 * @param mapReuse
		 *            The map containing the stencil node reuse registers
		 * @param options
		 *            Code generation options
		 * @return The translated instruction list implementing the computation
		 */
		private InstructionList generateComputation (Map<StencilNode, IRegisterOperand> mapReuse, CodeGeneratorRuntimeOptions options)
		{
			Map<Integer, InstructionListWithAssemblySectionState> map = m_mapCachedCodes.get (m_sdit);
			if (map == null)
				m_mapCachedCodes.put (m_sdit, map = new HashMap<> ());
			
			int nUnroll = options.getIntValue (InnermostLoopCodeGenerator.OPTION_INLINEASM_UNROLLFACTOR, 1);
			InstructionListWithAssemblySectionState cached = map.get (nUnroll);
			
			if (cached != null)
			{
				m_assemblySection.mergeAssemblySectionState (cached.getAssemblySectionState ());
				return cached.getInstructionList ();
			}
			
			m_assemblySection.reset ();
			AssemblyExpressionCodeGenerator cgExpr = new AssemblyExpressionCodeGenerator (
				m_assemblySection, m_data, mapReuse, m_mapTemporaries);

			InstructionList il = new InstructionList ();
			for (Stencil stencil : m_data.getStencilCalculation ().getStencilBundle ())
			{
				if (!StencilAnalyzer.isStencilConstant (stencil, m_data.getStencilCalculation ()))
				{
					il.addInstruction (new Comment (stencil.getStencilExpression ()));
					cgExpr.generate (stencil.getExpression (), stencil.getOutputNodes ().iterator ().next (), il, options);
				}
			}
			
			// translate the generic instruction list to the architecture-specific one
			// this also performs register allocation
			il = m_assemblySection.translate (il, m_assemblySection.getDatatype (), m_rgPreTranslateOptimizers, m_rgPreRegAllocOptimizers, m_rgPostTranslateOptimizers);
			
			map.put (nUnroll, new InstructionListWithAssemblySectionState (il, m_assemblySection));
			return il;
		}
		
		/**
		 * Assign registers to the stencil nodes, which are to be reused in unit
		 * stride direction within the innermost loop.
		 * 
		 * @param il
		 *            The instruction list to which the load instructions will
		 *            be added
		 */
		private void assignReuseRegisters (InstructionList il)
		{
			m_mapReuseNodesToRegisters = new HashMap<> ();
			m_mapRegistersToReuseNodes = new HashMap<> ();
			m_listReuseRegisterSets = new LinkedList<> ();
			
			// TODO: debug...
			if (true) return;
			
			for (StencilNodeSet set : findReuseStencilNodeSets ())
			{
				// sort nodes by unit stride direction coordinate
				List<StencilNode> listNodes = new ArrayList<> (set.size ());
				for (StencilNode n : set)
					listNodes.add (n);
				Collections.sort (listNodes, new Comparator<StencilNode> ()
				{
					@Override
					public int compare (StencilNode n1, StencilNode n2)
					{
//						return n1.getSpaceIndex ()[0] - n2.getSpaceIndex ()[0];
						Expression expr = ExpressionUtil.subtract (n1.getIndex ().getSpaceIndex (0), n2.getIndex ().getSpaceIndex (0));
						if (expr instanceof IntegerLiteral)
							return (int) ((IntegerLiteral) expr).getValue ();
						return 1;
					}
				});
				
				// request registers for reuse stencil nodes
				List<IOperand.IRegisterOperand> listSet = new LinkedList<> ();
				m_listReuseRegisterSets.add (listSet);
				
				int nPrevCoord = Integer.MIN_VALUE;
				for (StencilNode node : listNodes)
				{
					int nIdxMax = ExpressionUtil.getIntegerValue (node.getIndex ().getSpaceIndex (0));
					
					// add missing intermediates
					if (nPrevCoord != Integer.MIN_VALUE)
					{
						for (int i = nPrevCoord + 1; i < nIdxMax; i++)
						{
							IOperand.IRegisterOperand opReg = m_assemblySection.getFreeRegister (TypeRegisterType.SIMD);
							listSet.add (opReg);
							
							// load the value into the register
							StencilAssemblySection.OperandWithInstructions opGrid = m_assemblySection.getGrid (node, i - nIdxMax);
							il.addInstruction (new Instruction (TypeBaseIntrinsicEnum.LOAD_FPR_UNALIGNED, opGrid.getOp (), opReg), opGrid);
						}
					}
					
					Map<StencilNode, IOperand.IRegisterOperand> map = m_mapReuseNodesToRegisters.get (node.getName ());
					if (map == null)
						m_mapReuseNodesToRegisters.put (node.getName (), map = new HashMap<> ());
					
					// request a new register for the actual node
					IOperand.IRegisterOperand opReg = m_assemblySection.getFreeRegister (TypeRegisterType.SIMD);
					listSet.add (opReg);
					map.put (node, opReg);
					m_mapRegistersToReuseNodes.put (opReg, node);

					// load the value into the register
					StencilAssemblySection.OperandWithInstructions opGrid = m_assemblySection.getGrid (node, 0);
					if (true) throw new RuntimeException ("Handle pre and post instructions");
					il.addInstruction (new Instruction (TypeBaseIntrinsicEnum.LOAD_FPR_UNALIGNED, opGrid.getOp (), opReg), opGrid);
					
					nPrevCoord = nIdxMax;
				}
			}
		}
		
		/**
		 * Finds sets of stencil nodes which can be reused cyclically during
		 * iterating over the
		 * unit stride dimension.
		 * 
		 * @return An iterable over reuse stencil node sets
		 */
		private Iterable<StencilNodeSet> findReuseStencilNodeSets ()
		{
			// estimate the registers used
			int nAvailableRegisters = m_data.getArchitectureDescription ().getRegistersCount (TypeRegisterType.SIMD);
			
			// subtract the registers used for the calculations
			RegisterAllocator regcnt = new RegisterAllocator (m_data, m_assemblySection, null);
			int nRegsForCalculations = 0;
			for (Stencil stencil : m_data.getStencilCalculation ().getStencilBundle ())
				if (!StencilAnalyzer.isStencilConstant (stencil, m_data.getStencilCalculation ()))
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
			ReuseNodesCollector reuse = new ReuseNodesCollector (
				m_data.getStencilCalculation ().getStencilBundle ().getFusedStencil ().getAllNodes (), 0, getSIMDVectorLength ());
			reuse.addUnrollNodes (0, m_nUnrollFactor);
			return reuse.getSetsWithMaxNodesConstraint (nAvailableRegisters);
		}
		
		/**
		 * 
		 * @param il
		 */
		protected void rotateReuseRegisters (InstructionList il)
		{
			int nUnrollingFactor = m_options.getIntValue (InnermostLoopCodeGenerator.OPTION_INLINEASM_UNROLLFACTOR);
				
			for (List<IOperand.IRegisterOperand> listSet : m_listReuseRegisterSets)
			{
				IOperand.IRegisterOperand[] rgRegs = new IOperand.IRegisterOperand[listSet.size ()];
				listSet.toArray (rgRegs);
				int i = 0;
				
				// TODO: check correctness

				// swap registers
				for ( ; i < rgRegs.length - nUnrollingFactor; i++)
					il.addInstruction (new Instruction (TypeBaseIntrinsicEnum.LOAD_FPR_ALIGNED, rgRegs[i + nUnrollingFactor], rgRegs[i]));

				// load new values into the register that corresponds to the largest coordinates
				for ( ; i < rgRegs.length; i++)
				{
					StencilAssemblySection.OperandWithInstructions op = m_assemblySection.getGrid (m_mapRegistersToReuseNodes.get (rgRegs[i]), 0);
					il.addInstruction (new Instruction (TypeBaseIntrinsicEnum.LOAD_FPR_UNALIGNED, op.getOp (), rgRegs[i]), op);
				}
				
				// OLD CODE -->
//				IOperand.IRegisterOperand opPrev = null;
//				for (IOperand.IRegisterOperand op : listSet)
//				{
//					if (opPrev != null)
//						il.addInstruction (new Instruction (TypeBaseIntrinsicEnum.MOVE_FPR, op, opPrev));
//					opPrev = op;
//				}
//				
//				// load a new value into the register that corresponds to the largest coordinate
//				il.addInstruction (new Instruction (TypeBaseIntrinsicEnum.MOVE_FPR_UNALIGNED, m_assemblySection.getGrid (m_mapRegistersToReuseNodes.get (opPrev), 0), opPrev));
				// <--
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
		 * Returns the size (in Bytes) of the data type of the stencil
		 * computation.
		 * 
		 * @return The size (in Bytes) of the computation data type
		 */
		public int getBaseTypeSize ()
		{
			return ArchitectureDescriptionManager.getTypeSize (m_assemblySection.getDatatype ());
		}
		
		public IOperand.IRegisterOperand getCounterRegister ()
		{
			return m_regCounter;
		}

		/**
		 * Returns the instruction list implementing the header of the prolog
		 * loop.
		 * The prolog loop header precedes the prolog computation, which deals
		 * with non-aligned first grid points.
		 * 
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
		abstract public InstructionList generateUnrolledMainHeader ();
		
		/**
		 * 
		 * @return
		 */
		abstract public InstructionList generateUnrolledMainFooter ();
		
		/**
		 * 
		 * @return
		 */
		abstract public InstructionList generateSimpleMainHeader ();
		
		/**
		 * 
		 * @return
		 */
		abstract public InstructionList generateSimpleMainFooter ();

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

	protected CodeGeneratorSharedObjects m_data;
	
	/**
	 * Flag indicating whether the architecture supports SIMD intrinsics
	 */
	private boolean m_bArchSupportsSIMD;
	
	private boolean m_bHasAlignmentRestriction;
	
	/**
	 * Flag indicating whether the architecture supports unaligned data movement
	 */
	private boolean m_bArchSupportsUnalignedMoves;

	private Map<Traversable, Map<Integer, InstructionListWithAssemblySectionState>> m_mapCachedCodes;
	
	

	///////////////////////////////////////////////////////////////////
	// Implementation

	public InnermostLoopCodeGenerator (CodeGeneratorSharedObjects data)
	{
		m_data = data;
	
		Specifier specType = Globals.BASE_DATATYPES[0];
		m_bArchSupportsSIMD = m_data.getArchitectureDescription ().getSIMDVectorLength (specType) > 1;
		m_bArchSupportsUnalignedMoves = true;
		if (m_bArchSupportsSIMD)
		{
			m_bArchSupportsUnalignedMoves = 
				m_data.getArchitectureDescription ().getIntrinsic (TypeBaseIntrinsicEnum.LOAD_FPR_UNALIGNED.value (), specType) != null &&
				m_data.getArchitectureDescription ().getIntrinsic (TypeBaseIntrinsicEnum.STORE_FPR_UNALIGNED.value (), specType) != null;
		}
		
		// determine whether there are alignment restrictions on the vector data types
		m_bHasAlignmentRestriction = true;
		for (Specifier spec : Globals.BASE_DATATYPES)
			if (m_data.getArchitectureDescription ().getAlignmentRestriction (spec) == 1)
				m_bHasAlignmentRestriction = false;
		
		m_mapCachedCodes = new HashMap<> ();
	}
	
	@Override
	public StatementListBundle generate (Traversable sdit, CodeGeneratorRuntimeOptions options)
	{
		return newCodeGenerator ((SubdomainIterator) sdit, options).generate ();
	}
	
	/**
	 * Creates a new instance of the actual code generator.
	 * @param sdit
	 * @param options
	 * @return
	 */
	protected abstract InnermostLoopCodeGenerator.CodeGenerator newCodeGenerator (SubdomainIterator sdit, CodeGeneratorRuntimeOptions options);

	/**
	 * Determines whether the architecture supports unaligned moves of SIMD
	 * vectors.
	 * 
	 * @return <code>true</code> iff unaligned moves of SIMD vectors are
	 *         supported
	 */
	public boolean isUnalignedMoveSupported ()
	{
		return m_bArchSupportsUnalignedMoves;
	}
	
	public boolean hasAlignmentRestrictions ()
	{
		return m_bHasAlignmentRestriction;
	}

	/**
	 * Determines whether the architecture supports SIMD.
	 * 
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
