package ch.unibas.cs.hpwc.patus.codegen.backend.assembly.x86_64;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.log4j.Logger;

import ch.unibas.cs.hpwc.patus.arch.TypeRegister;
import ch.unibas.cs.hpwc.patus.arch.TypeRegisterClass;
import ch.unibas.cs.hpwc.patus.arch.TypeRegisterType;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;
import ch.unibas.cs.hpwc.patus.codegen.StencilNodeSet;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.Comment;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.IInstruction;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.IOperand;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.Instruction;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.InstructionList;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.StencilAssemblySection;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.StencilAssemblySection.OperandWithInstructions;
import ch.unibas.cs.hpwc.patus.representation.Stencil;
import ch.unibas.cs.hpwc.patus.representation.StencilNode;
import ch.unibas.cs.hpwc.patus.util.ExpressionUtil;
import ch.unibas.cs.hpwc.patus.util.IntArray;
import ch.unibas.cs.hpwc.patus.util.MathUtil;

public class X86_64PrefetchingCodeGenerator
{
	///////////////////////////////////////////////////////////////////
	// Constants
	
	private final static Logger LOGGER = Logger.getLogger (X86_64PrefetchingCodeGenerator.class);
	
	/**
	 * The prefetch instruction
	 */
	private final static String PREFETCH_INSTR = "prefetchnta";	

	
	///////////////////////////////////////////////////////////////////
	// Member Variables

	private CodeGeneratorSharedObjects m_data;
	private StencilAssemblySection m_as;
	
	private Map<PrefetchConfig, InstructionList> m_mapGeneratedCode;
	
	
	///////////////////////////////////////////////////////////////////
	// Implementation
	
	public X86_64PrefetchingCodeGenerator (CodeGeneratorSharedObjects data, StencilAssemblySection as)
	{
		m_data = data;
		m_as = as;
		
		m_mapGeneratedCode = new HashMap<> ();
	}
	
	/**
	 * Generates prefetching code.
	 * 
	 * @param bPrefetchHigherDimensions
	 *            Flag indicating whether higher dimensions (dimensions >= 2)
	 *            should be prefetched
	 * 
	 * @return The list of instructions implementing the prefetching
	 */
	public InstructionList generate (PrefetchConfig config)
	{
		InstructionList il = m_mapGeneratedCode.get (config);
		if (il != null)
			return il;
		
		InstructionList l = new InstructionList ();
		
		// do prefetching at all?
		if (!config.doPrefetching ())
		{
			m_mapGeneratedCode.put (config, l);
			return l;
		}
		
		l.addInstruction (new Comment ("Prefetching next y-line"));
		
		// get the register with the "+y_stride"
		StencilAssemblySection.OperandWithInstructions owiYOffset = getStrideOffsetRegister (1, true, l);
		if (owiYOffset == null)
		{
			// no offset register could be found/computed
			LOGGER.warn ("No y_stride register found in the stencil (stencil does not contain a [x,y+1,...] node) or y_stride is not an address (%basereg,%offsetreg). Unable to create prefetching code.");
			m_mapGeneratedCode.put (config, l);
			return l;
		}

		IOperand.Address addrYOffset = (IOperand.Address) owiYOffset.getOp ();
		IOperand.IRegisterOperand regYBase = addrYOffset.getRegBase ();
		IOperand.IRegisterOperand regYOffset = addrYOffset.getRegIndex ();
		if (regYOffset == null)
		{
			LOGGER.warn ("No y_stride (offset) register found. Unable to create prefetching code.");
			m_mapGeneratedCode.put (config, l);
			return l;
		}
					
		// if there is a scaling in the address, apply it
		if (addrYOffset.getScale () != 1)
			l.addInstruction (new Instruction ("shl", new IOperand.Immediate (MathUtil.log2 (addrYOffset.getScale ())), regYOffset));
		
		IOperand.Register regStack = getStackPointerRegister ();
		MultiplyByConstantCodeGenerator multiplyCG = new MultiplyByConstantCodeGenerator (regYOffset, new IOperand.Address (regStack));
		
		IntArray arrEligibleScalings = new IntArray (StencilAssemblySection.ELIGIBLE_ADDRESS_SCALING_FACTORS);
		
		boolean bYOffsetRegChanged = false;
		
		// generate the prefetching code proper
		StencilNode nodePrev = null;
		int nPrevCoord = 1;
		int nPrevDir = -1;
		IOperand.IRegisterOperand regOtherDirOffset = null;
		
		for (StencilNode node : X86_64PrefetchingCodeGenerator.sortNodesForPrefetching (getPrefetchNodeSet (config)))
		{
			OperandWithInstructions owiGrid = m_as.getGrid (node, 0);

			if (isParallelToYDirection (node))
			{
				boolean bCanPrefetch = false;
				if (owiGrid.getOp () instanceof IOperand.Address)
				{
					if (((IOperand.Address) owiGrid.getOp ()).getRegBase ().equals (regYBase))
						bCanPrefetch = true;
					else if ((owiGrid.getInstrPre () == null || owiGrid.getInstrPre ().length == 0) && (owiGrid.getInstrPost () == null || owiGrid.getInstrPost ().length == 0))
						bCanPrefetch = true;
				}
				
				if (!bCanPrefetch)
					continue;
				
				// stencil node is parallel to the prefetching direction				
				Integer nCoordEx = ExpressionUtil.getIntegerValueEx (node.getIndex ().getSpaceIndex (1));
				if (nCoordEx == null)
					continue;
				int nCoord = nCoordEx + 1;
				int nScale = 1;

				if (isParallelToYDirection (nodePrev))
				{
					// this and the previous node are both parallel to the y-direction:
					// increment the register by the difference

					// note: nodePrev can't be null here (if it is, isParallelToYDirection returns false)
					
					if ((nCoord % nPrevCoord) == 0)
					{
						if (bYOffsetRegChanged)
							l.addInstruction (new Instruction ("mov", new IOperand.Address (regStack), regYOffset));

						int nQuot = nCoord / nPrevCoord;
						if (arrEligibleScalings.contains (nQuot))
						{
							// we could use the scaling factor in the address to do the multiplication
							nScale = nQuot;
							bYOffsetRegChanged = false;
						}
						else
						{
							if (config.isPrefetchOnlyAddressable ())
								continue;

							multiplyCG.generate (l, nQuot, false);
							bYOffsetRegChanged = true;
						}
					}
					else if (config.isPrefetchOnlyAddressable ())
					{
						if (nCoord - nPrevCoord <= 2)
						{
							for (int i = nPrevCoord; i < nCoord; i++)
								l.addInstruction (new Instruction ("add", new IOperand.Address (regStack), regYOffset));
							bYOffsetRegChanged = true;
						}
						else
						{
							// general case
							// load from stack and multiply
							if (bYOffsetRegChanged)
								l.addInstruction (new Instruction ("mov", new IOperand.Address (regStack), regYOffset));
							multiplyCG.generate (l, nCoord, false);
							bYOffsetRegChanged = true;
						}
					}
					else
						continue;
				}
				else
				{
					// previous node was not parallel to the y-direction, but this one is
					// either the previous node was null or parallel to the x-direction
					
					if (bYOffsetRegChanged)
						l.addInstruction (new Instruction ("mov", new IOperand.Address (regStack), regYOffset));
					
					if (nCoord == 1 || arrEligibleScalings.contains (nCoord))
					{
						// use the scaling in the address for multiplication
						nScale = nCoord;
						bYOffsetRegChanged = false;
					}
					else
					{
						if (config.isPrefetchOnlyAddressable ())
							continue;
						
						multiplyCG.generate (l, nCoord, false);
						bYOffsetRegChanged = true;
					}
				}
				
				// emit prefetching code
				l.addInstruction (new Instruction (PREFETCH_INSTR, new IOperand.Address (((IOperand.Address) owiGrid.getOp ()).getRegBase (), regYOffset, nScale)));
				
				if (bYOffsetRegChanged)
					nPrevCoord = nCoord;
			}
			else
			{
				// not parallel to the prefetching (y) direction
								
				// if pre/post instructions are required, don't do any prefetching (for now)
				if ((owiGrid.getInstrPre () != null && owiGrid.getInstrPre ().length > 0) || (owiGrid.getInstrPost () != null && owiGrid.getInstrPost ().length > 0))
					continue;
				
				if (owiGrid.getOp () instanceof IOperand.Register)
				{
					if (bYOffsetRegChanged)
						l.addInstruction (new Instruction ("mov", new IOperand.Address (regStack), regYOffset));
					l.addInstruction (new Instruction (PREFETCH_INSTR, new IOperand.Address ((IOperand.Register) owiGrid.getOp (), regYOffset)));
					bYOffsetRegChanged = false;
				}
				else if (owiGrid.getOp () instanceof IOperand.Address)
				{
					IOperand.Address opAddr = (IOperand.Address) owiGrid.getOp ();

					// find the next multiple of UNITSTRIDE_ALIGNMENT which is less or equal to opAddr.getDisplacement ()
					long nOffset = opAddr.getDisplacement () < 0 ?
						((opAddr.getDisplacement () - config.getUnitStrideAlignment () + 1) / config.getUnitStrideAlignment ()) * config.getUnitStrideAlignment () :
						(opAddr.getDisplacement () / config.getUnitStrideAlignment ()) * config.getUnitStrideAlignment ();

					if (opAddr.getRegIndex () == null)
					{
						if (bYOffsetRegChanged)
							l.addInstruction (new Instruction ("mov", new IOperand.Address (regStack), regYOffset));
												
						l.addInstruction (new Instruction (PREFETCH_INSTR, new IOperand.Address (opAddr.getRegBase (), regYOffset, 1, nOffset)));
						bYOffsetRegChanged = false;
					}
					else if (!config.isPrefetchOnlyAddressable ())
					{
						// we currently perform prefetching only for nodes parallel to axes
						// TODO: remove this restriction (if stride register for a direction exists?)
						
						int nCurDir = getParallelAxisDirection (node);
						if (config.isOmitHigherDimensions () && nCurDir >= 2)
							continue;
						
						if (nCurDir != -1)
						{
							if (nPrevDir != nCurDir)
							{
								nPrevCoord = 0;
								if (bYOffsetRegChanged)
								{
									l.addInstruction (new Instruction ("mov", new IOperand.Address (regStack), regYOffset));
									bYOffsetRegChanged = false;
								}								
								
								// get the offset register in the positive direction along the current axis
								OperandWithInstructions owi = getStrideOffsetRegister (nCurDir, false, l);
								if (owi == null)
									continue;
								if (!(owi.getOp () instanceof IOperand.Address))
									continue;
								regOtherDirOffset = ((IOperand.Address) owi.getOp ()).getRegIndex ();
								if (regOtherDirOffset == null)
									continue;
							}
							
							Integer nCoordEx = ExpressionUtil.getIntegerValueEx (node.getIndex ().getSpaceIndex (nCurDir));
							if (nCoordEx == null)
								continue;
							int nCoord = nCoordEx;
							if (nPrevCoord > nCoord)
							{
								for (int i = 0; i > nCoord; i--)
									l.addInstruction (new Instruction ("sub", regOtherDirOffset, regYOffset));
								bYOffsetRegChanged = true;
							}
							else
							{
								addNTimes (regYOffset, nCoord - nPrevCoord, regOtherDirOffset, l);
								bYOffsetRegChanged = true;
							}
							
							l.addInstruction (new Instruction (PREFETCH_INSTR, new IOperand.Address (opAddr.getRegBase (), regYOffset, 1, nOffset)));
							
							nPrevCoord = nCoord;
						}
						nPrevDir = nCurDir;
						
 						// general code, but deoptimizes performance
						/*
						l.addInstruction (new Instruction ("lea", opAddr, regYOffset));
						l.addInstruction (new Instruction ("add", new IOperand.Address (regStack), regYOffset));
						l.addInstruction (new Instruction (PREFETCH_INSTR, new IOperand.Address (regYOffset)));
						bYOffsetRegChanged = true;
						*/
					}
				}
			}
			
			nodePrev = node;
		}
		
		// save and restore the register
		if (bYOffsetRegChanged)
		{
			l.addInstructionAtTop (new Instruction ("push", regYOffset));
			l.addInstruction (new Instruction ("pop", regYOffset));
		}
		
		// if we pushed the shifted value, un-shift it
		if (addrYOffset.getScale () != 1)
			l.addInstruction (new Instruction ("shr", new IOperand.Immediate (MathUtil.log2 (addrYOffset.getScale ()))));
		
		l.addInstructions (owiYOffset.getInstrPost ());
		
		m_mapGeneratedCode.put (config, l);
		return l;
	}
	
	/**
	 * Finds the stencil node with a (0,1,0,...,0) index (or
	 * (0,2<sup><i>i</i></sup>,0,...,0) for some integer <i>i</i> if there
	 * is no unit vector), obtains the register holding the offset to the
	 * stencil node and returns the corresponding
	 * {@link OperandWithInstructions} object. Any pre-code required to
	 * initialize the offset register correctly is added to the instruction
	 * list <code>l</code>, as well as the code computing the unit-offset in
	 * case no unit-vector was found.
	 * 
	 * @param set
	 *            The set of stencil nodes in which to search for the
	 *            unit-vector parallel to the y-axis
	 * @param l
	 *            The instruction list to which eventual instructions to
	 *            compute the unit offset are added
	 * @return An {@link OperandWithInstructions} object holding the offset
	 *         register
	 */
	private StencilAssemblySection.OperandWithInstructions getStrideOffsetRegister (int nDirection, boolean bAllowPreAndPostInstructions, InstructionList l)
	{
		StencilNode nodeOffset = null;
		int nLastPowerOf2 = Integer.MAX_VALUE;
		for (StencilNode node : m_data.getStencilCalculation ().getStencilBundle ().getFusedStencil ())
		{
			if (!isParallelToDirection (node, nDirection))
				continue;
			
			Integer nCoordEx =  ExpressionUtil.getIntegerValue (node.getIndex ().getSpaceIndex (nDirection));
			if (nCoordEx == null)
				continue;
			int nCoord = nCoordEx;
			if (nCoord == 1)
			{
				// we're done as soon as we've found a node that has the coordinates (0,...,0,1,0,...,0)
				nodeOffset = node;
				break;
			}
			else if (MathUtil.isPowerOfTwo (Math.abs (nCoord)))
			{
				// (0,2^i,0,...,0) is also acceptable, but search further
				if (Math.abs (nCoord) < nLastPowerOf2)
				{
					nodeOffset = node;
					nLastPowerOf2 = Math.abs (nCoord);
				}
			}
		}
		
		// no acceptable node could be found
		if (nodeOffset == null)
			return null;
		
		StencilAssemblySection.OperandWithInstructions op = m_as.getGrid (nodeOffset, 0, true);
		if (!bAllowPreAndPostInstructions)
		{
			if (op.getInstrPre () != null && op.getInstrPre ().length > 0)
				return null;
			if (op.getInstrPost () != null && op.getInstrPost ().length > 0)
				return null;
		}
		else
		{
			// we don't allow pre/post instructions that involve pseudo registers (i.e., require register allocation)
			// this messes up the translation...
			
			if (op.getInstrPre () != null && containsPseudoRegister (op.getInstrPre ()))
				return null;
			if (op.getInstrPost () != null && containsPseudoRegister (op.getInstrPost ()))
				return null;
		}
		
		// we only support prefetching if the node can be accessed by (%basereg,%offsetreg) 
		if (!(op.getOp () instanceof IOperand.Address))
			return null;

		// emit any pre-instructions
		l.addInstructions (op.getInstrPre ());

		// if the coordinate was not 1, compute the right offset
		Integer nCoord =  ExpressionUtil.getIntegerValue (nodeOffset.getIndex ().getSpaceIndex (nDirection));
		if (nCoord != null)
		{
			if (nCoord != 1)
			{
				if (nCoord < 0)
					l.addInstruction (new Instruction ("neg", op.getOp ()));
				l.addInstruction (new Instruction ("shl", new IOperand.Immediate (MathUtil.log2 (Math.abs (nCoord))), op.getOp ()));
			}
		}
		
		return op;
	}
	
	/**
	 * 
	 * @param setNodes
	 * @return
	 */
	private static Iterable<StencilNode> sortNodesForPrefetching (StencilNodeSet setNodes)
	{
		List<StencilNode> listNodes = new ArrayList<> (setNodes.size ());
		for (StencilNode node : setNodes)
			listNodes.add (node);
				
		Collections.sort (listNodes, new Comparator<StencilNode> ()
		{
			@Override
			public int compare (StencilNode n1, StencilNode n2)
			{
				// sort by ascending x, y, z, ... directions
				// if not parallel to a direction, the order doesn't matter
				
				int nDim = Math.min (n1.getIndex ().getSpaceIndexEx ().length, n2.getIndex ().getSpaceIndexEx ().length);
				for (int nDir = 0; nDir < nDim; nDir++)
				{
					if (isParallelToDirection (n1, nDir))
					{
						if (isParallelToDirection (n2, nDir))
							return ExpressionUtil.getIntegerValue (n1.getIndex ().getSpaceIndex (nDir)) - ExpressionUtil.getIntegerValue (n2.getIndex ().getSpaceIndex (nDir));
						
						// n1 is parallel to the direction nDir, but n2 is not => n1 comes before n2
						return -1;	// n1 < n2
					}
					else if (isParallelToDirection (n2, nDir))
					{
						// n2 is parallel to nDir, but n1 is not => n1 comes after n2
						return 1;	// n1 > n2
					}
				}
				
				return -1;
			}
		});

		return listNodes;
	}
	
	/**
	 * Determines whether the stencil node <code>n</code> is parallel to the
	 * y-axis.
	 * 
	 * @param n
	 *            The stencil node to examine
	 * @return <code>true</code> iff the spatial index of <code>n</code> is
	 *         parallel to the y-axis
	 */
	private static boolean isParallelToYDirection (StencilNode n)
	{
		return isParallelToDirection (n, 1);
	}
	
	private static boolean isParallelToDirection (StencilNode n, int nDir)
	{
		if (n == null || n.getIndex ().getSpaceIndexEx ().length <= nDir)
			return false;
		for (int i = 0; i < n.getIndex ().getSpaceIndexEx ().length; i++)
			if (i != nDir && !ExpressionUtil.isZero (n.getIndex ().getSpaceIndex (i)))
				return false;
		return true;			
	}
	
	/**
	 * Find the direction to which the node <code>node</code> is parallel to. If
	 * it isn't parallel to any axis, the method returns -1.
	 * 
	 * @param node
	 *            The stencil node
	 * @return The axis to which the node is parallel to, or -1 if it isn't
	 *         parallel to any axis
	 */
	private static int getParallelAxisDirection (StencilNode node)
	{
		int nDir = -1;
		for (int i = 0; i < node.getIndex ().getSpaceIndexEx ().length; i++)
		{
//			if (node.getSpaceIndex ()[i] != 0)
			if (!ExpressionUtil.isZero (node.getIndex ().getSpaceIndex (i)))
			{
				if (nDir != -1)
				{
					// a non-zero coordinate has already been found: the node isn't parallel to any axis
					return -1;
				}
				
				nDir = i;
			}
		}
		
		return nDir;
	}
			
	/**
	 * Find all the input nodes of the stencil with
	 * <ul>
	 * 	<li>the lowest offset in x-direction</li>
	 * 	<li>positive offsets in the y-direction</li>
	 * 	<li>non-zero offsets in z- and higher directions</li>
	 * </ul>
	 * These are the nodes we want to prefetch.
	 * @return
	 */
	private StencilNodeSet getPrefetchNodeSet (PrefetchConfig config)
	{
		Stencil stencil = m_data.getStencilCalculation ().getStencilBundle ().getFusedStencil ();
		
		StencilNodeSet setInput = new StencilNodeSet (stencil, StencilNodeSet.ENodeTypes.INPUT_NODES);
		StencilNodeSet setOutput = new StencilNodeSet ();
		
		int nMinT = stencil.getMinTimeIndex ();
		int nMaxT = stencil.getMaxTimeIndex ();
		if (config.isRestrictToLargestNodeSets ())
		{
			int nNodesCount = 0;
			for (int t = stencil.getMinTimeIndex (); t <= stencil.getMaxTimeIndex (); t++)
			{
				int nSize = setInput.restrict (t, null).size ();
				if (nNodesCount < nSize)
				{
					nNodesCount = nSize;
					nMinT = t;
				}
			}
			
			nMaxT = nMinT;
		}
		
		for (int t = nMinT; t <= nMaxT; t++)
		{
			int nLowestX = Integer.MAX_VALUE;
			int nHighestX = Integer.MIN_VALUE;
			StencilNode nodeLowestX = null;
			StencilNode nodeHighestX = null;
			
			for (StencilNode node : setInput.restrict (t, null))
			{
				int[] rgSpaceIdx = node.getSpaceIndex ();
				
				if (rgSpaceIdx.length == 0)
					continue;
				
				if (rgSpaceIdx[0] < nLowestX)
				{
					nLowestX = rgSpaceIdx[0];
					nodeLowestX = node;
				}
				if (rgSpaceIdx[0] > nHighestX)
				{
					nHighestX = rgSpaceIdx[0];
					nodeHighestX= node;
				}
				
				for (int i = 1; i < rgSpaceIdx.length; i++)
					if ((i == 1 && rgSpaceIdx[i] > 0) || (i > 1 && rgSpaceIdx[i] != 0))
					{
						setOutput.add (node);
						break;
					}
			}
			
			if (nodeLowestX != null)
				setOutput.add (nodeLowestX);
			if (nodeHighestX != null)
				setOutput.add (nodeHighestX);
		}
		
		return setOutput;
	}
	
	private IOperand.Register getStackPointerRegister ()
	{
		TypeRegisterClass clsStackPtr = null;
		for (TypeRegisterClass cls : m_data.getArchitectureDescription ().getRegisterClasses (TypeRegisterType.GPR))
		{
			clsStackPtr = cls;
			break;
		}
		
		if (clsStackPtr == null)
			throw new RuntimeException ("Unable to find the register class for the stack pointer register");
		
		TypeRegister reg = new TypeRegister ();
		reg.setClazz (clsStackPtr);
		reg.setName ("rsp");

		return new IOperand.Register (reg);
	}
	
	/**
	 * Determines whether one of the operands of one of the instructions in
	 * <code>rgInstructions</code> contains a pseudo-register.
	 * 
	 * @param rgInstructions
	 *            The array of instructions to examine
	 * @return <code>true</code> iff <code>rgInstructions</code> contains an
	 *         instruction with a pseudo-register as an operand
	 */
	private static boolean containsPseudoRegister (IInstruction[] rgInstructions)
	{
		for (IInstruction instr : rgInstructions)
		{
			if (instr instanceof Instruction)
			{
				for (IOperand op : ((Instruction) instr).getOperands ())
				{
					if (op instanceof IOperand.PseudoRegister)
						return true;
					if (op instanceof IOperand.Address)
					{
						IOperand.Address opAddr = (IOperand.Address) op;
						if (opAddr.getRegBase () instanceof IOperand.PseudoRegister)
							return true;
						if (opAddr.getRegIndex () != null && opAddr.getRegIndex () instanceof IOperand.PseudoRegister)
							return true;
					}
				}
			}
		}
		
		return false;
	}
	
	/**
	 * Adds <code>nFactor</code> * <code>regFactor</code> to the destination
	 * register, <code>regDest</code>.
	 * 
	 * @param regDest
	 *            The destination to which the product <code>nFactor</code> *
	 *            <code>regFactor</code> is added
	 * @param nFactor
	 *            The constant factor
	 * @param regFactor
	 *            The register holding the variable factor
	 * @param l
	 *            The instruction list to which the generated instructions are
	 *            added
	 */
	private static void addNTimes (IOperand.IRegisterOperand regDest, int nFactor, IOperand.IRegisterOperand regFactor, InstructionList l)
	{
		int[] rgLeaFactors = new int[StencilAssemblySection.ELIGIBLE_ADDRESS_SCALING_FACTORS.length];
		System.arraycopy (StencilAssemblySection.ELIGIBLE_ADDRESS_SCALING_FACTORS, 0, rgLeaFactors, 0, rgLeaFactors.length);
		Arrays.sort (rgLeaFactors);
		
		int nRemainingFactor = nFactor;
		for (int i = rgLeaFactors.length - 1; i >= 0; i--)
		{
			while (nRemainingFactor >= rgLeaFactors[i])
			{
				if (rgLeaFactors[i] == 1)
					l.addInstruction (new Instruction ("add", regFactor, regDest));
				else
					l.addInstruction (new Instruction ("lea", new IOperand.Address (regDest, regFactor, rgLeaFactors[i]), regDest));
				
				nRemainingFactor -= rgLeaFactors[i];
			}
		}
		
		for ( ; nRemainingFactor > 0; nRemainingFactor--)
			l.addInstruction (new Instruction ("add", regFactor, regDest));
	}
}
