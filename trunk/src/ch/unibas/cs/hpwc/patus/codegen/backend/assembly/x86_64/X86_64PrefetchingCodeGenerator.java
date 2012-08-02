package ch.unibas.cs.hpwc.patus.codegen.backend.assembly.x86_64;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import org.apache.log4j.Logger;

import ch.unibas.cs.hpwc.patus.arch.TypeRegister;
import ch.unibas.cs.hpwc.patus.arch.TypeRegisterClass;
import ch.unibas.cs.hpwc.patus.arch.TypeRegisterType;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;
import ch.unibas.cs.hpwc.patus.codegen.StencilNodeSet;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.Comment;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.IOperand;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.Instruction;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.InstructionList;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.StencilAssemblySection;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.StencilAssemblySection.OperandWithInstructions;
import ch.unibas.cs.hpwc.patus.representation.StencilNode;
import ch.unibas.cs.hpwc.patus.util.MathUtil;

public class X86_64PrefetchingCodeGenerator
{
	///////////////////////////////////////////////////////////////////
	// Constants
	
	private final static Logger LOGGER = Logger.getLogger (X86_64PrefetchingCodeGenerator.class);
	
	private final static String PREFETCH_INSTR = "prefetchnta";

	
	///////////////////////////////////////////////////////////////////
	// Member Variables

	private CodeGeneratorSharedObjects m_data;
	private StencilAssemblySection m_as;
	private InstructionList m_ilPrefetching;
	
	
	///////////////////////////////////////////////////////////////////
	// Implementation
	
	public X86_64PrefetchingCodeGenerator (CodeGeneratorSharedObjects data, StencilAssemblySection as)
	{
		m_data = data;
		m_as = as;
		m_ilPrefetching = null;
	}
	
	/**
	 * Generates prefetching code.
	 * 
	 * @return The list of instructions implementing the prefetching
	 */
	public InstructionList generate ()
	{
		if (m_ilPrefetching != null)
			return m_ilPrefetching;
		
		InstructionList l = new InstructionList ();
		l.addInstruction (new Comment ("Prefetching next y-line"));
		
		StencilNodeSet setPrefetch = getPrefetchNodeSet ();
		
		// get the register with the "+y_stride"
		StencilAssemblySection.OperandWithInstructions owiYOffset = getYStrideOffsetRegister (setPrefetch, l);
		if (owiYOffset == null)
		{
			// no offset register could be found/computed
			LOGGER.warn ("No y_stride register found in the stencil (stencil does not contain a [x,y+1,...] node) or y_stride is not an address (%basereg,%offsetreg). Unable to create prefetching code.");
			return l;
		}

		IOperand.Address addrYOffset = (IOperand.Address) owiYOffset.getOp ();
		IOperand.IRegisterOperand regYBase = addrYOffset.getRegBase ();
		IOperand.IRegisterOperand regYOffset = addrYOffset.getRegIndex ();
		if (regYOffset == null)
		{
			LOGGER.warn ("No y_stride (offset) register found. Unable to create prefetching code.");
			return l;
		}
					
		// if there is a scaling in the address, apply it
		if (addrYOffset.getScale () != 1)
			l.addInstruction (new Instruction ("shl", new IOperand.Immediate (MathUtil.log2 (addrYOffset.getScale ())), regYOffset));

		// save the value
		l.addInstruction (new Instruction ("push", regYOffset));
		
		IOperand.Register regStack = getStackPointerRegister ();
		MultiplyByConstantCodeGenerator multiplyCG = new MultiplyByConstantCodeGenerator (regYOffset, new IOperand.Address (regStack));
		
		boolean bYOffsetRegChanged = false;
		
		// generate the prefetching code proper
		StencilNode nodePrev = null;
		int nPrevCoord = 1;
		
		for (StencilNode node : X86_64PrefetchingCodeGenerator.sortNodesForPrefetching (setPrefetch))
		{
			if (isParallelToYDirection (node))
			{
				int nCoord = node.getSpaceIndex ()[1] + 1;
				int nScale = 1;

				if (isParallelToYDirection (nodePrev))
				{
					// note: nodePrev can't be null here (if it is, isParallelToYDirection returns false)
					
					// this and the previous node are both parallel to the y-direction:
					// increment the register by the difference
					if ((nCoord % nPrevCoord) == 0)
					{
						if (bYOffsetRegChanged)
							l.addInstruction (new Instruction ("mov", new IOperand.Address (regStack), regYOffset));

						int nQuot = nCoord / nPrevCoord;
						if (nQuot == 2 || nQuot == 4 || nQuot == 8)
						{
							// we could use the scaling factor in the address to do the multiplication
							nScale = nQuot;
							bYOffsetRegChanged = false;
						}
						else
						{
							multiplyCG.generate (l, nQuot, false);
							bYOffsetRegChanged = true;
						}
					}
					else if (nCoord - nPrevCoord <= 2)
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
				{
					// previous node was not parallel to the y-direction, but this one is
					// either the previous node was null or parallel to the x-direction
					
					if (bYOffsetRegChanged)
						l.addInstruction (new Instruction ("mov", new IOperand.Address (regStack), regYOffset));
					
					if (nCoord == 2 || nCoord == 4 || nCoord == 8)
					{
						// use the scaling in the address for multiplication
						nScale = nCoord;
						bYOffsetRegChanged = false;
					}
					else
					{
						multiplyCG.generate (l, nCoord, false);
						bYOffsetRegChanged = true;
					}
				}
				
				// emit prefetching code
				l.addInstruction (new Instruction (PREFETCH_INSTR, new IOperand.Address (regYBase, regYOffset, nScale)));
				
				if (bYOffsetRegChanged)
					nPrevCoord = nCoord;
			}
			else
			{
				// emit prefetching code
				OperandWithInstructions owiGrid = m_as.getGrid (node, 0);
				l.addInstructions (owiGrid.getInstrPre ());
				
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
					if (opAddr.getRegIndex () == null)
					{
						if (bYOffsetRegChanged)
							l.addInstruction (new Instruction ("mov", new IOperand.Address (regStack), regYOffset));
						
final long nMultiple = 64;	// cache line size
long nOffset = ((Math.abs (opAddr.getDisplacement ()) + nMultiple - 1) / nMultiple) * nMultiple;
if (opAddr.getDisplacement () < 0)
	nOffset = -nOffset;
						
						l.addInstruction (new Instruction (PREFETCH_INSTR, new IOperand.Address (opAddr.getRegBase (), regYOffset, 1, nOffset /*opAddr.getDisplacement ()*/)));
						bYOffsetRegChanged = false;
					}
					else
					{
/*
						l.addInstruction (new Instruction ("lea", opAddr, regYOffset));
						l.addInstruction (new Instruction ("add", new IOperand.Address (regStack), regYOffset));
						l.addInstruction (new Instruction (PREFETCH_INSTR, new IOperand.Address (regYOffset)));
						bYOffsetRegChanged = true;
*/
						bYOffsetRegChanged = false;
					}
				}
				
				l.addInstructions (owiGrid.getInstrPost ());
			}				
			
			nodePrev = node;
		}
		
		// restore
		l.addInstruction (new Instruction ("pop", regYOffset));
		
		// if we pushed the shifted value, un-shift it
		if (addrYOffset.getScale () != 1)
			l.addInstruction (new Instruction ("shr", new IOperand.Immediate (MathUtil.log2 (addrYOffset.getScale ()))));
		
		l.addInstructions (owiYOffset.getInstrPost ());
		
		return m_ilPrefetching = l;
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
	private StencilAssemblySection.OperandWithInstructions getYStrideOffsetRegister (StencilNodeSet set, InstructionList l)
	{
		StencilNode nodeYOffset = null;
		int nLastPowerOf2 = Integer.MAX_VALUE;
		for (StencilNode node : set)
		{
			if (!isParallelToYDirection (node))
				continue;
			
			int nCoord = node.getSpaceIndex ()[1];
			if (nCoord == 1)
			{
				// we're done as soon as we've found a node that has the coordinates (0,1,0,...,0)
				nodeYOffset = node;
				break;
			}
			else if (MathUtil.isPowerOfTwo (Math.abs (nCoord)))
			{
				// (0,2^i,0,...,0) is also acceptable, but search further
				if (Math.abs (nCoord) < nLastPowerOf2)
				{
					nodeYOffset = node;
					nLastPowerOf2 = Math.abs (nCoord);
				}
			}
		}
		
		// no acceptable node could be found
		if (nodeYOffset == null)
			return null;
		
		StencilAssemblySection.OperandWithInstructions op = m_as.getGrid (nodeYOffset, 0);
		
		// we only support prefetching if the node in y direction can be accessed by (%basereg,%offsetreg) 
		if (!(op.getOp () instanceof IOperand.Address))
			return null;

		// emit any pre-instructions
		l.addInstructions (op.getInstrPre ());

		// if the coordinate was not 1, compute the right offset
		if (nodeYOffset.getSpaceIndex ()[1] != 1)
		{
			if (nodeYOffset.getSpaceIndex ()[1] < 0)
				l.addInstruction (new Instruction ("neg", op.getOp ()));
			l.addInstruction (new Instruction ("shl", new IOperand.Immediate (MathUtil.log2 (Math.abs (nodeYOffset.getSpaceIndex ()[1]))), op.getOp ()));
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
				// sort by ascending x- and y-directions first, after that the order doesn't matter
				for (int nDir = 0; nDir <= 1; nDir++)
				{
					if (isParallelToDirection (n1, nDir))
					{
						if (isParallelToDirection (n2, nDir))
							return n1.getSpaceIndex ()[nDir] - n2.getSpaceIndex ()[nDir];
						
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
		if (n == null || n.getSpaceIndex ().length <= nDir)
			return false;
		for (int i = 0; i < n.getSpaceIndex ().length; i++)
			if (i != nDir && n.getSpaceIndex ()[i] != 0)
				return false;
		return true;			
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
	private StencilNodeSet getPrefetchNodeSet ()
	{
		StencilNodeSet set = new StencilNodeSet ();
		
		int nLowestX = Integer.MAX_VALUE;
		StencilNode nodeLowestX = null;
		
		for (StencilNode node : m_data.getStencilCalculation ().getStencilBundle ().getFusedStencil ())
		{
			if (node.getSpaceIndex ().length == 0)
				continue;
			
			if (node.getSpaceIndex ()[0] < nLowestX)
			{
				nLowestX = node.getSpaceIndex ()[0];
				nodeLowestX = node;
			}
			
			for (int i = 1; i < node.getSpaceIndex ().length; i++)
				if ((i == 1 && node.getSpaceIndex ()[i] > 0) || (i > 1 && node.getSpaceIndex ()[i] != 0))
				{
					set.add (node);
					break;
				}
		}
		
		if (nodeLowestX != null)
			set.add (nodeLowestX);

		return set;
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
}
