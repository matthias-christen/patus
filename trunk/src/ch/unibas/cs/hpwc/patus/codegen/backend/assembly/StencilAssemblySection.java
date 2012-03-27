package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.DepthFirstIterator;
import cetus.hir.Expression;
import cetus.hir.FloatLiteral;
import cetus.hir.IntegerLiteral;
import cetus.hir.NameID;
import cetus.hir.SizeofExpression;
import cetus.hir.Specifier;
import cetus.hir.Traversable;
import cetus.hir.UnaryExpression;
import cetus.hir.UnaryOperator;
import ch.unibas.cs.hpwc.patus.arch.TypeRegisterType;
import ch.unibas.cs.hpwc.patus.ast.StatementListBundle;
import ch.unibas.cs.hpwc.patus.ast.SubdomainIdentifier;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;
import ch.unibas.cs.hpwc.patus.codegen.MemoryObject;
import ch.unibas.cs.hpwc.patus.codegen.StencilNodeSet;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.IOperand.IRegisterOperand;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;
import ch.unibas.cs.hpwc.patus.representation.FindStencilNodeBaseVectors;
import ch.unibas.cs.hpwc.patus.representation.Stencil;
import ch.unibas.cs.hpwc.patus.representation.StencilNode;
import ch.unibas.cs.hpwc.patus.util.CodeGeneratorUtil;
import ch.unibas.cs.hpwc.patus.util.ExpressionUtil;
import ch.unibas.cs.hpwc.patus.util.IntArray;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 * Assembly section specialization for stencil computations.
 * 
 * @author Matthias-M. Christen
 */
public class StencilAssemblySection extends AssemblySection
{
	///////////////////////////////////////////////////////////////////
	// Constants
	
	public final static String INPUT_CONSTANTS_ARRAYPTR = "constants";
	
	
	///////////////////////////////////////////////////////////////////
	// Inner Types

	public static class OperandWithInstructions
	{
		private IInstruction[] m_rgInstrPre;
		private IOperand m_op;
		private IInstruction[] m_rgInstrPost;
		
		public OperandWithInstructions (IOperand op)
		{
			this (null, op, null);
		}
		
		public OperandWithInstructions (IInstruction[] rgInstrPre, IOperand op, IInstruction[] rgInstrPost)
		{
			m_rgInstrPre = rgInstrPre;
			m_op = op;
			m_rgInstrPost = rgInstrPost;
		}

		public IInstruction[] getInstrPre ()
		{
			return m_rgInstrPre;
		}

		public IOperand getOp ()
		{
			return m_op;
		}

		public IInstruction[] getInstrPost ()
		{
			return m_rgInstrPost;
		}
		
		@Override
		public String toString ()
		{
			StringBuilder sb = new StringBuilder ();
			
			sb.append ("Pre:  ");
			if (m_rgInstrPre == null)
				sb.append ("-\n");
			else
			{
				for (IInstruction instr : m_rgInstrPre)
				{
					sb.append (instr.toString ());
					sb.append ('\n');
				}
			}
			
			sb.append ("Op:   ");
			sb.append (m_op.toString ());
			sb.append ('\n');

			sb.append ("Post: ");
			if (m_rgInstrPost == null)
				sb.append ("-\n");
			else
			{
				for (IInstruction instr : m_rgInstrPost)
				{
					sb.append (instr.toString ());
					sb.append ('\n');
				}
			}

			return sb.toString ();
		}
	}
		

	///////////////////////////////////////////////////////////////////
	// Member Variables
	
	/**
	 * The identifier of the subdomain this assembly section is in.
	 * (Used to compute addresses of grids used as inputs for the computation.)
	 */
	private SubdomainIdentifier m_sdid;
	
	/**
	 * Code generation options
	 */
	private CodeGeneratorRuntimeOptions m_options;
	
	/**
	 * Additional generated C code to be inserted before the inline assembly section
	 */
	private StatementListBundle m_slbGeneratedCode;
	
	private Map<StencilNode, IOperand.IRegisterOperand> m_mapGrids;
	private List<IOperand.IRegisterOperand> m_listGridInputs;
	
	private Map<IntArray, IOperand.IRegisterOperand> m_mapStrides;
	private Map<IntArray, Expression> m_mapStrideExpressions;
	private boolean m_bUseNegOnStrides;
	
	private Map<Expression, Integer> m_mapConstantsAndParams;

	private FindStencilNodeBaseVectors m_baseVectors;
	
	/**
	 * The data type of the stencil calculation. Only one data type is supported, i.e.,
	 * the stencil computation can't be in mixed precisions.
	 */
	private Specifier m_specDatatype;

	
	///////////////////////////////////////////////////////////////////
	// Implementation
	
	public StencilAssemblySection (CodeGeneratorSharedObjects data, SubdomainIdentifier sdid, CodeGeneratorRuntimeOptions options)
	{
		super (data);
		
		m_sdid = sdid;
		m_options = options;
		
		m_slbGeneratedCode = new StatementListBundle ();

		m_mapGrids = new HashMap<> ();
		m_listGridInputs = new ArrayList<> ();
		m_mapStrides = new HashMap<> ();
		m_mapStrideExpressions = new HashMap<> ();
		m_mapConstantsAndParams = new HashMap<> ();
		m_bUseNegOnStrides = false;
		
		m_baseVectors = new FindStencilNodeBaseVectors (new int[] { 1, 2, 4, 8 });	// TODO: put this in architecture.xml
		m_specDatatype = null;
		
		m_data.getData ().getMemoryObjectManager ().clear ();
	}
	
	/**
	 * Determines the data type of the stencil computation.
	 * Throws an exception if the computation is in mixed precisions.
	 * 
	 * @return The data type of the stencil computation
	 */
	public Specifier getDatatype ()
	{
		if (m_specDatatype == null)
		{
			for (StencilNode node : m_data.getStencilCalculation ().getOutputBaseNodeSet ())
			{
				if (m_specDatatype == null)
					m_specDatatype = node.getSpecifier ();
				else
				{
					if (!m_specDatatype.equals (node.getSpecifier ()))
						throw new RuntimeException ("");
				}
			}
		}
		
		return m_specDatatype;
	}

	/**
	 * Creates the inputs to the assembly section (i.e., generates the values
	 * that are assigned to registers within the inline assembly input section).
	 */
	public void createInputs ()
	{
		int nInitialFreeRegisters = getFreeRegistersCount (TypeRegisterType.GPR) - 2;
		int nNumInputs = m_listInputs.size ();
		int nNumRegistersForGrids = getNumRegistersForGrids ();
		int nNumRegistersForStrides = countStrides ();
		
		// 1 additional register is needed for the constants array
		int nFreeRegisters = nInitialFreeRegisters - nNumInputs - nNumRegistersForGrids - nNumRegistersForStrides - 1;
		if (nFreeRegisters >= 0)
		{
			// enough free registers
			addGrids ();
			addStrides (m_mapStrideExpressions.keySet ());
			addConstants ();
		}
		else
		{
			// check whether be negating stride registers we can save enough
			Set<IntArray> setPositiveStrides = new HashSet<> ();
			for (IntArray v : m_mapStrideExpressions.keySet ())
				if (!setPositiveStrides.contains (v) && !setPositiveStrides.contains (v.neg ()))
					setPositiveStrides.add (v);
				
			int nSavedRegisters = nNumRegistersForStrides - setPositiveStrides.size ();			
			if (nFreeRegisters + nSavedRegisters >= 0)
			{
				addGrids ();
				addStrides (setPositiveStrides);
				m_bUseNegOnStrides = true;
				addConstants ();
			}
			else
				throw new RuntimeException ("no regs left");
		}
	}
	
	private int getNumRegistersForGrids ()
	{
		StencilNodeSet setAllGrids = m_data.getStencilCalculation ().getInputBaseNodeSet ().union (
			m_data.getStencilCalculation ().getOutputBaseNodeSet ());
		
		return setAllGrids.size ();
	}
	
	private void addGrids ()
	{
		// TODO: get reference nodes for "current" memory objects!!!
		StencilNodeSet setAllGrids = m_data.getStencilCalculation ().getInputBaseNodeSet ().union (
			m_data.getStencilCalculation ().getOutputBaseNodeSet ());
		
		if (setAllGrids.size () > 10)
		{
			// TODO: "too many grids"
			throw new RuntimeException ("Too many grids...");
		}

		for (StencilNode node : setAllGrids)
			addGrid (node);		
	}
	
	/**
	 * Adds the stride expressions as assembly section inputs.
	 * <p>Call {@link StencilAssemblySection#countStrides()} before calling this method.</p>
	 */
	private void addStrides (Iterable<IntArray> itBaseVectors)
	{
		for (IntArray arrBaseVector : itBaseVectors)
		{
			Expression exprStride = m_mapStrideExpressions.get (arrBaseVector);
			m_mapStrides.put (arrBaseVector, (IOperand.IRegisterOperand) addInput (exprStride, exprStride));
		}
	}
	
	private int countStrides ()
	{
		StencilNodeSet setNodes = m_data.getStencilCalculation ().getStencilBundle ().getFusedStencil ().getAllNodes ();
		for (StencilNode node : setNodes)
			m_baseVectors.addNode (node);
		
		// find base vectors
		m_baseVectors.run ();
		for (StencilNode node : setNodes)
		{
			int[] rgBaseVector = m_baseVectors.getBaseVector (node);
			if (rgBaseVector == null)
				continue;
			
			// for the stride computation, we neglect the coordinate in the 0-th dimension
			// (the unit stride direction), which will be handled by SIMD shuffles or unaligned moves
			IntArray v = new IntArray (rgBaseVector, true);
			v.set (0, 0);
			
			addStride (v, node);
		}
		
		return m_mapStrideExpressions.size ();
	}
	
	/**
	 * Adds a single grid pointer as input.
	 * 
	 * @param node
	 *            The stencil node corresponding to the grid
	 */
	protected void addGrid (StencilNode node)
	{
		if (m_mapGrids.containsKey (node))
			return;
		
		IOperand op = addInput (
			node,
			new UnaryExpression (
				UnaryOperator.ADDRESS_OF,
				m_data.getData ().getMemoryObjectManager ().getMemoryObjectExpression (
					m_sdid, node, null, true, true, false, m_slbGeneratedCode, m_options
				)
			)
		);
		m_listGridInputs.add ((IOperand.IRegisterOperand) op);
		
		// add the node to the grid
		m_mapGrids.put (node, (IOperand.IRegisterOperand) op);
		
		// add all other nodes which project to the same node
		MemoryObject mo = m_data.getData ().getMemoryObjectManager ().getMemoryObject (m_sdid, node, true);
		for (StencilNode n : m_data.getStencilCalculation ().getStencilBundle ().getFusedStencil ())
			if (n != node && mo.contains (n))
				m_mapGrids.put (n, (IOperand.IRegisterOperand) op);
	}
	
	/**
	 * 
	 * @param arrBaseVector The base vector used to access the stencil node
	 * @param node
	 */
	public void addStride (IntArray arrBaseVector, StencilNode node)
	{
		// don't do anything if the base vector is 0 (no strides needed)
		if (arrBaseVector == null || arrBaseVector.isZero ())
			return;
		
		// don't do anything if the base vector was added already
		if (m_mapStrideExpressions.containsKey (arrBaseVector))
			return;
		
		// get the (local) memory object corresponding to the node
		MemoryObject mo = m_data.getData ().getMemoryObjectManager ().getMemoryObject (m_sdid, node, true);
		
		// TODO: create temporary variables (=> common subexpression elimination)
		
		// calculate the stride
		Expression exprStride = null;
		for (int i = 1; i < mo.getDimensionality (); i++)
		{
			if (arrBaseVector.get (i) != 0)
			{
				Expression exprStridePart = ExpressionUtil.product (mo.getSize ().getCoords (), 0, i - 1);
				if (arrBaseVector.get (i) != 1)
					exprStridePart = new BinaryExpression (new IntegerLiteral (arrBaseVector.get (i)), BinaryOperator.MULTIPLY, exprStridePart);
				
				if (exprStride == null)
					exprStride = exprStridePart;
				else
					exprStride = new BinaryExpression (exprStride, BinaryOperator.ADD, exprStridePart);
			}
		}
		
		exprStride = new BinaryExpression (exprStride, BinaryOperator.MULTIPLY, new SizeofExpression (CodeGeneratorUtil.specifiers (getDatatype ())));
		m_mapStrideExpressions.put (arrBaseVector, exprStride);
	}

	private void addToConstantsAndParamsMap (Expression expr)
	{
		if (!m_mapConstantsAndParams.containsKey (expr))
			m_mapConstantsAndParams.put (expr, m_mapConstantsAndParams.size ());
	}
	
	/**
	 * Finds all the {@link FloatLiteral}s and {@link NameID}s which are stencil
	 * parameters in <code>trv</code> and adds them to the constants and
	 * parameters map <code>m_mapConstantsAndParams</code>.
	 * 
	 * @param trv
	 *            The {@link Traversable} to search for floating point literals
	 *            and stencil parameters
	 */
	private void findConstantsAndParams (Traversable trv)
	{
		for (DepthFirstIterator it = new DepthFirstIterator (trv); it.hasNext (); )
		{
			Object obj = it.next ();
			if (obj instanceof FloatLiteral)
				addToConstantsAndParamsMap ((FloatLiteral) obj);
			else if (obj instanceof NameID)
			{
				// if the NameID is a stencil parameter, add it to the map
				if (m_data.getStencilCalculation ().isArgument (((NameID) obj).getName ()))
					addToConstantsAndParamsMap ((NameID) obj);
			}
		}
	}

	/**
	 * Adds the constants in <code>listConstants</code> to an array and uses the
	 * address of the array as an input for the assembly section.
	 * 
	 * @param listConstants
	 *            The list of constant numbers
	 */
	public void addConstants ()
	{
		Map<NameID, Stencil> mapStencils = new HashMap<> ();

		// add constants/parameters and constant stencils
		for (Stencil stencil : m_data.getStencilCalculation ().getStencilBundle ())
		{
			if (!stencil.isConstant ())
				findConstantsAndParams (stencil.getExpression ());
			else
			{
				for (StencilNode nodeOut : stencil.getOutputNodes ())
				{
					NameID nidNode = new NameID (nodeOut.getName ());
					addToConstantsAndParamsMap (nidNode);
					mapStencils.put (nidNode, stencil);
				}
			}
		}
		
		// nothing to do if no constants have been found
		if (m_mapConstantsAndParams.size () == 0)
			return;
		
		Expression[] rgConstsAndParams = new Expression[m_mapConstantsAndParams.size ()];
		
		for (Expression expr : m_mapConstantsAndParams.keySet ())
		{
			int nIdx = m_mapConstantsAndParams.get (expr);
			
			Stencil stencil = mapStencils.get (expr);
			if (stencil != null)
				rgConstsAndParams[nIdx] = stencil.getExpression ();
			else					
				rgConstsAndParams[nIdx] = expr;
		}
		
		addInput (
			INPUT_CONSTANTS_ARRAYPTR,
			m_data.getCodeGenerators ().getSIMDScalarGeneratedIdentifiers ().createVectorizedScalars (
				rgConstsAndParams, getDatatype (), m_slbGeneratedCode, m_options)
		);
	}
	
	/**
	 * Determines the number of constants within the entire stencil bundle.
	 * 
	 * @return The number of constants used
	 */
	public int getConstantsAndParamsCount ()
	{
		return m_mapConstantsAndParams.size ();
	}
	
	/**
	 * Returns an iterable over all the constants and parameters saved in the
	 * constant/parameter array.
	 * 
	 * @return An iterable over constants and parameters saved in the
	 *         const/param array of the assembly section
	 */
	public Iterable<Expression> getConstantsAndParams ()
	{
		return m_mapConstantsAndParams.keySet ();
	}

	/**
	 * Returns the index of the constant or the stencil parameter
	 * <code>exprConstantOrParam</code> within the constants/parameters map or
	 * <code>-1</code> if no such constant/parameter exists
	 * in the map.
	 * 
	 * @param exprConstantOrParam
	 *            The constant or parameter for which to retrieve the index
	 * @return The index of <code>exprConstantOrParam</code> within the
	 *         constants/parameter map or <code>-1</code> if no such
	 *         constant/parameter exists
	 */
	public int getConstantOrParamIndex (Expression exprConstantOrParam)
	{
		Integer nIdx = m_mapConstantsAndParams.get (exprConstantOrParam);
		return nIdx == null ? -1 : nIdx;
	}

	/**
	 * Returns the address operand for the constant or stencil parameter
	 * <code>exprConstantOrParam</code>.
	 * 
	 * @param exprConstantOrParam
	 *            The expression (a constant, i.e., a floating point literal, or
	 *            a stencil parameter) for which to retrieve its address
	 * @return The address operand for <code>exprConstantOrParam</code>
	 */
	public IOperand getConstantOrParamAddress (Expression exprConstantOrParam)
	{
		int nConstParamIdx = getConstantOrParamIndex (exprConstantOrParam);
		if (nConstParamIdx == -1)
			throw new RuntimeException (StringUtil.concat ("No index for the constant of parameter ", exprConstantOrParam.toString ()));
		
		int nSIMDVectorLength = m_data.getArchitectureDescription ().getSIMDVectorLength (getDatatype ());
		
		return new IOperand.Address (
			getInput (StencilAssemblySection.INPUT_CONSTANTS_ARRAYPTR),
			nConstParamIdx * AssemblySection.getTypeSize (getDatatype ()) * nSIMDVectorLength);
	}

	/**
	 * Returns an iterable over the register containing the grid pointers.
	 * @return
	 */
	public Iterable<IOperand.IRegisterOperand> getGrids ()
	{
		return m_listGridInputs;
	}
	
	/**
	 * Determines whether the stencil nodes <code>node</code> and
	 * <code>nodeRef</code> are compatible, i.e., whether all their spatial
	 * indices coincide except in the first (the reuse) dimension.
	 * 
	 * @param node
	 *            The stencil node to check whether it is compatible with the
	 *            stencil node <code>nodeRef</code>
	 * @param nodeRef
	 *            The reference stencil node
	 * @return <code>true</code> iff the stencil nodes <code>node</code> and
	 *         <code>nodeRef</code> are compatible
	 */
	private static boolean isNodeCompatible (StencilNode node, StencilNode nodeRef)
	{
		if (nodeRef.getSpaceIndex ().length != node.getSpaceIndex ().length)
			return false;
		
		// compare all the coordinates except in the first dimension (i==0)
		for (int i = 1; i < nodeRef.getSpaceIndex ().length; i++)
			if (node.getSpaceIndex ()[i] != nodeRef.getSpaceIndex ()[i])
				return false;
		
		return true;
	}
	
	/**
	 * Finds a reference node, which is compatible with <code>node</code>, for
	 * an arbitrary stencil node, <code>node</code>. Two stencil nodes are
	 * <i>compatible</i> iff all their spatial coordinates coincide except the
	 * one in the first (the reuse) dimension. <code>null</code> is returned if
	 * no compatible node could be found.
	 * 
	 * @param node
	 *            The stencil node for which to find a compatible reference node
	 * @return A stencil node from the node set which is compatible with
	 *         <code>node</code> or <code>null</code> if no such node could be
	 *         found
	 */
	private StencilNode findReferenceNode (StencilNode node)
	{
		if (m_mapGrids.isEmpty ())
			return null;
		
		for (StencilNode nodeRef : m_mapGrids.keySet ())
			if (StencilAssemblySection.isNodeCompatible (node, nodeRef))
				return nodeRef;
		
		return null;
	}
	
	private OperandWithInstructions getStride (IntArray arrBaseVector)
	{
		if (arrBaseVector.isZero ())
			return new OperandWithInstructions (null);
			
		IOperand op = m_mapStrides.get (arrBaseVector);
		if (op != null)
			return new OperandWithInstructions (op);
		
		if (m_bUseNegOnStrides)
		{
			op = m_mapStrides.get (arrBaseVector.neg ());
			if (op != null)
			{
				return new OperandWithInstructions (
					new IInstruction[] { new Instruction ("neg", op) },
					op,
					new IInstruction[] { new Instruction ("neg", op) }
				);
			}
		}

		throw new RuntimeException ("Stride not found");
	}
	
	/**
	 * Returns the address to access the stencil node <code>node</code> shifted
	 * in unit stride direction by <code>nElementsShift</code>.
	 * 
	 * @param node
	 *            The stencil node for which to retrieve the address operand
	 * @param nElementsShift
	 *            The number of elements by which the node is shifted to the
	 *            right for unrolling
	 * @return The address operand to access <code>node</code>
	 */
	public OperandWithInstructions getGrid (StencilNode node, int nElementsShift)
	{
		StencilNode nodeLocal = node;
		int nElementsShiftLocal = nElementsShift;
		
		// TODO: if too many grids need to use LEA...
		IOperand.IRegisterOperand opBase = m_mapGrids.get (nodeLocal);
		if (opBase == null)
		{
			// no base node found in the map for "node"
			// find a reference node
			StencilNode nodeRef = findReferenceNode (nodeLocal);
			if (nodeRef == null)
				return null;
			
			nElementsShiftLocal += nodeLocal.getSpaceIndex ()[0] - nodeRef.getSpaceIndex ()[0];
			nodeLocal = nodeRef;
			opBase = m_mapGrids.get (nodeLocal);
		}
		
		// no index register is needed if the offset is only in the unit stride direction (dimension 0)
		boolean bHasOffsetInNonUnitStride = false || (nElementsShiftLocal > 0);
		if (!bHasOffsetInNonUnitStride)
		{
			for (int i = 1; i < nodeLocal.getSpaceIndex ().length; i++)
				if (nodeLocal.getSpaceIndex ()[i] != 0)
				{
					bHasOffsetInNonUnitStride = true;
					break;
				}
		}
		
		if (!m_data.getArchitectureDescription ().supportsUnalignedSIMD ())
			throw new RuntimeException ("Currently only architectures which support unaligned SIMD vector loads/stores are supported");
		
		// get the offset in unit stride direction (will become the displacement in inline assembly)
		Specifier specDatatype = getDatatype ();
		int nSIMDVectorLength = m_data.getArchitectureDescription ().getSIMDVectorLength (specDatatype);
		int nUnitStrideOffset = (nodeLocal.getSpaceIndex ()[0] + nElementsShiftLocal * nSIMDVectorLength) * AssemblySection.getTypeSize (specDatatype);
		
		if (!bHasOffsetInNonUnitStride)
			return new OperandWithInstructions (new IOperand.Address (opBase, nUnitStrideOffset));
		
		// general case
		// TODO: what if too many strides?

		IntArray v = new IntArray (m_baseVectors.getBaseVector (nodeLocal), true);
		v.set (0, 0);
		
		OperandWithInstructions opStride = getStride (v);
		return new OperandWithInstructions (
			opStride.getInstrPre (),
			new IOperand.Address (opBase, (IOperand.IRegisterOperand) opStride.getOp (), m_baseVectors.getScalingFactor (nodeLocal), nUnitStrideOffset),
			opStride.getInstrPost ()
		);
	}
		
	/**
	 * Returns the operand corresponding to the constant or parameter
	 * <code>exprConstantOrParam</code>.
	 * 
	 * @param exprConstantOrParam
	 *            The constant or stencil parameter
	 * @param specDatatype
	 *            The datatype
	 * @return The operand corresponding to <code>exprConstantOrParam</code>
	 */
	public IOperand getConstantOrParam (Expression exprConstantOrParam, Specifier specDatatype)
	{
		IOperand regBase = getInput (INPUT_CONSTANTS_ARRAYPTR);
		if (regBase == null)
			return null;
		return new IOperand.Address ((IRegisterOperand) regBase, m_mapConstantsAndParams.get (exprConstantOrParam) * AssemblySection.getTypeSize (specDatatype));
	}

	/**
	 * Returns the auxiliary statements generated for the assembly section.
	 * 
	 * @return The auxiliary statements to be inserted before the assembly section
	 */
	public StatementListBundle getAuxiliaryStatements ()
	{
		return m_slbGeneratedCode;
	}
}
