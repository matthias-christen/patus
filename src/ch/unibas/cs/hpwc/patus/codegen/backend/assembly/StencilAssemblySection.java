package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.log4j.Logger;

import cetus.hir.ArrayAccess;
import cetus.hir.AssignmentExpression;
import cetus.hir.AssignmentOperator;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.DepthFirstIterator;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.FloatLiteral;
import cetus.hir.IntegerLiteral;
import cetus.hir.NameID;
import cetus.hir.SizeofExpression;
import cetus.hir.Specifier;
import cetus.hir.Traversable;
import cetus.hir.Typecast;
import cetus.hir.UnaryExpression;
import cetus.hir.UnaryOperator;
import cetus.hir.UserSpecifier;
import cetus.hir.VariableDeclarator;
import ch.unibas.cs.hpwc.patus.analysis.StencilAnalyzer;
import ch.unibas.cs.hpwc.patus.arch.ArchitectureDescriptionManager;
import ch.unibas.cs.hpwc.patus.arch.TypeBaseIntrinsicEnum;
import ch.unibas.cs.hpwc.patus.arch.TypeRegisterType;
import ch.unibas.cs.hpwc.patus.ast.StatementListBundle;
import ch.unibas.cs.hpwc.patus.ast.SubdomainIdentifier;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;
import ch.unibas.cs.hpwc.patus.codegen.MemoryObject;
import ch.unibas.cs.hpwc.patus.codegen.StencilNodeSet;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;
import ch.unibas.cs.hpwc.patus.representation.FindStencilNodeBaseVectors;
import ch.unibas.cs.hpwc.patus.representation.Stencil;
import ch.unibas.cs.hpwc.patus.representation.StencilCalculation;
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
	
	private final static Logger LOGGER = Logger.getLogger (StencilAssemblySection.class);
	
	/**
	 * Factors by which the index operand in an address can be scaled
	 */
	// TODO: put this in architecture.xml
	public final static int[] ELIGIBLE_ADDRESS_SCALING_FACTORS = new int[] { 1, 2, 4, 8 };
	
	public final static String INPUT_GRIDS_ARRAYPTR = "_grids_";
	public final static String INPUT_STRIDE_ARRAYPTR = "_strides_";
	
	
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
	private List<IOperand> m_listGridInputs;
	private Map<IOperand, Boolean> m_mapGridLoaded;
	private Map<IOperand, Integer> m_mapGridIndices;
	private boolean m_bUseGridPointers;
	
	private Map<IntArray, IOperand.IRegisterOperand> m_mapStrides;
	private Map<IntArray, Expression> m_mapStrideExpressions;
	private boolean m_bUseNegOnStrides;
	private boolean m_bUseStridePointers;
	
	private FindStencilNodeBaseVectors m_baseVectors;
	
	private Map<Expression, Integer> m_mapConstantsAndParams;
	private Map<Expression, IOperand.IRegisterOperand> m_mapReusedConstantsAndParams;
	private Set<IOperand.IRegisterOperand> m_setConstantsAndParamsRegisters;

	/**
	 * The data type of the stencil calculation. Only one data type is supported, i.e.,
	 * the stencil computation can't be in mixed precisions.
	 */
	private Specifier m_specDatatype;
	
	/**
	 * The offset from the default center stencil node (to account for loop unrolling)
	 */
	private int[] m_rgOffset;

	
	///////////////////////////////////////////////////////////////////
	// Implementation
	
	public StencilAssemblySection (CodeGeneratorSharedObjects data, SubdomainIdentifier sdid, CodeGeneratorRuntimeOptions options)
	{
		super (data);
		
		m_sdid = sdid;
		m_options = options;
		
		m_slbGeneratedCode = new StatementListBundle ();

		m_mapGrids = new HashMap<> ();
		m_mapGridLoaded = new HashMap<> ();
		m_mapGridIndices = new HashMap<> ();
		m_listGridInputs = new ArrayList<> ();
		m_mapStrides = new HashMap<> ();
		m_mapStrideExpressions = new HashMap<> ();
		
		m_mapConstantsAndParams = new HashMap<> ();
		m_mapReusedConstantsAndParams = new HashMap<> ();
		m_setConstantsAndParamsRegisters = new HashSet<> ();

		m_bUseNegOnStrides = false;
		m_bUseGridPointers = false;
		m_bUseStridePointers = false;
		
		m_baseVectors = new FindStencilNodeBaseVectors (ELIGIBLE_ADDRESS_SCALING_FACTORS);
		m_specDatatype = null;
		
		m_rgOffset = (int[]) options.getObjectValue (CodeGeneratorRuntimeOptions.OPTION_INNER_UNROLLINGCONFIGURATION);
		
		//m_data.getData ().getMemoryObjectManager ().clear ();
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
			m_specDatatype = StencilAssemblySection.getDatatype (m_data.getStencilCalculation ());
		
		return m_specDatatype;
	}
	
	public static Specifier getDatatype (StencilCalculation stencil)
	{
		Specifier specDatatype = null;
		
		for (StencilNode node : stencil.getOutputBaseNodeSet ())
		{
			if (specDatatype == null)
			{
				specDatatype = node.getSpecifier ();
				if (specDatatype != null)
					break;
			}
			else
			{
				if (!specDatatype.equals (node.getSpecifier ()))
					throw new RuntimeException ("");
			}
		}
		
		if (specDatatype == null)
		{
			for (StencilNode node : stencil.getInputBaseNodeSet ())
			{
				if (specDatatype == null)
				{
					specDatatype = node.getSpecifier ();
					if (specDatatype != null)
						break;
				}
				else
				{
					if (!specDatatype.equals (node.getSpecifier ()))
						throw new RuntimeException ("");
				}
			}					
		}
		
		// default...
		if (specDatatype == null)
			specDatatype = Specifier.FLOAT;
		
		return specDatatype;
	}

	/**
	 * Creates the inputs to the assembly section (i.e., generates the values
	 * that are assigned to registers within the inline assembly input section).
	 */
	public void createInputs ()
	{
		// set to 0, 1, 2, 3, or 4 to manually select a code generation path (see below) for debugging purposes.
		// set to -1 for default behavior.
		final int nDebugSelectVariant = -1;

		int nInitialFreeRegisters = getFreeRegistersCount (TypeRegisterType.GPR) - 1;
		int nNumInputs = getInputsCount ();
		int nNumRegistersForGrids = getNumRegistersForGrids ();
		int nNumRegistersForStrides = countStrides ();
		
		// 1 additional register is needed for the constants array
		int nFreeRegisters = nInitialFreeRegisters - nNumInputs - nNumRegistersForGrids - nNumRegistersForStrides - 1;
		if (((nFreeRegisters >= 0)
			&& nDebugSelectVariant == -1) || nDebugSelectVariant == 0)
		{
			// enough free registers
			addGrids ();
			addStrides (m_mapStrideExpressions.keySet ());
			addConstants ();
		}
		else
		{			
			// check whether be negating stride registers we can save enough
			Set<IntArray> setPositiveStrides = getPositiveStrides ();			
			int nSavedRegisters = nNumRegistersForStrides - setPositiveStrides.size ();
			
			if (((nFreeRegisters + nSavedRegisters >= 0)
				&& nDebugSelectVariant == -1) || nDebugSelectVariant == 1)
			{
				addGrids ();
				addStrides (setPositiveStrides);
				m_bUseNegOnStrides = true;
				addConstants ();
			}
			else if (((nFreeRegisters + nNumRegistersForGrids - 2 >= 0)
				&& nDebugSelectVariant == -1) || nDebugSelectVariant == 2)
			{
				// there are enough registers for the individual strides if all the grids are addressed indirectly (LEA)
				// note that we need one register to save the grid pointers array, and at least one into which the
				// grid address is LEAed, hence we need at least 2 free registers
				
				addGridPointers ();
				addStrides (m_mapStrideExpressions.keySet ());
				addConstants ();
			}
			else if (((nFreeRegisters + nNumRegistersForGrids - 2 + nSavedRegisters >= 0)
				&& nDebugSelectVariant == -1) || nDebugSelectVariant == 3)
			{
				// there are enough registers for reduced strides (by negating)
				// if all the grids are addressed indirectly (LEA)

				addGridPointers ();
				addStrides (setPositiveStrides);
				m_bUseNegOnStrides = true;
				addConstants ();
			}
			else // default -- or nDebugSelectVariant == 4
			{
				// not enough registers; load both grids and strides indirectly

				addGridPointers ();
				addStridePointers ();
				addConstants ();
			}
		}
	}
	
	private Set<IntArray> getPositiveStrides ()
	{
		Set<IntArray> set = new HashSet<> ();

		for (IntArray v : m_mapStrideExpressions.keySet ())
		{
			IntArray vNeg = v.neg ();
			if (!set.contains (v) && !set.contains (vNeg))
			{
				// decide whether to add v or vNeg
				int nPosRefs = 0;
				int nNegRefs = 0;
				
				for (Stencil stencil : m_data.getStencilCalculation ().getStencilBundle ())
					for (StencilNode node : stencil)
					{
						int[] rgBaseVector = m_baseVectors.getBaseVector (node);
						if (v.equals (rgBaseVector))
							nPosRefs++;
						else if (vNeg.equals (rgBaseVector))
							nNegRefs++;
					}
				
				set.add (nPosRefs > nNegRefs ? v : vNeg);
			}
		}
		
		return set;
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
		
		for (StencilNode node : setAllGrids)
			m_listGridInputs.add (addGrid (node));
	}
	
	/**
	 * Creates an array holding all the grid pointers and initializes it with
	 * the respective pointers before the inline assembly section.
	 * The address of the array is added as an input to the inline assembly section.
	 */
	private void addGridPointers ()
	{
		m_bUseGridPointers = true;
		
		StencilNodeSet setAllGrids = m_data.getStencilCalculation ().getInputBaseNodeSet ().union (
			m_data.getStencilCalculation ().getOutputBaseNodeSet ());
		
		Specifier specDatatype = new UserSpecifier (new NameID (m_clsDefaultGPRClass.getDatatype ()));
		VariableDeclarator decl = m_data.getCodeGenerators ().getConstantGeneratedIdentifiers ().createDeclarator (
			INPUT_GRIDS_ARRAYPTR, specDatatype, true, setAllGrids.size ());
		
		addInput (INPUT_GRIDS_ARRAYPTR, decl.getID (), EAssemblySectionInputType.CONST_POINTER);
		
		int nIdx = 0;
		for (StencilNode node : setAllGrids)
		{
			m_slbGeneratedCode.addStatement (new ExpressionStatement (new AssignmentExpression (
				new ArrayAccess (decl.getID ().clone (), new IntegerLiteral (nIdx)),
				AssignmentOperator.NORMAL,
				new Typecast (CodeGeneratorUtil.specifiers (specDatatype), getGridPointer (node))
			)));
			
			IOperand op = addGrid (node);
			m_listGridInputs.add (new IOperand.Address (getInput (INPUT_GRIDS_ARRAYPTR), nIdx * m_clsDefaultGPRClass.getWidth () / 8));
			m_mapGridIndices.put (op, nIdx);
			
			nIdx++;
		}
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
			m_mapStrides.put (arrBaseVector, (IOperand.IRegisterOperand) addInput (exprStride, exprStride, EAssemblySectionInputType.CONSTANT));
		}
	}
	
	/**
	 * Counts the number of stride variables that ideally have to be used.
	 * 
	 * @return The ideal number of stride variables
	 */
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
	
	private void addStridePointers ()
	{
		m_bUseStridePointers = true;
		throw new RuntimeException ("Not implemented");
	}
	
	/**
	 * Creates an expression pointing to the grid at the location identified by
	 * the stencil node <code>node</code>.
	 * 
	 * @param node
	 *            The stencil node for which to create the grid address
	 * @return An expression pointing to the grid location at <code>node</code>
	 */
	private Expression getGridPointer (StencilNode node)
	{
		StencilNode nodeOffset = new StencilNode (node);
		nodeOffset.getIndex ().offsetInSpace (m_rgOffset);

		return new UnaryExpression (
			UnaryOperator.ADDRESS_OF,
			m_data.getData ().getMemoryObjectManager ().getMemoryObjectExpression (
				m_sdid, nodeOffset, null, true, true, false, m_slbGeneratedCode, m_options
			)
		);		
	}
	
	/**
	 * Adds a single grid pointer as input.
	 * 
	 * @param node
	 *            The stencil node corresponding to the grid
	 */
	protected IOperand addGrid (StencilNode node)
	{
		IOperand op = m_mapGrids.get (node);
		if (op != null)
			return op;
		
		op = m_bUseGridPointers ?
			new IOperand.PseudoRegister (TypeRegisterType.GPR) :
			addInput (node, getGridPointer (node), EAssemblySectionInputType.VAR_POINTER);
		
		// add the node to the grid
		m_mapGrids.put (node, (IOperand.IRegisterOperand) op);
		
		// add all other nodes which project to the same node
		MemoryObject mo = m_data.getData ().getMemoryObjectManager ().getMemoryObject (m_sdid, node, true);
		for (StencilNode n : m_data.getStencilCalculation ().getStencilBundle ().getFusedStencil ())
			if (n != node && mo.contains (n))
				m_mapGrids.put (n, (IOperand.IRegisterOperand) op);
		
		return op;
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
		StencilCalculation sc = m_data.getStencilCalculation ();
		
		for (DepthFirstIterator it = new DepthFirstIterator (trv); it.hasNext (); )
		{
			Object obj = it.next ();
			if (obj instanceof FloatLiteral)
				addToConstantsAndParamsMap ((FloatLiteral) obj);
			else if (obj instanceof NameID)
			{
				String strName = ((NameID) obj).getName ();
				
				// if the NameID is a stencil parameter, add it to the map
				if (sc.isParameter (strName))
					addToConstantsAndParamsMap ((NameID) obj);
				else if (sc.isReductionVariable (strName))
				{
					StencilCalculation.ReductionVariable rv = sc.getReductionVariable (strName);
					if (rv != null)
						addToConstantsAndParamsMap (rv.getLocalReductionVariable ());
				}
			}
			else if (obj instanceof StencilNode)
			{
				// if the stencil node is contained in the m_mapConstantsAndParams, it has to be
				// a constant (output) node, and since it is already in the map, we know when we
				// get here again that it is reused
				
				if (m_mapConstantsAndParams.containsKey (obj))
					m_mapReusedConstantsAndParams.put ((Expression) obj, null);
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
			if (!StencilAnalyzer.isStencilConstant (stencil, m_data.getStencilCalculation ()))
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
		
		// build the expression array
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
		
		Expression exprConstsAndParamsPtr = m_data.getCodeGenerators ().getSIMDScalarGeneratedIdentifiers ().createVectorizedScalars (
			rgConstsAndParams, getDatatype (), m_slbGeneratedCode, m_options
		);
		if (rgConstsAndParams.length == 1)
			exprConstsAndParamsPtr = new UnaryExpression (UnaryOperator.ADDRESS_OF, exprConstsAndParamsPtr);
		
		addInput (AssemblySection.INPUT_CONSTANTS_ARRAYPTR, exprConstsAndParamsPtr, EAssemblySectionInputType.CONST_POINTER);
	}
	
	/**
	 * Returns an iterable over the register containing the grid pointers.
	 * @return
	 */
	public Iterable<IOperand> getGrids ()
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
		if (nodeRef.getIndex ().getSpaceIndexEx ().length != node.getIndex ().getSpaceIndexEx ().length)
			return false;
		
		// compare all the coordinates except in the first dimension (i==0)
		for (int i = 1; i < nodeRef.getIndex ().getSpaceIndexEx ().length; i++)
			if (!(node.getIndex ().getSpaceIndex (i).equals (nodeRef.getIndex ().getSpaceIndex (i))))
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
		return getGrid (node, nElementsShift, false);
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
	 * @param bAlwaysLoadGrid
	 *            If set to <code>true</code>, if the grid isn't directly an
	 *            input (and consequentially has to be loaded from the input
	 *            grids pointer), always issues and instruction to load the
	 *            grid's address from the input pointer array.
	 * @return The address operand to access <code>node</code>
	 */
	public OperandWithInstructions getGrid (StencilNode node, int nElementsShift, boolean bAlwaysLoadGrid)
	{
		if (LOGGER.isDebugEnabled ())
			LOGGER.debug (StringUtil.concat ("Requesting operand for stencil node ", node.toString ()));

		StencilNode nodeLocal = node;
		int nElementsShiftLocal = nElementsShift;
		
		int nNodeLocalIdx0 = ExpressionUtil.getIntegerValue (nodeLocal.getIndex ().getSpaceIndex (0));
		
		IOperand.IRegisterOperand opBase = m_mapGrids.get (nodeLocal);
		if (opBase == null)
		{
			// no base node found in the map for "node"
			// find a reference node
			StencilNode nodeRef = findReferenceNode (nodeLocal);
			if (nodeRef == null)
				return null;
			
			nElementsShiftLocal += nNodeLocalIdx0 - ExpressionUtil.getIntegerValue (nodeRef.getIndex ().getSpaceIndex (0));
			nodeLocal = nodeRef;
			opBase = m_mapGrids.get (nodeLocal);
		}
		
		// load the grid address if we are using an array of grid pointers and the address hasn't been loaded yet
		IInstruction[] rgPreInstructions = null;
		if (IOperand.PseudoRegister.isPseudoRegisterOfType (opBase, TypeRegisterType.GPR))
		{
			if (bAlwaysLoadGrid || !m_mapGridLoaded.containsKey (opBase))
			{
				rgPreInstructions = new IInstruction[] {
					new Instruction (
						"mov",
						new IOperand.Address (getInput (INPUT_GRIDS_ARRAYPTR), m_mapGridIndices.get (opBase) * m_clsDefaultGPRClass.getWidth () / 8),
						opBase
					)
				};
				
				m_mapGridLoaded.put (opBase, true);
			}
		}
		
		// no index register is needed if the offset is only in the unit stride direction (dimension 0)
		boolean bHasOffsetInNonUnitStride = false || (nElementsShiftLocal > 0);
		if (!bHasOffsetInNonUnitStride)
		{
			int[] rgSpaceIdx = nodeLocal.getSpaceIndex ();
			for (int i = 1; i < rgSpaceIdx.length; i++)
				if (rgSpaceIdx[i] != 0)
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
		int nUnitStrideOffset = (nNodeLocalIdx0 + nElementsShiftLocal * nSIMDVectorLength) *
			ArchitectureDescriptionManager.getTypeSize (specDatatype);
		
		if (!bHasOffsetInNonUnitStride)
			return new OperandWithInstructions (rgPreInstructions, new IOperand.Address (opBase, nUnitStrideOffset), null);
		
		// general case
		// TODO: what if too many strides?

		IntArray v = new IntArray (m_baseVectors.getBaseVector (nodeLocal), true);
		v.set (0, 0);
		
		OperandWithInstructions opStride = getStride (v);
		
		// add the pre-instructions from loading the grid pointer
		IInstruction[] rgPreInstructionsNew = rgPreInstructions == null ?
			opStride.getInstrPre () :
			new IInstruction[(opStride.getInstrPre () == null ? 0 : opStride.getInstrPre ().length) + rgPreInstructions.length];
		if (rgPreInstructions != null)
			for (int i = rgPreInstructionsNew.length - rgPreInstructions.length; i < rgPreInstructionsNew.length; i++)
				rgPreInstructionsNew[i] = rgPreInstructions[i - rgPreInstructionsNew.length + rgPreInstructions.length];
		
		return new OperandWithInstructions (
			rgPreInstructionsNew,
			new IOperand.Address (opBase, (IOperand.IRegisterOperand) opStride.getOp (), m_baseVectors.getScalingFactor (nodeLocal), nUnitStrideOffset),
			opStride.getInstrPost ()
		);
	}
		
	protected void addToConstantsAndParamsMap (Expression expr)
	{
		if (!m_mapConstantsAndParams.containsKey (expr))
		{
			// if the constant or param hasn't been registered yet, add it and give it an ID (an index)
			m_mapConstantsAndParams.put (expr, m_mapConstantsAndParams.size ());
		}
		else
		{
			// if the constant or param has already been registered, mark it for reuse
			// (don't allocate any pseudo register yet into which it will be loaded; that way we can detect
			// while generating the code whether the load instruction has already been generated)
			
			m_mapReusedConstantsAndParams.put (expr, null);
		}
	}

	/**
	 * Determines the number of constants within the entire stencil bundle.
	 * 
	 * @return The number of constants used
	 */
	@Override
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
	public OperandWithInstructions getConstantOrParam (Expression exprConstantOrParam)
	{
		if (LOGGER.isDebugEnabled ())
			LOGGER.debug (StringUtil.concat ("Requesting operand for constant/param ", exprConstantOrParam.toString ()));
		
		IOperand.IRegisterOperand op = null;
		if (m_mapReusedConstantsAndParams.containsKey (exprConstantOrParam))
		{
			op = m_mapReusedConstantsAndParams.get (exprConstantOrParam);
			if (op != null)
				return new OperandWithInstructions (op);
			
			op = new IOperand.PseudoRegister (TypeRegisterType.SIMD);
			m_mapReusedConstantsAndParams.put (exprConstantOrParam, op);
			m_setConstantsAndParamsRegisters.add (op);
		}
		
		int nConstParamIdx = getConstantOrParamIndex (exprConstantOrParam);
		if (nConstParamIdx == -1)
			throw new RuntimeException (StringUtil.concat ("No index for the constant or parameter ", exprConstantOrParam.toString ()));
		
		IOperand opAddr = new IOperand.Address (
			getInput (AssemblySection.INPUT_CONSTANTS_ARRAYPTR),
			nConstParamIdx * m_data.getArchitectureDescription ().getSIMDVectorLengthInBytes ());
		
		IInstruction[] rgInstrMove = null;
		if (op != null)
		{
			// reuse a constant or param: load the value into a pseudo register
			rgInstrMove = new IInstruction[] { new Instruction (TypeBaseIntrinsicEnum.LOAD_FPR_ALIGNED, opAddr, op) };
		}
		
		return new OperandWithInstructions (rgInstrMove, op == null ? opAddr : op, null);
	}

	public boolean isConstantOrParam (IOperand.PseudoRegister reg)
	{
		return m_setConstantsAndParamsRegisters.contains (reg);
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

	public void reset ()
	{
		m_mapGridLoaded.clear ();
		
		for (Expression expr : m_mapConstantsAndParams.keySet ())
			m_mapReusedConstantsAndParams.put (expr, null);
	}
}
