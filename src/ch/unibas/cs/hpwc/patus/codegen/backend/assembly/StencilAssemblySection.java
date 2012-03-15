package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.DepthFirstIterator;
import cetus.hir.Expression;
import cetus.hir.FloatLiteral;
import cetus.hir.IntegerLiteral;
import cetus.hir.NameID;
import cetus.hir.Specifier;
import cetus.hir.Traversable;
import cetus.hir.UnaryExpression;
import cetus.hir.UnaryOperator;
import ch.unibas.cs.hpwc.patus.ast.StatementListBundle;
import ch.unibas.cs.hpwc.patus.ast.SubdomainIdentifier;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;
import ch.unibas.cs.hpwc.patus.codegen.MemoryObject;
import ch.unibas.cs.hpwc.patus.codegen.StencilNodeSet;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.IOperand.IRegisterOperand;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;
import ch.unibas.cs.hpwc.patus.representation.FindStencilNodeBaseVectors;
import ch.unibas.cs.hpwc.patus.representation.Stencil;
import ch.unibas.cs.hpwc.patus.representation.StencilCalculation.EArgumentType;
import ch.unibas.cs.hpwc.patus.representation.StencilNode;
import ch.unibas.cs.hpwc.patus.util.ExpressionUtil;
import ch.unibas.cs.hpwc.patus.util.IntArray;

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
		
		m_options = options;
		
		m_slbGeneratedCode = new StatementListBundle ();

		m_mapGrids = new HashMap<StencilNode, IOperand.IRegisterOperand> ();
		m_listGridInputs = new ArrayList<IOperand.IRegisterOperand> ();
		m_mapStrides = new HashMap<IntArray, IOperand.IRegisterOperand> ();
		m_mapConstantsAndParams = new HashMap<Expression, Integer> ();
		
		m_baseVectors = new FindStencilNodeBaseVectors (new int[] { 1, 2, 4, 8 });	// TODO: put this in architecture.xml
		m_specDatatype = null;
		
		m_data.getData ().getMemoryObjectManager ().clear ();
		createInputs ();
	}
	
	/**
	 * Determines the data type of the stencil computation.
	 * Throws an exception if the computation is in mixed precisions.
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
	 * Creates the inputs to the assembly section (i.e., generates the values that are
	 * assigned to registers within the inline assembly input section).
	 */
	protected void createInputs ()
	{
		addGrids ();
		addStrides ();
		
		// add constants
		for (Stencil stencil : m_data.getStencilCalculation ().getStencilBundle ())
			findConstants (stencil.getExpression ());
		
		// add stencil params
		for (String strParamName : m_data.getStencilCalculation ().getArguments (EArgumentType.PARAMETER))
			m_mapConstantsAndParams.put (new NameID (strParamName), m_mapConstantsAndParams.size ());
		
		addConstants ();
	}
	
	private void addGrids ()
	{
		// TODO: get reference nodes for "current" memory objects!!!
		StencilNodeSet setAllGrids = m_data.getStencilCalculation ().getInputBaseNodeSet ().union (m_data.getStencilCalculation ().getOutputBaseNodeSet ());
		
		if (setAllGrids.size () > 10)
		{
			// TODO: "too many grids"
			throw new RuntimeException ("Too many grids...");
		}

		for (StencilNode node : setAllGrids)
			addGrid (node);		
	}
	
	private void addStrides ()
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
	}
	
	/**
	 * Finds all the {@link FloatLiteral} in <code>trv</code> and calls
	 * {@link StencilAssemblySection#addConstant(double, Specifier)} for each of the literals.
	 * 
	 * @param trv
	 */
	private void findConstants (Traversable trv)
	{
		for (DepthFirstIterator it = new DepthFirstIterator (trv); it.hasNext (); )
		{
			Object obj = it.next ();
			if (obj instanceof FloatLiteral)
				m_mapConstantsAndParams.put ((FloatLiteral) obj, m_mapConstantsAndParams.size ());
		}
	}

	/**
	 * Adds a single grid pointer as input.
	 * @param node The stencil node corresponding to the grid
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
		if (m_mapStrides.containsKey (arrBaseVector))
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
		
		m_mapStrides.put (arrBaseVector, (IOperand.IRegisterOperand) addInput (exprStride, exprStride));
	}

	/**
	 * Adds the constants in <code>listConstants</code> to an array and uses the address of
	 * the array as an input for the assembly section. 
	 * @param listConstants The list of constant numbers
	 */
	public void addConstants ()
	{
		if (m_mapConstantsAndParams.size () == 0)
			return;
		
		Expression[] rgConstsAndParams = new Expression[m_mapConstantsAndParams.size ()];
		m_mapConstantsAndParams.keySet ().toArray (rgConstsAndParams);
		
		addInput (
			INPUT_CONSTANTS_ARRAYPTR,
			m_data.getCodeGenerators ().getSIMDScalarGeneratedIdentifiers ().createVectorizedScalars (
				rgConstsAndParams, getDatatype (), m_slbGeneratedCode, m_options)
		);
	}
	
	/**
	 * Determines the number of constants within the entire stencil bundle.
	 * @return The number of constants used
	 */
	public int getConstantsAndParamsCount ()
	{
		return m_mapConstantsAndParams.size ();
	}
	
	/**
	 * Returns an iterable over all the constants and parameters saved in the constant/parameter array.
	 * @return An iterable over constants and parameters saved in the const/param array of the assembly section
	 */
	public Iterable<Expression> getConstantsAndParams ()
	{
		return m_mapConstantsAndParams.keySet ();
	}

	/**
	 * Returns the index of the constant or the stencil parameter <code>exprConstantOrParam</code>
	 * within the constants/parameters map or <code>-1</code> if no such constant/parameter exists
	 * in the map.
	 * @param exprConstantOrParam The constant or parameter for which to retrieve the index
	 * @return The index of <code>exprConstantOrParam</code> within the constants/parameter map
	 * 	or <code>-1</code> if no such constant/parameter exists
	 */
	public int getConstantOrParamIndex (Expression exprConstantOrParam)
	{
		Integer nIdx = m_mapConstantsAndParams.get (exprConstantOrParam);
		return nIdx == null ? -1 : nIdx;
	}

	/**
	 * Returns an iterable over the register containing the grid pointers
	 * @return
	 */
	public Iterable<IOperand.IRegisterOperand> getGrids ()
	{
		return m_listGridInputs;
	}
	
	/**
	 * 
	 * @param node
	 * @param nElementsShift The number of elements by which the node is shifted to the right
	 * 	for unrolling
	 * @return
	 */
	public IOperand getGrid (StencilNode node, int nElementsShift)
	{
		// TODO: if too many grids need to use LEA...
		IOperand.IRegisterOperand opBase = m_mapGrids.get (node);
		
		// no index register is needed if the offset is only in the unit stride direction (dimension 0)
		boolean bHasOffsetInNonUnitStride = false || (nElementsShift > 0);
		if (!bHasOffsetInNonUnitStride)
		{
			for (int i = 1; i < node.getSpaceIndex ().length; i++)
				if (node.getSpaceIndex ()[i] != 0)
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
		int nUnitStrideOffset = (node.getSpaceIndex ()[0] + nElementsShift * nSIMDVectorLength) * AssemblySection.getTypeSize (specDatatype);
		
		if (!bHasOffsetInNonUnitStride)
			return new IOperand.Address (opBase, nUnitStrideOffset);
		
		// general case
		// TODO: what if too many strides?

		IntArray v = new IntArray (m_baseVectors.getBaseVector (node), true);
		v.set (0, 0);
		
		return new IOperand.Address (opBase, m_mapStrides.get (v), m_baseVectors.getScalingFactor (node), nUnitStrideOffset);
	}
		
	/**
	 * 
	 * @param fValue
	 * @param specDatatype
	 * @return
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
	 * @return
	 */
	public StatementListBundle getAuxiliaryStatements ()
	{
		return m_slbGeneratedCode;
	}
}
