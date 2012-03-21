/*******************************************************************************
 * Copyright (c) 2011 Matthias-M. Christen, University of Basel, Switzerland.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the GNU Lesser Public License v2.1
 * which accompanies this distribution, and is available at
 * http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
 * 
 * Contributors:
 *     Matthias-M. Christen, University of Basel, Switzerland - initial API and implementation
 ******************************************************************************/
package ch.unibas.cs.hpwc.patus.codegen;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.Expression;
import cetus.hir.Identifier;
import cetus.hir.IntegerLiteral;
import cetus.hir.NameID;
import cetus.hir.PointerSpecifier;
import cetus.hir.Specifier;
import cetus.hir.UnaryExpression;
import cetus.hir.UnaryOperator;
import cetus.hir.VariableDeclaration;
import cetus.hir.VariableDeclarator;
import ch.unibas.cs.hpwc.patus.arch.TypeDeclspec;
import ch.unibas.cs.hpwc.patus.ast.IStatementList;
import ch.unibas.cs.hpwc.patus.ast.SubdomainIdentifier;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;
import ch.unibas.cs.hpwc.patus.geometry.Border;
import ch.unibas.cs.hpwc.patus.geometry.Box;
import ch.unibas.cs.hpwc.patus.geometry.Point;
import ch.unibas.cs.hpwc.patus.geometry.Size;
import ch.unibas.cs.hpwc.patus.geometry.Vector;
import ch.unibas.cs.hpwc.patus.representation.StencilNode;
import ch.unibas.cs.hpwc.patus.util.ExpressionUtil;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 * Represents an array of contiguous memory locations.
 * <pre>Type* M_<i>sgit</i>_<i>vecidx</i>[<i>timeidx</i>][<i>spaceinfo</i>];</pre>
 *
 * @author Matthias-M. Christen
 */
public class MemoryObject
{
//	private final static Logger LOGGER = Logger.getLogger (MemoryObject.class);

	private final static boolean USE_MODULUS = false;


	///////////////////////////////////////////////////////////////////
	// Member Variables

	private CodeGeneratorSharedObjects m_data;

	/**
	 * The identifier for the memory object (in the generated code).
	 */
	private Identifier m_idMemoryObject;

//	private int[] m_rgSpatialIndex;

//	/**
//	 * The vector index of the stencil nodes from which the memory object
//	 * was constructed
//	 */
//	private int m_nVectorIndex;

	/**
	 * A reference stencil node
	 */
	private StencilNode m_nodeReference;

	/**
	 * The datatype of the memory object.
	 */
	private Specifier m_specDatatype;

	/**
	 * Flag specifying whether the time index of the reference node is
	 * used. If the time index is used, the memory objects are not
	 * grouped together in an array (the array index being the time
	 * index), but one distinct memory object exists for each time index.
	 * Time indices are used if pointer swapping is used.
	 */
	private boolean m_bUseTimeIndex;

	/**
	 * The reference box (the one of the strategy subdomain iterator)
	 */
	private Box m_boxReference;

	/**
	 * The ghost node layer that is required for this memory object.
	 * The border also determines the local origin, which is equal to
	 * {@link Border#getMin()}.
	 */
	private Border m_border;

	/**
	 * The mask used to project to this memory object (the projection unmask)
	 */
	private IMask m_maskProjectionToMemobj;
	
	/**
	 * The mask used to project to memory object classes (the projection mask)
	 */
	private IMask m_maskGlobalProjection;

	/**
	 * The parallelism level on which this memory object resides
	 */
	private int m_nParallelismLevel;

	/**
	 * A cache for index expressions
	 */
	private IndexExpressionCache m_cacheIndices;


	///////////////////////////////////////////////////////////////////
	// Working Member Variables

	/**
	 * Precalculated expression: localorigin - boxorigin
	 */
	private Size m_sizeLocalOrigin_minus_Origin;


	///////////////////////////////////////////////////////////////////
	// Implementation

	/**
	 *
	 * @param strName
	 * @param bIsBaseGrid
	 * @param specDatatype
	 * @param rgSpatialIndex
	 * @param nTimeIndex
	 * @param nodeReference A reference stencil node (an item in the equivalence class of stencil nodes (that map to the same memory object))
	 * @param nTimeIndicesCount Number of time indices (for the particular vector index)
	 * @param nVectorIndex
	 * @param boxReference
	 * @param border
	 * @param ptIndexOffset The offset in indexing as defined in the stencil definition in the grid parameters to the &quot;operation&quot;.
	 * 	If no grid box size is defined or if this isn't a base memory object, this should be <code>null</code>.
	 * @param bIsMemoryObjectReferenced Specifies whether the memory object referenced in the code and
	 * 	therefore if declarations have to be added
	 * @param cache
	 * @param data
	 */
	public MemoryObject (
		String strName,
		boolean bIsBaseGrid,
		boolean bUseTimeIndex,
		Specifier specDatatype,
		StencilNode nodeReference,
		int nTimeIndicesCount, int nSpaceIndicesCount,
		boolean bIsTimeAdvanceable,
		Box boxReference, Border border, Point ptIndexOffset,
		boolean bIsMemoryObjectReferenced,
		IMask maskProjectionToMemobj, IMask maskGlobalProjection,
		IndexExpressionCache cache, CodeGeneratorSharedObjects data)
	{
		m_data = data;

		m_bUseTimeIndex = bUseTimeIndex;
		m_specDatatype = specDatatype;

		m_nodeReference = new StencilNode (nodeReference);

		// reference box; move to the left if necessary to satisfy SIMD alignment restrictions
		m_boxReference = new Box (boxReference);

		int nSIMDVectorLength = getSIMDVectorLength ();
		if (nSIMDVectorLength > 1)
		{
			Size sizeOffset = new Size (m_boxReference.getDimensionality ());

			// shift the minimum coordinate to floor((minimum coord) / nSIMDVectorLength), i.e.
			// offset the box by
			//    (floor((min coord) / nSIMDVectorLength) - (min coord)) = -(min coord % nSIMDVectorLength)

			Expression exprCoordMin = m_boxReference.getMin ().getCoord (0);

			if (USE_MODULUS)
			{
				// ==> parsing errors (-> simplifying)
				sizeOffset.setCoord (0, new UnaryExpression (
					UnaryOperator.MINUS,
					new BinaryExpression (exprCoordMin.clone (), BinaryOperator.MODULUS, new IntegerLiteral (nSIMDVectorLength)))
				);
			}
			else
			{
				sizeOffset.setCoord (0, new BinaryExpression (
					new BinaryExpression (
						ExpressionUtil.floor (new BinaryExpression (exprCoordMin.clone (), BinaryOperator.DIVIDE, new IntegerLiteral (nSIMDVectorLength))),
						BinaryOperator.MULTIPLY,
						new IntegerLiteral (nSIMDVectorLength)
					),
					BinaryOperator.SUBTRACT,
					exprCoordMin.clone ()));
			}

			m_boxReference.offset (sizeOffset);
		}

		m_border = border == null ? new Border () : border;

		m_maskProjectionToMemobj = maskProjectionToMemobj;
		m_maskGlobalProjection = maskGlobalProjection;
		m_cacheIndices = cache;

		VariableDeclarator decl = new VariableDeclarator (new NameID (strName));

//			new ArraySpecifier (CodeGeneratorUtil.expressions (
//				new IntegerLiteral (nTimeIndicesCount),
//				nSpaceIndicesCount == 1 ? null : new IntegerLiteral (nSpaceIndicesCount))));
		m_idMemoryObject = new Identifier (decl);

		m_sizeLocalOrigin_minus_Origin = new Size (m_border.getMin ());
		m_sizeLocalOrigin_minus_Origin.subtract (m_boxReference.getMin ());
		if (ptIndexOffset != null)
			m_sizeLocalOrigin_minus_Origin.subtract (ptIndexOffset);

		// declare the memory object if it isn't coming from a base grid
		if (bIsMemoryObjectReferenced)
		{
			List<Specifier> listSpecifiers = new ArrayList<> ();

			// if not time-advanceable, create a constant pointer to a constant
			if (!bIsTimeAdvanceable  && !m_data.getData ().isCreatingInitialization ())
				listSpecifiers.add (Specifier.CONST);

			listSpecifiers.addAll (m_data.getArchitectureDescription ().getType (m_specDatatype));
			listSpecifiers.add (PointerSpecifier.UNQUALIFIED);
			listSpecifiers.addAll (m_data.getArchitectureDescription ().getDeclspecs (TypeDeclspec.RESTRICTEDPOINTER));
			listSpecifiers.add (Specifier.CONST);                                 // create a constant pointer (=> the variable can't be assigned another pointer)

			m_data.getData ().addDeclaration (new VariableDeclaration (listSpecifiers, decl));
		}
	}

	/**
	 * Returns the size of the memory object in number of elements (in each direction), enlarged, if necessary.
	 * @return The size of the mememory object
	 */
	public Size getSize ()
	{
		Box box = new Box (m_boxReference);

		// adjust box to comply with SIMD
		int nSIMDVectorLength = getSIMDVectorLength ();
		if (nSIMDVectorLength > 1)
		{
			// we assume that the minimum coordinate of the box is adjusted correctly (to multiples of the SIMD vector length)
			// (this is done in the constructor)
			// => adjust the max coordinate in unit stride direction (dim=0) if necessary, such that nSIMDVectorLength divides
			// (new max + 1)
			// (note that the "+1" is necessary because max coordinates are inclusive)

			// new max = ceil ((max+1) / nSIMDVectorLength) * nSIMDVectorLength - 1

			Expression exprMax = box.getMax ().getCoord (0);
			box.getMax ().setCoord (
				0,
				new BinaryExpression (
					new BinaryExpression (
						ExpressionUtil.ceil (ExpressionUtil.increment (exprMax), new IntegerLiteral (nSIMDVectorLength)),
						BinaryOperator.MULTIPLY,
						new IntegerLiteral (nSIMDVectorLength)),
					BinaryOperator.SUBTRACT,
					Globals.ONE.clone ()
				)
			);
		}

		box.addBorder (
//			new BinaryExpression (new BinaryExpression (exprTotalTimesteps.clone (), BinaryOperator.SUBTRACT, exprLocalTimestep.clone ()), BinaryOperator.ADD, new IntegerLiteral (1)),
			//new BinaryExpression (exprTotalTimesteps.clone (), BinaryOperator.SUBTRACT, exprLocalTimestep.clone ()),
			m_border, nSIMDVectorLength, m_maskProjectionToMemobj);

		// TODO: implement padding
		// size = ceil((oldsize + padding)/L) * L, L=lcm(alignment restrictions)

		return box.getSize ();
	}

	private int getSIMDVectorLength ()
	{
		return m_data.getOptions ().useNativeSIMDDatatypes () ?
			m_data.getArchitectureDescription ().getSIMDVectorLength (m_specDatatype) : 1;
	}

	public Border getBorder ()
	{
		return m_border;
	}

	/**
	 * Returns the identifier associated with the memory object.
	 * @return The memory object's identifier
	 */
	public Identifier getIdentifier ()
	{
		return m_idMemoryObject;
	}

	/**
	 * Specifies whether the time index of the reference node is
	 * used. If the time index is used, the memory objects are not
	 * grouped together in an array (the array index being the time
	 * index), but one distinct memory object exists for each time index.
	 * Time indices are used if pointer swapping is used.
	 */
	public boolean useTimeIndex ()
	{
		return m_bUseTimeIndex;
	}

	/**
	 * Calculates the linear index into the memory object for the stencil node <code>node</code>
	 * based at the reference point <code>ptReference</code> (which is a global index within the memobj's box).
	 * @param boxIterator The reference box, the iterator within the box corresponding to the memory object.
	 * 	Only the box's reference point is of interest
	 * @param node The stencil node
	 * @return The linear index into the memory object
	 */
	public Expression index (SubdomainIdentifier sdid, StencilNode node, Vector vecOffsetWithinMemObj, IStatementList slGeneratedCode, CodeGeneratorRuntimeOptions options)
	{
		return m_cacheIndices.getIndex (sdid, node, vecOffsetWithinMemObj, this, m_sizeLocalOrigin_minus_Origin, slGeneratedCode, options);
	}

	/**
	 *
	 * @param sdid The subdomain identifier from which the index point is calculated
	 * @param node The stencil node for which to compute the index into the memory object array
	 * @param vecOffsetWithinMemObj An offset from the index point defined by
	 * 	<code>sdid</code> or <code>null</code> if no offset is to be added. The offset point must
	 * 	lie within the same memory object.
	 * @param options Code generation options
	 * @return
	 */
	public Expression computeIndex (SubdomainIdentifier sdid, StencilNode node, Vector vecOffsetWithinMemObj, CodeGeneratorRuntimeOptions options)
	{
		// calculate the index (N-dimensional):

		// idx = localorigin +             (refpt - boxorigin) + (stencilnodeoffset - stencilrefoffset)
		//       <rel to stencil ref pt>   <global coords>       <keep vector in the memory object (plane, ...)>

		Point ptIndex = null;
		if (sdid == null)
			ptIndex = Point.getZero (getDimensionality ());
		else
			ptIndex = m_data.getData ().getGeneratedIdentifiers ().getIndexPoint (sdid).clone ().offset (m_maskProjectionToMemobj.apply (node.getIndex ().getSpaceIndex ()));

		// offset the index point
		if (vecOffsetWithinMemObj != null)
			ptIndex.offset (vecOffsetWithinMemObj);

		Size sizeLocalOriginMinusOriginSIMD = m_sizeLocalOrigin_minus_Origin;

		int nSIMDVectorLength = getSIMDVectorLength ();
		if (nSIMDVectorLength > 1)
		{
			// native SIMD datatypes
			sizeLocalOriginMinusOriginSIMD = new Size (m_sizeLocalOrigin_minus_Origin);
			sizeLocalOriginMinusOriginSIMD.setCoord (
				0,
				new BinaryExpression (
					ExpressionUtil.ceil (sizeLocalOriginMinusOriginSIMD.getCoord (0), new IntegerLiteral (nSIMDVectorLength)),
					BinaryOperator.MULTIPLY,
					new IntegerLiteral (nSIMDVectorLength)
				)
			);
		}

		ptIndex.add (sizeLocalOriginMinusOriginSIMD);

		// convert the point index to a linear index
		// convention: unit stride is in the first coordinate
		// index (x1,x2,...,xn) in box of size (w1,w2,...,wn) has the linear index
		//     x1 + x2*w1 + x3*w1*w2 + ... + xn*w1*w2*...*w{n-1} =                             (*)
		//       x1 + w1(x2 + w2(x3 + ... + w{n-2}(x{n-1} + w{n-1}*xn) ... ))
		//
		// SIMD + Padding in unit stride direction:
		// Account for SIMD by dividing the element index by the SIMD vector length
		// (size already accounts for SIMD and padding, but is in number of elements, not SIMD vectors!)

		Size size = getSize ();
		byte nDim = ptIndex.getDimensionality ();
		Expression exprIdx = ptIndex.getCoord (nDim - 1).clone ();
		for (int i = nDim - 2; i >= 0; i--)
		{
			// account for SIMD in the unit stride direction (dim=0)
			Expression exprCoord = ptIndex.getCoord (i).clone ();
			Expression exprSize = size.getCoord (i).clone ();
			if (i == 0 && nSIMDVectorLength > 1)
			{
				exprCoord = new BinaryExpression (exprCoord, BinaryOperator.DIVIDE, new IntegerLiteral (nSIMDVectorLength));
				exprSize = new BinaryExpression (exprSize, BinaryOperator.DIVIDE, new IntegerLiteral (nSIMDVectorLength));
			}

			// multiply / add according to formula (*)
			exprIdx = new BinaryExpression (
				new BinaryExpression (exprSize, BinaryOperator.MULTIPLY, exprIdx.clone ()),
				BinaryOperator.ADD,
				exprCoord);
		}

//		if (MemoryObject.LOGGER.isDebugEnabled ())
//		{
//			MemoryObject.LOGGER.debug (StringUtil.concat (
//				"iterator=", it == null ? "(null)" : it.getLoopHeadAnnotation (),
//				", node=", node.toString (),
//				", index=", exprIdx.toString ()));
//		}

		return exprIdx;//Symbolic.simplify (exprIdx);
	}

	/**
	 * Returns the datatype of the memory object.
	 * @return
	 */
	public Specifier getDatatype ()
	{
		return m_specDatatype;
	}

	public int[] getSpatialIndex ()
	{
		//return m_rgSpatialIndex;
		return m_nodeReference.getSpaceIndex ();
	}

	/**
	 *
	 * @return
	 */
	public int getVectorIndex ()
	{
		//return m_nVectorIndex;
		return m_nodeReference.getIndex ().getVectorIndex ();
	}

	public int getTimeIndex ()
	{
		return m_nodeReference.getIndex ().getTimeIndex ();
	}

	public StencilNode getReferenceStencilNode ()
	{
		return m_nodeReference;
	}

	/**
	 * Returns the parallelism level this memory object lives on.
	 * @return
	 */
	public int getParallelismLevel ()
	{
		return m_nParallelismLevel;
	}

	/**
	 * Returns the memory object's projection mask.
	 * @return The projection mask
	 */
	public IMask getProjectionMask ()
	{
		return m_maskProjectionToMemobj;
	}

	/**
	 * Determines whether the node <code>node</code> is contained in this memory object.
	 * @param node The node for which to determine whether it belongs to this memory object
	 * @return <code>true</code> iff the node <code>node</code> belongs to this memory object
	 */
	public boolean contains (StencilNode node)
	{
		if (node.getIndex ().getTimeIndex () != m_nodeReference.getIndex ().getTimeIndex ())
			return false;
		if (node.getIndex ().getVectorIndex () != m_nodeReference.getIndex ().getVectorIndex ())
			return false;
		
		return Arrays.equals (m_maskGlobalProjection.getEquivalenceClass (m_nodeReference), m_maskGlobalProjection.getEquivalenceClass (node));
	}

	/**
	 * Returns the memory objects dimensionality.
	 * By default, this is the dimensionality of the reference box.
	 * If, however, the size of the box is 1 in all the k last dimensions, <code>dimensionality-k</code> is returned
	 * @return
	 */
	public byte getDimensionality ()
	{
		byte nDim = m_boxReference.getDimensionality ();
		for (int i = nDim - 1; i >= 0; i--)
		{
			if (ExpressionUtil.isValue (m_boxReference.getSize ().getCoord (i), 1))
				nDim--;
			else
				break;
		}

		return nDim;
	}

	@Override
	public String toString ()
	{
		return StringUtil.concat (
			"Name: ", m_idMemoryObject.getName (), "\n",
			"Data type: ", m_specDatatype.toString (), "\n",
			"Reference box: ", m_boxReference.toString (), "\n",
			"Border: ", m_border.toString (), "\n");
	}
}
