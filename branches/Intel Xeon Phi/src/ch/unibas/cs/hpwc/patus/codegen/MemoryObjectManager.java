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
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.log4j.Logger;

import cetus.hir.ArrayAccess;
import cetus.hir.AssignmentExpression;
import cetus.hir.AssignmentOperator;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.CompoundStatement;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.ForLoop;
import cetus.hir.Identifier;
import cetus.hir.IntegerLiteral;
import cetus.hir.NameID;
import cetus.hir.PointerSpecifier;
import cetus.hir.Specifier;
import cetus.hir.Typecast;
import cetus.hir.UnaryExpression;
import cetus.hir.UnaryOperator;
import cetus.hir.VariableDeclaration;
import cetus.hir.VariableDeclarator;
import ch.unibas.cs.hpwc.patus.analysis.StrategyAnalyzer;
import ch.unibas.cs.hpwc.patus.ast.IStatementList;
import ch.unibas.cs.hpwc.patus.ast.RangeIterator;
import ch.unibas.cs.hpwc.patus.ast.StatementListBundle;
import ch.unibas.cs.hpwc.patus.ast.SubdomainIdentifier;
import ch.unibas.cs.hpwc.patus.ast.SubdomainIterator;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;
import ch.unibas.cs.hpwc.patus.geometry.Border;
import ch.unibas.cs.hpwc.patus.geometry.Box;
import ch.unibas.cs.hpwc.patus.geometry.Point;
import ch.unibas.cs.hpwc.patus.geometry.Size;
import ch.unibas.cs.hpwc.patus.geometry.Vector;
import ch.unibas.cs.hpwc.patus.representation.Stencil;
import ch.unibas.cs.hpwc.patus.representation.StencilCalculation;
import ch.unibas.cs.hpwc.patus.representation.StencilNode;
import ch.unibas.cs.hpwc.patus.symbolic.Symbolic;
import ch.unibas.cs.hpwc.patus.util.CodeGeneratorUtil;
import ch.unibas.cs.hpwc.patus.util.ExpressionUtil;
import ch.unibas.cs.hpwc.patus.util.StatementListBundleUtil;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 *
 * @author Matthias-M. Christen
 */
public class MemoryObjectManager
{
	private final static Logger LOGGER = Logger.getLogger (MemoryObjectManager.class);

	public final static String SUFFIX_OUTPUTGRID = "_out";


	///////////////////////////////////////////////////////////////////
	// Inner Types

	/**
	 * Class grouping memory objects into spatial reuse groups.
	 */
	protected static class SpatialMemoryObjects
	{
		private ProjectionMask m_maskProj;
		private List<MemoryObject> m_listMemoryObjects;

		public SpatialMemoryObjects (ProjectionMask mask)
		{
			m_maskProj = mask;
			m_listMemoryObjects = new LinkedList<> ();
		}

		public void addMemoryObject (MemoryObject mo)
		{
			m_listMemoryObjects.add (mo);
		}

		/**
		 * Retrieves a memory object for a stencil node class represented by <code>node</code>.
		 * @param node
		 * @return
		 */
		public MemoryObject getMemoryObject (StencilNode node)
		{
			int[] rgProjectedNode = m_maskProj.apply (node.getSpaceIndex ());
			for (MemoryObject mo : m_listMemoryObjects)
				if (Arrays.equals (rgProjectedNode, mo.getReferenceStencilNode ().getSpaceIndex ()))
					return mo;
			return null;
		}

		public void swapPointers ()
		{

		}

		@Override
		public String toString ()
		{
			StringBuilder sb = new StringBuilder ("[");
			boolean bFirst = true;
			for (MemoryObject mo : m_listMemoryObjects)
			{
				if (!bFirst)
					sb.append (", ");
				sb.append (mo.getIdentifier ().getName ());
				bFirst = false;
			}
			sb.append (']');
			return sb.toString ();
		}
	}

///	private class SpatialMemoryObjects
///	{
///		private StencilNodeSet m_setMemObjs;
///
///		/**
///		 * The total number of memory objects; the sum over the sizes of all equivalence classes
///		 * (equivalence under the &quot;no-reuse mask&quot; &not;&rho;)
///		 */
///		private int m_nTotalMemoryObjects;
///
///		/**
///		 * Maps the equivalence class index to the number of indices in that equivalence class
///		 * (i.e. the size of the pre-image under the &quot;no-reuse mask&quot; &not;&rho;).<br/>
///		 * Key: i &isin; &Sigma;/(&not;&rho &pi;)<br/>
///		 * Value: #{(&not;&rho &pi;)<sup>-1</sup> (i)}
///		 */
///		private Map<IntArray, Integer> m_mapNumberOfIndicesInEquivalenceClass;
///
///		/**
///		 * The start index in the set of ordered memory objects for a given equivalence class.<br/>
///		 * Key: i &isin; &Sigma;/(&not;&rho &pi;)<br/>
///		 * Value: min { Idx } &rarr; &Sigma;/(&not;&rho &pi;)
///		 */
///		private Map<IntArray, Integer> m_mapStartIndicesPerEquivalenceClass;
///
///		/**
///		 * The index for a given memory object within its equivalence class.<br/>
///		 * Key: i &isin; &Sigma;/&pi;<br/>
///		 * Value: Idx &rarr; &Sigma;/(&not;&rho &pi;)
///		 */
///		private Map<IntArray, Integer> m_mapSpatialIndexToEquivClassIndex;
///
///		private ProjectionMask m_maskProj;
///		private ReuseUnmask m_maskNoReuse;
///
///
///		public SpatialMemoryObjects (StencilNodeSet setMemObjs, ProjectionMask maskProj, ReuseUnmask maskNoReuse)
///		{
///			m_setMemObjs = setMemObjs;
///
///			m_maskProj = maskProj;
///			m_maskNoReuse = maskNoReuse;
///
///			m_mapNumberOfIndicesInEquivalenceClass = new HashMap<IntArray, Integer> ();
///			m_mapStartIndicesPerEquivalenceClass = new HashMap<IntArray, Integer> ();
///			m_mapSpatialIndexToEquivClassIndex = new HashMap<IntArray, Integer> ();
///
///			Map<IntArray, List<ISpaceIndexable>> mapEquivClasses = maskNoReuse.getEquivalenceClasses (setMemObjs);
///
///			// get the equivalence classes and sort them by indices
///			List<IntArray> listEquivClassesIndices = new ArrayList<IntArray> (mapEquivClasses.keySet ().size ());
///			listEquivClassesIndices.addAll (mapEquivClasses.keySet ());
///			Collections.sort (listEquivClassesIndices);
///
///			int nEquivClassStartIndex = 0;
///			for (IntArray arrSpaceIdx : listEquivClassesIndices)
///			{
///				// store the start indices
///				int nEquivalentIndicesCount = mapEquivClasses.get (arrSpaceIdx).size ();
///				m_mapNumberOfIndicesInEquivalenceClass.put (arrSpaceIdx, nEquivalentIndicesCount);
///				m_mapStartIndicesPerEquivalenceClass.put (arrSpaceIdx, nEquivClassStartIndex);
///				nEquivClassStartIndex += nEquivalentIndicesCount;
///
///				// build the mapping from spatial indices in the pre-image of the projects
///				// (the elements that are projected to the same equivalence class under maskNoReuse)
///				// to memory object indices
///				List<IntArray> listEquivalentIndices = new ArrayList<IntArray> (nEquivalentIndicesCount);
///				for (ISpaceIndexable si : mapEquivClasses.get (arrSpaceIdx))
///					listEquivalentIndices.add (new IntArray (si.getSpaceIndex ()));
///				Collections.sort (listEquivalentIndices);
///				int nEquivClassIdx = 0;
///				for (IntArray arrIdx : listEquivalentIndices)
///				{
///					m_mapSpatialIndexToEquivClassIndex.put (arrIdx, nEquivClassIdx);
///					nEquivClassIdx++;
///				}
///			}
///
///			m_nTotalMemoryObjects = nEquivClassStartIndex;
///
///			// display some info about the memory objects
///			MemoryObjectManager.LOGGER.debug (StringUtil.concat ("Created spatial memory object ", toString ()));
///		}
///
///		/**
///		 *
///		 * @return
///		 */
///		public int getTotalMemoryObjectsCount ()
///		{
///			return m_nTotalMemoryObjects;
///		}
///
///		/**
///		 *
///		 * @param rgIndex The index: the spatial part of the stencil node
///		 * @return
///		 */
///		public int getNumberOfIndicesInEquivalenceClass (int[] rgIndex)
///		{
///			Integer nCount = m_mapNumberOfIndicesInEquivalenceClass.get (
///				new IntArray (m_maskNoReuse.apply (m_maskProj.apply (rgIndex))));
///			return nCount == null ? 0 : nCount;
///		}
///
///		/**
///		 *
///		 * @param rgIndex
///		 * @return
///		 */
///		public int getStartIndexOfEquivalenceClass (int[] rgIndex)
///		{
///			Integer nLinearStartIdx = m_mapStartIndicesPerEquivalenceClass.get (
///				new IntArray (m_maskNoReuse.apply (m_maskProj.apply (rgIndex))));
///			return nLinearStartIdx == null ? 0 : nLinearStartIdx;
///		}
///
///		/**
///		 * Returns the linear memory object index of <code>rgIndex </code> within the equivalence
///		 * class of <code>rgIndex</code>.
///		 * @param rgIndex
///		 * @return
///		 */
///		public int getIndexInEquivalenceClass (int[] rgIndex)
///		{
///			Integer nLinearIdx = m_mapSpatialIndexToEquivClassIndex.get (
///				new IntArray (m_maskProj.apply (rgIndex)));
///			return nLinearIdx == null ? 0 : nLinearIdx;
///		}
///
///		@Override
///		public String toString ()
///		{
///			return StringUtil.concat (
///				m_setMemObjs == null ? "(empty)" : m_setMemObjs.toString (),
///				"\n\nindex in equivalence class: ", StringUtil.toString (m_mapSpatialIndexToEquivClassIndex),
///				"\n# cosets/equivalence class: ", StringUtil.toString (m_mapNumberOfIndicesInEquivalenceClass),
///				"\nstart indices in equiv cls: ", StringUtil.toString (m_mapStartIndicesPerEquivalenceClass));
///		}
///	}

	/**
	 * Bundles memory object information per {@link SubdomainIdentifier}.
	 */
	protected class MemoryObjects //implements Iterable<MemoryObject>
	{
		private SubdomainIdentifier m_identifier;
		private SubdomainIterator m_iterator;
		private StencilNodeSet m_setAllNodes;

		private ProjectionMask m_maskProj;
		private ProjectionUnmask m_maskProjectToMemobj;
		private ReuseMask m_maskReuse;
		private ReuseUnmask m_maskNoReuse;

		/**
		 * Stencil node set with stencil nodes corresponding to memory objects
		 */
		private StencilNodeSet m_setMemoryObjects;

		/**
		 * Equivalence classes of stencil nodes corresponding to memory objects, equivalence is under the
		 * &quot;no reuse&quot; mask, i.e. two memory objects are equivalent if they map to the same array
		 * of memory objects during iterating in the reuse direction
		 */
		private StencilNodeSet m_setMemoryObjectEquivClasses;

		//private Map<StencilNode, MemoryObject> m_mapStencilNodesToMemoryObjects;

//		/**
//		 * The list of all memory objects
//		 */
//		private List<MemoryObject> m_listMemoryObjects;

		/**
		 * Create the map that contains the number of memory objects for
		 * <ul>
		 * 	<li>each vector index (first dimension)</li>
		 * 	<li>each time index</li>
		 * </ul>
		 */
		private SpatialMemoryObjects[][] m_rgSpatialMemoryObjectInfo;

		/**
		 * The minimum time indices per vector index
		 */
		private int[] m_rgMinTimeIndexPerVectorIndex;

		/**
		 * The maximum time indices per vector index
		 */
		private int[] m_rgMaxTimeIndexPerVectorIndex;

///		/**
///		 * The maximum space info index per vector index
///		 */
///		private int[] m_rgMaxSpaceInfoIndexPerVectorIndex;

		/**
		 * Flag indicating whether the memory object is referenced directly
		 * (i.e. provides local data copies or is the base grid and doesn't have any
		 * children with provide local data copies)
		 */
		private boolean m_bIsMemoryObjectReferenced;


		/**
		 * Constructs a new object bundling the memory objects, the number of memory objects for
		 * given time blocking factors, the maximum time indices per subdomain identifier.
		 * This constructor calculates the desired information.
		 * @param it
		 * @param setAllNodes
		 * @param maskProj
		 */
		public MemoryObjects (
			SubdomainIdentifier sdid, SubdomainIterator it, StencilNodeSet setAllNodes,
			ProjectionMask maskProj, ProjectionUnmask maskProjToMemobj, ReuseMask maskReuse, ReuseUnmask maskNoReuse)
		{
			m_identifier = sdid;
			m_iterator = it;
			m_setAllNodes = setAllNodes;

			m_maskProj = maskProj;
			m_maskReuse = maskReuse;
			m_maskProjectToMemobj = maskProjToMemobj;
			m_maskNoReuse = maskNoReuse;

			m_setMemoryObjects = m_setAllNodes.applyMask (maskProj);
			m_setMemoryObjectEquivClasses = m_setMemoryObjects.applyMask (maskNoReuse);

//			m_mapStencilNodesToMemoryObjects = new HashMap<StencilNode, MemoryObject> ();

			// create the memory objects for the subdomain iterator it
			computeMemoryObjectSpatialInfo ();
			addMemoryObjects ();
		}

		/**
		 * Returns the stencil node set with nodes corresponding to memory objects.
		 * @return The stencil nodes corresponding to memory objects
		 */
		public StencilNodeSet getMemoryObjectStencilNodes ()
		{
			return m_setMemoryObjects;
		}

		/**
		 * Returns the memory object for the stencil node <code>node</code>.
		 * @param node The stencil node for which to return the corresponding memory object
		 * @return The memory obejct corresponding to <code>node</code>
		 */
		public MemoryObject getMemoryObject (StencilNode node)
		{
			SpatialMemoryObjects smos = m_rgSpatialMemoryObjectInfo[node.getIndex ().getVectorIndex ()][node.getIndex ().getTimeIndex () - m_setAll.getMinimumTimeIndex ()];
			return smos.getMemoryObject (node);
			//return m_mapStencilNodesToMemoryObjects.get (node);
		}

		/**
		 * Returns the projection mask.
		 * @return The projection mask
		 */
		public ProjectionMask getProjectionMask ()
		{
			return m_maskProj;
		}

		public ReuseUnmask getNoReuseMask ()
		{
			return m_maskNoReuse;
		}

//		/**
//		 * Returns the list of memory objects in this class.
//		 * @return
//		 */
//		public List<MemoryObject> getMemoryObjects ()
//		{
//			return m_listMemoryObjects;
//		}
//
//		@Override
//		public Iterator<MemoryObject> iterator ()
//		{
//			return m_listMemoryObjects.iterator ();
//		}

		public StencilNodeSet getStencilNodes ()
		{
			return m_setMemoryObjects;
		}

		/**
		 *
		 * @param sgid
		 * @return
		 */
		public StencilNodeSet getFrontStencilNodes ()
		{
			return m_setMemoryObjects.getFront (m_maskReuse.getReuseDimension ());
		}

		/**
		 *
		 * @return
		 */
		public StencilNodeSet getBackStencilNodes ()
		{
			final Object objDummy = new Object ();
			Map<StencilNode, Object> map = new HashMap<StencilNode, Object> ();
			for (StencilNode n : m_setMemoryObjects)
				map.put (n, objDummy);
			for (StencilNode n : getFrontStencilNodes ())
				map.remove (n);

			StencilNode[] rgStencilNodes = new StencilNode[map.size ()];
			int i = 0;
			for (StencilNode n : map.keySet ())
				rgStencilNodes[i++] = n;

			return new StencilNodeSet (rgStencilNodes);
		}

		public StencilNodeSet getInputStencilNodes ()
		{
			return m_setInput;
		}

		public StencilNodeSet getOutputStencilNodes ()
		{
			return m_setOutput;
		}

		/**
		 * Returning the subdomain identifier corresponding to the memory object.
		 * @return The subdomain identifier corresponding to the memory object
		 */
		public SubdomainIdentifier getIdentifier ()
		{
			return m_identifier;
		}

		/**
		 * Returns the maximum time index for a given vector index, <code>nVectorIndex</code>.
		 * @param nVectorIndex The vector index
		 * @return The maximum time index for the vector index <code>nVectorIndex</code>
		 */
		public int getMinTimeIndex (int nVectorIndex)
		{
			return m_rgMinTimeIndexPerVectorIndex[nVectorIndex];
		}

		public int getMinTimeIndex ()
		{
			return m_setAllNodes.getMinimumTimeIndex ();
		}

		/**
		 * Returns the maximum time index for a given vector index, <code>nVectorIndex</code>.
		 * @param nVectorIndex The vector index
		 * @return The maximum time index for the vector index <code>nVectorIndex</code>
		 */
		public int getMaxTimeIndex (int nVectorIndex)
		{
			return m_rgMaxTimeIndexPerVectorIndex[nVectorIndex];
		}

		public int getMaxTimeIndex ()
		{
			return m_setAllNodes.getMaximumTimeIndex ();
		}

///		/**
///		 * Returns the number of time indices for the memory object for a given vector index.
///		 * @param nVectorIndex
///		 * @return
///		 */
///		public int getTimeIndicesCount (int nVectorIndex)
///		{
///			// get the number of time indices (the 0-th entry is always null)
///			return m_rgSpatialMemoryObjectInfo[nVectorIndex].length - 1;
///		}
///
///		public int getMaxSpaceInfoCount (int nVectorIndex)
///		{
///			return m_rgMaxSpaceInfoIndexPerVectorIndex[nVectorIndex];
///		}

		/**
		 * Recursively determines whether the subdomain identifier <code>sdid</code> has a child
		 * grid that has memory objects with local data copies.
		 * @param sdid
		 * @return
		 */
		protected boolean hasChildMemoryObjectsWithLocalDataCopies (SubdomainIdentifier sdid)
		{
			for (SubdomainIdentifier sdidChild : m_data.getCodeGenerators ().getStrategyAnalyzer ().getChildGrids (sdid))
			{
				MemoryObjects mos = m_mapMemoryObjects.get (sdidChild);
				if (mos != null && hasLocalDataCopies (sdidChild))
					return true;
				return hasChildMemoryObjectsWithLocalDataCopies (sdidChild);
			}

			return false;
		}

		/**
		 * Creates the actual memory objects and adds them to the internal data structures.
		 */
		protected void addMemoryObjects ()
		{
			// get the base name for the memory objects
			Box boxReference = m_identifier.getSubdomain ().getBox ();

			// get the equivalence classes
			StencilNodeSet setMemObjs = m_setAllNodes.applyMask (m_maskProj);//.applyMask (m_maskNoReuse);

			// if pointer swapping is possible, create a memory object for each timestep, otherwise summarize them in an array
			boolean bCanUsePointerSwapping = canUsePointerSwapping (m_iterator);

			// create the memory objects
//			m_listMemoryObjects = new ArrayList<MemoryObject> ();


			//Stencil stencil = m_data.getStencilCalculation ().getStencilBundle ().getFusedStencil ();
			//int nVectorIndicesCount = stencil.getNumberOfVectorComponents ();

			int[] rgVectorIndices = m_setAllNodes.getVectorIndices ();

			int nMinVecIdx = rgVectorIndices[0];
			int nMaxVecIdx = rgVectorIndices[rgVectorIndices.length - 1];
			int nVectorIndicesCount = nMaxVecIdx - nMinVecIdx + 1;


			for (int nVecIdx = 0; nVecIdx < nVectorIndicesCount; nVecIdx++)
			{
				// TODO: !!!time indices if no pointer swapping can be used!!!
				for (Integer nTimeIdx = bCanUsePointerSwapping ? m_rgMinTimeIndexPerVectorIndex[nVecIdx] : null; nTimeIdx != null && nTimeIdx <= m_rgMaxTimeIndexPerVectorIndex[nVecIdx]; nTimeIdx++)
				{
					// get the first stencil node for the vector index nVecIdx
					for (StencilNode node : setMemObjs.restrict (nTimeIdx, nVecIdx))
					{
						boolean bIsBaseMemoryObject = m_iterator == null;

						// create the memory object name; if we can use pointer swapping, include time information in the name
						String strName = MemoryObjectManager.createMemoryObjectName (
							bIsBaseMemoryObject && bCanUsePointerSwapping ? null : m_identifier,	// use no identifier for base memory objects if pointer swapping can be used => creates same name as kernel argument identifier
							node,
							bIsBaseMemoryObject ? null : m_maskProj,
							bCanUsePointerSwapping);

						m_bIsMemoryObjectReferenced = hasLocalDataCopies (m_identifier);

						// determine whether, if these are base memory objects, the memory objects are accessed
						// directly and no pointer swapping is used (and hence have to be declared)
						if (!m_bIsMemoryObjectReferenced && bIsBaseMemoryObject && !bCanUsePointerSwapping)
							m_bIsMemoryObjectReferenced = !hasChildMemoryObjectsWithLocalDataCopies (m_identifier);

						// get the border and the offset to the min grid bounds point
						Border border = getBorder (nVecIdx, bIsBaseMemoryObject);

						Point ptOffset = null;
						/* OFFSET ALREADY ACCOUNTED FOR IN THE REFERENCE BOX
						if (bIsBaseMemoryObject)
						{
							StencilCalculation sc = m_data.getStencilCalculation ();
							ArgumentType argType = sc.getArgumentType (strName);

							if (argType != null && (argType instanceof StencilCalculation.GridType) && ((StencilCalculation.GridType) argType).getBoxDimension () != null)
								ptOffset = ((StencilCalculation.GridType) argType).getBoxDimension ().getMin ();
							else
							{
								ptOffset = new Point (sc.getDomainSize ().getMin ());
								ptOffset.subtract (border.getMin ());
							}
						}*/

						// create and add the memory object
						MemoryObjectManager.LOGGER.debug (StringUtil.concat ("Adding memory object ", strName));
						MemoryObject mo = new MemoryObject (
							strName,
							bIsBaseMemoryObject,
							bIsBaseMemoryObject && bCanUsePointerSwapping,	// ??? use time index
							node.getSpecifier (),
							//node.getSpaceIndex (), nVecIdx,
							node,

	///						bCanUsePointerSwapping ? 1 : getTimeIndicesCount (nVecIdx),	// only one time index if we can use pointer swapping
							1,	/// always use pointer swapping (==> TODO: define in strategy!!)

							///getMaxSpaceInfoCount (nVecIdx),
							1,

							node.getIndex ().isAdvanceableInTime (),
							boxReference, border, ptOffset,
							m_bIsMemoryObjectReferenced,
							m_maskProjectToMemobj, m_maskProj,
							m_cacheIndices, m_data);

//						m_listMemoryObjects.add (mo);
						m_rgSpatialMemoryObjectInfo[nVecIdx][nTimeIdx == null ? 0 : nTimeIdx - m_setAll.getMinimumTimeIndex ()].addMemoryObject (mo);
//						m_mapStencilNodesToMemoryObjects.put (node, mo);
					}
				}
			}
		}

		/**
		 * Find the number of memory objects that need to be allocated for time blocking factors.
		 */
		protected void computeMemoryObjectSpatialInfo ()
		{
			int[] rgVectorIndices = m_setAllNodes.getVectorIndices ();

			int nMinVecIdx = rgVectorIndices[0];
			int nMaxVecIdx = rgVectorIndices[rgVectorIndices.length - 1];
			int nVectorIndicesCount = nMaxVecIdx - nMinVecIdx + 1;

			// TODO: consolidate stencil.getMinTimeIndex (), ... functions
//			StencilBundle stencil = m_data.getStencilCalculation ().getStencilBundle ();
//			int nMinTimeIdx = stencil.getMinTimeIndex ();
//			int nMaxTimeIdx = stencil.getMaxTimeIndex ();
///			int nMinTimeIdx = m_setAllNodes.getMinimumTimeIndex ();
///			int nMaxTimeIdx = m_setAllNodes.getMaximumTimeIndex ();

			// timeblocking can only be applied for iterators of co-dimension <= 1 (i.e. plane or cube iterators --
			// stick iterators have too little efficiency: too many data have to be reloaded or recomputed; the
			// shape and iteration direction would need to be adjusted, cf. Strzodka: Cache Oblivious Parallelograms
			// in Iterative Stencil Computations)
///			boolean bCanTimeblock = m_maskProj.getDimension () <= 1 && m_data.getCodeGenerators ().getStrategyAnalyzer ().isTimeblocked ();

			// check whether the memory object occurs in a timeblocked loop
			// if yes, throw an exception since this case isn't supported (see above)
///			if (!bCanTimeblock && m_iterator != null)
///			{
///				if (!ExpressionUtil.isValue (m_data.getCodeGenerators ().getStrategyAnalyzer ().getTimeBlockingFactor (m_iterator /*, m_identifier*/), 1))
///					throw new RuntimeException ("Can't create timeblocked loop for subdomain iterators of co-dimension > 1");
///			}

			// compute the maximum time blocking factor
///			Expression exprMaxTimesteps = m_iterator == null ? null : m_data.getCodeGenerators ().getStrategyAnalyzer ().getMaximumTimstepOfTemporalIterator (m_iterator);
///			int nMaxTimeBlockingFactor = bCanTimeblock ?
///				Math.min (
///					exprMaxTimesteps instanceof IntegerLiteral ? (int) ((IntegerLiteral) exprMaxTimesteps).getValue () : MemoryObjectManager.MAX_TIMEBLOCKING_FACTOR,
///					MemoryObjectManager.MAX_TIMEBLOCKING_FACTOR) :
///				1;
///
			m_rgSpatialMemoryObjectInfo = new SpatialMemoryObjects[nVectorIndicesCount][m_setAll.getMaximumTimeIndex () - m_setAll.getMinimumTimeIndex () + 1];

			m_rgMaxTimeIndexPerVectorIndex = new int[nVectorIndicesCount];
			m_rgMinTimeIndexPerVectorIndex = new int[nVectorIndicesCount];
///			m_rgMaxSpaceInfoIndexPerVectorIndex = new int[nVectorIndicesCount];
			Arrays.fill (m_rgMinTimeIndexPerVectorIndex, 0);
			Arrays.fill (m_rgMaxTimeIndexPerVectorIndex, 0);
///			Arrays.fill (m_rgMaxSpaceInfoIndexPerVectorIndex, 0);

			for (int nVecIdx : rgVectorIndices)
			{
				StencilNodeSet set = m_setAllNodes.restrict (null, nVecIdx);

				// create the spatial memory object information objects
				for (int j = 0; j < m_rgSpatialMemoryObjectInfo[nVecIdx].length; j++)
					m_rgSpatialMemoryObjectInfo[nVecIdx][j] = new SpatialMemoryObjects (m_maskProj);

				// find the maximum time index
				m_rgMaxTimeIndexPerVectorIndex[nVecIdx] = Integer.MIN_VALUE;
				m_rgMinTimeIndexPerVectorIndex[nVecIdx] = Integer.MAX_VALUE;
				for (StencilNode node : set)
				{
					int nTimeIdx = node.getIndex ().getTimeIndex ();
					if (nTimeIdx < m_rgMinTimeIndexPerVectorIndex[nVecIdx])
						m_rgMinTimeIndexPerVectorIndex[nVecIdx] = nTimeIdx;
					if (nTimeIdx > m_rgMaxTimeIndexPerVectorIndex[nVecIdx])
						m_rgMaxTimeIndexPerVectorIndex[nVecIdx] = nTimeIdx;
				}

				// compute the number of memory objects per time blocking factor
///				StencilNodeSet setUnion = new StencilNodeSet (set);

				// iterate over time blocking factors, assign memory object per time blocking factor and local timestep
///				for (int nTimeBlockingFactor = 1; nTimeBlockingFactor <= nMaxTimeBlockingFactor; nTimeBlockingFactor++)
///				{
///					// allocate the spatial memory objects
///					m_rgSpatialMemoryObjectInfo[nVecIdx][nTimeBlockingFactor] = new SpatialMemoryObjects[nTimeBlockingFactor + nMaxTimeIdx - nMinTimeIdx];
///
///					// add the nodes for the time blocking factor to the set
///					setUnion = setUnion.union (set.addTemporalOffset (nTimeBlockingFactor - 1));
///
///					// assign the memory objects at local timesteps
///					for (int nLocalTimeStep = 0; nLocalTimeStep < nTimeBlockingFactor + nMaxTimeIdx - nMinTimeIdx; nLocalTimeStep++)
///					{
///						MemoryObjectManager.LOGGER.debug (StringUtil.concat ("Creating memory objects for subdomain ", m_identifier,
///							", vector_index=", nVecIdx,
///							", time_blocking_factor=", nTimeBlockingFactor,
///							", local_timestep=", nLocalTimeStep));
///
///						m_rgSpatialMemoryObjectInfo[nVecIdx][nTimeBlockingFactor][nLocalTimeStep] =
///							new SpatialMemoryObjects (
///								setUnion.applyMask (m_maskProj).restrict (nLocalTimeStep + nMinTimeIdx, null),
///								m_maskProj,
///								m_maskNoReuse);
///
///						// fill the maximum space index array
///						m_rgMaxSpaceInfoIndexPerVectorIndex[nVecIdx] = Math.max (
///							m_rgMaxSpaceInfoIndexPerVectorIndex[nVecIdx],
///							m_rgSpatialMemoryObjectInfo[nVecIdx][nTimeBlockingFactor][nLocalTimeStep].getTotalMemoryObjectsCount ());
///					}
///				}
			}

			// add the lookup tables for equivalence class count, start indices and spatial-to-linear
			// indices within an equivalence class
///			addLookupTables (nMaxTimeBlockingFactor, nMinTimeIdx, nMaxTimeIdx);
		}


///		/**
///		 * Returns the expression that expresses the the index of the memory object in the memory object array
///		 * partitioned to equivalence classes. (An equivalence class contains all the memory objects that are mapped
///		 * to the same &quot;reuse class&quot;, i.e. that will be cyclically reused.)
///		 * @param node
///		 * @param exprLocalTimestep
///		 * @return
///		 */
///		public Expression getIndexInEquivalenceClass (StencilNode node, Expression exprTimeBlockingFactor, Expression exprLocalTimestep)
///		{
///			int nVectorIndex = node.getIndex ().getVectorIndex ();
///			if ((exprTimeBlockingFactor instanceof IntegerLiteral) && (exprLocalTimestep instanceof IntegerLiteral))
///			{
///				return new IntegerLiteral (
///					m_rgSpatialMemoryObjectInfo
///						[nVectorIndex]
///						[(int) ((IntegerLiteral) exprTimeBlockingFactor).getValue ()]
///						[(int) ((IntegerLiteral) exprLocalTimestep).getValue ()].
///							getIndexInEquivalenceClass (node.getSpaceIndex ()));
///			}
///
///			return new ArrayAccess (
///				m_data.getData ().getGeneratedIdentifiers ().getMemoryObjectIndexIdentifier (m_identifier, nVectorIndex).clone (),
///				CodeGeneratorUtil.expressions (
///					exprTimeBlockingFactor.clone (),
///					exprLocalTimestep.clone (),
///					new IntegerLiteral (m_setMemoryObjects.getLinearSpatialIndex (node.getSpaceIndex ()))));
///		}

///		/**
///		 * Returns the number of memory objects
///		 * @param node The stencil node for which to retrieve the number of memory objects corresponding to the node
///		 * @param exprTimeBlockingFactor The time blocking factor for which to retrieve the number of memory objects (or 1 if no time blocking)
///		 * @param exprLocalTimestep The local timestep within the time blocking loop (or 0 if no time blocking)
///		 * @return
///		 */
///		public Expression getNumberOfMemoryObjectsInEquivalenceClass (StencilNode node, Expression exprTimeBlockingFactor, Expression exprLocalTimestep)
///		{
///			int nVectorIndex = node.getIndex ().getVectorIndex ();
///			if ((exprTimeBlockingFactor instanceof IntegerLiteral) && (exprLocalTimestep instanceof IntegerLiteral))
///			{
///				return new IntegerLiteral (
///					m_rgSpatialMemoryObjectInfo
///						[nVectorIndex]
///						[(int) ((IntegerLiteral) exprTimeBlockingFactor).getValue ()]
///						[(int) ((IntegerLiteral) exprLocalTimestep).getValue ()].
///							getNumberOfIndicesInEquivalenceClass (node.getSpaceIndex ()));
///			}
///
///			return new ArrayAccess (
///				m_data.getData ().getGeneratedIdentifiers ().getMemoryObjectCountIdentifier (m_identifier, nVectorIndex).clone (),
///				CodeGeneratorUtil.expressions (
///					exprTimeBlockingFactor.clone (),
///					exprLocalTimestep.clone (),
///					new IntegerLiteral (m_setMemoryObjectEquivClasses.getLinearSpatialIndex (node.getSpaceIndex ()))));
///		}

///		/**
///		 *
///		 * @param node
///		 * @param exprLocalTimestep
///		 * @return
///		 */
///		public Expression getEquivalenceClassStartIndex (StencilNode node, Expression exprTimeBlockingFactor, Expression exprLocalTimestep)
///		{
///			int nVectorIndex = node.getIndex ().getVectorIndex ();
///			if ((exprTimeBlockingFactor instanceof IntegerLiteral) && (exprLocalTimestep instanceof IntegerLiteral))
///			{
///				return new IntegerLiteral (
///					m_rgSpatialMemoryObjectInfo
///						[nVectorIndex]
///						[(int) ((IntegerLiteral) exprTimeBlockingFactor).getValue ()]
///						[(int) ((IntegerLiteral) exprLocalTimestep).getValue ()].
///							getStartIndexOfEquivalenceClass (node.getSpaceIndex ()));
///			}
///
///			return new ArrayAccess (
///				m_data.getData ().getGeneratedIdentifiers ().getMemoryObjectStartIndexIdentifier (m_identifier, nVectorIndex).clone (),
///				CodeGeneratorUtil.expressions (
///					exprTimeBlockingFactor.clone (),
///					exprLocalTimestep.clone (),
///					new IntegerLiteral (m_setMemoryObjectEquivClasses.getLinearSpatialIndex (node.getSpaceIndex ()))));
///		}

		public boolean areMemoryObjectsReferenced ()
		{
			return m_bIsMemoryObjectReferenced;
		}

		@Override
		public String toString ()
		{
			return StringUtil.concat (
				"iterator:  ", m_iterator == null ? m_identifier.getName () : m_iterator.getLoopHeadAnnotation (), "\n\n",
				"all nodes:\n", m_setAllNodes.toString (), "\n\n",
				"memobjs:\n", m_setMemoryObjects.toString (), "\n\n",
				"memobj equiv classes:\n", m_setMemoryObjectEquivClasses.toString ());
		}
	}


	///////////////////////////////////////////////////////////////////
	// Constants

	public static final int MAX_TIMEBLOCKING_FACTOR = 16;


	///////////////////////////////////////////////////////////////////
	// Member Variables

	/**
	 * The shared data object (used for the information about the stencil structure)
	 */
	private CodeGeneratorSharedObjects m_data;

	/**
	 * Holds the memory objects (per subdomain iterator, which is the key into the map).
	 * Only the memory objects for non-timeblocking are stored here.
	 */
	private Map<SubdomainIdentifier, MemoryObjects> m_mapMemoryObjects;

	/**
	 * The input stencil node set
	 */
	private StencilNodeSet m_setInput;

	/**
	 * The output stencil node set
	 */
	private StencilNodeSet m_setOutput;

	/**
	 * Set of all stencil nodes (input (union) output)
	 */
	private StencilNodeSet m_setAll;

///	private int[] m_rgMinSpaceIndex;

	/**
	 * The border (ghost node zone) required for each memory object per vector index
	 * (the vector index is the key into the map)
	 */
	private Map<Integer, Border> m_mapBorders;

	/**
	 * The maximum border if all base memory objects are to be of the same size.
	 * TODO: make this more general and allow for (optional) grid size specification in the
	 * stencil specification
	 */
	private Border m_borderMaximum;

	/**
	 * Expression cache for memory object indices
	 */
	private IndexExpressionCache m_cacheIndices;

	private Map<SubdomainIterator, Boolean> m_mapCanUsePointerSwapping;
	private int m_nSwapCount;


	///////////////////////////////////////////////////////////////////
	// Implementation

	/**
	 * Creates the memory object manager.
	 */
	public MemoryObjectManager (CodeGeneratorSharedObjects data)
	{
		m_data = data;
		m_cacheIndices = new IndexExpressionCache (m_data);
		m_mapCanUsePointerSwapping = new HashMap<> ();
		m_nSwapCount = 0;

		m_mapMemoryObjects = new HashMap<> ();

		// generate the stencil node sets
		Stencil stencil = m_data.getStencilCalculation ().getStencilBundle ().getFusedStencil ();
		m_setInput = new StencilNodeSet (stencil, StencilNodeSet.ENodeTypes.INPUT_NODES);
		m_setOutput = new StencilNodeSet (stencil, StencilNodeSet.ENodeTypes.OUTPUT_NODES);
		m_setAll = m_setInput.union (m_setOutput);

///		m_rgMinSpaceIndex = stencil.getMinSpaceIndex ();

		// compute the borders for the individual memory objects
		// (computeBorders will only compute the borders for individual vector indices,
		// the border "multiplication" due to local time stepping will be calculated
		// on the fly)
		m_mapBorders = new HashMap<> ();
		m_borderMaximum = null;
	}

	public void initialize ()
	{
		computeBorders ();

		// create the map that will contain the number of memory objects for each time
		// blocking factor add base memory objects, i.e. the grids that are passed to the kernel
		addBaseMemoryObjects ();
	}

	/**
	 * Computes the borders required for the stencil computation.
	 */
	protected void computeBorders ()
	{
		Stencil stencil = m_data.getStencilCalculation ().getStencilBundle ().getFusedStencil ();

		// compute the border (required ghost zone layer thickness) per vector index
		for (int nVecIdx : m_setInput.getVectorIndices ())
		{
			int[] rgMinSpaceIdx = stencil.getMinSpaceIndexByVectorIndex (nVecIdx);
			int[] rgMaxSpaceIdx = stencil.getMaxSpaceIndexByVectorIndex (nVecIdx);

			// make positive (if extends in negative direction)
			for (int i = 0; i < rgMinSpaceIdx.length; i++)
				rgMinSpaceIdx[i] = -rgMinSpaceIdx[i];

			m_mapBorders.put (nVecIdx, new Border (new Size (rgMinSpaceIdx), new Size (rgMaxSpaceIdx)));
		}
	}

	/**
	 * Determines the border for a memory object corresponding to a stencil node with a specific
	 * vector index, <code>nVecIdx</code>.<br/>
	 * If this is a base memory object and bounds are specified in the stencil specification,
	 * calculate the border from the domain size and the bounds; otherwise determine the border
	 * from the stencil nodes.
	 * @param nVecIdx
	 * @param bIsBaseMemoryObject
	 * @return
	 */
	public Border getBorder (int nVecIdx, boolean bIsBaseMemoryObject)
	{
		StencilCalculation sc = m_data.getStencilCalculation ();
		StencilCalculation.GridType argType = sc.getArgumentType (nVecIdx);

		if (bIsBaseMemoryObject && argType != null && argType.getBoxDimension () != null)
		{
			Box boxDomain = sc.getDomainSize ();
			Box boxBounds = argType.getBoxDimension ();

			Expression[] rgMin = new Expression[boxDomain.getDimensionality ()];
			Expression[] rgMax = new Expression[boxDomain.getDimensionality ()];

			for (int i = 0; i < boxDomain.getDimensionality (); i++)
			{
				// calculate min and max border sizes
				// note that both min and max border sizes are made positive

				rgMin[i] = new BinaryExpression (
					boxDomain.getMin ().getCoord (i).clone (),
					BinaryOperator.SUBTRACT,
					boxBounds.getMin ().getCoord (i).clone ());

				rgMax[i] = new BinaryExpression (
					boxBounds.getMax ().getCoord (i).clone (),
					BinaryOperator.SUBTRACT,
					boxDomain.getMax ().getCoord (i).clone ());
			}

			return new Border (new Size (rgMin), new Size (rgMax));
		}


		// if stencil is specified to have equally sized base grids, return the maximum border
		if (sc.getGridSizeOptions () == StencilCalculation.EGridSizeOptions.EQUAL_BASE && bIsBaseMemoryObject)
		{
			if (m_borderMaximum == null)
			{
				// find the maximum sizes
				byte nDim = m_data.getStencilCalculation ().getDimensionality ();
				int[] rgMin = new int[nDim];
				int[] rgMax = new int[nDim];

				for (Border b : m_mapBorders.values ())
				{
					Size sizeMin = b.getMin ();
					Size sizeMax = b.getMax ();

					for (int i = 0; i < nDim; i++)
					{
						Expression exprMin = sizeMin.getCoord (i);
						Expression exprMax = sizeMax.getCoord (i);
						int nMin = exprMin == null ? 0 : ExpressionUtil.getIntegerValue (exprMin);
						int nMax = exprMax == null ? 0 : ExpressionUtil.getIntegerValue (exprMax);

						rgMin[i] = Math.max (nMin, rgMin[i]);
						rgMax[i] = Math.max (nMax, rgMax[i]);
					}
				}

				// create the new border
				m_borderMaximum = new Border (new Size (rgMin), new Size (rgMax));
			}

			return m_borderMaximum;
		}

		return m_mapBorders.get (nVecIdx);
	}

	/**
	 * Builds the name for a particular memory object array.
	 * 
	 * @param sdid
	 *            The strategy grid identifier
	 * @param node
	 *            The stencil node
	 * @return The memory object identifier name for the memory object array for
	 *         a grid identifier and stencil node
	 */
	public static String createMemoryObjectName (SubdomainIdentifier sdid, StencilNode node, ProjectionMask mask, boolean bIncludeTimeInformation)
	{
		String strTimeInformation = null;
		if (bIncludeTimeInformation)
		{
			int nTimeIdx = node.getIndex ().getTimeIndex ();
			strTimeInformation = StringUtil.concat ("_", StringUtil.num2IdStr (nTimeIdx));
		}

		StringBuilder sbSpaceInformation = new StringBuilder ();
		if (mask != null)
		{
			int[] rgInfo = mask.apply (node.getSpaceIndex ());
			for (int nInfo : rgInfo)
			{
				if (nInfo < 0)
					sbSpaceInformation.append ('m');
				sbSpaceInformation.append (Math.abs (nInfo));
				sbSpaceInformation.append ('_');
			}
			sbSpaceInformation.append ('_');
		}

		return StringUtil.concat (
			sdid == null ? null : sdid.getName (), sdid == null ? null : "__",
			sbSpaceInformation.toString (),
			node.getName (), "_", String.valueOf (node.getIndex ().getVectorIndex ()),
			strTimeInformation);
	}

	/**
	 * Determines whether the iterator belonging to <code>sgid</code> is on a
	 * parallelism level
	 * that has its own local data copies.
	 * 
	 * @param sdid
	 *            The subdomain identifier for which to determine whether local
	 *            data copies are desired
	 * @return <code>true</code> iff <code>sgid</code> has local data copies
	 */
	protected boolean hasLocalDataCopies (SubdomainIdentifier sdid)
	{
		SubdomainIterator it = m_data.getCodeGenerators ().getStrategyAnalyzer ().getIteratorForSubdomainIdentifier (sdid);
		return it == null ? false : m_data.getArchitectureDescription ().hasExplicitLocalDataCopies (it.getParallelismLevel ());
	}

	/**
	 * Finds a subdomain iterator with memory objects (above in the hierarchy or
	 * identical to <code>it</code>) and returns the list of memory objects.
	 * 
	 * @param sdid
	 *            The subdomain iterator above or in which to look for memory
	 *            objects
	 * @param bOnlyFindMemoryObjectsWithLocalDataCopies
	 *            Finds only memory objects that are
	 *            attached to a subdomain iterator on a parallelism level that
	 *            has its own local data copies
	 * @return The list of memory objects belonging to <code>it</code> or a
	 *         subdomain iterator above <code>it</code>
	 */
	protected MemoryObjects findMemoryObjects (SubdomainIdentifier sdid, boolean bOnlyFindMemoryObjectsWithLocalDataCopies)
	{
		//sdid = getActualSubdomainIdentifier (sdid);

		// find the subdomain above or in sdid with memory objects attached to it
		while (sdid != null)
		{
			MemoryObjects mos = m_mapMemoryObjects.get (sdid);
			if (mos != null)
			{
				if (bOnlyFindMemoryObjectsWithLocalDataCopies)
				{
					if (hasLocalDataCopies (sdid))
						return mos;
				}
				else
					return mos;
			}

			sdid = m_data.getCodeGenerators ().getStrategyAnalyzer ().getParentGrid (sdid);
		}

		// no memory objects found; return the root memory objects
		return m_mapMemoryObjects.get (m_data.getCodeGenerators ().getStrategyAnalyzer ().getRootSubdomain ());
	}

	/**
	 * Returns the parent memory object of the memory object used in the iterator <code>it</code>.
	 * E.g., data is loaded from/stored to the parent memory object.
	 * @param it
	 * @param nVectorIdx
	 * @return
	 */
	public MemoryObject getParentMemoryObject (SubdomainIterator it, StencilNode node)
	{
		// find the subdomain above it.getIterator () in the subdomain hierarchy with memory objects attached to it
		MemoryObjects mosParent = findMemoryObjects (it.getDomainIdentifier (), false);
		if (mosParent == null)
			throw new RuntimeException ("Error while finding the parent memory objects");

		MemoryObject mo = mosParent.getMemoryObject (node);
		if (mo != null)
			return mo;

//		List<MemoryObject> listParentObjects = mosParent.getMemoryObjects ();
//		if (listParentObjects == null || listParentObjects.size () == 0)
//			throw new RuntimeException ("Error while finding the parent memory objects");
//
//		// find the memory object in which the stencil node lies in
//		for (MemoryObject mo : listParentObjects)
//		{
//			if (/*mo.getTimeIndex () == node.getIndex ().getTimeIndex () &&*/ mo.getVectorIndex () == nVectorIdx)
//				return mo;
//		}

		// no matching memory object could be found
		throw new RuntimeException ("Error while finding the parent memory objects");
	}

	/**
	 * Returns the parent memory object of the memory object used in the iterator <code>it</code>.
	 * E.g., data is loaded from/stored to the parent memory object.
	 * @param it
	 * @param mo
	 * @return
	 */
	public MemoryObject getParentMemoryObject (SubdomainIterator it, MemoryObject mo)
	{
		return getParentMemoryObject (it, mo.getReferenceStencilNode ());
	}

	/**
	 * Adds the base memory objects to the memory object manager.
	 * Base memory objects are the memory portions that are passed to the
	 * kernel.
	 */
	protected void addBaseMemoryObjects ()
	{
		// get the base subdomain identifier
		SubdomainIdentifier sdidBase = m_data.getStrategy ().getBaseDomain ();

		// construct masks; assume this is a full box (no non-trivial projections/reuse)
		// (if the size of sdidBase was used, non-trivial projection/reuse masks would be
		// created if the box is only 1 layer high in one dimension)
		Vector v = Vector.getZeroVector (sdidBase.getDimensionality ());
		ProjectionMask maskProj = new ProjectionMask (v);
		ReuseMask maskReuse = new ReuseMask (v);
		ReuseUnmask maskNoReuse = new ReuseUnmask (v);

		// get the stencil node set containing all the nodes that we have to take into account
		StencilNodeSet setAllNodes = getAllStencilNodes (
			maskProj, maskReuse, m_data.getArchitectureDescription ().supportsAsynchronousIO (1));

		// construct the MemoryObjects object and store it in the map
		m_mapMemoryObjects.put (sdidBase, new MemoryObjects (sdidBase, null, setAllNodes, maskProj, new ProjectionUnmask (v), maskReuse, maskNoReuse));
	}

	/**
	 * Creates the memory objects and creates the allocation code if it has been
	 * specified that the memory objects hold a local data copy.
	 * 
	 * @param it
	 *            The subdomain iterator for which to create the memory objects
	 */
	public void allocateMemoryObjects (SubdomainIterator it, StatementListBundle slbGenerated, CodeGeneratorRuntimeOptions options)
	{
		ProjectionMask maskProj = new ProjectionMask (it);
		ReuseMask maskReuse = new ReuseMask (it);
		ReuseUnmask maskNoReuse = new ReuseUnmask (it);

		StencilNodeSet setAllNodes = getAllStencilNodes (
			maskProj, maskReuse, m_data.getArchitectureDescription ().supportsAsynchronousIO (it.getParallelismLevel ()));

		SubdomainIdentifier sdid = it.getIterator ();
		MemoryObjects mos = new MemoryObjects (sdid, it, setAllNodes, maskProj, new ProjectionUnmask (it), maskReuse, maskNoReuse);
		m_mapMemoryObjects.put (sdid, mos);

		// initialize the iteration counter
		Identifier idCounter = m_data.getData ().getGeneratedIdentifiers ().getLoopCounterIdentifier (it.getIterator ());
		slbGenerated.addStatement (new ExpressionStatement (new AssignmentExpression (idCounter.clone (), AssignmentOperator.NORMAL, new IntegerLiteral (0))));

		// allocate for each local timestep

//XXX ??????? how to replace this?
		StatementListBundle slbInner = slbGenerated;
		Expression exprTimestepsCount = m_data.getCodeGenerators ().getStrategyAnalyzer ().getMaximumTotalTimestepsCount (it);
		ForLoop loop = null;
		if (m_data.getCodeGenerators ().getStrategyAnalyzer ().isTimeblocked ())
		{
			Identifier idLoopIdx = m_data.getData ().getGeneratedIdentifiers ().getTimeIndexIdentifier (sdid);
			loop = new ForLoop (
				new ExpressionStatement (new AssignmentExpression (idLoopIdx.clone (), AssignmentOperator.NORMAL, new IntegerLiteral (0))),
				new BinaryExpression (idLoopIdx, BinaryOperator.COMPARE_LT, exprTimestepsCount),
				new UnaryExpression (UnaryOperator.PRE_INCREMENT, idLoopIdx.clone ()),
				new CompoundStatement ());
			slbInner = new StatementListBundle ();
		}

		// create the allocation code
		for (StencilNode n : mos.getStencilNodes ())
		{
			MemoryObject mo = getMemoryObject (sdid, n, true);
			m_data.getCodeGenerators ().getBackendCodeGenerator ().allocateData (
				n, mo, getMemoryObjectExpression (sdid, n, null, false, false, false, slbInner, options), it.getParallelismLevel (), slbInner);
		}

		if (loop != null)
		{
			StatementListBundle slbLoop = new StatementListBundle (loop);
			StatementListBundleUtil.addToLoopBody (slbLoop, slbInner);
			slbGenerated.addStatements (slbLoop);
		}
//XXX <----------
	}

	/**
	 *
	 * @param maskProj
	 * @param maskReuse
	 * @param bHasAsyncIO
	 * @return
	 */
	protected StencilNodeSet getAllStencilNodes (ProjectionMask maskProj, ReuseMask maskReuse, boolean bHasAsyncIO)
	{
		// construct the preload set
		StencilNodeSet setDeferredIO = null;
		if (bHasAsyncIO && maskReuse.getReuseDimension () >= 0)
		{
			// TODO: scale the vector if non-unit steps are performed

			// offset the union of the input and output nodes by one in the reuse direction
			// and get the front nodes in the reuse direction
			// TODO: distinguish between load and store asynchronicity!!
			setDeferredIO = m_setAll.addSpatialOffset (maskReuse.getVector ()).getFront (maskReuse.getReuseDimension ());
		}

		// the memory object set contains the input, output and preload nodes
		// as well as all the nodes between the min and max nodes in reuse direction
		// (that's what the "fill" method does)
		StencilNodeSet setAllNodes = new StencilNodeSet (m_setAll);
		if (setDeferredIO != null)
			setAllNodes = setAllNodes.union (setDeferredIO);
		if (maskReuse.getReuseDimension () >= 0)
		{
			// TODO: adjust for non-unity steps
			setAllNodes = setAllNodes.fill (maskReuse.getReuseDimension ());
		}

		return setAllNodes;
	}

//	/**
//	 * Finds the {@link MemoryObject} with vector index <code>nVectorIndex</code> in the <code>list</code>
//	 * of memory objects. Returns <code>null</code> if no such memory object can be found.
//	 * @param list The list to search for a memory object with vector index <code>nVectorIndex</code>
//	 * @param nVectorIndex The vector index to look for
//	 * @return The memory object in <code>list</code> with vector index <code>nVectorIndex</code> or <code>null</code>
//	 */
//	protected MemoryObject findMemoryObjectVectorIndex (List<MemoryObject> list, int nVectorIndex, int nTimeIndex)
//	{
//		for (MemoryObject mo : list)
//			if (mo.getVectorIndex () == nVectorIndex && (!mo.useTimeIndex () || (mo.useTimeIndex () && mo.getTimeIndex () == nTimeIndex)))
//				return mo;
//		return null;
//	}

//	/**
//	 * Returns the subdomain identifier that is actually referenced by <code>sgid</code>:
//	 * Subdomain identifiers can be subscripted with other subdomain identifiers, in which
//	 * case this method finds the inner most subdomain identifier and returns it.
//	 * @param sgid
//	 * @return
//	 */
//	protected SubdomainIdentifier getActualSubdomainIdentifier (SubdomainIdentifier sgid)
//	{
//		if (true)
//			return sgid;
//
//		//XXX !!!rgSpatialIndex[i] is always an expression!!!
//		for (SubdomainIdentifier s = sgid; ; )
//		{
//			Expression[] rgSpatialIndex = null;//s.getSpatialIndex ();
//			if (rgSpatialIndex == null || rgSpatialIndex.length > 1 || !(rgSpatialIndex[0] instanceof SubdomainIdentifier))
//				return s;
//			s = (SubdomainIdentifier) rgSpatialIndex[0];
//		}
//	}
//
	/**
	 * Gets the memory object for the subdomain identifier <code>sdid</code> and the stencil node
	 * <code>node</code> (representing its class of nodes).
	 * @param sdid
	 * @param node
	 * @param bIsDataAccess
	 * @return
	 */
	public MemoryObject getMemoryObject (SubdomainIdentifier sdid, StencilNode node, boolean bIsDataAccess)
	{
		if (MemoryObjectManager.LOGGER.isDebugEnabled ())
			MemoryObjectManager.LOGGER.debug (StringUtil.concat ("Getting memory object for subdomain identifier ", sdid, " and stencil node ", node));

		// find the memory objects for the subdomain identifier
		MemoryObjects mos = findMemoryObjects (/*getActualSubdomainIdentifier*/ (sdid), bIsDataAccess);
		if (mos == null)
			throw new RuntimeException ("Failed to find memory objects");

		// find the memory object with the desired vector index
		MemoryObject mo = mos.getMemoryObject (node); //findMemoryObjectVectorIndex (mos.getMemoryObjects (), node.getIndex ().getVectorIndex (), node.getIndex ().getTimeIndex ());
		if (mo == null)
			throw new RuntimeException ("Failed to find memory objects");

		return mo;
	}

	/**
	 * Constructs an expression to access a memory object array.
	 * 
	 * @param sdid
	 *            The subdomain identifier for which to generate the memory
	 *            object expression
	 * @param node
	 *            The stencil node that is accessed
	 * @param vecOffsetWithinMemObj
	 *            An offset from the index point defined by <code>sdid</code> or
	 *            <code>null</code> if no offset is to be added. The offset
	 *            point must lie within the same memory object.
	 * @param bIsDataAccess
	 * @param bIndex
	 *            Tells the method to index the memory object to access a single
	 *            grid point if set to <code>true</code>, if set to <code>false</code>
	 *            a pointer to the memory object is returned
	 * @param bAccessParentMemoryObject
	 *            Specify whether the parent memory object should be accessed
	 *            instead of the memory object associated with <code>sdid</code>.
	 * @param slGenerated
	 *            The list of statements to which index calculations will be
	 *            added. Can be <code>null</code> if <code>bIndex</code> is set
	 *            to <code>false</code>
	 * @param options
	 * @return
	 */
	public Expression getMemoryObjectExpression (
		SubdomainIdentifier sdid, StencilNode node, Vector vecOffsetWithinMemObj,
		boolean bIsDataAccess, boolean bIndex, boolean bAccessParentMemoryObject,
		IStatementList slGenerated, CodeGeneratorRuntimeOptions options)
	{
		boolean bNoVectorize = options.getBooleanValue (CodeGeneratorRuntimeOptions.OPTION_NOVECTORIZE, false);
		boolean bUseSIMD = m_data.getArchitectureDescription ().useSIMD ();
		boolean bUseNativeSIMD = m_data.getOptions ().useNativeSIMDDatatypes ();

		// check whether the node represents a scalar and return the identifier for the scalar variable in that case
		if (node.isScalar ())
		{
			// determine whether to create special vector identifiers:
			// if vectorization is turned on and no native vector datatypes are used and vectorization is enabled for the current code generation phase
			boolean bCreateVectorizedIdentifier = bUseNativeSIMD || (bUseSIMD && !bNoVectorize);

			VariableDeclarator decl = new VariableDeclarator (new NameID (node.getName ()));
			m_data.getData ().addDeclaration (new VariableDeclaration (
				bCreateVectorizedIdentifier ?
					m_data.getArchitectureDescription ().getType (node.getSpecifier ()) :
					CodeGeneratorUtil.specifiers (node.getSpecifier ()),
				decl));

			return new Identifier (decl);
		}

		// get the actual subdomain identifier (subdomain identifiers might be subscripted with other subdomain identifiers)
		SubdomainIdentifier sdidActual = sdid;//getActualSubdomainIdentifier (sdid);

		// get the strategy time offset
		MemoryObjects mos = findMemoryObjects (sdidActual, bIsDataAccess);
		if (mos == null)
			throw new RuntimeException (StringUtil.concat ("Can't find memory object for subdomain identifier ", sdidActual.getName ()));

		StrategyAnalyzer analyzer = m_data.getCodeGenerators ().getStrategyAnalyzer ();
		SubdomainIterator it = analyzer.getIteratorForSubdomainIdentifier (sdidActual);
		if (it == null)
			it = analyzer.getOuterMostSubdomainIterator ();

///		Expression exprTimeblockingFactor = analyzer.getTimeBlockingFactor (it/*, sdidActual*/);
///		Expression exprLocalTimestep = analyzer.getLocalTimestep (it, sdidActual, sdidActual.getTemporalIndex ());
		Identifier idInnermostEnclosingTimeIndex = analyzer.getEnclosingTemporalLoop (it).getLoopIndex ();

		int nVectorIndex = node.getIndex ().getVectorIndex ();
		int nMinStencilNodeTimeIndex = mos.getMinTimeIndex (nVectorIndex);
		int nMaxStencilNodeTimeIndex = mos.getMaxTimeIndex (nVectorIndex);

		// calculate the time index
		Expression exprTimeIndex = null;
///		if (mos.getMaxSpaceInfoCount (nVectorIndex) == 1)
		{
			// only one equivalence class: swap among local timesteps

			if (!canUsePointerSwapping (it))
			{
				// (t<inner most enclosing time idx> + local_timestep + stencilnode_timeindex{normalized to min -> 0}) % #local_timesteps
				exprTimeIndex = Symbolic.simplify (
					new BinaryExpression (
						ExpressionUtil.sum (
							idInnermostEnclosingTimeIndex.clone (),
							///exprLocalTimestep,
							new IntegerLiteral (node.getIndex ().getTimeIndex () - nMinStencilNodeTimeIndex)),
						BinaryOperator.MODULUS,

						///new BinaryExpression (exprTimeblockingFactor, BinaryOperator.ADD, new IntegerLiteral (nMaxStencilNodeTimeIndex - nMinStencilNodeTimeIndex))	//???
						new IntegerLiteral (nMaxStencilNodeTimeIndex - nMinStencilNodeTimeIndex)
						///
					),
					Symbolic.ALL_VARIABLES_INTEGER);
			}
		}
///		else
///		{
///			// more than one equivalence class: swap within local timestep
///
///			// global_timestep - (t<inner most enclosing time idx> + local_timestep + stencilnode_timeindex{normalized to min -> 0})
///			exprTimeIndex = Symbolic.simplify (
///				new BinaryExpression (
///					m_data.getCodeGenerators ().getStrategyAnalyzer ().getTimeIndexVariable (),
///					BinaryOperator.SUBTRACT,
///					ExpressionUtil.sum (
///						idInnermostEnclosingTimeIndex.clone (),
///						///exprLocalTimestep,
///						new IntegerLiteral (node.getIndex ().getTimeIndex () - nMinStencilNodeTimeIndex)
///					)
///				),
///				Symbolic.ALL_VARIABLES_INTEGER);
///		}

		// calculate the spatial information (the second array index)
		Expression exprSpatialInfo = null;
///		if (mos.getMaxSpaceInfoCount (nVectorIndex) > 1)
///		{
///			// (loop_counter + idx_in_equiv_class) % num_equiv_classes + equiv_class_start_index
///			exprSpatialInfo = Symbolic.simplify (
///				new BinaryExpression (
///					new BinaryExpression (
///						new BinaryExpression (
///							m_data.getData ().getGeneratedIdentifiers ().getLoopCounterIdentifier (mos.getIdentifier ()).clone (),
///							BinaryOperator.ADD,
///							mos.getIndexInEquivalenceClass (node, exprTimeblockingFactor, exprLocalTimestep)
///						),
///						BinaryOperator.MODULUS,
///						mos.getNumberOfMemoryObjectsInEquivalenceClass (node, exprTimeblockingFactor, exprLocalTimestep)
///					),
///					BinaryOperator.ADD,
///					mos.getEquivalenceClassStartIndex (node, exprTimeblockingFactor, exprLocalTimestep)
///				),
///				Symbolic.ALL_VARIABLES_INTEGER);
///		}

		// calculate the spatial index
		MemoryObject mo = bAccessParentMemoryObject ? getParentMemoryObject (it, node) : getMemoryObject (sdidActual, node, bIsDataAccess);
		Expression exprSpatialIndex = null;
		if (bIndex)
			exprSpatialIndex = mo.index (sdid, node, vecOffsetWithinMemObj, slGenerated, options).clone ();

		// build the complete array access expression
		if (exprTimeIndex == null && exprSpatialInfo == null && exprSpatialIndex == null)
			return mo.getIdentifier ().clone ();

		return getMemoryObjectExpression (node, mo, exprTimeIndex, exprSpatialInfo, exprSpatialIndex, options);
	}

	public Expression getMemoryObjectExpression (StencilNode node, MemoryObject mo,
		Expression exprTimeIndex, Expression exprSpatialInfo, Expression exprSpatialIndex,
		CodeGeneratorRuntimeOptions options)
	{
		boolean bNoVectorize = options.getBooleanValue (CodeGeneratorRuntimeOptions.OPTION_NOVECTORIZE, false);
		boolean bUseSIMD = m_data.getArchitectureDescription ().useSIMD ();
		boolean bUseNativeSIMD = m_data.getOptions ().useNativeSIMDDatatypes ();

		Expression exprResult = new ArrayAccess (
			mo.getIdentifier ().clone (),
			CodeGeneratorUtil.expressions (exprTimeIndex, exprSpatialInfo, exprSpatialIndex));

		// if no native SIMD types are used, but this call is within a vectorized section,
		// we need to cast the types to native SIMD types
		boolean bIsStencilCalculation = options.hasValue (CodeGeneratorRuntimeOptions.OPTION_STENCILCALCULATION, CodeGeneratorRuntimeOptions.VALUE_STENCILCALCULATION_STENCIL);
		if (!bUseNativeSIMD && bUseSIMD && !bNoVectorize && bIsStencilCalculation)
		{
			List<Specifier> listSpecs = new ArrayList<> ();
			listSpecs.addAll (m_data.getArchitectureDescription ().getType (node.getSpecifier ()));
			listSpecs.add (PointerSpecifier.UNQUALIFIED);

			return new UnaryExpression (
				UnaryOperator.DEREFERENCE,
				new Typecast (listSpecs, new UnaryExpression (UnaryOperator.ADDRESS_OF, exprResult)));
		}

		return exprResult;
	}

	public StencilNodeSet getInputStencilNodes (SubdomainIdentifier sdid)
	{
		MemoryObjects mos = m_mapMemoryObjects.get (sdid);
		return mos == null ? new StencilNodeSet () : mos.getInputStencilNodes ();
	}

	public StencilNodeSet getFrontStencilNodes (SubdomainIdentifier sdid)
	{
		MemoryObjects mos = m_mapMemoryObjects.get (sdid);
		return mos == null ? new StencilNodeSet () : mos.getFrontStencilNodes ().restrict (mos.getMinTimeIndex (), null);
	}

	public StencilNodeSet getBackStencilNodes (SubdomainIdentifier sdid)
	{
		MemoryObjects mos = m_mapMemoryObjects.get (sdid);
		return mos == null ? new StencilNodeSet () : mos.getBackStencilNodes ().restrict (mos.getMinTimeIndex (), null);
	}

	public StencilNodeSet getOutputStencilNodes (SubdomainIdentifier sdid)
	{
		MemoryObjects mos = m_mapMemoryObjects.get (sdid);
		return mos == null ? new StencilNodeSet () : mos.getOutputStencilNodes ();
	}

	/**
	 * Determines whether the memory object belonging to <code>sdid</code> are referenced directly.
	 * If <code>sdid</code> does not have any memory objects, <code>false</code> is returned.
	 * @param sdid
	 * @return
	 */
	public boolean areMemoryObjectsReferenced (SubdomainIdentifier sdid)
	{
		MemoryObjects mos = m_mapMemoryObjects.get (sdid);
		if (mos == null)
			return false;
		return mos.areMemoryObjectsReferenced ();
	}

	/**
	 * Determine whether pointer swapping can be used for the current
	 * configuration.<br/>
	 * Pointer swapping can be used if
	 * <ul>
	 * 	<li>There is no timeblocking</li>
	 * 	<li></li>
	 * </ul>
	 * If no pointer swapping can be used, the grid arrays are indexed as
	 * u[temporal][spaceinfo][idx], where temporal is an expression similar
	 * to (t+k)%m, where t is the local timestep.
	 * 
	 * @return <code>true</code> iff pointer swapping can be used
	 */
	public boolean canUsePointerSwapping (SubdomainIterator it)
	{
		Boolean bCanUsePointerSwapping = m_mapCanUsePointerSwapping.get (it);
		if (bCanUsePointerSwapping != null)
			return bCanUsePointerSwapping;

		StrategyAnalyzer analyzer = m_data.getCodeGenerators ().getStrategyAnalyzer ();

		Expression exprTimeblockingFactor = analyzer.getTimeBlockingFactor (it);
		if (ExpressionUtil.isValue (exprTimeblockingFactor, 1))
			bCanUsePointerSwapping = true;
		else
		{
			// TODO: implement other cases

			bCanUsePointerSwapping = false;
		}

		m_mapCanUsePointerSwapping.put (it, bCanUsePointerSwapping);
		return bCanUsePointerSwapping;
	}

	/**
	 * Generates the code swapping the memory object pointers used within the
	 * temporal loop <code>loopTemporal</code>.
	 * 
	 * @param loopTemporal
	 *            The temporal loop that swaps the memory object pointers
	 * @param slb
	 *            The statement list bundle to which the code is added
	 */
	public void swapMemoryObjectPointers (RangeIterator loopTemporal, StatementListBundle slb, CodeGeneratorRuntimeOptions options)
	{
		// only generate the swapping code in the stencil calculation
		if (options.hasValue (CodeGeneratorRuntimeOptions.OPTION_STENCILCALCULATION, CodeGeneratorRuntimeOptions.VALUE_STENCILCALCULATION_STENCIL) ||
			options.hasValue (CodeGeneratorRuntimeOptions.OPTION_STENCILCALCULATION, CodeGeneratorRuntimeOptions.VALUE_STENCILCALCULATION_VALIDATE))
		{
			StrategyAnalyzer analyzer = m_data.getCodeGenerators ().getStrategyAnalyzer ();

			generateResultAssignment (loopTemporal, analyzer, slb, options);
			generateSwappingCode (loopTemporal, analyzer, slb, options);
		}
	}

	private void generateResultAssignment (RangeIterator loopTemporal, StrategyAnalyzer analyzer, StatementListBundle slb, CodeGeneratorRuntimeOptions options)
	{
		// assign to the output variables if loopTemporal is the outermost temporal iterator
		if (loopTemporal == null || loopTemporal.isMainTemporalIterator ())
		{
			for (StencilNode node : m_data.getStencilCalculation ().getOutputBaseNodeSet ())
			{
				Expression exprGrid = new NameID (StringUtil.concat (MemoryObjectManager.createMemoryObjectName (null, node, null, true), SUFFIX_OUTPUTGRID));

				// dereference if the _out grid is a double pointer (usually it is, it only isn't in validation)
				if (!options.hasValue (CodeGeneratorRuntimeOptions.OPTION_STENCILCALCULATION, CodeGeneratorRuntimeOptions.VALUE_STENCILCALCULATION_VALIDATE))
					exprGrid = new UnaryExpression (UnaryOperator.DEREFERENCE, exprGrid);

				slb.addStatement (new ExpressionStatement (new AssignmentExpression (
					exprGrid,
					AssignmentOperator.NORMAL,
					getMemoryObjectExpression (analyzer.getRootSubdomain (), node, null, false, false, false, slb, options))));
			}
		}
	}

	private void generateSwappingCode (RangeIterator loopTemporal, StrategyAnalyzer analyzer, StatementListBundle slb, CodeGeneratorRuntimeOptions options)
	{
		if (loopTemporal == null)
			return;

		///
		Stencil stencil = m_data.getStencilCalculation ().getStencilBundle ().getFusedStencil ();
		int nMinVecIdx = Integer.MAX_VALUE;
		int nMaxVecIdx = 0;
		for (StencilNode node : stencil.getAllNodes ())
		{
			nMinVecIdx = Math.min (nMinVecIdx, node.getIndex ().getVectorIndex ());
			nMaxVecIdx = Math.max (nMinVecIdx, node.getIndex ().getVectorIndex ());
		}
		int nVectorIndicesCount = nMaxVecIdx - nMinVecIdx + 1;
		///

		// find inner most subdomain iterator before the next temporal loop
		Set<SubdomainIdentifier> setSubdomainIdentifiers = new HashSet<> ();
		for (SubdomainIterator it : analyzer.getRelativeInnerMostSubdomainIterators (loopTemporal))
		{
			if (canUsePointerSwapping (it))
				setSubdomainIdentifiers.add (/*getActualSubdomainIdentifier*/ (it.getIterator ()));
		}

		// create the swapping code
		for (SubdomainIdentifier sdid : setSubdomainIdentifiers)
		{
			MemoryObjects mos = findMemoryObjects (sdid, false);
			if (mos != null)
			{
				for (int nVectorIndex = 0; nVectorIndex < nVectorIndicesCount; nVectorIndex++)
				{
///					if (mos.getMaxSpaceInfoCount (nVectorIndex) == 1)
					{
						// only one equivalence class: swap among local timesteps

						// only swap if more than one memory object for this vector index
						int nMemObjsCount = mos.getMemoryObjectStencilNodes ().size ();
						if (nMemObjsCount > 1)
						{
							// get the memory objects in the right order (sort by time index)
							ArrayList<StencilNode> listNodes = new ArrayList<> (nMemObjsCount);
							for (StencilNode node : mos.getMemoryObjectStencilNodes ())
								if (node.getIndex ().getVectorIndex () == nVectorIndex && node.getIndex ().isAdvanceableInTime ())	// ??? (added 4/4/2011)
									listNodes.add (node);

							if (listNodes.size () > 1)
							{
								Collections.sort (listNodes, new Comparator<StencilNode> ()
								{
									@Override
									public int compare (StencilNode n1, StencilNode n2)
									{
										return n1.getIndex ().getTimeIndex () - n2.getIndex ().getTimeIndex ();
									}
								});

								// TODO: need to check whether distinct time indices?

								// create the swap code
								generateSwappingBlock (sdid, listNodes, slb, options);
							}
						}
					}
///					else
///					{
///						// more than one equivalence class: swap within local timestep
///
///						for (int nTimeIdx : mos.getMemoryObjectStencilNodes ().getTimeIndices ())
///						{
///							ArrayList<StencilNode> listNodes = new ArrayList<StencilNode> ();
///							for (StencilNode node : mos.getMemoryObjectStencilNodes ().restrict (nTimeIdx, null))
///								listNodes.add (node);
///
///							// TODO: need to sort listMemObjs somehow?
///
///							// create the swap code
///							generateSwappingBlock (sdid, listNodes, slb, options);
///						}
///					}
				}
			}
		}
	}

	/**
	 * Generates one block of swapping code.
	 * @param sdid
	 * @param listNodes
	 * @param slb
	 */
	private void generateSwappingBlock (SubdomainIdentifier sdid, ArrayList<StencilNode> listNodes, StatementListBundle slb, CodeGeneratorRuntimeOptions options)
	{
		// create the temporary variable
		VariableDeclarator decl = new VariableDeclarator (new NameID (StringUtil.concat ("tmp_swap_", m_nSwapCount++)));

		Specifier specType = listNodes.get (0).getSpecifier ();
		List<Specifier> listSpecifiers = new ArrayList<> ();
		if (m_data.getOptions ().useNativeSIMDDatatypes () && m_data.getArchitectureDescription ().useSIMD ())
			listSpecifiers.addAll (m_data.getArchitectureDescription ().getType (specType));
		else
			listSpecifiers.add (specType);
		listSpecifiers.add (PointerSpecifier.UNQUALIFIED);

		m_data.getData ().addDeclaration (new VariableDeclaration (listSpecifiers, decl));
		Identifier idTmp = new Identifier (decl);

		// create the swapping assignment statements
		for (int i = 0; i <= listNodes.size (); i++)
		{
			Expression exprLHS = i == 0 ? idTmp.clone () : getMemoryObjectExpression (sdid, listNodes.get (i - 1), null, false, false, false, slb, options);
			Expression exprRHS = i == listNodes.size () ? idTmp.clone () : getMemoryObjectExpression (sdid, listNodes.get (i), null, false, false, false, slb, options);

			slb.addStatement (new ExpressionStatement (new AssignmentExpression (exprLHS, AssignmentOperator.NORMAL, exprRHS)));
		}
	}

	/**
	 * Resets the flags in the {@link IndexExpressionCache} signifying that
	 * after the reset the indices will be recomputed.
	 */
	public void resetIndices ()
	{
		m_cacheIndices.resetIndices ();
	}

	/**
	 * Resets the memory object manager.
	 */
	public void clear ()
	{
		m_cacheIndices.clear ();
		m_mapMemoryObjects.clear ();
		initialize ();
	}

	@Override
	public String toString ()
	{
		StringBuilder sb = new StringBuilder ();
		for (SubdomainIdentifier id : m_mapMemoryObjects.keySet ())
		{
			sb.append ("+-----------------------+\n");
			sb.append ("| ");
			sb.append (id.getName ());
			sb.append ("\n");
			sb.append ("+-----------------------+\n\n");
			sb.append (m_mapMemoryObjects.get (id).toString ());
			sb.append ("\n\n\n");
		}

		return sb.toString ();
	}
}
