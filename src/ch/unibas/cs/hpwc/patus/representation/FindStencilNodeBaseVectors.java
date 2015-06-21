package ch.unibas.cs.hpwc.patus.representation;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import cetus.hir.Specifier;
import ch.unibas.cs.hpwc.patus.util.IntArray;
import ch.unibas.cs.hpwc.patus.util.MathUtil;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 * <p>This class finds a minimum set of base vectors with which the spatial indices of stencil nodes
 * can be expressed, given a set of admissible scaling factors, i.e., each of the stencil node
 * spatial vectors are representable by scaling one of the base vectors by a factor with factors
 * from a pre-defined set.</p>
 * 
 * <p>The intention of this class is to use a minimum number of registers for address calculations.
 * On x86(_64) architectures, address calculations can have the form</p>
 * <pre>    addr = base + scale * idx + displacement</pre>
 * <p>were <code>scale</code> is an element of {1, 2, 4, 8}, <code>base</code> and <code>scale</code>
 * are registers, and <code>displacement</code> is a constant.</p> 
 * 
 * @author Matthias-M. Christen
 */
public class FindStencilNodeBaseVectors
{
	///////////////////////////////////////////////////////////////////
	// Inner Types

	public static class BaseVectorWithScalingFactor
	{
		int m_rgBaseVector[];
		int m_nScalingFactor;
		
		public BaseVectorWithScalingFactor (int[] rgBaseIndex, int nScalingFactor)
		{
			set (rgBaseIndex, nScalingFactor);
		}

		public int[] getBaseVector ()
		{
			return m_rgBaseVector;
		}
		
		public void setBaseVector (int[] rgBaseVector)
		{
			m_rgBaseVector = new int[rgBaseVector.length];
			System.arraycopy (rgBaseVector, 0, m_rgBaseVector, 0, rgBaseVector.length);			
		}

		public int getScalingFactor ()
		{
			return m_nScalingFactor;
		}
		
		public void setScalingFactor (int nScalingFactor)
		{
			m_nScalingFactor = nScalingFactor;
		}
		
		public void set (int[] rgBaseVector, int nScalingFactor)
		{
			setBaseVector (rgBaseVector);
			setScalingFactor (nScalingFactor);			
		}
		
		@Override
		public String toString ()
		{
			StringBuilder sb = new StringBuilder ("[");
			boolean bFirst = true;
			for (int a : m_rgBaseVector)
			{
				if (!bFirst)
					sb.append (", ");
				bFirst = false;
				sb.append (a);
			}
			sb.append ("] * ");
			sb.append (m_nScalingFactor);
			
			return sb.toString ();
		}
	}
	
	
	///////////////////////////////////////////////////////////////////
	// Member Variables

	/**
	 * The collection of stencil nodes for which to find common
	 * base vectors and scaling factors
	 */
	private Collection<StencilNode> m_collNodes;
	
	/**
	 * The set of base vectors
	 */
	private Set<IntArray> m_setBaseVector;
	
	/**
	 * Maps actual vectors to a base vector-scaling factor pair
	 */
	private Map<IntArray, BaseVectorWithScalingFactor> m_mapNodes;
	
	/**
	 * 
	 */
	private Map<IntArray, List<StencilNode>> m_mapBaseVectors2StencilNodes;
	
	/**
	 * The list of admissible scaling factors.
	 * HW architectures might restrict the scaling factors in address calculations;
	 * e.g., the possible values on x86(_64) architectures are {1, 2, 4, 8}.
	 */
	private int[] m_rgAdmissibleScalingFactors;
	
	/**
	 * The dimension of the vectors (spatial indices)
	 */
	private int m_nDimension;
	
	private IntArray m_arrZero;
	
	
	///////////////////////////////////////////////////////////////////
	// Implementation

	public FindStencilNodeBaseVectors (int[] rgAdmissibleScalingFactors)
	{
		m_collNodes = new ArrayList<> ();
		m_rgAdmissibleScalingFactors = rgAdmissibleScalingFactors;

		m_setBaseVector = new HashSet<> ();
		m_mapNodes = new HashMap<> ();
		m_mapBaseVectors2StencilNodes = new HashMap<> ();

		m_arrZero = null;		
		m_nDimension = -1;
	}
	
	public FindStencilNodeBaseVectors (Collection<StencilNode> collNodes, int[] rgAdmissibleScalingFactors)
	{
		this (rgAdmissibleScalingFactors);
		m_collNodes.addAll (collNodes);
		
		// check for consistent dimension
		m_nDimension = -1;
		for (StencilNode node : collNodes)
			checkDimension (node);
	}
	
	public void addNode (StencilNode node)
	{
		if (checkDimension (node))
			m_collNodes.add (node);
	}
	
	/**
	 * Invoke this method to find the base vectors and scaling factors for the
	 * nodes that have been added.
	 */
	public void run ()
	{
		m_setBaseVector.clear ();
		m_mapNodes.clear ();
		m_mapBaseVectors2StencilNodes.clear ();
		
		for (StencilNode node : m_collNodes)
			processNode (node);
	}
	
	/**
	 * Returns the base vector for the stencil node <code>node</code>.
	 * 
	 * @param node
	 *            The node for which to find the base vector
	 * @return The base vector corresponding to the stencil node
	 *         <code>node</code>
	 */
	public int[] getBaseVector (StencilNode node)
	{
		BaseVectorWithScalingFactor m = m_mapNodes.get (new IntArray (node.getSpaceIndex ()));
		return m == null ? null : m.getBaseVector ();
	}
	
	/**
	 * Returns the scaling factor for the stencil node <code>node</code>.
	 * 
	 * @param node
	 *            The node for which to find the scaling factor
	 * @return The scaling factor corresponding to the stencil node
	 *         <code>node</code>
	 */
	public int getScalingFactor (StencilNode node)
	{
		BaseVectorWithScalingFactor m = m_mapNodes.get (new IntArray (node.getSpaceIndex ()));
		return m == null ? 0 : m.getScalingFactor ();		
	}
	
	/**
	 * Returns an iterable over all the base vectors.
	 * 
	 * @return An iterable over the base vectors
	 */
	public Iterable<IntArray> getBaseVectors ()
	{
		return m_setBaseVector;
	}
	
	/**
	 * Checks whether the dimensions of the nodes' spatial indices are
	 * consistent.
	 * 
	 * @param node
	 *            The node to check against the nodes that have been already
	 *            added
	 * @return <code>true</code> iff the dimensions of the spatial indices of
	 *         the added stencil nodes is consistent
	 */
	private boolean checkDimension (StencilNode node)
	{
		if (m_nDimension == -1)
		{
			m_nDimension = node.getIndex ().getSpaceIndex ().length;
			m_arrZero = new IntArray (IntArray.getArray (m_nDimension, 0));
		}
		else if (node.getIndex ().getSpaceIndex ().length != m_nDimension)
		{
			throw new RuntimeException (StringUtil.concat (
				"Stencil nodes must have spatial indices of equal dimensions. Dimension is ",
				m_nDimension,
				", but the dimension of ",
				node.toString (),
				" is ", node.getIndex ().getSpaceIndex ().length, "."));
		}
		
		return true;
	}
	
	/**
	 * Determines whether <code>rgVector</code> has <code>nZerosCount</code>
	 * leading zeros.
	 * 
	 * @param rgVector
	 *            The index to check
	 * @param nZerosCount
	 *            The number of leading zeros
	 * @return <code>true</code> iff there are exactly <code>nZerosCount</code>
	 *         leading zeros
	 */
	private static boolean hasZeros (int[] rgVector, int nZerosCount)
	{
		if (nZerosCount > rgVector.length)
			return false;
		
		// check for leading zeros
		for (int i = 0; i < nZerosCount; i++)
			if (rgVector[i] != 0)
				return false;
		
		// the coordinate after nZerosCount must not be 0 (unless all of the entries should be 0)
		return nZerosCount == rgVector.length || rgVector[nZerosCount] != 0;
	}
	
	/**
	 * Checks whether the coordinate <code>nNodeCoord</code> can be represented
	 * by a base vector
	 * with coordinate <code>nCoord</code>.
	 * 
	 * @param nOldBaseCoord
	 *            A coordinate of the base vector
	 * @param nNodeCoord
	 *            A coordinate of the spatial index of a stencil node to check
	 * @return the admissible scaling factor if the coordinate
	 *         <code>nNodeCoord</code> of the spatial index
	 *         of the stencil node can be represented by the base vector with
	 *         coordinate <code>nCoord</code>, or 0 if it can't.
	 */
	private int getAdmissibleBaseCoord (int nOldBaseCoord, int nNodeCoord)
	{
		// the absolute value of the potential new base coordinate
		// we don't know the sign yet
		int nPotentialNewBaseCoord = MathUtil.getGCD (nOldBaseCoord, nNodeCoord);
		
		// check whether the old base coord is admissible in the new potential base
		int nScalingFactorOldBaseCoord = getAdmissibleScalingFactor (nOldBaseCoord, nPotentialNewBaseCoord);
		if (nScalingFactorOldBaseCoord != 0)
		{
			int nScalingFactorNodeCoord = getAdmissibleScalingFactor (nNodeCoord, nPotentialNewBaseCoord);
			if (nScalingFactorNodeCoord != 0)
			{
				// common base found
				return nPotentialNewBaseCoord;
			}
		}
		
		// no common base found; try -nPotentialNewBaseCoord
		nScalingFactorOldBaseCoord = getAdmissibleScalingFactor (nOldBaseCoord, -nPotentialNewBaseCoord);
		if (nScalingFactorOldBaseCoord != 0)
		{
			int nScalingFactorNodeCoord = getAdmissibleScalingFactor (nNodeCoord, -nPotentialNewBaseCoord);
			if (nScalingFactorNodeCoord != 0)
			{
				// common base found
				return -nPotentialNewBaseCoord;
			}			
		}
		
		// no common base found
		return 0;
	}
	
	private int getAdmissibleScalingFactor (int nTest, int nPotentialBase)
	{
		if (nTest == nPotentialBase)
			return 1;

		for (int k : m_rgAdmissibleScalingFactors)
			if (nTest == k * nPotentialBase)
				return k;
						
		return 0;
	}
	
	/**
	 * Processes a single node: finds a base vector and a scaling factor for
	 * <code>node</code>.
	 * 
	 * @param node
	 *            The node for which to find a base vector-scaling factor pair
	 */
	private void processNode (StencilNode node)
	{
		List<IntArray> listAdmissibleVectors = getAdmissibleBaseVectors (node);
		if (listAdmissibleVectors != null)
		{
			// listAdmissibleVectors now contains all the vectors that satisfy (*)
			// pick the best one
			
			if (listAdmissibleVectors.size () == 0)
			{
				// no match found: add the index of the stencil node as a new index
				addNewBaseVector (node);
			}
			else
			{
				int[] rgSpaceIdx = node.getSpaceIndex ();
				
				// find the index that requires the least base index additions
				int nNonAdmissibleMin = Integer.MAX_VALUE;
				IntArray arrBest = null;
				int nNewBaseCoordBest = 1;
				for (IntArray arr : listAdmissibleVectors)
				{
					int nOldBaseCoord = FindStencilNodeBaseVectors.getFirstNonZero (arr.get ());
					int nNodeCoord = FindStencilNodeBaseVectors.getFirstNonZero (rgSpaceIdx);
					
					int nNewBaseCoord = getAdmissibleBaseCoord (nOldBaseCoord, nNodeCoord);
					
					int nIdxNonZero = FindStencilNodeBaseVectors.getFirstNonZeroIndex (rgSpaceIdx);
					
					int nNonAdmissibleCount = 0;
					List<StencilNode> listDependentVectors = m_mapBaseVectors2StencilNodes.get (arr);
					if (listDependentVectors != null)
					{
						for (StencilNode n : listDependentVectors)
							if (getAdmissibleBaseCoord (nNewBaseCoord, n.getSpaceIndex ()[nIdxNonZero]) != nNewBaseCoord)
							{
								// base change required
								nNonAdmissibleCount++;
							}
					}
					
					if (nNonAdmissibleCount < nNonAdmissibleMin)
					{
						nNonAdmissibleMin = nNonAdmissibleCount;
						arrBest = arr;
						nNewBaseCoordBest = nNewBaseCoord;
					}
				}
				
				insertNewVector (node, arrBest, nNewBaseCoordBest);
			}			
		}
	}
	
	/**
	 * Adds a new base vector based on the stencil node <code>node</code>.
	 * 
	 * @param node
	 *            The stencil node from which to create a new base vector
	 */
	private void addNewBaseVector (StencilNode node)
	{
		int[] rgSpaceIdx = node.getSpaceIndex ();
		m_setBaseVector.add (new IntArray (rgSpaceIdx));
		m_mapNodes.put (new IntArray (rgSpaceIdx), new BaseVectorWithScalingFactor (rgSpaceIdx, 1));
		addToBaseVectorMap (new IntArray (rgSpaceIdx), node);
	}
	
	/**
	 * 
	 * @param node
	 * @param arrBest
	 * @param nDenominatorBest
	 */
	private void insertNewVector (StencilNode node, IntArray arrBest, int nNewBaseCoord)
	{
		if (arrBest == null)
			addNewBaseVector (node);
		else
		{
			int[] rgVector = null;
			
			// do we need to create a new base node?
			int nOldBaseCoord = getFirstNonZero (arrBest.get ());
			if (nOldBaseCoord != nNewBaseCoord)
			{
				// replace the base vector
				m_setBaseVector.remove (arrBest);
				
				int nDenominator = nOldBaseCoord / nNewBaseCoord;
				int[] rgVectorNew = new int[m_nDimension];
				int nFirstNonzeroCoordNew = 0;
				for (int i = 0; i < m_nDimension; i++)
				{
					rgVectorNew[i] = arrBest.get (i) / nDenominator;
					if (rgVectorNew[i] != 0 && nFirstNonzeroCoordNew == 0)
						nFirstNonzeroCoordNew = rgVectorNew[i];
				}
				m_setBaseVector.add (new IntArray (rgVectorNew));
				
				// update the scaling factors
				List<StencilNode> listNodes = m_mapBaseVectors2StencilNodes.get (arrBest);
				
				m_mapBaseVectors2StencilNodes.put (new IntArray (rgVectorNew), listNodes);
				m_mapBaseVectors2StencilNodes.remove (arrBest);
				
				for (StencilNode n : listNodes)
				{
					int[] rgSpaceIdx = n.getSpaceIndex ();
					if (getAdmissibleBaseCoord (nFirstNonzeroCoordNew, getFirstNonZero (rgSpaceIdx)) != 0)
					{
						BaseVectorWithScalingFactor m = m_mapNodes.get (new IntArray (rgSpaceIdx));
						if (m != null)
							m.set (rgVectorNew, m.getScalingFactor () * nDenominator);
					}
					else
					{
						// the node isn't admissible with the new scaling factor; add a new base node
						addNewBaseVector (n);
					}
				}
			
				rgVector = rgVectorNew;
			}
			else
				rgVector = arrBest.get ();
			
			// add the new stencil node
			int[] rgSpaceIdx = node.getSpaceIndex ();
			m_mapNodes.put (new IntArray (rgSpaceIdx), new BaseVectorWithScalingFactor (rgVector, getFirstNonZero (rgSpaceIdx) / nNewBaseCoord));
			addToBaseVectorMap (new IntArray (rgVector), node);
		}
	}
	
	/**
	 * Returns the first non-zero value in the array <code>rgVector</code>.
	 * 
	 * @param rgVector
	 *            The array to search
	 * @return The first non-zero value of <code>rgVector</code>
	 */
	private static int getFirstNonZero (int[] rgVector)
	{
		for (int i : rgVector)
			if (i != 0)
				return i;
		return 0;
	}
	
	/**
	 * Returns the index of the coordinate at which the first non-zero element
	 * of <code>rgVector</code> is found.
	 * 
	 * @param rgVector
	 *            The array to search
	 * @return The index of the first non-zero element of <code>rgVector</code>
	 */
	private static int getFirstNonZeroIndex (int[] rgVector)
	{
		for (int i = 0; i < rgVector.length; i++)
			if (rgVector[i] != 0)
				return i;
		return -1;
	}
	
	/**
	 * 
	 * @param arrKey
	 * @param node
	 */
	private void addToBaseVectorMap (IntArray arrKey, StencilNode node)
	{
		List<StencilNode> list = m_mapBaseVectors2StencilNodes.get (arrKey);
		if (list == null)
			m_mapBaseVectors2StencilNodes.put (arrKey, list = new ArrayList<> ());
		list.add (node);
	}
	
	/**
	 * 
	 * @param node
	 * @return The list of admissible vectors or <code>null</code> if the node is the center node (zero vector)
	 */
	private List<IntArray> getAdmissibleBaseVectors (StencilNode node)
	{
		// try to find an vector v that satisfies
		// (*)    node.v = k * (sgn (v) * gcd (v, node.v))
		// for k in m_rgAdmissibleScalingFactors (and v representing a coordinate of the vector)
		
		int nCoordIdx = 0;
		int[] rgSpaceIdx = node.getSpaceIndex ();
		
		// skip zeros
		while (nCoordIdx < m_nDimension && rgSpaceIdx[nCoordIdx] == 0)
			nCoordIdx++;
		
		if (nCoordIdx >= m_nDimension)
		{
			// all entries are 0, i.e., center node				
			if (!m_setBaseVector.contains (m_arrZero))
			{
				m_setBaseVector.add (m_arrZero);
				m_mapNodes.put (m_arrZero, new BaseVectorWithScalingFactor (m_arrZero.get (), 0));
				addToBaseVectorMap (m_arrZero, node);
			}
			return null;
		}
		
		int nFirstNonzeroCoordIdx = nCoordIdx;
		
		// if not all entries are zero start looking for a matching vector
		List<IntArray> listAdmissibleVectors = new ArrayList<> ();
		for (IntArray arr : m_setBaseVector)
		{
			nCoordIdx = nFirstNonzeroCoordIdx;
			if (!FindStencilNodeBaseVectors.hasZeros (arr.get (), nFirstNonzeroCoordIdx))
				continue;

			// test whether the first non-zero coordinate is admissible (i.e., satisfies (*))
			int nCoord = arr.get (nCoordIdx);
			int nNodeCoord = rgSpaceIdx[nCoordIdx];
			if (getAdmissibleBaseCoord (nCoord, nNodeCoord) == 0)
				continue;

			// check if the ratio for the remaining coordinates stays constant
			nCoordIdx++;
			boolean bIsAdmissible = true;
			for ( ; nCoordIdx < m_nDimension; nCoordIdx++)
			{
				if (arr.get (nCoordIdx) * nNodeCoord != rgSpaceIdx[nCoordIdx] * nCoord)
				{
					bIsAdmissible = false;
					break;
				}
			}
			
			if (bIsAdmissible)
				listAdmissibleVectors.add (arr);
		}
		
		return listAdmissibleVectors;
	}
	
	public static void main (String[] args)
	{
		Collection<StencilNode> nodes = new ArrayList<> ();
		nodes.add (new StencilNode ("n0", Specifier.FLOAT, new Index (0, new int[] { 2, 4, 6 }, 0)));
		nodes.add (new StencilNode ("n1", Specifier.FLOAT, new Index (0, new int[] { 1, 2, 3 }, 0)));
		nodes.add (new StencilNode ("n2", Specifier.FLOAT, new Index (0, new int[] { -1, -2, -3 }, 0)));
		nodes.add (new StencilNode ("n3", Specifier.FLOAT, new Index (0, new int[] { 8, 16, 24 }, 0)));
		nodes.add (new StencilNode ("n4", Specifier.FLOAT, new Index (0, new int[] { 6, 12, 18 }, 0)));
		nodes.add (new StencilNode ("n5", Specifier.FLOAT, new Index (0, new int[] { 9, 18, 27 }, 0)));
		nodes.add (new StencilNode ("n5", Specifier.FLOAT, new Index (0, new int[] { 24, 48, 72 }, 0)));

		FindStencilNodeBaseVectors find = new FindStencilNodeBaseVectors (nodes, new int[] { 1, 2, 4, 8 });
		find.run ();
		
		for (IntArray v : find.getBaseVectors ())
			System.out.println (v);
		System.out.println ();
		for (StencilNode node : nodes)
			System.out.println (StringUtil.concat (node, " --> ", new IntArray (find.getBaseVector (node)), " * ", find.getScalingFactor (node)));
	}
}
