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
package ch.unibas.cs.hpwc.patus.representation;

import java.util.Comparator;
import java.util.NoSuchElementException;
import java.util.Set;
import java.util.TreeSet;

import cetus.hir.Expression;
import cetus.hir.IntegerLiteral;

/**
 * Some utility functions used to find information and manipulate
 * index sets.
 * @author Matthias-M. Christen
 */
public class IndexSetUtil
{
	///////////////////////////////////////////////////////////////////
	// Inner Types

	/**
	 * A comparator for space indices.
	 */
	public static class SpaceIndexComparator implements Comparator<Expression[]>
	{
		@Override
		public int compare (Expression[] rgIdx1, Expression[] rgIdx2)
		{
			for (int i = 0; i < Math.min (rgIdx1.length, rgIdx2.length); i++)
			{
				if ((rgIdx1[i] instanceof IntegerLiteral) && (rgIdx2[i] instanceof IntegerLiteral))
				{
					long v1 = ((IntegerLiteral) rgIdx1[i]).getValue ();
					long v2 = ((IntegerLiteral) rgIdx2[i]).getValue ();
					
					if (v1 < v2)
						return -1;
					else if (v1 > v2)
						return 1;
				}
			}

			return 0;
		}
	}

	/**
	 * A comparator for space indices.
	 */
	public static class SpaceIndexComparatorOld implements Comparator<int[]>
	{
		@Override
		public int compare (int[] rgIdx1, int[] rgIdx2)
		{
			for (int i = 0; i < Math.min (rgIdx1.length, rgIdx2.length); i++)
			{
				if (rgIdx1[i] < rgIdx2[i])
					return -1;
				else if (rgIdx1[i] > rgIdx2[i])
					return 1;
			}

			return 0;
		}
	}

//	/**
//	 * A class encapsulating an <code>int[]</code> space index.
//	 */
//	private static class SpaceIndex
//	{
//		private int[] m_rgIndex;
//
////		public SpaceIndex (int[] rgIndex)
////		{
////			this (rgIndex, rgIndex.length);
////		}
//
//		public SpaceIndex (int[] rgIndex, int nDimension)
//		{
//			setIndex (rgIndex, nDimension);
//		}
//
//		public void setIndex (int[] rgIndex, int nDimension)
//		{
//			if (rgIndex.length == nDimension)
//				m_rgIndex = rgIndex;
//			else
//			{
//				m_rgIndex = new int[nDimension];
//				System.arraycopy (rgIndex, 0, m_rgIndex, 0, Math.min (rgIndex.length, nDimension));
//			}
//		}
//
////		public int[] getIndex ()
////		{
////			return m_rgIndex;
////		}
//
//		@Override
//		public boolean equals (Object obj)
//		{
//			if (obj instanceof SpaceIndex)
//				return Arrays.equals (m_rgIndex, ((SpaceIndex) obj).m_rgIndex);
//			else if (obj instanceof int[])
//				return Arrays.equals (m_rgIndex, (int[]) obj);
//			return false;
//		}
//
//		@Override
//		public int hashCode ()
//		{
//			int nHashCode = 0;
//			int nMultiplicator = 1;
//			for (int m : m_rgIndex)
//			{
//				nHashCode += m * nMultiplicator;
//				nMultiplicator *= 3;
//			}
//
//			return nHashCode;
//		}
//
//		@Override
//		public String toString ()
//		{
//			StringBuffer sb = new StringBuffer ("(");
//			boolean bFirst = true;
//			for (int m : m_rgIndex)
//			{
//				if (!bFirst)
//					sb.append (", ");
//				sb.append (m);
//				bFirst = false;
//			}
//			sb.append (')');
//
//			return sb.toString ();
//		}
//	}


	///////////////////////////////////////////////////////////////////
	// Constants

	/**
	 * A comparator for space indices
	 */
	public final static SpaceIndexComparator SPACE_INDEX_COMPARATOR = new SpaceIndexComparator ();


	///////////////////////////////////////////////////////////////////
	// Functions

	/**
	 * Returns the minimum spatial index that occurs in the index set.
	 * <b>Note</b> that only the first entry in the index is considered.
	 * @return The minimum space index
	 */
	public static int getMinSpaceIndex0 (Set<Index> setIndices)
	{
		int nMinSpaceIndex = Integer.MAX_VALUE;
		for (Index idx : setIndices)
			if (idx.getSpaceIndex ()[0] < nMinSpaceIndex)
				nMinSpaceIndex = idx.getSpaceIndex ()[0];
		return nMinSpaceIndex;
	}

	/**
	 * Returns the minimum spatial index that occurs in the index set for
	 * a given time index <code>nTimeIndex</code>.
	 * <b>Note</b> that only the first entry in the spatial index is considered.
	 * @return The minimum space index for time index <code>nTimeIndex</code>
	 * @throws NoSuchElementException if there is no index with time index
	 * 	equal to <code>nTimeIndex</code> in the set <code>setIndices</code>
	 */
	public static int getMinSpaceIndex0 (Set<Index> setIndices, int nTimeIndex) throws NoSuchElementException
	{
		int nMinSpaceIndex = Integer.MAX_VALUE;
		for (Index idx : setIndices)
			if (idx.getSpaceIndex ()[0] < nMinSpaceIndex && idx.getTimeIndex () == nTimeIndex)
				nMinSpaceIndex = idx.getSpaceIndex ()[0];

		if (nMinSpaceIndex == Integer.MAX_VALUE)
			throw new NoSuchElementException ();

		return nMinSpaceIndex;
	}

	/**
	 * Returns the minimum spatial index that occurs in the index set for
	 * a given time index <code>nTimeIndex</code>.
	 * <b>Note</b> that only the first entry in the spatial index is considered.
	 * @return The minimum space index for time index <code>nTimeIndex</code>
	 * @throws NoSuchElementException if there is no index with time index
	 * 	equal to <code>nTimeIndex</code> in the set <code>setIndices</code>
	 */
	public static int getMinSpaceIndex0 (Set<Index> setIndices, int nTimeIndex, int nVectorIndex) throws NoSuchElementException
	{
		int nMinSpaceIndex = Integer.MAX_VALUE;
		for (Index idx : setIndices)
			if (idx.getSpaceIndex ()[0] < nMinSpaceIndex && idx.getTimeIndex () == nTimeIndex && idx.getVectorIndex () == nVectorIndex)
				nMinSpaceIndex = idx.getSpaceIndex ()[0];

		if (nMinSpaceIndex == Integer.MAX_VALUE)
			throw new NoSuchElementException ();

		return nMinSpaceIndex;
	}

	/**
	 * Returns the minimum spatial index of the index set <code>setIndices</code>
	 * in the sense of a lower bound, i.e. the method computes an elementwise
	 * minimum and returns a vector containing these minimum values.
	 * @return The elementwise minimum space index
	 * @see VectorUtil#getMinimum(int[]...)
	 */
	public static int[] getMinSpaceIndex (Set<Index> setIndices)
	{
		// get the dimension of the spatial index space
		int nSpaceIndexDimension = IndexSetUtil.getSpaceDimension (setIndices);

		// create the index that will contain the maximum values
		int[] rgMinIndex = new int[nSpaceIndexDimension];
		for (int i = 0; i < nSpaceIndexDimension; i++)
			rgMinIndex[i] = Integer.MAX_VALUE;

		// find maximum indices
		for (Index idx : setIndices)
			for (int i = 0; i < idx.getSpaceIndex ().length; i++)
				if (idx.getSpaceIndex ()[i] < rgMinIndex[i])
					rgMinIndex[i] = idx.getSpaceIndex ()[i];

		return rgMinIndex;
	}

	/**
	 * Returns the minimum spatial index that occurs in the index set for
	 * a given time index <code>nTimeIndex</code>.
	 * @return The minimum space index for time index <code>nTimeIndex</code>
	 * @throws NoSuchElementException if there is no index with time index
	 * 	equal to <code>nTimeIndex</code> in the set <code>setIndices</code>
	 */
	public static int[] getMinSpaceIndexByTimeIndex (Set<Index> setIndices, int nTimeIndex) throws NoSuchElementException
	{
		// get the dimension of the spatial index space
		int nSpaceIndexDimension = IndexSetUtil.getSpaceDimension (setIndices);

		// create the index that will contain the maximum values
		int[] rgMinIndex = new int[nSpaceIndexDimension];
		for (int i = 0; i < nSpaceIndexDimension; i++)
			rgMinIndex[i] = Integer.MAX_VALUE;

		// find maximum indices
		for (Index idx : setIndices)
			if (idx.getTimeIndex () == nTimeIndex)
				for (int i = 0; i < idx.getSpaceIndex ().length; i++)
					if (idx.getSpaceIndex ()[i] < rgMinIndex[i])
						rgMinIndex[i] = idx.getSpaceIndex ()[i];

		if (rgMinIndex[0] == Integer.MAX_VALUE)
			throw new NoSuchElementException ();

		return rgMinIndex;
	}

	/**
	 * Returns the minimum spatial index that occurs in the index set for
	 * a given time index <code>nTimeIndex</code>.
	 * @return The minimum space index for time index <code>nTimeIndex</code>
	 * @throws NoSuchElementException if there is no index with time index
	 * 	equal to <code>nTimeIndex</code> in the set <code>setIndices</code>
	 */
	public static int[] getMinSpaceIndexByVectorIndex (Set<Index> setIndices, int nVectorIndex) throws NoSuchElementException
	{
		// get the dimension of the spatial index space
		int nSpaceIndexDimension = IndexSetUtil.getSpaceDimension (setIndices);

		// create the index that will contain the maximum values
		int[] rgMinIndex = new int[nSpaceIndexDimension];
		for (int i = 0; i < nSpaceIndexDimension; i++)
			rgMinIndex[i] = Integer.MAX_VALUE;

		// find maximum indices
		for (Index idx : setIndices)
			if (idx.getVectorIndex () == nVectorIndex)
				for (int i = 0; i < idx.getSpaceIndex ().length; i++)
					if (idx.getSpaceIndex ()[i] < rgMinIndex[i])
						rgMinIndex[i] = idx.getSpaceIndex ()[i];

		if (rgMinIndex[0] == Integer.MAX_VALUE)
			throw new NoSuchElementException ();

		return rgMinIndex;
	}

	/**
	 * Returns the minimum spatial index that occurs in the index set for
	 * a given time index <code>nTimeIndex</code>.
	 * @return The minimum space index for time index <code>nTimeIndex</code>
	 * @throws NoSuchElementException if there is no index with time index
	 * 	equal to <code>nTimeIndex</code> in the set <code>setIndices</code>
	 */
	public static int[] getMinSpaceIndex (Set<Index> setIndices, int nTimeIndex, int nVectorIndex) throws NoSuchElementException
	{
		// get the dimension of the spatial index space
		int nSpaceIndexDimension = IndexSetUtil.getSpaceDimension (setIndices);

		// create the index that will contain the maximum values
		int[] rgMinIndex = new int[nSpaceIndexDimension];
		for (int i = 0; i < nSpaceIndexDimension; i++)
			rgMinIndex[i] = Integer.MAX_VALUE;

		// find maximum indices
		for (Index idx : setIndices)
			if (idx.getTimeIndex () == nTimeIndex && idx.getVectorIndex () == nVectorIndex)
				for (int i = 0; i < idx.getSpaceIndex ().length; i++)
					if (idx.getSpaceIndex ()[i] < rgMinIndex[i])
						rgMinIndex[i] = idx.getSpaceIndex ()[i];

		if (rgMinIndex[0] == Integer.MAX_VALUE)
			throw new NoSuchElementException ();

		return rgMinIndex;
	}

	/**
	 * Returns the maximum spatial index that occurs in the index set.
	 * <b>Note</b> that only the first entry in the spatial index is considered.
	 * @return The maximum space index
	 */
	public static int getMaxSpaceIndex0 (Set<Index> setIndices)
	{
		int nMaxSpaceIndex = Integer.MIN_VALUE;
		for (Index idx : setIndices)
			if (idx.getSpaceIndex ()[0] > nMaxSpaceIndex)
				nMaxSpaceIndex = idx.getSpaceIndex ()[0];
		return nMaxSpaceIndex;
	}

	/**
	 * Returns the maximum spatial index that occurs in the index set for
	 * a given time index <code>nTimeIndex</code>.
	 * <b>Note</b> that only the first entry in the spatial index is considered.
	 * @return The maximum space index for time index <code>nTimeIndex</code>
	 * @throws NoSuchElementException if there is no index with time index
	 * 	equal to <code>nTimeIndex</code> in the set <code>setIndices</code>
	 */
	public static int getMaxSpaceIndex0 (Set<Index> setIndices, int nTimeIndex) throws NoSuchElementException
	{
		int nMaxSpaceIndex = Integer.MIN_VALUE;
		for (Index idx : setIndices)
			if (idx.getSpaceIndex ()[0] > nMaxSpaceIndex && idx.getTimeIndex () == nTimeIndex)
				nMaxSpaceIndex = idx.getSpaceIndex ()[0];

		if (nMaxSpaceIndex == Integer.MIN_VALUE)
			throw new NoSuchElementException ();

		return nMaxSpaceIndex;
	}

	/**
	 * Returns the maximum spatial index that occurs in the index set for
	 * a given time index <code>nTimeIndex</code>.
	 * <b>Note</b> that only the first entry in the spatial index is considered.
	 * @return The maximum space index for time index <code>nTimeIndex</code>
	 * @throws NoSuchElementException if there is no index with time index
	 * 	equal to <code>nTimeIndex</code> in the set <code>setIndices</code>
	 */
	public static int getMaxSpaceIndex0 (Set<Index> setIndices, int nTimeIndex, int nVectorIndex) throws NoSuchElementException
	{
		int nMaxSpaceIndex = Integer.MIN_VALUE;
		for (Index idx : setIndices)
			if (idx.getSpaceIndex ()[0] > nMaxSpaceIndex && idx.getTimeIndex () == nTimeIndex && idx.getVectorIndex () == nVectorIndex)
				nMaxSpaceIndex = idx.getSpaceIndex ()[0];

		if (nMaxSpaceIndex == Integer.MIN_VALUE)
			throw new NoSuchElementException ();

		return nMaxSpaceIndex;
	}

	/**
	 * Returns the maximum spatial index of the index set <code>setIndices</code>
	 * in the sense of an upper bound, i.e. the method computes an elementwise
	 * maximum and returns a vector containing these maximum values.
	 * @return The elementwise maximum space index
	 * @see VectorUtil#getMaximum(int[]...)
	 */
	public static int[] getMaxSpaceIndex (Set<Index> setIndices)
	{
		// get the dimension of the spatial index space
		int nSpaceIndexDimension = IndexSetUtil.getSpaceDimension (setIndices);

		// create the index that will contain the maximum values
		int[] rgMaxIndex = new int[nSpaceIndexDimension];
		for (int i = 0; i < nSpaceIndexDimension; i++)
			rgMaxIndex[i] = Integer.MIN_VALUE;

		// find maximum indices
		for (Index idx : setIndices)
			for (int i = 0; i < idx.getSpaceIndex ().length; i++)
				if (idx.getSpaceIndex ()[i] > rgMaxIndex[i])
					rgMaxIndex[i] = idx.getSpaceIndex ()[i];

		return rgMaxIndex;
	}

	/**
	 * Returns the maximum spatial index that occurs in the index set for
	 * a given time index <code>nTimeIndex</code>.
	 * @return The maximum space index for time index <code>nTimeIndex</code>
	 * @throws NoSuchElementException if there is no index with time index
	 * 	equal to <code>nTimeIndex</code> in the set <code>setIndices</code>
	 */
	public static int[] getMaxSpaceIndexByTimeIndex (Set<Index> setIndices, int nTimeIndex) throws NoSuchElementException
	{
		// get the dimension of the spatial index space
		int nSpaceIndexDimension = IndexSetUtil.getSpaceDimension (setIndices);

		// create the index that will contain the maximum values
		int[] rgMaxIndex = new int[nSpaceIndexDimension];
		for (int i = 0; i < nSpaceIndexDimension; i++)
			rgMaxIndex[i] = Integer.MIN_VALUE;

		// find maximum indices
		for (Index idx : setIndices)
			if (idx.getTimeIndex () == nTimeIndex)
				for (int i = 0; i < idx.getSpaceIndex ().length; i++)
					if (idx.getSpaceIndex ()[i] > rgMaxIndex[i])
						rgMaxIndex[i] = idx.getSpaceIndex ()[i];

		if (rgMaxIndex[0] == Integer.MIN_VALUE)
			throw new NoSuchElementException ();

		return rgMaxIndex;
	}

	/**
	 * Returns the maximum spatial index that occurs in the index set for
	 * a given time index <code>nTimeIndex</code>.
	 * @return The maximum space index for time index <code>nTimeIndex</code>
	 * @throws NoSuchElementException if there is no index with time index
	 * 	equal to <code>nTimeIndex</code> in the set <code>setIndices</code>
	 */
	public static int[] getMaxSpaceIndexByVectorIndex (Set<Index> setIndices, int nVectorIndex) throws NoSuchElementException
	{
		// get the dimension of the spatial index space
		int nSpaceIndexDimension = IndexSetUtil.getSpaceDimension (setIndices);

		// create the index that will contain the maximum values
		int[] rgMaxIndex = new int[nSpaceIndexDimension];
		for (int i = 0; i < nSpaceIndexDimension; i++)
			rgMaxIndex[i] = Integer.MIN_VALUE;

		// find maximum indices
		for (Index idx : setIndices)
			if (idx.getVectorIndex () == nVectorIndex)
				for (int i = 0; i < idx.getSpaceIndex ().length; i++)
					if (idx.getSpaceIndex ()[i] > rgMaxIndex[i])
						rgMaxIndex[i] = idx.getSpaceIndex ()[i];

		if (rgMaxIndex[0] == Integer.MIN_VALUE)
			throw new NoSuchElementException ();

		return rgMaxIndex;
	}

	/**
	 * Returns the maximum spatial index that occurs in the index set for
	 * a given time index <code>nTimeIndex</code>.
	 * @return The maximum space index for time index <code>nTimeIndex</code>
	 * @throws NoSuchElementException if there is no index with time index
	 * 	equal to <code>nTimeIndex</code> in the set <code>setIndices</code>
	 */
	public static int[] getMaxSpaceIndex (Set<Index> setIndices, int nTimeIndex, int nVectorIndex) throws NoSuchElementException
	{
		// get the dimension of the spatial index space
		int nSpaceIndexDimension = IndexSetUtil.getSpaceDimension (setIndices);

		// create the index that will contain the maximum values
		int[] rgMaxIndex = new int[nSpaceIndexDimension];
		for (int i = 0; i < nSpaceIndexDimension; i++)
			rgMaxIndex[i] = Integer.MIN_VALUE;

		// find maximum indices
		for (Index idx : setIndices)
			if (idx.getTimeIndex () == nTimeIndex && idx.getVectorIndex () == nVectorIndex)
				for (int i = 0; i < idx.getSpaceIndex ().length; i++)
					if (idx.getSpaceIndex ()[i] > rgMaxIndex[i])
						rgMaxIndex[i] = idx.getSpaceIndex ()[i];

		if (rgMaxIndex[0] == Integer.MIN_VALUE)
			throw new NoSuchElementException ();

		return rgMaxIndex;
	}

	/**
	 * Finds the minimum time index with in the set of indices, <code>setIndices</code>.
	 * @param setIndices The index set to search for the minimum time index
	 * @return The minimum time index of the indices contained in <code>setIndices</code>
	 */
	public static int getMinTimeIndex (Set<Index> setIndices)
	{
		int nMinTimeIndex = Integer.MAX_VALUE;
		for (Index idx : setIndices)
			if (idx.getTimeIndex () < nMinTimeIndex)
				nMinTimeIndex = idx.getTimeIndex ();
		return nMinTimeIndex;
	}

	/**
	 * Finds the maximum time index with in the set of indices, <code>setIndices</code>.
	 * @param setIndices The index set to search for the maximum time index
	 * @return The maximum time index of the indices contained in <code>setIndices</code>
	 */
	public static int getMaxTimeIndex (Set<Index> setIndices)
	{
		int nMaxTimeIndex = Integer.MIN_VALUE;
		for (Index idx : setIndices)
			if (idx.getTimeIndex () > nMaxTimeIndex)
				nMaxTimeIndex = idx.getTimeIndex ();
		return nMaxTimeIndex;
	}

	/**
	 * Counts the number of indices in <code>setIndices</code> that have time
	 * index equal to <code>nTimeIndex</code>.
	 * @param setIndices The index set
	 * @param nTimeIndex The time index constraint
	 * @return The number of indices in <code>setIndices</code> that have their
	 * 	time index set to <code>nTimeIndex</code>
	 */
	public static int getIndicesCount (Set<Index> setIndices, int nTimeIndex)
	{
		int nCounter = 0;
		for (Index idx : setIndices)
			if (idx.getTimeIndex () == nTimeIndex)
				nCounter++;
		return nCounter;
	}

	/**
	 * Creates and returns a new set of indices that are clones
	 * of the indices in the set <code>set</code>.
	 * @param set The set to clone
	 * @return A copy of <code>set</code>
	 */
	public static Set<Index> copy (Iterable<Index> set)
	{
		Set<Index> setCopy = new TreeSet<> ();
		for (Index idx : set)
			setCopy.add (new Index (idx));
		return setCopy;
	}

	/**
	 * Returns an index set containing all the elements of set A that
	 * are not contained in set B.
	 * @param setA set A
	 * @param setB set B
	 * @return An index set with the elements that are only contained in A
	 */
	public static Set<Index> getElemsOfANotContainedInB (Set<Index> setA, Set<Index> setB)
	{
		Set<Index> setDifference = new TreeSet<> ();
		for (Index idx : setA)
			if (!setB.contains (idx))
				setDifference.add (idx);
		return setDifference;
	}

	/**
	 * Calculates the union of sets <code>setA</code> and <code>setB</code>.
	 * @param setA
	 * @param setB
	 * @return The union of <code>setA</code> and <code>setB</code>
	 */
	public static Set<Index> union (Set<Index> setA, Set<Index> setB)
	{
		Set<Index> setUnion = new TreeSet<> ();

		for (Index idx : setA)
			setUnion.add (idx);
		for (Index idx : setB)
			setUnion.add (idx);

		return setUnion;
	}

	/**
	 * Calculates the intersection of sets <code>setA</code> and <code>setB</code>.
	 * @param setA
	 * @param setB
	 * @return The intersection of <code>setA</code> and <code>setB</code>
	 */
	public static Set<Index> intersection (Set<Index> setA, Set<Index> setB)
	{
		Set<Index> setIntersection = new TreeSet<> ();

		for (Index idx : setA)
			if (setB.contains (idx))
				setIntersection.add (idx);

		return setIntersection;
	}

	/**
	 * Determines whether the sets A and B are equal.
	 * @param setA set A
	 * @param setB set B
	 * @return <code>true</code> if and only if A and B are equal, i.e.
	 * 	contain the same elements
	 */
	public static boolean areSetsEqual (Set<Index> setA, Set<Index> setB)
	{
		// if A is null, B must be null or an empty set
		if (setA == null)
			return setB == null || setB.size () == 0;

		// A isn't null; if B is, the sets aren't equal
		if (setB == null)
			return false;

		// now we know that both sets aren't null
		// equal sets must have the same size
		if (setA.size () != setB.size ())
			return false;

		// sets have equal size
		// now check individual set elements
		for (Index idx : setA)
			if (!setB.contains (idx))
				return false;

		return true;
	}

//	/**
//	 * Returns a <code>String</code> representation of the indices contained
//	 * in <code>set</code>.
//	 * @param set The index set to represent as a string
//	 * @return A string representing the indices in <code>set</code>
//	 */
//	public static String getSpaceIndicesString (Set<Index> set)
//	{
//		// sort the indices
//		List<int[]> listIndices = new LinkedList<> ();
//		for (Index idx : set)
//			listIndices.add (idx.getSpaceIndex ());
//		Collections.sort (listIndices, IndexSetUtil.SPACE_INDEX_COMPARATOR);
//
//		// create a string buffer with the indices
//		StringBuffer sb = new StringBuffer ();
//		boolean bFirst = true;
//
//		for (int[] rgIdx : listIndices)
//		{
//			if (!bFirst)
//				sb.append (", ");
//
//			if (rgIdx.length == 1)
//				sb.append (rgIdx[0]);
//			else
//			{
//				sb.append ('(');
//				for (int i = 0; i < rgIdx.length; i++)
//				{
//					if (i > 0)
//						sb.append (", ");
//					sb.append (rgIdx[i]);
//				}
//				sb.append (')');
//			}
//
//			bFirst = false;
//		}
//
//		return sb.toString ();
//	}

	/**
	 * Returns the dimension of the spatial indices in <code>set</code>.
	 * @param set The set of indices for which to find the spatial dimension
	 * @return The dimension of the spatial indices in the index set <code>set</code>
	 */
	public static int getSpaceDimension (Set<Index> set)
	{
		// find the dimension of the index in space
		int nSpaceIndexDimension = 0;
		for (Index idx : set)
			nSpaceIndexDimension = Math.max (nSpaceIndexDimension, idx.getSpaceIndex ().length);

		return nSpaceIndexDimension;
	}

	/**
	 * Returns the index following <code>rgCurrentIndex</code>.
	 * @param rgCurrentIndex The current index for which to retrieve the index following this one
	 * @param rgMinIndex The minimum index
	 * @param rgMaxIndex The maximum index
	 * @return
	 * @throws NoSuchElementException If no further index exists (respecting the bounds
	 * 	<code>rgMinIndex</code> &mdash; <code>rgMaxIndex</code>
	 */
	public static int[] getNextSpaceIndex (int[] rgCurrentIndex, int[] rgMinIndex, int[] rgMaxIndex) throws NoSuchElementException
	{
		int[] rgNextIndex = new int[rgCurrentIndex.length];
		int nCarry = 1;
		for (int i = rgCurrentIndex.length - 1; i >= 0; i--)
		{
			rgNextIndex[i] = rgCurrentIndex[i] + nCarry;

			if (rgNextIndex[i] > rgMaxIndex[i])
			{
				nCarry = 1;
				rgNextIndex[i] = rgMinIndex[i];
			}
			else
				nCarry = 0;
		}

		// if carry isn't 0 at the end, we already are at the end of the
		// available points
		if (nCarry > 0)
			throw new NoSuchElementException ();

		return rgNextIndex;
	}

//	/**
//	 * Determines whether the index set <code>set</code> constitutes a contiguous
//	 * space, i.e. no index in the hypercube spanned by the indices in the set
//	 * is omitted.
//	 * @param set The set to test
//	 * @return <code>true</code> if and only if all the indices in the hypercube
//	 * 	spanned by the indices in <code>set</code> are actually contained in the
//	 * 	<code>set</code>.
//	 */
//	public static boolean isContiguousIntervalInSpace (Set<Index> set)
//	{
//		// find the dimension of the index in space
//		int nSpaceIndexDimension = IndexSetUtil.getSpaceDimension (set);
//
//		int[] rgMinIndex = new int[nSpaceIndexDimension];
//		int[] rgMaxIndex = new int[nSpaceIndexDimension];
//
//		for (int i = 0; i < nSpaceIndexDimension; i++)
//		{
//			rgMinIndex[i] = Integer.MAX_VALUE;
//			rgMaxIndex[i] = Integer.MIN_VALUE;
//		}
//
//		// find minimum and maximum indices
//		for (Index idx : set)
//		{
//			for (int i = 0; i < idx.getSpaceIndex ().length; i++)
//			{
//				if (idx.getSpaceIndex ()[i] < rgMinIndex[i])
//					rgMinIndex[i] = idx.getSpaceIndex ()[i];
//				if (idx.getSpaceIndex ()[i] > rgMaxIndex[i])
//					rgMaxIndex[i] = idx.getSpaceIndex ()[i];
//			}
//		}
//
//		// calculate the number of index points that have to lie within the hypercube
//		int nProductSpaceDimension = 1;
//		for (int i = 0; i < nSpaceIndexDimension; i++)
//			nProductSpaceDimension *= rgMaxIndex[i] - rgMinIndex[i] + 1;
//
//		// if there are less index points in the set than in the product space, the set
//		// obviously doesn't contain all the points
//		if (set.size () < nProductSpaceDimension)
//			return false;
//
//		// create the set with all the indices that have to be present
//		// and add the indices of the set to the test set
//		Set<SpaceIndex> setFullSpaceIndices = new TreeSet<> ();
//		for (Index idx : set)
//			setFullSpaceIndices.add (new SpaceIndex (idx.getSpaceIndex (), nSpaceIndexDimension));
//
//		// if the number of indices in the test set equals the number of
//		// expected indices, return true, false otherwise
//		return setFullSpaceIndices.size () == nProductSpaceDimension;
//	}
}
