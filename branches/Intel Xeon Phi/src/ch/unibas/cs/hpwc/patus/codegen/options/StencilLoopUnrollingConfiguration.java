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
package ch.unibas.cs.hpwc.patus.codegen.options;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import ch.unibas.cs.hpwc.patus.util.DomainPointEnumerator;

/**
 * Code generator configuration for stencil loop unrolling.
 * Provides a facility to enumerate the unrolling space.
 *
 * @author Matthias-M. Christen
 */
public class StencilLoopUnrollingConfiguration
{
	///////////////////////////////////////////////////////////////////
	// Static Members

	public final static int NO_UNROLLING_LIMIT = -1;


	/**
	 *
	 * @param nDimensionality
	 * @return
	 */
	public static Iterable<int[]> getDefaultSpace (int nDimensionality)
	{
		int[] rgDefault = new int[nDimensionality];
		Arrays.fill (rgDefault, 0);
		List<int[]> list = new ArrayList<> (1);
		list.add (rgDefault);
		return list;
	}


	///////////////////////////////////////////////////////////////////
	// Member Variables

	private Map<Integer, Integer> m_mapPerDimensionUnrollings;
	private int m_nMaxDimension;


	///////////////////////////////////////////////////////////////////
	// Implementation

	public StencilLoopUnrollingConfiguration ()
	{
		m_mapPerDimensionUnrollings = new TreeMap<> ();
		m_nMaxDimension = -1;
	}

	/**
	 *
	 * @param rgUnrolling
	 * @param rgMaxUnrollings
	 */
	public StencilLoopUnrollingConfiguration (int nDimensions, int[] rgUnrolling, int[] rgMaxUnrollings)
	{
		this ();
		for (int i = 0; i < nDimensions; i++)
		{
			setUnrollingForDimension (
				i,
				i < rgUnrolling.length ? rgUnrolling[i] : 1,
				i < rgMaxUnrollings.length ? rgMaxUnrollings[i] : StencilLoopUnrollingConfiguration.NO_UNROLLING_LIMIT
			);
		}
	}

	/**
	 *
	 * @param nDimension
	 * @param nUnrollingFactor
	 * @param nMaxUnrolling
	 */
	public void setUnrollingForDimension (int nDimension, int nUnrollingFactor, int nMaxUnrolling)
	{
		m_mapPerDimensionUnrollings.put (
			nDimension,
			nMaxUnrolling == StencilLoopUnrollingConfiguration.NO_UNROLLING_LIMIT ? nUnrollingFactor : Math.min (nUnrollingFactor, nMaxUnrolling));

		if (nDimension > m_nMaxDimension)
			m_nMaxDimension = nDimension;
	}

	public int getUnrollingFactor (int nDimension)
	{
		Integer nUnrollingFactor = m_mapPerDimensionUnrollings.get (nDimension);
		return nUnrollingFactor == null ? 1 : nUnrollingFactor;
	}

	public boolean isLoopUnrolled (int nDimension)
	{
		return getUnrollingFactor (nDimension) > 1;
	}

	public Iterable<int[]> getConfigurationSpace ()
	{
		return getConfigurationSpace (m_nMaxDimension + 1);
	}

	/**
	 * Returns an iterable over the entire configuration space, e.g. if the
	 * loop unrolling configuration is set to
	 * 
	 * <pre>
	 * 	dim 1 -> 2
	 * 	dim 2 -> 3,
	 * </pre>
	 * 
	 * the iterable enumerates the space
	 * 
	 * <pre>
	 * 	{ 0, 0 }, { 1, 0 }, { 0, 1 }, { 1, 1 }, { 0, 2 }, { 1, 2 }
	 * </pre>
	 * 
	 * @return
	 */
	public Iterable<int[]> getConfigurationSpace (int nDimensionality)
	{
		// create an array for the specified dimensionality containing the unrollings
		// in each dimension (up to nDimensionality)
		final int[] rgUnrollings = getUnrollings (nDimensionality);

		DomainPointEnumerator dpe = new DomainPointEnumerator ();
		for (int nUnrolling : rgUnrollings)
			dpe.addDimension (0, nUnrolling - 1);
		return dpe;
	}

	/**
	 *
	 * @return
	 */
	public int[] getUnrollings ()
	{
		return getUnrollings (m_nMaxDimension + 1);
	}

	/**
	 *
	 * @param nDimensionality
	 * @return
	 */
	public int[] getUnrollings (int nDimensionality)
	{
		final int[] rgUnrollings = new int[nDimensionality];
		Arrays.fill (rgUnrollings, 1);

		for (int nDim : m_mapPerDimensionUnrollings.keySet ())
			if (nDim < nDimensionality)
				rgUnrollings[nDim] = m_mapPerDimensionUnrollings.get (nDim);

		return rgUnrollings;
	}

	@Override
	public String toString ()
	{
		StringBuilder sb = new StringBuilder ("{ ");
		boolean bFirst = true;

		for (int nDim : m_mapPerDimensionUnrollings.keySet ())
		{
			if (!bFirst)
				sb.append (", ");

			sb.append ("dim ");
			sb.append (nDim);
			sb.append (" -> ");
			sb.append (m_mapPerDimensionUnrollings.get (nDim));

			bFirst = false;
		}
		sb.append (" }");

		return sb.toString ();
	}

	/**
	 * Returns an integer representation of the unrolling configuration
	 * (that can be used for the code branching system in which parameter values
	 * are integers).
	 * 
	 * @return An integer representation of the unrolling configuration
	 */
	public int toInteger ()
	{
		int[] rgUnrollings = getUnrollings ();
		int n = 0;
		for (int nUnrolling : rgUnrollings)
			n = 100 * n + nUnrolling;
		return n;
	}

	@Override
	public StencilLoopUnrollingConfiguration clone ()
	{
		StencilLoopUnrollingConfiguration config = new StencilLoopUnrollingConfiguration ();
		config.m_mapPerDimensionUnrollings.putAll (m_mapPerDimensionUnrollings);
		return config;
	}

	@Override
	public boolean equals (Object obj)
	{
		if (!(obj instanceof StencilLoopUnrollingConfiguration))
			return false;

		StencilLoopUnrollingConfiguration slucOther = (StencilLoopUnrollingConfiguration) obj;
		if (m_mapPerDimensionUnrollings.size () != slucOther.m_mapPerDimensionUnrollings.size ())
			return false;

		for (Integer nKey : m_mapPerDimensionUnrollings.keySet ())
			if (!m_mapPerDimensionUnrollings.get (nKey).equals (slucOther.m_mapPerDimensionUnrollings.get (nKey)))
				return false;

		return true;
	}

	@Override
	public int hashCode ()
	{
		int nHash = 0;
		for (int i = 0; i < m_nMaxDimension; i++)
		{
			Integer j = m_mapPerDimensionUnrollings.get (i);
			nHash = 10 * nHash + (j == null ? 0 : j);
		}

		return nHash;
	}
}
