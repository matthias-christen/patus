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
package ch.unibas.cs.hpwc.patus.autotuner;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.MathContext;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 *
 * @author Matthias-M. Christen
 */
public class Histogram<TypeValue extends Number, TypeSample>
{
	///////////////////////////////////////////////////////////////////
	// Constants

	private static final int DEFAULT_BARS_COUNT = 10;


	///////////////////////////////////////////////////////////////////
	// Member Variables

	private Map<Range<TypeValue>, List<TypeSample>> m_mapHistogram;
	private Map<TypeSample, TypeValue> m_mapValues;

	private List<Range<TypeValue>> m_listRanges;

	private TypeValue m_vMinAcceptable;
	private TypeValue m_vMaxAcceptable;

	/**
	 * The number of histogram bars
	 */
	private int m_nBarsCount;

	/**
	 * Flag indicating whether the histogram structure has been created
	 */
	private boolean m_bIsHistogramCreated;


	///////////////////////////////////////////////////////////////////
	// Implementation

	public Histogram ()
	{
		m_mapValues = new HashMap<> ();
		m_mapHistogram = null;
		m_listRanges = null;

		m_vMinAcceptable = null;
		m_vMaxAcceptable = null;

		m_nBarsCount = DEFAULT_BARS_COUNT;
		m_bIsHistogramCreated = false;
	}

	/**
	 * Sets the number of histogram bars
	 * @param nBarsCount
	 */
	public void setBarsCount (int nBarsCount)
	{
		m_nBarsCount = nBarsCount;
		m_bIsHistogramCreated = false;
	}

	/**
	 *
	 * @param min
	 * @param max
	 */
	public void setAcceptableRange (TypeValue min, TypeValue max)
	{
		m_vMinAcceptable = min;
		m_vMaxAcceptable = max;
	}

	/**
	 * Adds a sample to the histogram.
	 * @param value The value (value of the objective funciton) of the sample
	 * @param sample The sample itself
	 */
	public void addSample (TypeValue value, TypeSample sample)
	{
		m_mapValues.put (sample, value);
		m_bIsHistogramCreated = false;
	}

	private void addRange (Range<TypeValue> range)
	{
		m_listRanges.add (range);
		m_mapHistogram.put (range, new LinkedList<TypeSample> ());
	}

	/**
	 * Finds the range for the value <code>value</code>.
	 * @return
	 */
	private Range<TypeValue> findRange (TypeValue value)
	{
		Range<TypeValue> range = null;
		for (Range<TypeValue> r : m_listRanges)
		{
			if (r.inRange (value))
			{
				range = r;
				break;
			}
		}

		return range;
	}

	@SuppressWarnings("unchecked")
	private Range<TypeValue> createRange (double fMin, double fMax, TypeValue tvTemplate)
	{
		if (tvTemplate instanceof AtomicInteger)
			return (Range<TypeValue>) new Range<> (new AtomicInteger ((int) fMin), new AtomicInteger ((int) fMax));
		if (tvTemplate instanceof AtomicLong)
			return (Range<TypeValue>) new Range<> (new AtomicLong ((long) fMin), new AtomicLong ((long) fMax));
		if (tvTemplate instanceof BigDecimal)
			return (Range<TypeValue>) new Range<> (new BigDecimal (fMin, MathContext.DECIMAL64), new BigDecimal (fMax, MathContext.DECIMAL64));
		if (tvTemplate instanceof BigInteger)
			return (Range<TypeValue>) new Range<> (new BigInteger (String.valueOf (fMin)), new BigInteger (String.valueOf (fMax)));
		if (tvTemplate instanceof Byte)
			return (Range<TypeValue>) new Range<> ((byte) fMin, (byte) fMax);
		if (tvTemplate instanceof Double)
			return (Range<TypeValue>) new Range<> (fMin, fMax);
		if (tvTemplate instanceof Float)
			return (Range<TypeValue>) new Range<> ((float) fMin, (float) fMax);
		if (tvTemplate instanceof Integer)
			return (Range<TypeValue>) new Range<> ((int) fMin, (int) fMax);
		if (tvTemplate instanceof Long)
			return (Range<TypeValue>) new Range<> ((long) fMin, (long) fMax);
		if (tvTemplate instanceof Short)
			return (Range<TypeValue>) new Range<> ((short) fMin, (short) fMax);

		return null;
	}

	/**
	 * Creates the histogram structure.
	 */
	public void create ()
	{
		// create new histogram data
		m_mapHistogram = new TreeMap<> ();
		m_listRanges = new ArrayList<> (10);

		// find the minimum and maximum values of the "objective function"
		TypeValue valTemplate = null;
		double fMin = Double.MAX_VALUE;
		double fMax = 0;
		for (TypeValue val : m_mapValues.values ())
		{
			double fVal = val.doubleValue ();
			if (m_vMinAcceptable != null && fVal < m_vMinAcceptable.doubleValue ())
				continue;
			if (m_vMaxAcceptable != null && fVal > m_vMaxAcceptable.doubleValue ())
				continue;

			fMin = Math.min (fVal, fMin);
			fMax = Math.max (fVal, fMax);

			if (valTemplate == null)
				valTemplate = val;
		}

		// create a list of ranges
		double fSpread = (fMax - fMin) / m_nBarsCount;
		for (int i = 0; i < m_nBarsCount; i++)
			addRange (createRange (i * fSpread + fMin, (i + 1) * fSpread + fMin, valTemplate));

		// add the samples to the histogram map
		for (TypeSample sample : m_mapValues.keySet ())
		{
			Range<TypeValue> range = findRange (m_mapValues.get (sample));
			if (range != null)
				m_mapHistogram.get (range).add (sample);
			else
				System.err.println (StringUtil.concat ("Cannot find range for ", sample));
		}

		m_bIsHistogramCreated = true;
	}

	public void print ()
	{
		System.out.println (toString ());
	}

	@SuppressWarnings("static-method")
	private void printBar (Range<TypeValue> range, double fPercentage, StringBuilder sb)
	{
		sb.append ('[');

		int i = 0;
		for ( ; i < (int) Math.ceil (fPercentage / 5); i++)
			sb.append ('#');
		for ( ; i < 20; i++)
			sb.append ('_');

		sb.append ("] ");

		NumberFormat nf = NumberFormat.getInstance();
		nf.setMaximumFractionDigits (2);
		nf.setMinimumFractionDigits (2);
		sb.append (nf.format (fPercentage));
		sb.append ("% -- ");

		sb.append (range.getMin ());
		sb.append ("..");
		sb.append (range.getMax ());

		sb.append ('\n');
	}

	@Override
	public String toString ()
	{
		if (!m_bIsHistogramCreated)
			create ();

		StringBuilder sb = new StringBuilder ();

		int nSample = 1;
		for (Range<TypeValue> range : m_listRanges)
		{
			List<TypeSample> listSamples = m_mapHistogram.get (range);
			Collections.sort (listSamples, new Comparator<TypeSample> ()
			{
				@Override
				public int compare (TypeSample s1, TypeSample s2)
				{
					double fVal1 = m_mapValues.get (s1).doubleValue ();
					double fVal2 = m_mapValues.get (s2).doubleValue ();
					return fVal1 < fVal2 ? -1 : (fVal1 == fVal2 ? 0 : 1);
				}
			});

			printBar (range, (listSamples.size () * 100.0) / m_mapValues.size (), sb);

			for (TypeSample sample : listSamples)
			{
				sb.append (nSample);
				sb.append (". ");

				sb.append (StringUtil.toString (sample));
				sb.append (" -- ");
				sb.append (m_mapValues.get (sample));
				sb.append ('\n');

				nSample++;
			}
			sb.append ('\n');
		}

		return sb.toString ();
	}
}
