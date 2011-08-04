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
package ch.unibas.cs.hpwc.patus.ast;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 *
 * @author Matthias-M. Christen
 */
public class Parameter implements Cloneable, Iterable<Integer>
{
	///////////////////////////////////////////////////////////////////
	// Member Variables

	/**
	 * The name of the parameter
	 */
	private String m_strName;

	/**
	 * The values the parameter can take
	 */
	private List<Integer> m_listValues;


	///////////////////////////////////////////////////////////////////
	// Implementation

	public Parameter (String strName)
	{
		m_strName = strName;
		if (m_strName == null)
			throw new NullPointerException ("Parameter names must not be null.");

		m_listValues = new ArrayList<Integer> ();
	}

	public void addValue (int nValue)
	{
		m_listValues.add (nValue);
	}

	public String getName ()
	{
		return m_strName;
	}

	public int[] getValues ()
	{
		int[] rgValues = new int[m_listValues.size ()];
		int i = 0;
		for (Integer nValue : m_listValues)
			rgValues[i++] = nValue;
		return rgValues;
	}

	@Override
	public Iterator<Integer> iterator ()
	{
		return m_listValues.iterator ();
	}

	public boolean isCompatible (Parameter p)
	{
		if (!equals (p))
			return false;
		return m_listValues.equals (p.m_listValues);
	}

	@Override
	public boolean equals (Object obj)
	{
		if (obj == null)
			return false;
		if (!(obj instanceof Parameter))
			return false;

		return m_strName.equals (((Parameter) obj).getName ());
	}

	@Override
	public int hashCode ()
	{
		return m_strName.hashCode ();
	}

	@Override
	public String toString ()
	{
		return StringUtil.concat (m_strName, ": { ", StringUtil.join (m_listValues, ", "), " }");
	}

	@Override
	protected Parameter clone () throws CloneNotSupportedException
	{
		Parameter p = new Parameter (m_strName);
		p.m_listValues.addAll (m_listValues);
		return p;
	}
}
