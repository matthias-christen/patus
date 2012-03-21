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

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 *
 * @author Matthias-M. Christen
 */
public class ParameterAssignment implements Iterable<Parameter>
{
	///////////////////////////////////////////////////////////////////
	// Member Variables

	private Map<Parameter, Integer> m_mapParameters;

	private boolean m_bIsDeprecated;


	///////////////////////////////////////////////////////////////////
	// Implementation

	public ParameterAssignment ()
	{
		m_mapParameters = new HashMap<> ();
		m_bIsDeprecated = false;
	}

	public ParameterAssignment (Parameter param, int nValue)
	{
		this ();
		setParameter (param, nValue);
	}

	public void setParameter (Parameter param, int nValue)
	{
		m_mapParameters.put (param, nValue);
	}

	public Integer getParameterValueOrNull (Parameter param)
	{
		return m_mapParameters.get (param);
	}

	public int getParameterValue (Parameter param)
	{
		Integer nValue = m_mapParameters.get (param);
		if (nValue == null)
			throw new RuntimeException (StringUtil.concat ("No such parameter: ", param.getName ()));
		return nValue;
	}

	public void setDeprecated ()
	{
		m_bIsDeprecated = true;
	}

	public boolean isDeprecated ()
	{
		return m_bIsDeprecated;
	}

	public boolean matches (Parameter param, int nParamValue)
	{
		Integer nVal = getParameterValueOrNull (param);
		if (nVal == null)
			return false;
		return nVal == nParamValue;
	}

	@Override
	public Iterator<Parameter> iterator ()
	{
		return m_mapParameters.keySet ().iterator ();
	}

	public int getParametersCount ()
	{
		return m_mapParameters.size ();
	}

	@Override
	public boolean equals (Object obj)
	{
		if (obj == null)
			return false;
		if (!(obj instanceof ParameterAssignment))
			return false;
		return m_mapParameters.equals (((ParameterAssignment) obj).m_mapParameters);
	}

	@Override
	public int hashCode ()
	{
		return m_mapParameters.hashCode ();
	}

	@Override
	public ParameterAssignment clone ()
	{
		ParameterAssignment pa = new ParameterAssignment ();
		pa.m_mapParameters.putAll (m_mapParameters);
		return pa;
	}

	@Override
	public String toString ()
	{
		StringBuilder sb = new StringBuilder ("{ ");
		boolean bFirst = true;
		for (Parameter p : m_mapParameters.keySet ())
		{
			if (!bFirst)
				sb.append (", ");
			sb.append (p.getName ());
			sb.append ('=');
			sb.append (m_mapParameters.get (p));
			bFirst = false;
		}
		sb.append (" }");
		if (m_bIsDeprecated)
			sb.append ("*");

		return sb.toString ();
	}
}
