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

import java.util.HashMap;
import java.util.Map;

/**
 * Class encapsulating code generation options that are set by the code generation
 * modules at code generation time.
 *
 * @author Matthias-M. Christen
 */
public class CodeGeneratorRuntimeOptions implements Cloneable
{
	///////////////////////////////////////////////////////////////////
	// Constants

	public final static String OPTION_LOOPUNROLLINGFACTOR = "LoopUnrollingFactor";

	public final static String OPTION_STENCILLOOPUNROLLINGFACTOR = "StencilLoopUnrollingFactor";

	public final static String OPTION_STENCILCALCULATION = "StencilCalculation";
	public final static int VALUE_STENCILCALCULATION_STENCIL = 0;
	public final static int VALUE_STENCILCALCULATION_INITIALIZE = 1;
	public final static int VALUE_STENCILCALCULATION_VALIDATE = 2;

	/**
	 * Suppresses vectorization of the stencil calculation code (if the value is
	 * {@link Boolean#TRUE})
	 */
	public final static String OPTION_NOVECTORIZE = "NoVectorize";

	/**
	 * The loop unrolling configuration used to generated the inner-most loop
	 * when used with a special code generator. The value of the option is
	 * <code>int[]</code>: the offset from the default center node.
	 */
	public static final String OPTION_INNER_UNROLLINGCONFIGURATION = "InnerMostUnrollingConfig";

	/**
	 * Specifies whether in the current code branch boundary checks are generated.
	 * The value is boolean.
	 */
	public static final String OPTION_DOBOUNDARYCHECKS = "DoBoundaryChecks";


	///////////////////////////////////////////////////////////////////
	// Member Variables

	private Map<String, Object> m_mapOptions;


	///////////////////////////////////////////////////////////////////
	// Implementation

	public CodeGeneratorRuntimeOptions ()
	{
		m_mapOptions = new HashMap<> ();
	}

	public void setOption (String strOption, Object objValue)
	{
		m_mapOptions.put (strOption, objValue);
	}

	public void removeOption (String strOption)
	{
		m_mapOptions.remove (strOption);
	}

	public void removeAllOptions ()
	{
		m_mapOptions.clear ();
	}

	public Object getObjectValue (String strOption)
	{
		return m_mapOptions.get (strOption);
	}

	public Boolean getBooleanValue (String strOption)
	{
		Object obj = m_mapOptions.get (strOption);
		if (obj == null || !(obj instanceof Boolean))
			return null;
		return (Boolean) obj;
	}

	public boolean getBooleanValue (String strOption, boolean bDefault)
	{
		Boolean b = getBooleanValue (strOption);
		return b == null ? bDefault : b;
	}

	public Integer getIntValue (String strOption)
	{
		Object obj = m_mapOptions.get (strOption);
		if (obj == null || !(obj instanceof Number))
			return null;
		return ((Number) obj).intValue ();
	}

	public int getIntValue (String strOption, int nDefault)
	{
		Integer n = getIntValue (strOption);
		return n == null ? nDefault : n;
	}

	public String getStringValue (String strOption)
	{
		Object obj = m_mapOptions.get (strOption);
		if (obj == null || !(obj instanceof String))
			return null;
		return (String) obj;
	}

	public String getStringValue (String strOption, String strDefault)
	{
		String s = getStringValue (strOption);
		return s == null ? strDefault : s;
	}
	
	/**
	 * Determines whether the option <code>strOption</code> has the value <code>oValue</code>.
	 * @param strOption The option to check
	 * @param oValue The expected value
	 * @return <code>true</code> iff the option <code>strOption</code> is set to the value <code>oValue</code>
	 */
	public boolean hasValue (String strOption, Object oValue)
	{
		Object oActualValue = getObjectValue (strOption);
		if (oValue == null)
			return oActualValue == null;
		return oValue.equals (oActualValue);
	}

	@Override
	public boolean equals (Object obj)
	{
		if (obj == null)
			return false;
		if (!(obj instanceof CodeGeneratorRuntimeOptions))
			return false;

		return m_mapOptions.equals (((CodeGeneratorRuntimeOptions) obj).m_mapOptions);
	}

	@Override
	public int hashCode ()
	{
		return m_mapOptions.hashCode ();
	}

	@Override
	public CodeGeneratorRuntimeOptions clone ()
	{
		CodeGeneratorRuntimeOptions options = new CodeGeneratorRuntimeOptions ();
		options.m_mapOptions.putAll (m_mapOptions);
		return options;
	}

	@Override
	public String toString ()
	{
		return m_mapOptions.toString ();
	}
}
