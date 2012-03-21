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
package ch.unibas.cs.hpwc.patus.config;

import java.util.ArrayList;
import java.util.List;

import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 * 
 * @author Matthias-M. Christen
 */
public class ConfigurationProperty
{
	///////////////////////////////////////////////////////////////////
	// Inner Types

	public enum EPropertyType
	{
		INTEGER,
		STRING,
		LIST,
		FILE;
		
		public static EPropertyType fromString (String s)
		{
			if ("integer".equals (s))
				return INTEGER;
			if ("string".equals (s))
				return STRING;
			if ("list".equals (s))
				return LIST;
			if ("file".equals (s))
				return FILE;
			
			throw new RuntimeException ("Type '" + s + "' is unknown!");
		}
	}
	

	///////////////////////////////////////////////////////////////////
	// Member Variables

	/**
	 * The key of the configuration property. The key is assembled automatically
	 * and has the form 'Category'.'Name'.
	 * The key is used to store the value of the property in the configuration file.
	 */
	private String m_strKey;
	
	/**
	 * The property's category
	 */
	private String m_strCategory;
	
	/**
	 * The property's name (used as caption in the configuration dialog)
	 */
	private String m_strName;
	
	/**
	 * The type of the property.
	 * Depending on the type the values in {@link ConfigurationProperty#m_listValues} have a different meaning.
	 * The type is used in the UI to create the matching UI component.
	 */
	private EPropertyType m_type;
	
	/**
	 * List of values, e.g. minima and maxima values (if the property has type {@link EPropertyType#INTEGER}), or
	 * a list of possible values the user can select from (in a combo box &mdash; if the property has type
	 * {@link EPropertyType#STRING}) 
	 */
	private List<Object> m_listValues;
	
	/**
	 * The property's current value.
	 */
	private String m_strValue;
	
	/**
	 * The property's default value.
	 */
	private String m_strDefaultValue;
		
	
	///////////////////////////////////////////////////////////////////
	// Implementation

	/**
	 * Constructs a property object.
	 * @param strCategory
	 * @param strName
	 * @param type
	 * @param strDefaultValue
	 * @param rgValues
	 */
	public ConfigurationProperty (String strCategory, String strName, EPropertyType type, String strDefaultValue, Object... rgValues)
	{
		m_strCategory = strCategory;
		m_strName = strName;
		m_type = type;
		
		m_strValue = null;
		m_strDefaultValue = strDefaultValue;
		
		m_strKey = StringUtil.concat (strCategory, ".", strName);
		
		// set values
		m_listValues = new ArrayList<> (rgValues.length);
		for (Object objValue : rgValues)
			m_listValues.add (objValue);
	}
	
	/**
	 * Returns the category this property belongs to.
	 * In the configuration dialog, each category will be displayed in its own tab page.
	 * @return The property category
	 */
	public String getCategory ()
	{
		return m_strCategory;
	}
	
	/**
	 * Returns the name of the property. The name is also used as caption in the configuration dialog.
	 * @return The property name
	 */
	public String getName ()
	{
		return m_strName;
	}
	
	/**
	 * Returns the property's key with which the property is identified in the configuration file.
	 * @return The property key
	 */
	public String getKey ()
	{
		return m_strKey;
	}
	
	/**
	 * Returns the property's value
	 * @return The value
	 */
	public String getValue ()
	{
		return m_strValue;
	}
	
	/**
	 * Sets the property's value.
	 * @param strValue The new value
	 */
	public void setValue (String strValue)
	{
		m_strValue = strValue;
	}
	
	/**
	 * The property's default value.
	 * @return The default value
	 */
	public String getDefaultValue ()
	{
		return m_strDefaultValue;
	}
	
	/**
	 * Returns the value to display for this property.
	 * If the value hasn't been set yet, the default value is returned, otherwise the value is returned.
	 * @return The value that should be displayed for this property
	 */
	public String getDisplayValue ()
	{
		return m_strValue == null ? m_strDefaultValue : m_strValue;
	}
	
	/**
	 * Returns the type of the property.
	 * Depending on the type the values in {@link ConfigurationProperty#m_listValues} have a different meaning.
	 * The type is used in the UI to create the matching UI component.
	 * @return The property type
	 */
	public EPropertyType getType ()
	{
		return m_type;
	}
	
	/**
	 * Returns the list of values, e.g. minima and maxima values (if the property has type {@link EPropertyType#INTEGER}), or
	 * a list of possible values the user can select from (in a combo box &mdash; if the property has type
	 * {@link EPropertyType#STRING}).
	 * @return List of type-dependent property values
	 */
	public List<?> getValues ()
	{
		return m_listValues;
	}

}
