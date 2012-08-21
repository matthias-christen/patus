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
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

import cetus.hir.CompoundStatement;
import cetus.hir.Declaration;
import cetus.hir.Statement;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 *
 * @author Matthias-M. Christen
 */
public class StatementListBundle implements Iterable<ParameterAssignment>, IStatementList
{
	///////////////////////////////////////////////////////////////////
	// Constants

	public final static Parameter DEFAULT_PARAM = new Parameter ("__default__");
	
	private final static Comparator<Parameter> PARAM_COMPARATOR = new Comparator<Parameter> ()
	{
		@Override
		public int compare (Parameter p1, Parameter p2)
		{
			return p1.getName ().compareTo (p2.toString ());
		}
	};
	
	private final static Comparator<ParameterAssignment> PARAMASSIGNMENT_COMPARATOR = new Comparator<ParameterAssignment> ()
	{
		@Override
		public int compare (ParameterAssignment pa1, ParameterAssignment pa2)
		{
			Set<Parameter> setParams = new TreeSet<> (PARAM_COMPARATOR);				
			for (Parameter p : pa1)
				setParams.add (p);
			for (Parameter p : pa2)
				setParams.add (p);
			
			for (Parameter p : setParams)
			{
				Integer nVal1 = pa1.getParameterValueOrNull (p);
				Integer nVal2 = pa2.getParameterValueOrNull (p);
				
				if (nVal1 == null)
					nVal1 = Integer.MIN_VALUE;
				if (nVal2 == null)
					nVal2 = Integer.MIN_VALUE;
				
				if (nVal1.intValue () != nVal2.intValue ())
					return nVal1 - nVal2;
			}

			return 0;
		}
	};


	///////////////////////////////////////////////////////////////////
	// Member Variables

	private Collection<Parameter> m_listParameters;
	private Map<ParameterAssignment, StatementList> m_mapStatementLists;
	
	/**
	 * Map of statement lists from which new statement lists are derived when a new value is added to a parameter
	 */
	private Map<Parameter, StatementList> m_mapDeriveFromStatementLists;


	///////////////////////////////////////////////////////////////////
	// Implementation

	public StatementListBundle ()
	{
		this (new StatementList (new LinkedList<Statement> ()));
	}

	public StatementListBundle (List<Statement> listStatements)
	{
		this (new StatementList (listStatements));
	}

	public StatementListBundle (CompoundStatement cmpstmt)
	{
		this (new StatementList (cmpstmt));
	}

	public StatementListBundle (Statement stmt)
	{
		this (new StatementList (stmt));
	}

	public StatementListBundle (Statement stmt, String strTag)
	{
		this (new StatementList (stmt, strTag));
	}

	public StatementListBundle (StatementList sl)
	{
		// we want the list of parameters and the sl map to be sorted
		m_listParameters = new PriorityQueue<> (10, PARAM_COMPARATOR);
		m_mapStatementLists = new TreeMap<> (PARAMASSIGNMENT_COMPARATOR);
		
		m_mapDeriveFromStatementLists = new HashMap<> ();

		Parameter paramDefault = StatementListBundle.DEFAULT_PARAM.clone ();
		m_listParameters.add (paramDefault);
		m_mapStatementLists.put (new ParameterAssignment (paramDefault, 0), sl);
	}

	/**
	 * Returns an iterable over all the parameters.
	 * 
	 * @return An iterable over all statement list bundle parameters
	 */
	public Iterable<Parameter> getParameters ()
	{
		return m_listParameters;
	}

	/**
	 * Retrieves the statement list belonging to the parameter assignment
	 * <code>pa</code>.
	 * 
	 * @param pa
	 *            The parameter assignment for which to retrieve the list of
	 *            statements
	 * @return The statement list belonging to <code>pa</code>
	 */
	public StatementList getStatementList (ParameterAssignment pa)
	{
		return m_mapStatementLists.get (pa);
	}

	/**
	 * Replaces the statement list at the parameter assignment <code>pa</code>
	 * by the new statement list, <code>slNew</code>.
	 * 
	 * @param pa
	 *            The parameter assigment whose statement list to replace
	 * @param slNew
	 *            The new statement list, which will be replace the current
	 *            statements belonging to the parameter assignment
	 *            <code>pa</code>
	 */
	public void replaceStatementList (ParameterAssignment pa, StatementList slNew)
	{
		// add missing parameters to list
		if (!m_mapStatementLists.containsKey (pa))
			for (Parameter param : pa)
				if (!containsParameter (param))
					m_listParameters.add (param);
		
		m_mapStatementLists.put (pa, slNew);
	}

	/**
	 * Replaces the statement lists at all available parameter assignments of
	 * the replacing statement list bundle, <code>slb</code>.
	 * 
	 * @param slb
	 *            The statement list bundle which replaces statement lists of
	 *            <code>this</code> bundle
	 */
	public void replaceStatementLists (StatementListBundle slb)
	{
		for (ParameterAssignment pa : slb)
			replaceStatementList (pa, slb.getStatementList (pa));
	}

	/**
	 * Adds a declaration to all the statement lists in the bundle.
	 * @param declaration The declaration to add
	 */
	@Override
	public void addDeclaration (Declaration declaration)
	{
		for (StatementList sl : m_mapStatementLists.values ())
			sl.addDeclaration (declaration);
	}

	/**
	 * Adds a declaration on the statement list with the parameter assignment <code>pa</code>
	 * only. All other statement lists are not affected.
	 * If no statement list is associated with <code>pa</code>, no operation is carried out.
	 * @param declaration
	 * @param pa
	 */
	public void addDeclaration (Declaration declaration, Parameter param, int nParamValue)
	{
		if (param == null)
			addDeclaration (declaration);
		else
		{
			for (StatementList sl : getStatementLists (param, nParamValue))
				sl.addDeclaration (declaration);
		}
	}

	/**
	 * Adds the statement <code>stmt</code> to all the existing code branches.
	 * @param stmt The statement to add
	 */
	@Override
	public void addStatement (Statement stmt)
	{
		for (StatementList sl : m_mapStatementLists.values ())
			sl.addStatement (stmt);
	}

	public void addStatement (Statement stmt, String strTag)
	{
		for (StatementList sl : m_mapStatementLists.values ())
			sl.addStatement (stmt, strTag);
	}

	/**
	 * Adds the statement <code>stmt</code> for a new parameter/value, i.e.,
	 * creates a new code branch or adds the statements to all the branches that
	 * have the parameter <code>param</code> set to <code>nParamValue</code> (if
	 * the code branch(es) already exist(s)).
	 * 
	 * @param stmt
	 *            The statement to add
	 * @param param
	 *            The parameter
	 * @param nParamValue
	 *            The parameter value defining to which statement list of the
	 *            bundle the statement <code>stmt</code> is added
	 */
	public void addStatement (Statement stmt, Parameter param, int nParamValue)
	{
		if (param == null)
			addStatement (stmt);
		else
		{
			for (StatementList sl : getStatementLists (param, nParamValue))
				sl.addStatement (stmt);
		}
	}

	public void addStatement (Statement stmt, String strTag, Parameter param, int nParamValue)
	{
		if (param == null)
			addStatement (stmt, strTag);
		else
		{
			for (StatementList sl : getStatementLists (param, nParamValue))
				sl.addStatement (stmt, strTag);
		}
	}

	/**
	 *
	 * @param stmt
	 */
	@Override
	public void addStatementAtTop (Statement stmt)
	{
		for (StatementList sl : m_mapStatementLists.values ())
			sl.addStatementAtTop (stmt);
	}

	public void addStatementAtTop (Statement stmt, String strTag)
	{
		for (StatementList sl : m_mapStatementLists.values ())
			sl.addStatementAtTop (stmt, strTag);
	}

	/**
	 *
	 * @param stmt
	 * @param param
	 * @param nParamValue
	 */
	public void addStatementAtTop (Statement stmt, Parameter param, int nParamValue)
	{
		if (param == null)
			addStatementAtTop (stmt);
		else
		{
			for (StatementList sl : getStatementLists (param, nParamValue))
				sl.addStatementAtTop (stmt);
		}
	}

	public void addStatementAtTop (Statement stmt, String strTag, Parameter param, int nParamValue)
	{
		if (param == null)
			addStatement (stmt, strTag);
		else
		{
			for (StatementList sl : getStatementLists (param, nParamValue))
				sl.addStatementAtTop (stmt, strTag);
		}
	}

	public void addStatements (List<Statement> listStatements)
	{
		for (StatementList sl : m_mapStatementLists.values ())
			sl.addStatements (listStatements);
	}

	public void addStatements (List<Statement> listStatements, String strTag)
	{
		for (StatementList sl : m_mapStatementLists.values ())
			sl.addStatements (listStatements, strTag);
	}

	public void addStatements (List<Statement> listStatements, Parameter param, int nParamValue)
	{
		if (param == null)
			addStatements (listStatements);
		else
		{
			for (StatementList sl : getStatementLists (param, nParamValue))
				sl.addStatements (listStatements);
		}
	}

	public void addStatements (List<Statement> listStatements, String strTag, Parameter param, int nParamValue)
	{
		if (param == null)
			addStatements (listStatements, strTag);
		else
		{
			for (StatementList sl : getStatementLists (param, nParamValue))
				sl.addStatements (listStatements, strTag);
		}
	}
	
	public void addStatementsAtTop (List<Statement> listStatements)
	{
		for (StatementList sl : m_mapStatementLists.values ())
			sl.addStatementsAtTop (listStatements);
	}

	public void addStatementsAtTop (List<Statement> listStatements, String strTag)
	{
		for (StatementList sl : m_mapStatementLists.values ())
			sl.addStatementsAtTop (listStatements, strTag);
	}

	public void addStatementsAtTop (List<Statement> listStatements, Parameter param, int nParamValue)
	{
		if (param == null)
			addStatementsAtTop (listStatements);
		else
		{
			for (StatementList sl : getStatementLists (param, nParamValue))
				sl.addStatementsAtTop (listStatements);
		}
	}

	public void addStatementsAtTop (List<Statement> listStatements, String strTag, Parameter param, int nParamValue)
	{
		if (param == null)
			addStatementsAtTop (listStatements, strTag);
		else
		{
			for (StatementList sl : getStatementLists (param, nParamValue))
				sl.addStatementsAtTop (listStatements, strTag);
		}
	}
	
	/**
	 * Adds another statement list bundle to this one.
	 * @param slb
	 */
	public void addStatements (StatementListBundle slb)
	{
		if (slb.isEmpty ())
			return;

		compatibilize (slb);
		for (ParameterAssignment pa : m_mapStatementLists.keySet ())
		{
			if (!pa.isDeprecated ())
			{
				StatementList sl = slb.getStatementList (pa);
				if (sl != null)
					m_mapStatementLists.get (pa).addStatements (sl);
			}
		}
	}

	public void addStatementsAtTop (StatementListBundle slb)
	{
		if (slb.isEmpty ())
			return;

		compatibilize (slb);
		for (ParameterAssignment pa : m_mapStatementLists.keySet ())
		{
			if (pa.isDeprecated ())
				continue;

			StatementList sl = slb.getStatementList (pa);
			if (sl != null)
				m_mapStatementLists.get (pa).addStatementsAtTop (sl);
		}
	}

	public void addStatements (StatementListBundle slb, Parameter param, int... rgParamValues)
	{
		if (slb.isEmpty ())
			return;
		
		if (param == null)
			addStatements (slb);
		else
		{
			// make sure that the param is contained in "this" statement list bundle
			ensureParamExists (param, rgParamValues);
	
			compatibilize (slb);
			
			for (ParameterAssignment pa : this)
			{
				if (pa.isDeprecated ())
					continue;
				
				// add the statement list once to the pa's statement list if one of the values in rgParamValues matches
				for (int nParamValue : rgParamValues)
				{
					if (pa.matches (param, nParamValue))
					{
						getStatementList (pa).addStatements (slb.getStatementList (pa));
						break;
					}
				}
			}
		}
	}

	/**
	 * Adds the {@link CompoundStatement} contents of a statement list in <code>slb</code>
	 * to the respective statement list in this bundle.
	 * @param slb
	 */
	public void addCompoundStatement (StatementListBundle slb)
	{
		compatibilize (slb);
		for (ParameterAssignment pa : m_mapStatementLists.keySet ())
			if (!pa.isDeprecated ())
				m_mapStatementLists.get (pa).addStatement (slb.getStatementList (pa).getCompoundStatement ());
	}

	@Override
	public Iterator<ParameterAssignment> iterator ()
	{
		return m_mapStatementLists.keySet ().iterator ();
	}

	/**
	 * Gets an arbitrary {@link StatementList} (but non-deprecated) from the statement bundle.
	 * @return
	 */
	public StatementList getDefaultList ()
	{
		// get the first statement list
		StatementList sl = null;
		StatementList slDeprecated = null;
		for (ParameterAssignment pa : m_mapStatementLists.keySet ())
		{
			if (pa.isDeprecated ())
			{
				if (slDeprecated == null)
					slDeprecated = m_mapStatementLists.get (pa);
			}
			else
			{
				sl = m_mapStatementLists.get (pa);
				break;
			}
		}

		return sl == null ? slDeprecated : sl;
	}

	/**
	 * Gets an arbitrary {@link CompoundStatement} from the statement bundle.
	 * @return
	 */
	public Statement getDefault ()
	{
		// get the default statement list
		StatementList sl = getDefaultList ();
		return sl == null ? null : sl.getCompoundStatement ();
	}

	/**
	 * Returns the number of statement lists in the bundle.
	 * @return The number of statement lists
	 */
	public int size ()
	{
		return m_mapStatementLists.size ();
	}

	public boolean isEmpty ()
	{
		if (size () == 0)
			return true;
		for (StatementList sl : m_mapStatementLists.values ())
			if (!sl.isEmpty ())
				return false;
		return true;
	}
	
	private void ensureParamExists (Parameter param, int... rgParamValues)
	{
		// try find the param assignments that have param set to nParamValue
		boolean bAllValuesFound = true;
		List<Integer> listMissingValues = new ArrayList<> (rgParamValues.length);
		for (int nParamValue : rgParamValues)
		{
			boolean bValueFound = false;
			for (ParameterAssignment pa : m_mapStatementLists.keySet ())
			{
				if (pa.matches (param, nParamValue))
				{
					bValueFound = true;
					break;
				}
			}
			
			if (!bValueFound)
			{
				bAllValuesFound = false;
				listMissingValues.add (nParamValue);
			}
		}

		if (bAllValuesFound)
			return;
		
		boolean bContainsParameter = containsParameter (param);

		// get a template value on which the missing values will be based (only used if the parameter isn't contained in the slb yet)
		int nTemplateValue = -1;
		if (bContainsParameter)
			nTemplateValue = param.getValues ()[0];

		// parameter / value was not found
		for (int nParamValue : listMissingValues)
			param.addValue (nParamValue);

		if (!bContainsParameter)
		{
			// new parameter
			// discard all deprecated branches, mark all existing branches as deprecated

			List<ParameterAssignment> listDeprecated = new LinkedList<> ();
			for (ParameterAssignment pa : m_mapStatementLists.keySet ())
				if (pa.isDeprecated ())
					listDeprecated.add (pa);
			if (listDeprecated.size () > 0 && m_mapStatementLists.size () > 1)
				for (ParameterAssignment pa : listDeprecated)
					m_mapStatementLists.remove (pa);

			for (ParameterAssignment pa : m_mapStatementLists.keySet ())
				pa.setDeprecated ();

			// add the new parameter
			m_listParameters.add (param);
		}
		else
		{
			// parameter was found, but only some values were not found
			StatementList slDeriveFrom = m_mapDeriveFromStatementLists.get (param);
			if (slDeriveFrom == null)
				throw new RuntimeException (StringUtil.concat ("No statement list found from which a branch for ", param.getName (), "=", listMissingValues.toString (), " could be derived."));
			
			Map<ParameterAssignment, StatementList> mapTmp = new HashMap<> ();
			for (ParameterAssignment pa : m_mapStatementLists.keySet ())
			{
				if (pa.matches (param, nTemplateValue))
				{
					for (int nParamValue : listMissingValues)
					{
						ParameterAssignment paNew = pa.clone ();
						paNew.setParameter (param, nParamValue);
						mapTmp.put (paNew, slDeriveFrom.clone ());
					}
				}
			}
			
			m_mapStatementLists.putAll (mapTmp);
		}

		// create a new copy of the branches from the branches marked as deprecated with the new parameter/value added
		Map<ParameterAssignment, StatementList> mapTmp = new HashMap<> ();
		for (ParameterAssignment paOld : m_mapStatementLists.keySet ())
		{
			if (paOld.isDeprecated ())
			{
				for (int nParamValue : rgParamValues)
				{
					ParameterAssignment paNew = paOld.clone ();
					paNew.setParameter (param, nParamValue);
					
					if (m_mapStatementLists.containsKey (paNew))
						continue;
	
					StatementList slOld = m_mapStatementLists.get (paOld);
					if (!m_mapDeriveFromStatementLists.containsKey (param))
						m_mapDeriveFromStatementLists.put (param, slOld);
					
					StatementList slNew = slOld.clone ();
					mapTmp.put (paNew, slNew);
				}
			}
		}

		m_mapStatementLists.putAll (mapTmp);
	}

	/**
	 * Returns all the statement lists that match the parameter assignment <code>pa</code>.
	 * If <code>pa</code> contains all the parameters that are currently assigned,
	 * @param param
	 * @param nParamValue
	 * @return
	 */
	private Iterable<StatementList> getStatementLists (Parameter param, int nParamValue)
	{
		ensureParamExists (param, nParamValue);
		
		List<StatementList> list = new LinkedList<> ();
		for (ParameterAssignment pa : m_mapStatementLists.keySet ())
			if (pa.matches (param, nParamValue))
				list.add (m_mapStatementLists.get (pa));
		return list;
	}

	/**
	 * Makes this statement list bundle and <code>slb</code> compatible,
	 * i.e. adds statement lists to either of this or <code>slb</code> so they both share
	 * the same parameter assignments after the method call.
	 * @param slb
	 */
	public void compatibilize (StatementListBundle slb)
	{
		compatibilizeInternal (slb);
		slb.compatibilizeInternal (this);
	}

	/**
	 * Performs a linear search for the parameter named <code>strParamName</code>
	 * and returns <code>null</code> if the parameter can't be found.
	 * @param strParamName The name of the parameter to find
	 * @return The parameter object or <code>null</code>
	 */
	private Parameter findParameter (String strParamName)
	{
		for (Parameter param : m_listParameters)
			if (param.getName ().equals (strParamName))
				return param;
		return null;
	}

	/**
	 * Determines whether the parameter <code>param</code> has been defined
	 * in this statement list bundle.
	 * @param param
	 * @return
	 */
	private boolean containsParameter (Parameter param)
	{
		return findParameter (param.getName ()) != null;
	}

	/**
	 * Adds all the parameters that are in <code>slb</code>, but not in this object
	 * to this object (including the corresponding statement lists).
	 * @param slb
	 */
	private void compatibilizeInternal (StatementListBundle slb)
	{
		for (Parameter paramOther : slb.m_listParameters)
		{
			Parameter paramThis = findParameter (paramOther.getName ());
			if (paramThis != null)
			{
				// check whether the params are compatible
				// this should never happen
				if (!paramThis.isCompatible (paramOther))
					throw new RuntimeException ("Parameters are not compatible");
			}
			else
			{
				// there is no parameter corresponding to paramOther in this statement list bundle
				// add the parameter with all the values defined in paramOther

				Map<ParameterAssignment, StatementList> mapTmp = new TreeMap<> (PARAMASSIGNMENT_COMPARATOR);
				for (ParameterAssignment paThis : m_mapStatementLists.keySet ())
				{
					if (paThis.isDeprecated ())
						continue;
					
					StatementList slThis = m_mapStatementLists.get (paThis);
					for (int nValue : paramOther)
					{
						ParameterAssignment paNew = paThis.clone ();
						paNew.setParameter (paramOther, nValue);
						mapTmp.put (paNew, slThis.clone ());
					}
				}

				m_mapStatementLists = mapTmp;
				m_listParameters.add (paramOther);
			}
		}
	}

	@Override
	public String toString ()
	{
		return toString (100);
	}
	
	public String toLongString ()
	{
		return toString (Integer.MAX_VALUE);
	}
	
	public String toString (int nCharLimit)
	{
		StringBuilder sb = new StringBuilder ();
		for (ParameterAssignment pa : m_mapStatementLists.keySet ())
		{
			sb.append (pa.toString ());
			sb.append (":\n\n");

			StatementList sl = m_mapStatementLists.get (pa);
			String strCode = sl == null ? "<null>" : sl.toString ();
			if (strCode.length () > nCharLimit)
			{
				sb.append (strCode.substring (0, nCharLimit));
				sb.append ("...");
			}
			else
				sb.append (strCode);
			sb.append ("\n\n\n");
		}

		return sb.toString ();
	}
}
