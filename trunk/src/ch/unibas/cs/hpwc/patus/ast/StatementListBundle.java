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
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import cetus.hir.CompoundStatement;
import cetus.hir.Declaration;
import cetus.hir.Statement;

/**
 *
 * @author Matthias-M. Christen
 */
public class StatementListBundle implements Iterable<ParameterAssignment>, IStatementList
{
	///////////////////////////////////////////////////////////////////
	// Constants

	public final static Parameter DEFAULT_PARAM = new Parameter ("__default__");
	public final static ParameterAssignment DEFAULT_ASSIGNMENT = new ParameterAssignment (StatementListBundle.DEFAULT_PARAM, 0);


	///////////////////////////////////////////////////////////////////
	// Member Variables

	private List<Parameter> m_listParameters;
	private Map<ParameterAssignment, StatementList> m_mapStatementLists;


	///////////////////////////////////////////////////////////////////
	// Implementation

	/**
	 * Creates a new statement list bundle.
	 */
	public StatementListBundle ()
	{
		this (new StatementList (new LinkedList<Statement> ()));
	}

	/**
	 *
	 * @param listStatements
	 */
	public StatementListBundle (List<Statement> listStatements)
	{
		this (new StatementList (listStatements));
	}

	/**
	 *
	 * @param cmpstmt
	 */
	public StatementListBundle (CompoundStatement cmpstmt)
	{
		this (new StatementList (cmpstmt));
	}

	/**
	 *
	 * @param stmt
	 */
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
		m_listParameters = new ArrayList<Parameter> ();
		m_mapStatementLists = new HashMap<ParameterAssignment, StatementList> ();

		m_listParameters.add (StatementListBundle.DEFAULT_PARAM);
		m_mapStatementLists.put (StatementListBundle.DEFAULT_ASSIGNMENT, sl);
	}

	/**
	 * Returns an iterable over all the parameters.
	 * @return
	 */
	public Iterable<Parameter> getParameters ()
	{
		return m_listParameters;
	}

	/**
	 *
	 * @param pa
	 * @return
	 */
	public StatementList getStatementList (ParameterAssignment pa)
	{
		return m_mapStatementLists.get (pa);
	}

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
		for (StatementList sl : getStatementLists (param, nParamValue))
			sl.addDeclaration (declaration);
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
	 * @param stmt
	 * @param param
	 * @param nParamValue
	 */
	public void addStatement (Statement stmt, Parameter param, int nParamValue)
	{
		for (StatementList sl : getStatementLists (param, nParamValue))
			sl.addStatement (stmt);
	}

	public void addStatement (Statement stmt, String strTag, Parameter param, int nParamValue)
	{
		for (StatementList sl : getStatementLists (param, nParamValue))
			sl.addStatement (stmt, strTag);
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
		for (StatementList sl : getStatementLists (param, nParamValue))
			sl.addStatementAtTop (stmt);
	}

	public void addStatementAtTop (Statement stmt, String strTag, Parameter param, int nParamValue)
	{
		for (StatementList sl : getStatementLists (param, nParamValue))
			sl.addStatementAtTop (stmt, strTag);
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
		for (StatementList sl : getStatementLists (param, nParamValue))
			sl.addStatements (listStatements);
	}

	public void addStatements (List<Statement> listStatements, String strTag, Parameter param, int nParamValue)
	{
		for (StatementList sl : getStatementLists (param, nParamValue))
			sl.addStatements (listStatements, strTag);
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
		for (StatementList sl : getStatementLists (param, nParamValue))
			sl.addStatementsAtTop (listStatements);
	}

	public void addStatementsAtTop (List<Statement> listStatements, String strTag, Parameter param, int nParamValue)
	{
		for (StatementList sl : getStatementLists (param, nParamValue))
			sl.addStatementsAtTop (listStatements, strTag);
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
			StatementList sl = slb.getStatementList (pa);
			if (sl != null)
				m_mapStatementLists.get (pa).addStatements (sl);
		}
	}

//	public void addStatements (StatementListBundle slb, Parameter param, int nParamValue)
//	{
//		compatibilize (slb);
//		for (StatementList sl : getStatementList (param, nParamValue))
//			sl.addStatements (slb.g)
//	}

	/**
	 * Adds the {@link CompoundStatement} contents of a statement list in <code>slb</code>
	 * to the respective statement list in this bundle.
	 * @param slb
	 */
	public void addCompoundStatement (StatementListBundle slb)
	{
		compatibilize (slb);
		for (ParameterAssignment pa : m_mapStatementLists.keySet ())
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

	/**
	 * Returns all the statement lists that match the parameter assignment <code>pa</code>.
	 * If <code>pa</code> contains all the parameters that are currently assigned,
	 * @param param
	 * @param nParamValue
	 * @return
	 */
	private Iterable<StatementList> getStatementLists (Parameter param, int nParamValue)
	{
		List<StatementList> list = new LinkedList<StatementList> ();

		// try find the param assignments that have param set to nParamValue
		boolean bParamExists = false;
		for (ParameterAssignment pa : m_mapStatementLists.keySet ())
		{
			if (pa.matches (param, nParamValue))
			{
				list.add (m_mapStatementLists.get (pa));
				bParamExists = true;
			}
		}

		if (!bParamExists)
		{
			param.addValue (nParamValue);

			if (!containsParameter (param))
			{
				// new parameter
				// discard all deprecated branches, mark all existing branches as deprecated

				List<ParameterAssignment> listDeprecated = new LinkedList<ParameterAssignment> ();
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

			// create a new copy of the branches from the branches marked as deprecated with the new parameter/value added
			Map<ParameterAssignment, StatementList> mapTmp = new HashMap<ParameterAssignment, StatementList> ();
			for (ParameterAssignment paOld : m_mapStatementLists.keySet ())
			{
				if (paOld.isDeprecated ())
				{
					ParameterAssignment paNew = paOld.clone ();
					paNew.setParameter (param, nParamValue);

					StatementList slNew = m_mapStatementLists.get (paOld).clone ();
					mapTmp.put (paNew, slNew);
					list.add (slNew);
				}
			}
			m_mapStatementLists.putAll (mapTmp);
		}

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

				Map<ParameterAssignment, StatementList> mapTmp = new HashMap<ParameterAssignment, StatementList> ();
				for (ParameterAssignment paThis : m_mapStatementLists.keySet ())
				{
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
		StringBuilder sb = new StringBuilder ();
		for (ParameterAssignment pa : m_mapStatementLists.keySet ())
		{
			sb.append (pa.toString ());
			sb.append (":\n\n");

			StatementList sl = m_mapStatementLists.get (pa);
			String strCode = sl == null ? "<null>" : sl.toString ();
			if (false)//if (strCode.length () > 100)
			{
				sb.append (strCode.substring (0, 100));
				sb.append ("...");
			}
			else
				sb.append (strCode);
			sb.append ("\n\n\n");
		}

		return sb.toString ();
	}
}
