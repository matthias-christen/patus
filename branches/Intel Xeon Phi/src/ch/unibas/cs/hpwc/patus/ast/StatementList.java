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

import cetus.hir.Annotation;
import cetus.hir.AnnotationStatement;
import cetus.hir.CommentAnnotation;
import cetus.hir.CompoundStatement;
import cetus.hir.Declaration;
import cetus.hir.DeclarationStatement;
import cetus.hir.Statement;
import cetus.hir.Traversable;
import ch.unibas.cs.hpwc.patus.util.CodeGeneratorUtil;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

public class StatementList implements IStatementList, Iterable<Statement>, Cloneable
{
	///////////////////////////////////////////////////////////////////
	// Member Variables

	private CompoundStatement m_cmpstmt;
	private List<Statement> m_listStatements;
	private List<Declaration> m_listDeclarations;

	private Map<Statement, String> m_mapTags;


	///////////////////////////////////////////////////////////////////
	// Implementation

	public StatementList ()
	{
		this (new ArrayList<Statement> ());
	}

	public StatementList (CompoundStatement cmpstmt)
	{
		m_cmpstmt = cmpstmt;
		m_listStatements = null;
		m_listDeclarations = new LinkedList<> ();
		m_mapTags = new HashMap<> ();
	}

	public StatementList (List<Statement> listStatements)
	{
		m_cmpstmt = null;
		m_listStatements = listStatements;
		m_listDeclarations = new LinkedList<> ();
		m_mapTags = new HashMap<> ();
	}

	public StatementList (Statement stmt)
	{
		this (new ArrayList<Statement> ());
		addStatement (stmt);
	}

	public StatementList (Statement stmt, String strTag)
	{
		this (stmt);
		m_mapTags.put (stmt, strTag);
	}

	/**
	 * Copy constructor.
	 * @param list
	 */
	public StatementList (StatementList list)
	{
		m_cmpstmt = list.m_cmpstmt != null ? list.m_cmpstmt.clone () : null;

		if (list.m_listStatements == null)
			m_listStatements = null;
		else
		{
			m_listStatements = new LinkedList<> ();
			for (Statement stmt : list.m_listStatements)
				m_listStatements.add (stmt.clone ());
		}

		m_listDeclarations = new LinkedList<> ();
		for (Declaration decl : list.m_listDeclarations)
			m_listDeclarations.add (decl.clone ());
	}

	/**
	 * Removes all the statements and declarations from the statement list.
	 */
	public void clear ()
	{
		if (m_cmpstmt != null)
		{
			List<Traversable> listChildren = new ArrayList<> (m_cmpstmt.getChildren ().size ());
			listChildren.addAll (m_cmpstmt.getChildren ());
			for (Traversable trv : listChildren)
			{
				m_cmpstmt.removeChild (trv);
				trv.setParent (null);
			}
		}

		if (m_listStatements != null)
			m_listStatements.clear ();

		m_listDeclarations.clear ();
	}

	@Override
	public void addStatement (Statement stmt)
	{
		if (stmt == null)
			return;

		if (m_cmpstmt != null)
			m_cmpstmt.addStatement (stmt.getParent () != null ? stmt.clone () : stmt);
		if (m_listStatements != null)
			m_listStatements.add (stmt);
	}

	/**
	 * Adds a statement and gives it the tag <code>strTag</code>.
	 * @param stmt The statement to add
	 * @param strTag The tag to add to the statement <code>stmt</code>
	 */
	public void addStatement (Statement stmt, String strTag)
	{
		addStatement (stmt);
		m_mapTags.put (stmt, strTag);
	}

	public void addAnnotation (Annotation annotation)
	{
		if (annotation instanceof CommentAnnotation)
			((CommentAnnotation) annotation).setOneLiner (true);
		addStatement (new AnnotationStatement (annotation));
	}

	public void addStatements (List<Statement> listStatements)
	{
		for (Statement stmt : listStatements)
			addStatement (stmt);
	}

	public void addStatements (List<Statement> listStatements, String strTag)
	{
		addStatements (listStatements);
		for (Statement stmt : listStatements)
			m_mapTags.put (stmt, strTag);
	}

	public void addStatements (StatementList sl)
	{
		// add statements
		if (sl.m_cmpstmt != null)
		{
			for (Traversable trv : sl.m_cmpstmt.getChildren ())
				if (trv instanceof Statement)
					addStatement ((Statement) trv);
		}
		else if (sl.m_listStatements != null)
			for (Statement stmt : sl.m_listStatements)
				addStatement (stmt);

		// add declarations
		m_listDeclarations.addAll (sl.m_listDeclarations);
	}

	public void addStatements (StatementList sl, String strTag)
	{
		addStatements (sl);
		for (Statement stmt : sl)
			m_mapTags.put (stmt, strTag);
	}

	@Override
	public void addStatementAtTop (Statement stmt)
	{
		if (stmt == null)
			return;

		if (m_cmpstmt != null)
			CodeGeneratorUtil.addStatementAtTop (m_cmpstmt, stmt);
		if (m_listStatements != null)
			m_listStatements.add (0, stmt);
	}

	public void addStatementAtTop (Statement stmt, String strTag)
	{
		addStatementAtTop (stmt);
		m_mapTags.put (stmt, strTag);
	}

	public void addStatementsAtTop (Statement... rgStatements)
	{
		if (rgStatements == null)
			return;

		if (m_cmpstmt != null)
			CodeGeneratorUtil.addStatementsAtTop (m_cmpstmt, rgStatements);
		if (m_listStatements != null)
		{
			List<Statement> listStatements = new ArrayList<> (rgStatements.length);
			for (Statement stmt : rgStatements)
				listStatements.add (stmt);
			m_listStatements.addAll (0, listStatements);
		}
	}

	public void addStatementsAtTop (String strTag, Statement... rgStatements)
	{
		addStatementsAtTop (rgStatements);
		for (Statement stmt : rgStatements)
			m_mapTags.put (stmt, strTag);
	}

	public void addStatementsAtTop (List<Statement> listStatements)
	{
		if (listStatements == null)
			return;

		if (m_cmpstmt != null)
			CodeGeneratorUtil.addStatementsAtTop (m_cmpstmt, listStatements);
		if (m_listStatements != null)
			m_listStatements.addAll (0, listStatements);
	}

	public void addStatementsAtTop (List<Statement> listStatements, String strTag)
	{
		addStatementsAtTop (listStatements);
		for (Statement stmt : listStatements)
			m_mapTags.put (stmt, strTag);
	}

	public void addStatementsAtTop (StatementList sl)
	{
		if (sl == null)
			return;

		if (m_cmpstmt != null)
			CodeGeneratorUtil.addStatementsAtTop (m_cmpstmt, sl.getStatementsAsList ());
		if (m_listStatements != null)
			m_listStatements.addAll (0, sl.getStatementsAsList ());
	}

	public void addStatementsAtTop (StatementList sl, String strTag)
	{
		addStatementsAtTop (sl);
		for (Statement stmt : sl)
			m_mapTags.put (stmt, strTag);
	}

	/**
	 * Determines whether the statement <code>stmt</code> within the statement list has the tag <code>strTag</code>.
	 * @param stmt The statement for which to determine whether it has the tag <code>strTag</code>
	 * @param strTag The tag to check. If <code>null</code>, always <code>true</code> is returned
	 * @return <code>true</code> iff <code>stmt</code> has the tag <code>strTag</code>
	 */
	public boolean hasTag (Statement stmt, String strTag)
	{
		if (m_mapTags == null || stmt == null || strTag == null)
			return true;

		String strStmtTag = m_mapTags.get (stmt);
		if (strStmtTag == null)
			return false;
		return strStmtTag.equals (strTag);
	}

	public String getTag (Statement stmt)
	{
		return m_mapTags.get (stmt);
	}

	public CompoundStatement getCompoundStatement ()
	{
		if (m_cmpstmt != null)
			return m_cmpstmt;

		if (m_listStatements != null)
		{
			CompoundStatement cmpstmt = new CompoundStatement ();
			for (Statement stmt : m_listStatements)
			{
				stmt.setParent (null);
				if (stmt instanceof DeclarationStatement)
				{
					Declaration decl = ((DeclarationStatement) stmt).getDeclaration ();
					decl.setParent (null);
					cmpstmt.addDeclaration (decl);
				}
				else
					cmpstmt.addStatement (stmt);
			}

			return cmpstmt;
		}

		return null;
	}

	public List<Statement> getStatementsAsList ()
	{
		List<Statement> listStatements = new ArrayList<> (m_listStatements == null ? (m_cmpstmt == null ? 0 : m_cmpstmt.getChildren ().size ()) : m_listStatements.size ());
		if (m_listStatements != null)
			listStatements.addAll (m_listStatements);
		else if (m_cmpstmt != null)
		{
			for (Traversable trv : m_cmpstmt.getChildren ())
				if (trv instanceof Statement)
					listStatements.add ((Statement) trv);
		}

		return listStatements;
	}

	public Statement[] getStatementsAsArray ()
	{
		Statement[] rgStatements = new Statement[m_listStatements == null ? (m_cmpstmt == null ? 0 : m_cmpstmt.getChildren ().size ()) : m_listStatements.size ()];
		if (m_listStatements != null)
			m_listStatements.toArray (rgStatements);
		else if (m_cmpstmt != null)
		{
			int i = 0;
			for (Traversable trv : m_cmpstmt.getChildren ())
				if (trv instanceof Statement)
					rgStatements[i++] = (Statement) trv;
		}

		return rgStatements;
	}

	@Override
	public Iterator<Statement> iterator ()
	{
		if (m_cmpstmt != null)
		{
			List<Statement> listStatements = new ArrayList<> (m_cmpstmt.getChildren ().size ());
			for (Traversable trv : m_cmpstmt.getChildren ())
				if (trv instanceof Statement)
					listStatements.add ((Statement) trv);
			return listStatements.iterator ();
		}

		if (m_listStatements != null)
			return m_listStatements.iterator ();

		return null;
	}

	@Override
	public void addDeclaration (Declaration declaration)
	{
		m_listDeclarations.add (declaration);
	}

	public List<Declaration> getDeclarations ()
	{
		return m_listDeclarations;
	}

	@Override
	public String toString ()
	{
		if (m_cmpstmt != null)
			return m_cmpstmt.toString ();
		return StringUtil.join (m_listStatements, "\n");
	}

	public String toStringWithDeclarations ()
	{
		if (m_listDeclarations.size () == 0)
			return toString ();
		return StringUtil.concat (StringUtil.join (m_listDeclarations, ";\n"), ";\n", toString ());
	}

	@Override
	protected StatementList clone ()
	{
		return new StatementList (this);
	}

	public boolean isEmpty ()
	{
		if (m_listDeclarations.size () > 0)
			return false;
		if (m_listStatements != null)
			return m_listStatements.isEmpty ();
		if (m_cmpstmt != null)
			return m_cmpstmt.getChildren ().size () == 0;
		return true;
	}
}
