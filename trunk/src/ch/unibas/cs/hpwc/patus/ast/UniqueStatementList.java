package ch.unibas.cs.hpwc.patus.ast;

import java.util.HashSet;
import java.util.Set;

import cetus.hir.Statement;

public class UniqueStatementList extends StatementList
{
	///////////////////////////////////////////////////////////////////
	// Member Variables
	
	private Set<String> m_setStatements;

	
	///////////////////////////////////////////////////////////////////
	// Implementation

	public UniqueStatementList ()
	{
		m_setStatements = new HashSet<> ();
	}
	
	private boolean existsStatement (Statement stmt)
	{
		return m_setStatements.contains (stmt.toString ());
	}
	
	@Override
	public void addStatement (Statement stmt)
	{
		if (stmt == null || existsStatement (stmt))
			return;

		m_setStatements.add (stmt.toString ());
		super.addStatement (stmt);
	}
	
	@Override
	public void addStatementAtTop (Statement stmt)
	{
		if (stmt == null || existsStatement (stmt))
			return;

		m_setStatements.add (stmt.toString ());
		super.addStatementAtTop (stmt);
	}
}
