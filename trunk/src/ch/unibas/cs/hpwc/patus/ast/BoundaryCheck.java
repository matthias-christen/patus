package ch.unibas.cs.hpwc.patus.ast;

import java.io.PrintWriter;

import cetus.hir.Statement;

public class BoundaryCheck extends Statement
{
	///////////////////////////////////////////////////////////////////
	// Member Variables

	private SubdomainIterator m_sdit;
	private Statement m_stmtWithChecks;
	private Statement m_stmtWithoutChecks;

	
	///////////////////////////////////////////////////////////////////
	// Implementation

	public BoundaryCheck (SubdomainIterator it, Statement stmtWithChecks, Statement stmtWithoutChecks)
	{
		m_sdit = it;
		m_stmtWithChecks = stmtWithChecks;
		m_stmtWithoutChecks = stmtWithoutChecks;
		
		if (stmtWithChecks != null)
			addChild (stmtWithChecks);
		if (stmtWithoutChecks != null)
			addChild (stmtWithoutChecks);
	}

	public SubdomainIterator getSubdomainIterator ()
	{
		return m_sdit;
	}

	public Statement getWithChecks ()
	{
		return m_stmtWithChecks;
	}

	public Statement getWithoutChecks ()
	{
		return m_stmtWithoutChecks;
	}
	
	public BoundaryCheck clone ()
	{
		return new BoundaryCheck (m_sdit.clone (), m_stmtWithChecks.clone (), m_stmtWithoutChecks.clone ());
	}
	
	@Override
	public void print (PrintWriter out)
	{
		out.print ("if (check_bnds (");
		out.print (m_sdit.getIterator ().toString ());
		out.println ("))");
		m_stmtWithChecks.print (out);
		out.println ("\nelse");
		m_stmtWithoutChecks.print (out);
	}
}
