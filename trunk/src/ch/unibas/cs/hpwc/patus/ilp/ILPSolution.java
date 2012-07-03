package ch.unibas.cs.hpwc.patus.ilp;

public class ILPSolution
{
	///////////////////////////////////////////////////////////////////
	// Constants
	
	/* package */ final static int STATUS_OPTIMAL = 0;
	/* package */ final static int STATUS_INFEASIBLE = 1;
	/* package */ final static int STATUS_LIMIT_REACHED = 2;
	/* package */ final static int STATUS_ABANDONED = 3;
	/* package */ final static int STATUS_NOSOLUTIONFOUND = 4;
	
	
	///////////////////////////////////////////////////////////////////
	// Inner Types

	public enum ESolutionStatus
	{
		OPTIMAL (STATUS_OPTIMAL),
		INFEASIBLE (STATUS_INFEASIBLE),
		SOLUTION_LIMIT_REACHED (STATUS_LIMIT_REACHED),
		ABANDONED (STATUS_ABANDONED),
		NOSOLUTIONFOUND (STATUS_NOSOLUTIONFOUND),
		UNKNOWN (-1);
		
		
		private int m_nCode;
		
		private ESolutionStatus (int nCode)
		{
			m_nCode = nCode;
		}
		
		public int getCode ()
		{
			return m_nCode;
		}
		
		public static ESolutionStatus fromCode (int nCode)
		{
			switch (nCode)
			{
			case STATUS_OPTIMAL:
				return OPTIMAL;
			case STATUS_INFEASIBLE:
				return INFEASIBLE;
			case STATUS_LIMIT_REACHED:
				return SOLUTION_LIMIT_REACHED;
			case STATUS_ABANDONED:
				return ABANDONED;
			case STATUS_NOSOLUTIONFOUND:
				return NOSOLUTIONFOUND;
			}
			
			return UNKNOWN;
		}
	}

	
	///////////////////////////////////////////////////////////////////
	// Member Variables
	
	private double[] m_rgSolution;
	private double m_fObjective;
	private ESolutionStatus m_status;

	
	///////////////////////////////////////////////////////////////////
	// Implementation

	/* package */ ILPSolution (int nCode, double[] rgSolution, double fObjective)
	{
		m_rgSolution = rgSolution;
		m_fObjective = fObjective;
		m_status = ESolutionStatus.fromCode (nCode);
	}
	
	public double[] getSolution ()
	{
		return m_rgSolution;
	}
	
	public double getObjective ()
	{
		return m_fObjective;
	}
	
	public ESolutionStatus getStatus ()
	{
		return m_status;
	}
	
	@Override
	public String toString ()
	{
		StringBuilder sb = new StringBuilder (m_status.toString ());
		if (m_status.equals (ESolutionStatus.OPTIMAL))
		{
			sb.append ("\n\nObjective value: ");
			sb.append (m_fObjective);
			sb.append ("\n\nSolution:");
			
			for (int i = 0; i < m_rgSolution.length; i++)
			{
				sb.append ("\nVar ");
				sb.append (i);
				sb.append (" = ");
				sb.append (m_rgSolution[i]);
			}
		}
		
		return sb.toString ();
	}
}
