package ch.unibas.cs.hpwc.patus.ilp;

/**
 * 
 * @author Matthias-M. Christen
 */
public class ILPSolver
{
	///////////////////////////////////////////////////////////////////
	// Member Variables
	
	private static ILPSolver m_inst;

	
	///////////////////////////////////////////////////////////////////
	// Static Members (Singleton Pattern)

	static
	{
		System.loadLibrary ("ilpsolver");		
		m_inst = new ILPSolver ();
	}
	
	public static ILPSolver getInstance ()
	{
		return m_inst;
	}

	
	///////////////////////////////////////////////////////////////////
	// Implementation

	private ILPSolver ()
	{
	}
			
	/**
	 * 
	 * @param model
	 * @return
	 */
	public ILPSolution solve (ILPModel model)
	{
		double[] rgSolution = new double[model.getVariablesCount ()];
		double[] rgObjective = new double[1];
		
		int nCode = solveInternal (model, rgSolution, rgObjective);
		
		return new ILPSolution (nCode, rgSolution, rgObjective[0]);
	}
	
	/**
	 * 
	 * @param model
	 * @param rgSolution
	 * @return
	 */
	private native int solveInternal (ILPModel model, double[] rgSolution, double[] rgObjective);
}
