package ch.unibas.cs.hpwc.patus.ilp;

import ch.unibas.cs.hpwc.patus.ilp.ILPSolution.ESolutionStatus;

/**
 * This class encapsulates the external ILP/MIP solver.
 * Currently, the COIN-OR solver <a
 * href="http://www.coin-or.org/projects/Cbc.xml">Cbc</a> is used.
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
	 * Solves the integer linear programming / mixed integer programming problem
	 * defined in <code>model</code>.
	 * 
	 * @param model
	 *            The ILP model to solve
	 * @return A solution object which can be used to retrieve the solver exit
	 *         status, the solution vector, and the value of the objective
	 *         function
	 */
	public ILPSolution solve (ILPModel model)
	{
		return solve (model, -1);
	}
			
	/**
	 * Solves the integer linear programming / mixed integer programming problem
	 * defined in <code>model</code>.
	 * 
	 * @param model
	 *            The ILP model to solve
	 * @param nTimeLimit
	 *            A time limit in seconds after which the solver is canceled. If
	 *            set to a negative value, no time limit is imposed
	 * @return A solution object which can be used to retrieve the solver exit
	 *         status, the solution vector, and the value of the objective
	 *         function
	 */
	public ILPSolution solve (ILPModel model, int nTimeLimit)
	{
		double[] rgSolution = new double[model.getVariablesCount ()];
		double[] rgObjective = new double[1];
		
		int nCode = solveInternal (model, rgSolution, rgObjective, nTimeLimit);
		
		return new ILPSolution (nCode, rgSolution, rgObjective[0]);
	}
	
	/**
	 * Calls the external solver.
	 * 
	 * @param model
	 *            The ILP model to solve
	 * @param rgSolution
	 *            This array will contain the solution when the method returns
	 * @param rgObjective
	 *            This array (containing one element) will contain the value of
	 *            the objective function, once the method returns
	 * @param nTimeLimit
	 *            A time limit in seconds after which the solver is canceled. If
	 *            set to a negative value, no time limit is imposed
	 * @return A solver status code, as defined in {@link ESolutionStatus}
	 */
	private native int solveInternal (ILPModel model, double[] rgSolution, double[] rgObjective, int nTimeLimit);
}
