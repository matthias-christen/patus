package ch.unibas.cs.hpwc.patus.autotuner;

/**
 *
 * @author Matthias-M. Christen
 */
public interface IOptimizer
{
	/**
	 * Runs the optimizer for the executable <code>run</code>.
	 * @param run The executable
	 */
	public abstract void optimize (IRunExecutable run);

	/**
	 * Returns the optimal result parameters (values of the &quot;decision variables&quot;).
	 * @return The result parameter set
	 */
	public abstract int[] getResultParameters ();

	/**
	 * Returns the timing for the optimal parameters determined by the optimizer.
	 * @return The timing for the optimal parameter set
	 */
	public abstract double getResultTiming ();

	/**
	 * Returns the program output for the optimum found.
	 * @return The program output for the optimal run
	 */
	public abstract String getProgramOutput ();

	/**
	 * Returns the optimizer's / method name.
	 * @return The optimizer name
	 */
	public abstract String getName ();
	
	public abstract boolean checkBounds ();
	
	public abstract void setCheckBounds (boolean bCheckBounds);
}
