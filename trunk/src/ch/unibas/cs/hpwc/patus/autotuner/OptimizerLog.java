package ch.unibas.cs.hpwc.patus.autotuner;

import org.apache.log4j.Logger;

import ch.unibas.cs.hpwc.patus.util.StringUtil;

public class OptimizerLog
{
	///////////////////////////////////////////////////////////////////
	// Constants

	private final static Logger LOGGER = Logger.getLogger (OptimizerLog.class);


	///////////////////////////////////////////////////////////////////
	// Implementation

	public static void step (int nIteration, int[] rgConfiguration, double fValue)
	{
		OptimizerLog.step (nIteration, rgConfiguration, fValue, "");
	}

	public static void step (int nIteration, int[] rgConfiguration, double fValue, String strMessage)
	{
		LOGGER.info (StringUtil.concat (
			"Iteration ", String.valueOf (nIteration), ": config = [ ", StringUtil.join (rgConfiguration, " "), "], value = ", String.valueOf (fValue), ".  ", strMessage));
	}

	public static void terminate (int nIteration, int nMaxIterations)
	{
		if (nIteration <= nMaxIterations)
			LOGGER.info ("Optimizer terminated.");
		else
			LOGGER.info ("Maximum optimizer iterations exceeded");
	}
}
