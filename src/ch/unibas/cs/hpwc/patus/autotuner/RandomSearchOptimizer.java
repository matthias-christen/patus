package ch.unibas.cs.hpwc.patus.autotuner;


public class RandomSearchOptimizer extends AbstractOptimizer
{
	private final static int MAX_SEARCHES = 500;
	
	private int[] m_rgCurrentParams = null;
		
	
	@Override
	public void optimize (IRunExecutable run)
	{
		StringBuilder sbResult = new StringBuilder ();
		for (int i = 0; i < MAX_SEARCHES; i++)
		{
			createNewParams (run);
						
			sbResult.setLength (0);
			double fObjValue = run.execute (m_rgCurrentParams, sbResult, checkBounds ());
			
			OptimizerLog.step (i, m_rgCurrentParams, fObjValue);

			if (fObjValue < getResultTiming ())
			{
				setResultTiming (fObjValue);
				setResultParameters (m_rgCurrentParams);				
				setProgramOutput (sbResult);
			}
		}
	}
	
	private void createNewParams (IRunExecutable run)
	{
		if (m_rgCurrentParams == null)
			m_rgCurrentParams = new int[run.getParametersCount ()];

		OptimizerUtil.getRandomPointWithinBounds (m_rgCurrentParams, run.getParameterLowerBounds (), run.getParameterUpperBounds ());
	}

	@Override
	public String getName ()
	{
		return "RandomSearch";
	}
}
