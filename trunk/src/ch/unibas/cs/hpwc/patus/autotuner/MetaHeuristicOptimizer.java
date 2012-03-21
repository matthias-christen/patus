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
package ch.unibas.cs.hpwc.patus.autotuner;

import org.apache.log4j.Logger;
import org.jgap.Chromosome;
import org.jgap.Configuration;
import org.jgap.FitnessFunction;
import org.jgap.Gene;
import org.jgap.Genotype;
import org.jgap.IChromosome;
import org.jgap.InvalidConfigurationException;
import org.jgap.impl.DefaultConfiguration;
import org.jgap.impl.IntegerGene;

public class MetaHeuristicOptimizer extends AbstractOptimizer
{
	private static final int MAX_EVOLUTIONS = 200;
	private static final int POPULATION_SIZE = 50;

	private final static Logger LOGGER = Logger.getLogger (MetaHeuristicOptimizer.class);
	

	@Override
	public void optimize (final IRunExecutable run)
	{
		Configuration config = new DefaultConfiguration ();
		config.setPreservFittestIndividual (true);
		
		try
		{
			config.setFitnessFunction (new FitnessFunction ()
			{
				private static final long serialVersionUID = 1L;
				private Double m_fBias = null;

				@Override
				protected double evaluate (IChromosome chromosome)
				{
					StringBuilder sbResult = new StringBuilder ();					
					double fResult = run.execute (MetaHeuristicOptimizer.getParametersFromChromosome (chromosome), sbResult, checkBounds ());
					chromosome.setApplicationData (sbResult.toString ());

					// JGAP maximizes the fitness function, and only positive results are allowed
					
					//return Math.exp (-fResult / 1e8);
					
					//if (m_fBias == null)
					//	m_fBias = fResult * 10;
					//return m_fBias - fResult;
					
					if (m_fBias == null && fResult != Double.MAX_VALUE && fResult != 0)
						m_fBias = fResult;
					if (m_fBias == null)
						return Math.exp (-fResult);
					return Math.exp (-fResult / m_fBias);
				}
			});
			
			// create the chromosome
			Gene[] rgSampleGenes = new Gene[run.getParametersCount ()];
			for (int i = 0; i < run.getParametersCount (); i++)
				rgSampleGenes[i] = new IntegerGene (config, run.getParameterLowerBounds ()[i], run.getParameterUpperBounds ()[i]);
			config.setSampleChromosome (new Chromosome (config, rgSampleGenes));

			config.setPopulationSize (POPULATION_SIZE);
			
			// create population and evolve it
			Genotype population = Genotype.randomInitialGenotype (config);
			for (int i = 0; i < MAX_EVOLUTIONS; i++)
			{
				population.evolve ();
				LOGGER.info (population.getFittestChromosome ());
			}
			
			setResult (population.getFittestChromosome ());
		}
		catch (InvalidConfigurationException e)
		{
			e.printStackTrace ();
		}		
	}
	
	private void setResult (IChromosome chromosome)
	{
	    setResultParameters (getParametersFromChromosome (chromosome));
	    setResultTiming (chromosome.getFitnessValue ());
	    setProgramOutput ((String) chromosome.getApplicationData ());
	}
	
	private static int[] getParametersFromChromosome (IChromosome chromosome)
	{
		int rgParams[] = new int[chromosome.getGenes ().length];
		for (int i = 0; i < rgParams.length; i++)
			rgParams[i] = (Integer) chromosome.getGene (i).getAllele ();

		return rgParams;
	}
	
	@Override
	public String getName ()
	{
		return "Metaheuristic";
	}
}
