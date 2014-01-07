package ch.unibas.cs.hpwc.patus.codegen.backend.openmp;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.log4j.Logger;
import org.jgap.Chromosome;
import org.jgap.Configuration;
import org.jgap.DefaultFitnessEvaluator;
import org.jgap.FitnessFunction;
import org.jgap.Gene;
import org.jgap.Genotype;
import org.jgap.IChromosome;
import org.jgap.InvalidConfigurationException;
import org.jgap.Population;
import org.jgap.RandomGenerator;
import org.jgap.event.EventManager;
import org.jgap.impl.BestChromosomesSelector;
import org.jgap.impl.ChromosomePool;
import org.jgap.impl.CrossoverOperator;
import org.jgap.impl.GABreeder;
import org.jgap.impl.IntegerGene;
import org.jgap.impl.MutationOperator;
import org.jgap.impl.StockRandomGenerator;

import cetus.hir.Expression;
import cetus.hir.FunctionCall;
import cetus.hir.NameID;

public class PermutatorGenetic
{
	private final static int POPULATION_SIZE = 10000;

	private final static int MAX_EVOLUTIONS = 20;

	private final static Logger LOGGER = Logger.getLogger (PermutatorGenetic.class);

	
	public abstract static class Operator
	{
		private int m_nOperandsCount;

		private int m_nElementPerOperand;

		private List<Integer> m_listIDs;
		
		private Map<Integer, List<Selector>> m_mapSelectors;
		
		private NameID m_nidFnxName;

		
		public Operator (String strFnxName, int nOperandsCount, int nElementsPerOperand, Selector[] rgSelectors)
		{
			m_nOperandsCount = nOperandsCount;
			m_nElementPerOperand = nElementsPerOperand;
			m_nidFnxName = new NameID (strFnxName);

			if (rgSelectors.length != nElementsPerOperand)
				throw new RuntimeException ("The number of selectors must correspond to the number of elements per operand");

			// build the selector map
			m_listIDs = new ArrayList<> ();
			m_mapSelectors = new HashMap<> ();
			int nSelectorSetSize = -1;
			int nIndex = 0;
			for (Selector s : rgSelectors)
			{
				s.setIndex (nIndex);
				if (nSelectorSetSize == -1)
					nSelectorSetSize = s.getSelectSet ().length;
				else if (s.getSelectSet ().length != nSelectorSetSize)
					throw new RuntimeException ("All selector sets must have the same size");

				List<Selector> list = m_mapSelectors.get (s.getID ());
				if (list == null)
				{
					m_listIDs.add (s.getID ());
					m_mapSelectors.put (s.getID (), list = new ArrayList<> ());
				}
				list.add (s);

				nIndex++;
			}
		}

		public int getOperandsCount ()
		{
			return m_nOperandsCount;
		}
		
		public NameID getNameID ()
		{
			return m_nidFnxName;
		}

		public IChromosome apply (IChromosome firstMate, IChromosome secondMate, RandomGenerator generator, Configuration config) throws InvalidConfigurationException
		{
			Gene[] rgGenes = new Gene[m_nElementPerOperand];
			for (int i = 0; i < m_nElementPerOperand; i++)
				rgGenes[i] = new IntegerGene (config, 0, 2 * m_nElementPerOperand - 1);
			
			int[] rgOpConfig = new int[m_mapSelectors.size ()];

			int nIdx = 0;
			for (int nID : m_listIDs)
			{
				List<Selector> list = m_mapSelectors.get (nID);
				rgOpConfig[nIdx] = generator.nextInt (list.get (0).getArgsCount ());
				for (Selector s : list)
					rgGenes[s.getIndex ()].setAllele (s.getResult (firstMate, secondMate, rgOpConfig[nIdx], m_nElementPerOperand));
				nIdx++;
			}
			
			List<Operand> listParent = new ArrayList<> (2);
			listParent.add ((Operand) firstMate.getApplicationData ());
			if (m_nOperandsCount >= 2)
				listParent.add ((Operand) secondMate.getApplicationData ());

			IChromosome c = new Chromosome (config, rgGenes);
			c.setApplicationData (new Operand (null, null, listParent, this, rgOpConfig));
			return c;
		}
		
		public Expression getFunctionInvocation (Operand op)
		{
			return new FunctionCall (m_nidFnxName.clone (), getFunctionArguments (op));
		}
		
		public List<Expression> getFunctionArguments (Operand op)
		{
			List<Operand> listParentOps = op.getParentOperands ();
			List<Expression> listArgs = new ArrayList<> (listParentOps.size () + 1);
			
			for (Operand o : listParentOps)
				listArgs.add (o.getExpression ());
			if (op.getOperatorConfiguration () != null)
				listArgs.add (getControlExpression (op.getOperatorConfiguration ()));
			
			return listArgs;
		}
		
		public abstract Expression getControlExpression (int[] rgConfig);
	}

	public static class Selector
	{
		private int m_nID;

		private int m_nIndex;

		private int[] m_rgSelectSet;

		public Selector (int nID, int[] rgSelectSet)
		{
			m_nID = nID;
			m_rgSelectSet = rgSelectSet;
		}

		public int getID ()
		{
			return m_nID;
		}

		/* package */void setIndex (int nIndex)
		{
			m_nIndex = nIndex;
		}

		public int getIndex ()
		{
			return m_nIndex;
		}

		public int[] getSelectSet ()
		{
			return m_rgSelectSet;
		}
		
		public int getArgsCount ()
		{
			return m_rgSelectSet.length;
		}

		public int getResult (List<Operand> listOperands, int nArgNum)
		{
			int nIdx = m_rgSelectSet[nArgNum];
			int nEltsCount = listOperands.get (0).getElementsCount ();
			return listOperands.get (nIdx / nEltsCount).getElements ()[nIdx % nEltsCount];
		}
		
		public int getResult (IChromosome firstMate, IChromosome secondMate, int nArgNum, int nEltsCount)
		{
			int nIdx = m_rgSelectSet[nArgNum];
			
			IChromosome c = firstMate;
			if (nIdx >= nEltsCount)
			{
				c = secondMate;
				nIdx -= nEltsCount;
			}
			
			return (Integer) c.getGene (nIdx).getAllele ();
		}

		@Override
		public String toString ()
		{
			StringBuilder sb = new StringBuilder ("s");
			sb.append (m_nID);
			sb.append (" (");
			sb.append (m_nIndex);
			sb.append (") --> ");
			sb.append (Arrays.toString (m_rgSelectSet));

			return sb.toString ();
		}
	}

	public static class Operand
	{
		private NameID m_nidName;
		
		private int[] m_rgElements;

		private List<Operand> m_listParentOperands;

		private Operator m_operator;
		
		private int[] m_rgOperatorConfiguration;
		
		private int m_nOperatorEvaluationsCount;
		

		public Operand (String strName, int nElementsCount)
		{
			this (strName, new int[nElementsCount], null, null, null);
			Arrays.fill (m_rgElements, -1);
		}

		public Operand (String strName, int[] rgElements)
		{
			this (strName, rgElements, null, null, null);
		}

		public Operand (String strName, int[] rgElements, List<Operand> listParentOperands, Operator op, int[] rgOperatorConfiguration)
		{
			m_nidName = strName == null ? null : new NameID (strName);
			m_rgElements = rgElements;
			m_listParentOperands = listParentOperands;
			m_operator = op;
			m_rgOperatorConfiguration = rgOperatorConfiguration;
			
			m_nOperatorEvaluationsCount = 0;
			if (listParentOperands != null)
			{
				for (Operand parent : listParentOperands)
					m_nOperatorEvaluationsCount += parent.getOperatorEvaluationsCount ();
				m_nOperatorEvaluationsCount++;
			}
		}

		public int[] getElements ()
		{
			return m_rgElements;
		}

		public int getElementsCount ()
		{
			return m_rgElements.length;
		}

		public List<Operand> getParentOperands ()
		{
			return m_listParentOperands;
		}

		public Operator getOperator ()
		{
			return m_operator;
		}
		
		public int[] getOperatorConfiguration ()
		{
			return m_rgOperatorConfiguration;
		}
		
		public int getOperatorEvaluationsCount ()
		{
			return m_nOperatorEvaluationsCount;
		}

		public boolean contains (int n)
		{
			for (int i = 0; i < m_rgElements.length; i++)
				if (m_rgElements[i] == n)
					return true;
			return false;
		}
		
		public Expression getExpression ()
		{
			if (m_operator == null)
				return m_nidName == null ? new NameID ("Op?") : m_nidName.clone ();
			return m_operator.getFunctionInvocation (this);
		}

		@Override
		public boolean equals (Object obj)
		{
			if (!(obj instanceof Operand))
				return false;
			return Arrays.equals (m_rgElements, ((Operand) obj).getElements ());
		}

		@Override
		public int hashCode ()
		{
			return Arrays.hashCode (m_rgElements);
		}

		@Override
		public String toString ()
		{
			//return Arrays.toString (m_rgElements);
			return String.valueOf (m_nOperatorEvaluationsCount);
		}
	}

	
	private int m_nElementsCount;
	private List<Operand> m_listInputOperands;
	private Operand m_opResult;
	private Operator[] m_rgOperators;
	
	private Configuration m_config;
	
	
	public PermutatorGenetic (final int nElementsCount, final List<Operand> listInputOperands, final Operand opResult, final Operator[] rgOperators)
	{
		m_nElementsCount = nElementsCount;
		m_listInputOperands = listInputOperands;
		m_opResult = opResult;
		m_rgOperators = rgOperators;
		
		m_config = new Configuration ();
		
		try
		{
			m_config.setBreeder (new GABreeder ());
			m_config.setRandomGenerator (new StockRandomGenerator ());
			m_config.setEventManager (new EventManager ());
			BestChromosomesSelector bestChromsSelector = new BestChromosomesSelector (m_config, 0.90d);
			bestChromsSelector.setDoubletteChromosomesAllowed (true);
			m_config.addNaturalSelector (bestChromsSelector, false);
			m_config.setMinimumPopSizePercent (0);
			m_config.setSelectFromPrevGen (1.0d);
			m_config.setKeepPopulationSizeConstant (true);
			m_config.setFitnessEvaluator (new DefaultFitnessEvaluator ());
			m_config.setChromosomePool (new ChromosomePool ());
	
			m_config.setPreservFittestIndividual (true);
	
			m_config.setFitnessFunction (new FitnessFunction ()
			{
				private static final long serialVersionUID = 1L;
	
				@Override
				protected double evaluate (IChromosome chromosome)
				{
					// reward numbers at the correct position, reward numbers occurring in the result,
					// and reward shorter path lengths
					
					double fResult = 0;
					int nIdx = 0;
					for (Gene g : chromosome.getGenes ())
					{
						int nValue = (Integer) g.getAllele ();
						if (opResult.getElements ()[nIdx] == nValue)
							fResult += 10;
						else if (opResult.contains (nValue))
							fResult += 1;
						nIdx++;
					}
					
					int nOpEvals = ((Operand) chromosome.getApplicationData ()).getOperatorEvaluationsCount ();
					if (nOpEvals == 0)
						return fResult;
					return fResult + 20.0 / nOpEvals;
				}
			});
	
			// create the chromosome
			Gene[] rgSampleGenes = new Gene[nElementsCount];
			for (int i = 0; i < nElementsCount; i++)
				rgSampleGenes[i] = new IntegerGene (m_config, 0, 2 * nElementsCount - 1);
			Chromosome chromSample = new Chromosome (m_config, rgSampleGenes);
			chromSample.setApplicationData (new Operand (null, 0));
			m_config.setSampleChromosome (chromSample);
	
			m_config.setPopulationSize (POPULATION_SIZE);
	
			m_config.addGeneticOperator (new CrossoverOperator (m_config)
			{
				private static final long serialVersionUID = 1L;
	
				@SuppressWarnings({ "rawtypes", "unchecked" })
				@Override
				protected void doCrossover (IChromosome firstMate, IChromosome secondMate, List a_candidateChromosomes, RandomGenerator generator)
				{
					try
					{
						for (Operator op : rgOperators)
							a_candidateChromosomes.add (op.apply (firstMate, secondMate, generator, m_config));
					}
					catch (InvalidConfigurationException e)
					{
						e.printStackTrace();
					}
				}
			});
			m_config.addGeneticOperator (new MutationOperator (m_config, 0));
		}
		catch (InvalidConfigurationException e)
		{
			e.printStackTrace ();
		}
	}

	public Operand findSequence ()
	{
		for ( ; ; )
		{
			Operand opResult = findOneSequence ();
			if (opResult.equals (m_opResult))
			{
				//System.out.println (StringUtil.concat ("[", opResult.getOperatorEvaluationsCount (), "]: ", opResult.getExpression ().toString ()));
				//if (opResult.getOperatorEvaluationsCount () <= 3)
					return opResult;
			}
		}
	}
	
	private Operand findOneSequence ()
	{
		try
		{
			// create population and evolve it
			IChromosome[] rgChromosomes = new IChromosome[m_listInputOperands.size ()];
			int nIdx = 0;
			for (Operand op : m_listInputOperands)
			{
				Gene[] rgGenes = new Gene[m_nElementsCount];
				for (int i = 0; i < m_nElementsCount; i++)
				{
					rgGenes[i] = new IntegerGene (m_config, 0, 2 * m_nElementsCount - 1);
					rgGenes[i].setAllele (op.getElements ()[i]);
				}
				
				rgChromosomes[nIdx] = new Chromosome (m_config, rgGenes);
				rgChromosomes[nIdx].setApplicationData (op);
				nIdx++;
			}
			
			// evolve the population
			Population population = new Population (m_config, rgChromosomes);
			Genotype genotype = new Genotype (m_config, population);
			for (int i = 0; i < MAX_EVOLUTIONS; i++)
			{
				genotype.evolve ();
				//LOGGER.info (genotype.getFittestChromosome ());
			}
			
			IChromosome chromFittest = genotype.getFittestChromosome ();
			LOGGER.info (chromFittest);
			
			Operand opRes = (Operand) chromFittest.getApplicationData ();
			opRes.m_rgElements = new int[m_nElementsCount];
			for (int i = 0; i < m_nElementsCount; i++)
				opRes.m_rgElements[i] = (Integer) chromFittest.getGene (i).getAllele ();
			
			return opRes;
		}
		catch (InvalidConfigurationException e)
		{
			e.printStackTrace ();
		}

		return null;
	}
}
