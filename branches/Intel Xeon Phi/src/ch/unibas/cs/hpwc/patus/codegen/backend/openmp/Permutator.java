package ch.unibas.cs.hpwc.patus.codegen.backend.openmp;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class Permutator
{
	private final static int MAX_ITER = 100;
	
	
	public static class Operator
	{
		private int m_nOperandsCount;
		private int m_nElementPerOperand;
		
		private Selector[] m_rgSelectors;
		private Map<Integer, List<Selector>> m_mapSelectors;
		
		
		public Operator (int nOperandsCount, int nElementsPerOperand, Selector[] rgSelectors)
		{
			m_nOperandsCount = nOperandsCount;
			m_nElementPerOperand = nElementsPerOperand;
			
			if (rgSelectors.length != nElementsPerOperand)
				throw new RuntimeException ("The number of selectors must correspond to the number of elements per operand");
			
			// build the selector map
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
					m_mapSelectors.put (s.getID (), list = new ArrayList<> ());
				list.add (s);
				
				nIndex++;
			}
			
			m_rgSelectors = rgSelectors;
		}
		
		public int getOperandsCount ()
		{
			return m_nOperandsCount;
		}
		
		public List<Operand> apply (List<Operand> listOperands, Operand opDesiredResult)
		{
			List<Operand> listResults = new ArrayList<> ();
			
			// find selector configurations such that each desired element is contained at least once
			boolean[] rgSelectorApplied = new boolean[m_nElementPerOperand];
			for (int i = 0; i < m_nElementPerOperand; i++)
			{
				Arrays.fill (rgSelectorApplied, false);
				Operand opResult = new Operand (m_nElementPerOperand);		

				for (int j = 0; j < m_nElementPerOperand; j++)
				{
					// if a selector has already been applied to this element, continue
					if (rgSelectorApplied[j])
						continue;
					
					// get the list of selectors to apply together with the one to apply at the current index
					List<Selector> listSelectors = m_mapSelectors.get (m_rgSelectors[j].getID ());
					if (listSelectors == null)
						continue;
					
					// find the best argument and apply the selectors
					int nArgNum = -1;
					int nDesiredValue = opDesiredResult.getElements ()[i];
					for (Selector selChooseArg : listSelectors)
					{
						int nArgNumNew = selChooseArg.findBestArgNum (j, listOperands, nDesiredValue, opDesiredResult, opResult);
						if (nArgNumNew == -1)
							continue;
						nArgNum = nArgNumNew;
						if (selChooseArg.getResult (listOperands, nArgNum) == nDesiredValue)
							break;
					}
					
					// use default argument if none could be found
					if (nArgNum == -1)
						nArgNum = 0;
					
					for (Selector selApply : listSelectors)
					{
						selApply.apply (opResult, selApply.getIndex (), listOperands, nArgNum);
						rgSelectorApplied[selApply.getIndex ()] = true;
					}
				}
				
				listResults.add (opResult);
			}
			
			return listResults;
		}
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
		
		/* package */ void setIndex (int nIndex)
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
		
		public int getResult (List<Operand> listOperands, int nArgNum)
		{
			int nIdx = m_rgSelectSet[nArgNum];
			int nEltsCount = listOperands.get (0).getElementsCount ();
			return listOperands.get (nIdx / nEltsCount).getElements ()[nIdx % nEltsCount];
		}
		
		public int findBestArgNum (int nArgIdx, List<Operand> listOperands, int nDesiredValue, Operand opDesiredResult, Operand opCurrentResult)
		{
			int nDesired = opDesiredResult.getElements ()[nArgIdx];
			
			// get a list of all possible results for the nArgIdx-th argument
			int rgPossible[] = new int[m_rgSelectSet.length];
			for (int i = 0; i < m_rgSelectSet.length; i++)
				rgPossible[i] = getResult (listOperands, i);
			
			// if the current result doesn't contain the desired value yet, try to set the nArgIdx-th argument to the desired value
			if (!opCurrentResult.contains (nDesiredValue))
				for (int i = 0; i < m_rgSelectSet.length; i++)
					if (rgPossible[i] == nDesiredValue)
						return i;

			// if the current result doesn't contain the desired value yet, try to set the nArgIdx-th argument to the desired value
			if (!opCurrentResult.contains (nDesired))
				for (int i = 0; i < m_rgSelectSet.length; i++)
					if (rgPossible[i] == nDesired)
						return i;
			
			for (int j = 0; j < opDesiredResult.getElementsCount (); j++)
			{
				int n = opDesiredResult.getElements ()[j];
				if (n != nDesiredValue && n != nDesired && !opCurrentResult.contains (n))
					for (int i = 0; i < m_rgSelectSet.length; i++)
						if (rgPossible[i] == n)
							return i;
			}
			
			return -1;
		}

		public void apply (Operand opResult, int nArgIdx, List<Operand> listOperands, int nArgNum)
		{
			opResult.getElements ()[nArgIdx] = getResult (listOperands, nArgNum);
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
		private int[] m_rgElements;
		private List<Operand> m_listParentOperands;
		private Operator m_operator;
		
		public Operand (int nElementsCount)
		{
			this (new int[nElementsCount], null, null);
			Arrays.fill (m_rgElements, -1);
		}
		
		public Operand (int[] rgElements)
		{
			this (rgElements, null, null);
		}
		
		public Operand (int[] rgElements, List<Operand> listParentOperands, Operator op)
		{
			m_rgElements = rgElements;
			m_listParentOperands = listParentOperands;
			m_operator = op;
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
		
		public boolean contains (int n)
		{
			for (int i = 0; i < m_rgElements.length; i++)
				if (m_rgElements[i] == n)
					return true;
			return false;
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
			return Arrays.toString (m_rgElements);
		}
	}
	
	
	private Set<Operand> m_setAllOperands;

	
	public Permutator ()
	{		
		m_setAllOperands = new HashSet<> ();
	}
	
	public Operand findSequence (List<Operand> listInputOperands, Operand opResult, Operator[] rgOperators)
	{
		m_setAllOperands.clear ();
		m_setAllOperands.addAll (listInputOperands);
		
		List<Operand> listInput = new ArrayList<> ();
		listInput.addAll (listInputOperands);
		List<List<Operand>> listInputLocal = new ArrayList<> ();
		
		for (int i = 0; i < MAX_ITER; i++)
		{
			for (Operator op : rgOperators)
			{
				// build the input arguments
				listInputLocal.clear ();
				if (op.getOperandsCount () == 1)
				{
					for (Operand o : listInput)
					{
						List<Operand> l = new ArrayList<> (1);
						l.add (o);
						listInputLocal.add (l);
					}
				}
				else if (op.getOperandsCount () == 2)
				{
					for (int j = 0; j < listInput.size (); j++)
						for (int k = 0; k <= j; k++)
						{
							List<Operand> l = new ArrayList<> (2);
							l.add (listInput.get (j));
							l.add (listInput.get (k));
							listInputLocal.add (l);
						}
				}
				else
					throw new RuntimeException ("Only operators with 1 or 2 arguments supported");
				
				// apply the operator
				for (List<Operand> l : listInputLocal)
				{
					List<Operand> listResult = op.apply (l, opResult);
					for (Operand opRes : listResult)
					{
						m_setAllOperands.add (opRes);
						if (opRes.equals (opResult))
							return opRes;
					}
				}
			}
			
			listInput.clear ();
			listInput.addAll (m_setAllOperands);
		}
		
		return null;
	}
}
