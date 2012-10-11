package ch.unibas.cs.hpwc.patus.grammar.stencil2;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import cetus.hir.BinaryExpression;
import cetus.hir.DepthFirstIterator;
import cetus.hir.Expression;
import cetus.hir.IDExpression;
import cetus.hir.IntegerLiteral;
import ch.unibas.cs.hpwc.patus.representation.StencilNode;
import ch.unibas.cs.hpwc.patus.symbolic.Symbolic;
import ch.unibas.cs.hpwc.patus.util.AnalyzeTools;
import ch.unibas.cs.hpwc.patus.util.DomainPointEnumerator;
import ch.unibas.cs.hpwc.patus.util.ExpressionUtil;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

public class LocalVars
{
	///////////////////////////////////////////////////////////////////
	// Inner Types
	
	public static class ValueAssignment
	{
		private Map<String, Integer> m_mapAssignment;
		
		public ValueAssignment ()
		{
			m_mapAssignment = new HashMap<> ();
		}
		
		public ValueAssignment (int[] rgValues, Collection<String> collVarNames)
		{
			this ();
			
			int j = 0;
			for (String strLoc : collVarNames)
				set (strLoc, rgValues[j++]);
		}
		
		public Integer get (String strVariableName)
		{
			return m_mapAssignment.get (strVariableName);
		}
		
		public void set (String strVariableName, int nValue)
		{
			m_mapAssignment.put (strVariableName, nValue);
		}
		
		@Override
		public String toString ()
		{
			return m_mapAssignment.toString ();
		}
	}
	
	protected class ExpandedExpressionIterator implements Iterator<Expression>
	{
		private Expression m_exprToExpand;
		private Collection<String> m_collLocalVars;
		private Iterator<int[]> m_itDPE;
		
		private Expression m_exprNext;
		
		
		public ExpandedExpressionIterator (Expression exprToExpand)
		{
			m_exprToExpand = exprToExpand;
			
			// build the index space
			DomainPointEnumerator dpe = new DomainPointEnumerator ();
			for (String strLoc : m_collLocalVars = collectLocalVariables (exprToExpand))
			{
				Range range = get (strLoc);
				if (range != null)
					dpe.addDimension (new DomainPointEnumerator.MinMax (range.getStart (), range.getEnd ()));
			}
			
			m_itDPE = dpe.iterator ();
			computeNext ();
		}

		@Override
		public boolean hasNext ()
		{
			return m_exprNext != null;
		}

		@Override
		public Expression next ()
		{
			Expression expr = m_exprNext;
			computeNext ();
			return expr;
		}
		
		private void computeNext ()
		{
			m_exprNext = null;
			
			if (!m_itDPE.hasNext ())
				return;
			
			for ( ; m_itDPE.hasNext (); )
			{
				int[] rgValues = m_itDPE.next ();
				ValueAssignment va = new ValueAssignment (rgValues, m_collLocalVars);
			
				if (evaluatePredicates (va))
				{
					m_exprNext = substituteLocalVariables (m_exprToExpand, va);
					break;
				}
			}
		}

		@Override
		public void remove ()
		{
			// Can't remove elements
			throw new RuntimeException ("Elements cannot be removed from this iterator");			
		}		
	}
	

	///////////////////////////////////////////////////////////////////
	// Member Variables

	private Map<String, Range> m_mapVariables;
	
	private List<Expression> m_listPredicates;
	
	
	///////////////////////////////////////////////////////////////////
	// Implementation

	public LocalVars ()
	{
		m_mapVariables = new HashMap<> ();
		m_listPredicates = new ArrayList<> ();
	}
	
	/**
	 * Adds a new local variable.
	 * 
	 * @param strVariableName
	 *            The name of the variable to add
	 * @param range
	 *            The value range of the variable
	 */
	public void addVariable (String strVariableName, Range range)
	{
		m_mapVariables.put (strVariableName, range);
	}
	
	public void addPredicate (Expression exprPredicate)
	{
		m_listPredicates.add (exprPredicate);
	}
	
	public boolean hasVariable (String strVariableName)
	{
		return m_mapVariables.containsKey (strVariableName);
	}
	
	public Range get (String strVariableName)
	{
		return m_mapVariables.get (strVariableName);
	}
	
	public boolean containsLocalVariable (Expression expr)
	{
		for (DepthFirstIterator it = new DepthFirstIterator (expr); it.hasNext (); )
		{
			Object o = it.next ();
			if (o instanceof StencilNode)
			{
				for (Expression e : ((StencilNode) o).getIndex ().getSpaceIndexEx ())
					if (containsLocalVariable (e))
						return true;
			}
			else if (o instanceof IDExpression)
			{
				if (hasVariable (((IDExpression) o).getName ()))
					return true;
			}
		}
		
		return false;
	}
	
	public Collection<String> collectLocalVariables (Expression expr)
	{
		Set<String> set = new HashSet<> ();
		collectLocalVariablesRecursive (expr, set);
		return set;
	}
	
	private void collectLocalVariablesRecursive (Expression expr, Set<String> set)
	{
		for (DepthFirstIterator it = new DepthFirstIterator (expr); it.hasNext (); )
		{
			Object o = it.next ();
			if (o instanceof StencilNode)
			{
				for (Expression e : ((StencilNode) o).getIndex ().getSpaceIndexEx ())
					collectLocalVariablesRecursive (e, set);						
			}
			else if (o instanceof IDExpression)
			{
				if (hasVariable (((IDExpression) o).getName ()))
					set.add (((IDExpression) o).getName ());
			}
		}
	}
	
	public Iterable<Expression> expand (final Expression expr)
	{
		return new Iterable<Expression>()
		{			
			@Override
			public Iterator<Expression> iterator ()
			{
				return new ExpandedExpressionIterator (expr);
			}
		};
	}
	
	public boolean evaluatePredicates (ValueAssignment values)
	{
		for (Expression exprPred : m_listPredicates)
		{
			Expression e = substituteLocalVariables (exprPred, values);
			
			if (e instanceof BinaryExpression && AnalyzeTools.isComparisonOperator (((BinaryExpression) e).getOperator ()))
			{
				BinaryExpression bexpr = (BinaryExpression) e;
				Expression exprLHS = bexpr.getLHS ();
				Expression exprRHS = bexpr.getRHS ();
				
				if (!(exprLHS instanceof IntegerLiteral))
					throw new RuntimeException (StringUtil.concat ("LHS ", exprLHS.toString (), " is no integer literal."));
				if (!(exprRHS instanceof IntegerLiteral))
					throw new RuntimeException (StringUtil.concat ("RHS ", exprRHS.toString (), " is no integer literal."));
				
				if (!ExpressionUtil.compare ((IntegerLiteral) exprLHS, bexpr.getOperator (), (IntegerLiteral) exprRHS))
					return false;				
			}
			else
				throw new RuntimeException (StringUtil.concat (e.toString (), " is no comparison expression."));
		}
		
		return true;
	}

	private Expression substituteLocalVariables (Expression expr, ValueAssignment values)
	{
		boolean bContainsStencilNodes = false;
		Expression exprResult = expr.clone ();
		
		for (DepthFirstIterator it = new DepthFirstIterator (exprResult); it.hasNext (); )
		{
			Object o = it.next ();
			
			if (o instanceof StencilNode)
			{
				Expression[] rgExprCoords = ((StencilNode) o).getIndex ().getSpaceIndexEx ();
				for (int i = 0; i < rgExprCoords.length; i++)
					rgExprCoords[i] = substituteLocalVariables (rgExprCoords[i], values);
				((StencilNode) o).getIndex ().setSpaceIndex (rgExprCoords);
				bContainsStencilNodes = true;
				
				if (o == exprResult)
					return (StencilNode) o;
			}
			else if (o instanceof IDExpression)
			{
				Integer nValue = values.get (((IDExpression) o).getName ());
				if (nValue != null)
				{
					if (o == exprResult)
						return new IntegerLiteral (nValue);
					else
						((Expression) o).swapWith (new IntegerLiteral (nValue));
				}
			}
		}
		
		if (!bContainsStencilNodes)
			return Symbolic.simplify (exprResult);
		return exprResult;
	}
}
