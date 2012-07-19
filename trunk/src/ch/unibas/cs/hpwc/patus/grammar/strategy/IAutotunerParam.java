package ch.unibas.cs.hpwc.patus.grammar.strategy;

import java.util.ArrayList;
import java.util.List;

import cetus.hir.Expression;
import ch.unibas.cs.hpwc.patus.codegen.Globals;

public interface IAutotunerParam
{
	public class AutotunerListParam implements IAutotunerParam
	{
		private List<Expression> m_listValues;
		
		public AutotunerListParam (Expression... values)
		{
			m_listValues = new ArrayList<> (values.length + 10);
			for (Expression v : values)
				m_listValues.add (v);
		}
		
		public void addValue (Expression exprValue)
		{
			m_listValues.add (exprValue);
		}
		
		@Override
		public String toString ()
		{
			StringBuilder sb = new StringBuilder ();
			
			boolean bFirst = true;
			for (Expression v : m_listValues)
			{
				if (!bFirst)
					sb.append (',');
				sb.append (v.toString ());
				bFirst = false;
			}

			return sb.toString ();
		}
	}
	
	public class AutotunerRangeParam implements IAutotunerParam
	{
		private Expression m_exprStart;
		private Expression m_exprStep;
		private Expression m_exprEnd;
		private boolean m_bIsMultiplicative;
		
		public AutotunerRangeParam (Expression exprStart, Expression exprEnd)
		{
			this (exprStart, Globals.ONE.clone (), exprEnd);
		}
		
		public AutotunerRangeParam (Expression exprStart, Expression exprStep, Expression exprEnd)
		{
			this (exprStart, exprStep, false, exprEnd);
		}
		
		public AutotunerRangeParam (Expression exprStart, Expression exprStep, boolean bIsMultiplicative, Expression exprEnd)
		{
			m_exprStart = exprStart;
			m_exprStep = exprStep;
			m_exprEnd = exprEnd;
			m_bIsMultiplicative = bIsMultiplicative;
		}

		@Override
		public String toString ()
		{
			StringBuilder sb = new StringBuilder ();
			
			sb.append (m_exprStart.toString ());
			sb.append (':');
			if (m_bIsMultiplicative)
				sb.append ('*');
			sb.append (m_exprStep.toString ());
			sb.append (':');
			sb.append (m_exprEnd.toString ());
			
			return sb.toString ();
		}
	}
}
