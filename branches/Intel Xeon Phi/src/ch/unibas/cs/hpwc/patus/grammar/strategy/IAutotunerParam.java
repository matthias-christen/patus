package ch.unibas.cs.hpwc.patus.grammar.strategy;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import cetus.hir.DepthFirstIterator;
import cetus.hir.Expression;
import cetus.hir.IDExpression;
import cetus.hir.NameID;
import ch.unibas.cs.hpwc.patus.codegen.Globals;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

public interface IAutotunerParam
{
	public String replaceIdentifiers (Map<String, Integer> mapArgName2ArgIdx);
	
	
	/**
	 * Helper class to replace identifiers in expressions by argument indices.
	 */
	public class AutotunerParamUtil
	{
		public static Expression replaceIdentifiers (Expression exprOrig, Map<String, Integer> mapArgName2ArgIdx)
		{
			if (exprOrig instanceof IDExpression)
				return AutotunerParamUtil.getReplacementIdentifier ((IDExpression) exprOrig, mapArgName2ArgIdx);

			Expression expr = exprOrig.clone ();
			for (DepthFirstIterator it = new DepthFirstIterator (expr); it.hasNext (); )
			{
				Object o = it.next ();
				if (o instanceof IDExpression)
					((IDExpression) o).swapWith (AutotunerParamUtil.getReplacementIdentifier ((IDExpression) o, mapArgName2ArgIdx));
			}
			return expr;
		}
		
		public static NameID getReplacementIdentifier (IDExpression idOrig, Map<String, Integer> mapArgName2ArgIdx)
		{
			return new NameID (StringUtil.concat ("$$", mapArgName2ArgIdx.get ((idOrig).getName ())));
		}
	}

	
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
		public String replaceIdentifiers (Map<String, Integer> mapArgName2ArgIdx)
		{
			List<Expression> listReplaced = new ArrayList<> (m_listValues.size ());
			for (Expression exprOrig : m_listValues)
				listReplaced.add (AutotunerParamUtil.replaceIdentifiers (exprOrig, mapArgName2ArgIdx));

			return AutotunerListParam.list2String (listReplaced);
		}
		
		private static String list2String (List<Expression> list)
		{
			StringBuilder sb = new StringBuilder ();
			
			boolean bFirst = true;
			for (Expression v : list)
			{
				if (!bFirst)
					sb.append (',');
				sb.append (v.toString ());
				bFirst = false;
			}

			return sb.toString ();			
		}
		
		@Override
		public String toString ()
		{
			return AutotunerListParam.list2String (m_listValues);
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
		public String replaceIdentifiers (Map<String, Integer> mapArgName2ArgIdx)
		{
			return toString (
				AutotunerParamUtil.replaceIdentifiers (m_exprStart, mapArgName2ArgIdx),
				AutotunerParamUtil.replaceIdentifiers (m_exprStep, mapArgName2ArgIdx),
				AutotunerParamUtil.replaceIdentifiers (m_exprEnd, mapArgName2ArgIdx)
			);
		}
		
		private String toString (Expression exprStart, Expression exprStep, Expression exprEnd)
		{
			StringBuilder sb = new StringBuilder ();
			
			sb.append (exprStart.toString ());
			sb.append (':');
			if (m_bIsMultiplicative)
				sb.append ('*');
			sb.append (exprStep.toString ());
			sb.append (':');
			sb.append (exprEnd.toString ());
			
			return sb.toString ();			
		}
		
		@Override
		public String toString ()
		{
			return toString (m_exprStart, m_exprStep, m_exprEnd);
		}
	}
}
