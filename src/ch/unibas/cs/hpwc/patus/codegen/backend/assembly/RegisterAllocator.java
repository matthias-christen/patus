package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import java.util.HashMap;
import java.util.Map;

import cetus.hir.BinaryExpression;
import cetus.hir.Expression;
import cetus.hir.FunctionCall;
import cetus.hir.Literal;
import cetus.hir.UnaryExpression;
import ch.unibas.cs.hpwc.patus.codegen.Globals;
import ch.unibas.cs.hpwc.patus.representation.StencilNode;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

public class RegisterAllocator
{
	///////////////////////////////////////////////////////////////////
	// Member Variables

	private Map<Expression, Integer> m_mapRegisterUsage;
	
	private Map<StencilNode, Boolean> m_mapReuseStencilNodes;
	

	///////////////////////////////////////////////////////////////////
	// Implementation
	
	public RegisterAllocator (Map<StencilNode, Boolean> mapReuseStencilNodes)
	{
		m_mapRegisterUsage = new HashMap<Expression, Integer> ();
		m_mapReuseStencilNodes = mapReuseStencilNodes;
	}
	
	/**
	 * Computes the number of registers needed to compute the expression <code>expr</code>
	 * using the Sethi-Ullman algorithm.
	 * @param expr
	 * @return
	 */
	public int countRegistersNeeded (Expression expr)
	{
		if (expr instanceof BinaryExpression)
		{
			int nLHSRegsCount = countRegistersNeeded (((BinaryExpression) expr).getLHS ());
			int nRHSRegsCount = countRegistersNeeded (((BinaryExpression) expr).getRHS ());
			
			int nRegsUsed = nLHSRegsCount == nRHSRegsCount ? nLHSRegsCount + 1 : Math.max (nLHSRegsCount, nRHSRegsCount);
			m_mapRegisterUsage.put (expr, nRegsUsed);
			return nRegsUsed;
		}
		else if (expr instanceof UnaryExpression)
		{
			int nRegsUsed = countRegistersNeeded (((UnaryExpression) expr).getExpression ());
			m_mapRegisterUsage.put (expr, nRegsUsed);
			return nRegsUsed;
		}
		else if (expr instanceof FunctionCall)
		{
			Expression exprFuncName = ((FunctionCall) expr).getName ();
			if (exprFuncName.equals (Globals.FNX_FMA) || exprFuncName.equals (Globals.FNX_FMS))
			{
				
			}
			else
				throw new RuntimeException (StringUtil.concat ("The function '", exprFuncName.toString (), "' is currently not supported."));
		}
		else if (expr instanceof StencilNode)
		{
			int nRegsUsed = m_mapReuseStencilNodes == null ? 1 : (m_mapReuseStencilNodes.containsKey (expr) ? 0 : 1);
			m_mapRegisterUsage.put (expr, nRegsUsed);
			return nRegsUsed;
		}
		else if (expr instanceof Literal)
		{
			m_mapRegisterUsage.put (expr, 0);
			return 0;
		}
		
		return 0;
	}
}
