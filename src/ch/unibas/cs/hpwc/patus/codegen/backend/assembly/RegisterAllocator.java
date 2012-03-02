package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.DepthFirstIterator;
import cetus.hir.Expression;
import cetus.hir.FunctionCall;
import cetus.hir.Literal;
import cetus.hir.NameID;
import cetus.hir.Traversable;
import cetus.hir.UnaryExpression;
import ch.unibas.cs.hpwc.patus.arch.TypeArchitectureType.Intrinsics.Intrinsic;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;
import ch.unibas.cs.hpwc.patus.codegen.Globals;
import ch.unibas.cs.hpwc.patus.representation.StencilNode;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 * Allocates registers for the stencil computations.
 * 
 * TODO: use live analysis and graph coloring of the interference graph
 * (interference graph: nodes are variables (=> registers), nodes are connected by
 * an edge if the corresponding variables are live at the same time)
 * 
 * @author Matthias-M. Christen
 */
public class RegisterAllocator
{
	///////////////////////////////////////////////////////////////////
	// Member Variables
	
	private CodeGeneratorSharedObjects m_data;
	
	private StencilAssemblySection m_assemblySection;

	private Map<Expression, Integer> m_mapRegisterUsage;
	
	private Map<StencilNode, Boolean> m_mapReuseStencilNodes;
	

	///////////////////////////////////////////////////////////////////
	// Implementation
	
	public RegisterAllocator (CodeGeneratorSharedObjects data, StencilAssemblySection as, Map<StencilNode, Boolean> mapReuseStencilNodes)
	{
		m_data = data;
		m_assemblySection = as;
		m_mapReuseStencilNodes = mapReuseStencilNodes;

		m_mapRegisterUsage = new HashMap<Expression, Integer> ();
	}
	
	/**
	 * Computes the number of registers needed to compute the expression <code>expr</code>
	 * using the Sethi-Ullman algorithm.
	 * @param expr
	 * @return
	 */
	public int countRegistersNeeded (Expression expr)
	{
		if (RegisterAllocator.isAddSubSubtree (expr))
		{
			m_mapRegisterUsage.put (expr, 1);
			return 1;
		}
		
		if (expr instanceof BinaryExpression)
		{
			int nLHSRegsCount = countRegistersNeeded (((BinaryExpression) expr).getLHS ());
			int nRHSRegsCount = countRegistersNeeded (((BinaryExpression) expr).getRHS ());
			
			int nRegsUsed = nLHSRegsCount == nRHSRegsCount ? nLHSRegsCount + 1 : Math.max (nLHSRegsCount, nRHSRegsCount);
			m_mapRegisterUsage.put (expr, nRegsUsed);
			return nRegsUsed;
		}
		
		if (expr instanceof UnaryExpression)
		{
			int nRegsUsed = countRegistersNeeded (((UnaryExpression) expr).getExpression ());
			m_mapRegisterUsage.put (expr, nRegsUsed);
			return nRegsUsed;
		}
		
		if (expr instanceof FunctionCall)
		{
			Expression exprFuncName = ((FunctionCall) expr).getName ();
			
			// fused multiply-add / multiply-subtract
			if (exprFuncName.equals (Globals.FNX_FMA) || exprFuncName.equals (Globals.FNX_FMS))
			{
				int nArgsCount = ((FunctionCall) expr).getArguments ().size ();
				if (nArgsCount != 3)
					throw new RuntimeException ("FMA/FMS expect 3 parameters");
				
				int[] rgRegsCount = new int[nArgsCount];
				int i = 0;
				for (Object objArg : ((FunctionCall) expr).getArguments ())
				{
					if (objArg instanceof Expression)
					{
						rgRegsCount[i] = countRegistersNeeded ((Expression) objArg);
						i++;
					}
				}
				
				// check whether all reg counts are equal, and if yes, set the number of the used registers
				// to the count+1, or to the max of the reg counts if they are not equal
				boolean bAllEqual = true;
				for (int j = 1; j < nArgsCount; j++)
					if (rgRegsCount[j - 1] != rgRegsCount[j])
					{
						bAllEqual = false;
						break;
					}
				
				int nRegsUsed = rgRegsCount[0];				
				if (bAllEqual)
					nRegsUsed++;
				else
				{
					for (int j = 1; j < nArgsCount; j++)
						nRegsUsed = Math.max (nRegsUsed, rgRegsCount[j]);
				}
				
				m_mapRegisterUsage.put (expr, nRegsUsed);
				return nRegsUsed;				
			}
			else
				throw new RuntimeException (StringUtil.concat ("The function '", exprFuncName.toString (), "' is currently not supported."));
		}
		
		if (expr instanceof StencilNode)
		{
			int nRegsUsed = -1;
			if (canBeMemoryOperand (expr))
				nRegsUsed = 0;
			else if (m_mapReuseStencilNodes == null)
				nRegsUsed = 1;
			else
				nRegsUsed = m_mapReuseStencilNodes.containsKey (expr) ? 0 : 1;

			m_mapRegisterUsage.put (expr, nRegsUsed);
			return nRegsUsed;
		}
		
		if (expr instanceof Literal)
		{
			// assume we have the constant already in a register
			m_mapRegisterUsage.put (expr, 0);
			return 0;
		}
		
		return 0;
	}
	
	/**
	 * Returns the number of registers used to compute the expression <code>expr</code>.
	 * @param expr The expression for which to retrieve the number of registers used for its computation
	 * @return The number of registers used to compute <code>expr</code>
	 */
	public int getNumRegistersUsed (Expression expr)
	{
		Integer nNumRegs = m_mapRegisterUsage.get (expr);
		if (nNumRegs != null)
			return nNumRegs;
		return Integer.MAX_VALUE;
	}
	
	/**
	 * Determines whether the expression <code>expr</code> can be a memory operand of the instruction
	 * corresponding to the operation the expression occurs in.
	 * @param expr
	 * @return
	 */
	private boolean canBeMemoryOperand (Expression expr)
	{
		Traversable trvParent = expr.getParent ();
		Argument arg = null;
		
		if (trvParent instanceof UnaryExpression)
		{
			Intrinsic intrinsic = m_data.getArchitectureDescription ().getIntrinsic (((UnaryExpression) trvParent).getOperator (), m_assemblySection.getDatatype ());
			Argument[] rgArgs = Arguments.parseArguments (intrinsic.getArguments ());
			
			arg = Arguments.getFirstInput (rgArgs);
		}
		else if (trvParent instanceof BinaryExpression)
		{
			Intrinsic intrinsic = m_data.getArchitectureDescription ().getIntrinsic (((BinaryExpression) trvParent).getOperator (), m_assemblySection.getDatatype ());
			Argument[] rgArgs = Arguments.parseArguments (intrinsic.getArguments ());
			
			if (expr == ((BinaryExpression) trvParent).getLHS ())
				arg = Arguments.getLHS (rgArgs);
			else if (expr == ((BinaryExpression) trvParent).getRHS ())
				arg = Arguments.getRHS (rgArgs);
			else
				throw new RuntimeException ("Child expression of BinaryExpression is neither its LHS nor its RHS.");
		}
		else if (trvParent instanceof FunctionCall)
		{
			Expression exprFuncName = ((FunctionCall) expr).getName ();
			
			// fused multiply-add / multiply-subtract
			if (exprFuncName.equals (Globals.FNX_FMA) || exprFuncName.equals (Globals.FNX_FMS))
			{
				Intrinsic intrinsic = m_data.getArchitectureDescription ().getIntrinsic ((FunctionCall) trvParent, m_assemblySection.getDatatype ());
				Argument[] rgArgs = Arguments.parseArguments (intrinsic.getArguments ());
				
				for (int i = 0; i < ((FunctionCall) trvParent).getNumArguments (); i++)
					if (expr == ((FunctionCall) trvParent).getArgument (i))
					{
						arg = Arguments.getNamedArgument (rgArgs, Globals.PARAMSTRINGS_FMA[i]);
						break;
					}
				if (arg == null)
					throw new RuntimeException ("Child expression of FunctionCall is none of its arguments.");
			}
			else
				throw new RuntimeException (StringUtil.concat ("The function '", exprFuncName.toString (), "' is currently not supported."));
		}
		
		if (arg == null)
			return false;
		
		return arg.isMemory ();
	}

	/**
	 * Detects whether only additions and subtractions are done in the expression <code>expr</code>.
	 * @param expr The expression to check
	 * @return <code>true</code> iff only additions and subtractions are performed in <code>expr</code>
	 */
	public static boolean isAddSubSubtree (Expression expr)
	{
		for (DepthFirstIterator it = new DepthFirstIterator (expr); it.hasNext (); )
		{
			Object oChild = it.next ();
			
			if (oChild instanceof BinaryExpression)
			{
				BinaryOperator op = ((BinaryExpression) oChild).getOperator ();
				if (!op.equals (BinaryOperator.ADD) && !op.equals (BinaryOperator.SUBTRACT))
					return false;
			}
			else if (oChild instanceof FunctionCall)
				return false;
		}
		
		return true;
	}
	
	/**
	 * Assuming that <code>expr</code> is a binary expression tree with only additions and
	 * subtractions, this method creates a flat list of (operator, expression) pairs
	 * equivalent to <code>expr</code>. 
	 * @param expr The expression to linearize
	 * @param list An empty list that will contain the linearized version on output
	 * @param op Set to {@link BinaryOperator#ADD} when calling the method
	 */
	public static void linearizeAddSubSubtree (Expression expr, List<AddSub> list, BinaryOperator op)
	{
		if (expr instanceof BinaryExpression)
		{
			BinaryExpression bexpr = (BinaryExpression) expr;
			linearizeAddSubSubtree (bexpr.getLHS (), list, op);
			linearizeAddSubSubtree (bexpr.getRHS (), list,
				op.equals (BinaryOperator.SUBTRACT) ?
					(bexpr.getOperator ().equals (BinaryOperator.ADD) ? BinaryOperator.SUBTRACT : BinaryOperator.ADD) :
					bexpr.getOperator ()
			);
		}
		else
			list.add (new AddSub (op, expr));
	}

	
	public static void main (String[] args)
	{
		List<AddSub> list = new LinkedList<AddSub> ();
		RegisterAllocator.linearizeAddSubSubtree (
			new BinaryExpression (
				new BinaryExpression (
					new BinaryExpression (new NameID ("a1"), BinaryOperator.ADD, new NameID ("a2")),
					BinaryOperator.SUBTRACT,
					new BinaryExpression (new NameID ("a3"), BinaryOperator.SUBTRACT, new NameID ("a4"))),
				BinaryOperator.SUBTRACT,
				new BinaryExpression (new NameID ("a5"), BinaryOperator.ADD, new NameID ("a6"))
			),
			list, BinaryOperator.ADD);
		
		System.out.println (list);
	}
}
