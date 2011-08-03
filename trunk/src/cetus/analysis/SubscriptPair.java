package cetus.analysis;
import java.util.*;
import cetus.hir.*;

/**
 * Creates a pair of affine subscripts where subscript is a single dimension of an array reference
 */
public class SubscriptPair {
	
	/* Store normalized expression if affine */
	private Expression subscript1, subscript2;
	/* Statements that contain the subscripts */
	private Statement stmt1, stmt2;
	/* Loops from which indices are present in the subscript pair */
	private LinkedList<Loop> present_loops;
	/* All loops from the enclosing loop nest */
	private LinkedList<Loop> enclosing_loops;
	/* Loop information for the enclosing loop nest */
	private HashMap<Loop, LoopInfo> enclosing_loops_info;

	//public SubscriptPair (Expression s1, Expression s2, LinkedList<Loop> nest, HashMap <Loop,LoopInfo> loopinfo)
	// Modified for dd tests that need to access the IR.
	public SubscriptPair(
		Expression s1, Expression s2, // Two subscripts (possibly orphans)
		Statement st1, Statement st2, // Two statements containing s1,s2 (in IR)
		LinkedList<Loop> nest, HashMap <Loop,LoopInfo> loopinfo)
	{
		/* All symbols present in affine expressions */
		List<Identifier> symbols_in_expressions;
		List symbols_in_s1, symbols_in_s2;

		this.subscript1 = s1;
		this.subscript2 = s2;
		this.stmt1 = st1;
		this.stmt2 = st2;
		this.enclosing_loops = nest;
		this.enclosing_loops_info = loopinfo;

		Set<Symbol> symbols_s1 = DataFlowTools.getUseSymbol((Traversable)s1);
		Set<Symbol> symbols_s2 = DataFlowTools.getUseSymbol((Traversable)s2);
		present_loops = new LinkedList<Loop>();		
		for (Loop loop: nest)
		{
			LoopInfo info = loopinfo.get(loop);
			Expression index = info.getLoopIndex();
			if (symbols_s1.contains(((Identifier)index).getSymbol()) ||
					symbols_s2.contains(((Identifier)index).getSymbol()))
				present_loops.addLast(loop);
		}
	}

	protected HashMap<Loop,LoopInfo> getEnclosingLoopsInfo()
	{
		return enclosing_loops_info;
	}
	
	protected LinkedList<Loop> getEnclosingLoopsList()
	{
		return enclosing_loops;
	}
	
	protected LinkedList<Loop> getPresentLoops()
	{
		return present_loops;
	}
	
	protected Expression getSubscript1()
	{
		return subscript1;
	}
	
	protected Expression getSubscript2()
	{
		return subscript2;
	}

	protected Statement getStatement1()
	{
		return stmt1;
	}

	protected Statement getStatement2()
	{
		return stmt2;
	}
	
	protected int getComplexity()
	{
		return present_loops.size();
	}

	public String toString()
	{
		StringBuilder str = new StringBuilder(80);
		str.append("[SUBSCRIPT-PAIR] "+subscript1+", "+subscript2+"\n");
		for ( Loop loop : enclosing_loops )
			str.append("  enclosed by "+enclosing_loops_info.get(loop)+"\n");
		for ( Loop loop : present_loops )
			str.append("  relevant with "+enclosing_loops_info.get(loop)+"\n");
		return str.toString();
	}
}
