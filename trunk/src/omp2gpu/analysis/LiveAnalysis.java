package omp2gpu.analysis; 

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.TreeMap;

import cetus.analysis.CFGraph;
import cetus.analysis.DFANode;
import cetus.analysis.RangeDomain;
import cetus.analysis.Section;
import cetus.hir.DataFlowTools;
import cetus.hir.Expression;
import cetus.hir.PrintTools;
import cetus.hir.Symbol;
import cetus.hir.SymbolTools;
import cetus.hir.Tools;
import cetus.hir.Traversable;

/**
 * @author Seyong Lee <lee222@purdue.edu>
 *         ParaMount Group 
 *         School of ECE, Purdue University
 *
 * LiveAnalysis performs live-out analysis of the program at ArraySection level
 * (it handles both scalar and array-with-Section, i.e. A[a:b])
 * Live-out is a union of Upward-Exposed-Use set of all successors of each node
 *
 * Input  : CFGraph cfg
 * Output : LIVE set and USE set for each node in cfg
 *   USE(m) is Upward-Exposed-Use set of node m
 *   DEF(m) is DEF set of node m
 *
 * LIVE(exit-node) = {} : only intra-procedural analysis
 *
 * FOREACH successor node m 
 * 	LIVE(n) += { Use(m) + (LIVE(m) - DEF(m)) } = { LocalUSE(m) + LIVE(m) - DEF(m) }
 * END
 *
 * NOTE: the run() method returns a CFGraph, which contains all LiveAnalysis results. 
 */
public class LiveAnalysis
{
	private int debug_tab=0;
	private int debug_level;

	private Traversable input_code;
	private CFGraph cfg;
	private Set<Symbol> shared_vars;

	public LiveAnalysis(Traversable input, CFGraph i_cfg, Set<Symbol> ishvars)
	{
		input_code = input;
		cfg = i_cfg;
		shared_vars = ishvars;
		debug_level = PrintTools.getVerbosity();
	}

	public String getPassName()
	{
		return new String("[LiveAnalysis]");
	}
	
    public void run()
	{
		computeLive();

		display();
	}
    
	private void computeLive()
	{
		int count = 0;

		PrintTools.println("[computeLive] strt *****************************", 3);
		PrintTools.print("              shared variables in the input: ", 3);
		PrintTools.println("{" + PrintTools.collectionToString(shared_vars, ",") + "}", 3);

		Set<Symbol> def_symbols = DataFlowTools.getDefSymbol(input_code);
		List<Symbol> remove_vars = new ArrayList<Symbol>();
		for(Symbol s : def_symbols) {
			String sname = s.getSymbolName();
			// PrintTools.println(sname, 0);
			if( sname.startsWith("sh__") || sname.startsWith("red__") || sname.startsWith("_bid") || 
					sname.startsWith("_gtid") || sname.startsWith("_ti_100") || sname.startsWith("row_temp_")
					|| sname.endsWith("__extended") ) {
				remove_vars.add(s);
			}				
		}
		
		for(Symbol s : remove_vars)
			def_symbols.remove(s);
		
		
		TreeMap work_list = new TreeMap();

		// Enter the exit node in the work_list
		List<DFANode> exit_nodes = cfg.getExitNodes();
		if (exit_nodes.size() > 1)
		{
			PrintTools.println("[WARNING in computeLive()] multiple exits in the program", 2);
		}

		for ( DFANode exit_node : exit_nodes )
			work_list.put((Integer)exit_node.getData("top-order"), exit_node);

		// Do iterative steps
		while ( !work_list.isEmpty() )
		{
			if ( count++ > (cfg.size()*10) ) {
				PrintTools.println(cfg.toDot("tag,ir,ueuse", 3), 0);
				PrintTools.println("cfg size = " + cfg.size(), 0);
				Tools.exit("[computeLive] infinite loop!");
			}

			DFANode node = (DFANode)work_list.remove(work_list.lastKey());

			RangeDomain rd = (RangeDomain)node.getData("range");

			Section.MAP curr_map = new Section.MAP();

			// calculate the current live_out to check if there is any change
			for ( DFANode succ : node.getSuccs() )
			{
				Section.MAP succ_map = (Section.MAP)succ.getData("ueuse");
				curr_map = curr_map.unionWith(succ_map, rd);
			}

			// retrieve previous live_out
			Section.MAP prev_map = (Section.MAP)node.getData("live_out");

			if ( prev_map == null || !prev_map.equals(curr_map) )
			{
				// since live_out has been changed, we update it.
				node.putData("live_out", curr_map);

				// compute Upward-Exposed-Use set = LocalUEUse + (Live - DEF)
				computeUseSet(node, rd, def_symbols);

				for ( DFANode pred : node.getPreds() )
					work_list.put(pred.getData("top-order"), pred);
			}
		}

		PrintTools.println("[computeLive] done *****************************", 3);
	}

	private boolean isKernelBoundaryBarrierNode(DFANode node)
	{
		String tag = (String)node.getData("tag");
		String type = (String)node.getData("type");
		if (tag != null && tag.equals("barrier"))
		{
			if (type != null && (type.equals("S2P") || type.equals("P2S")) ) return true;
		}
		return false;
	}

	// compute Upward-Exposed-Use set 
	// USE = UEUse+(LIVE-DEF) = (LocalUSE-DEF)+(LIVE-DEF) = (LocalUSE+LIVE)-DEF
	private void computeUseSet(DFANode node, RangeDomain rd, Set<Symbol> def_symbols)
	{
		PrintTools.println("[computeUseSet] strt (node: " + node.getData("ir"), 5);

		Section.MAP ueuse = null;

		// live_out should not be null
		Section.MAP live_out = (Section.MAP)node.getData("live_out");

		if ( isKernelBoundaryBarrierNode(node) )
		{
			ueuse = new Section.MAP();
			node.putData("ueuse", ueuse);
		} else {
			// calculate upward-exposed-use set USE = (LocalUSE + LIVE)

			// calculate local USE set and local DEF set
			Section.MAP local_use, local_def;
			Section.MAP dipa_use = node.getData("use");
			if ( dipa_use == null )
				local_use = new Section.MAP();
			else
				local_use = dipa_use;

			Section.MAP dipa_def = node.getData("def");
			if ( dipa_def == null )
				local_def = new Section.MAP();
			else
				local_def = dipa_def;

			Object o = CFGraph.getIR(node);
			if ( o instanceof Traversable )
			{
				for (Expression e : DataFlowTools.getUseSet((Traversable)o) )
				{
					Symbol use_symbol = SymbolTools.getSymbolOf(e);
					PrintTools.println("  locally found used symbol: " + use_symbol, 6);
					if (use_symbol != null && shared_vars.contains(use_symbol) )
					{
						PrintTools.println("  locally found used symbol2: " + use_symbol, 6);
						Section.MAP new_section_map = DataFlowTools.getUseSectionMap(e, rd, def_symbols);
						PrintTools.println("  local use section" + new_section_map, 6);
						local_use = local_use.unionWith(new_section_map, rd);
					}
				}

				for (Expression e : DataFlowTools.getDefSet((Traversable)o) )
				{
					Section.MAP new_section_map = DataFlowTools.getDefSectionMap(e, rd, def_symbols);
					local_def = local_def.unionWith(new_section_map, rd);
				}
			}

			PrintTools.println("  local_use" + local_use, 8);
			PrintTools.println("  local_def" + local_def, 8);

			// upward-exposed USE = USE - DEF
			PrintTools.println("  live_out" + live_out, 6);

			// debugged on May 11
			if (live_out.isEmpty())
			{
				ueuse = local_use;
			}
			else
			{	// the following two statements should not be exchanged.
				ueuse = live_out.differenceFrom(local_def, rd);
				ueuse = local_use.unionWith(ueuse, rd);
			}

			PrintTools.println("  ueuse" + ueuse, 6);

			// insert upward-exposed USE set 
			node.putData("ueuse", ueuse);
		}

		PrintTools.println("  final ueuse" + ueuse, 5);

		PrintTools.println("[computeUseSet] done", 5);
	}

	private boolean isBarrierNode(DFANode node)
	{
		String tag = (String)node.getData("tag");
		if ( tag != null && tag.equals("barrier") )
			return true;
		else
			return false;
	}

		
/*
	// Assumption: there is no out-of-bound accesses in the program
	// convert [-INF:INF] to declared [lb:ub] assuming that 
	private void elaborate()
	{
		for ( int i=0; i<cfg.size(); i++)
		{
			DFANode node = cfg.getNode(i);
			Section.MAP live_out = (Section.MAP)node.getData("live_out");
			for (Symbol symbol : live_out.keySet() )
			{
				if ( isArray(symbol) || isPointer(symbol) )
				{
					Section section : live_out.get(symbol);
					for (Section.ELEMENT element : section)
					{
						int dimension_count = 0;
						for (Expression expr : element)
						{
							if (expr instanceof RangeExpression)
							{
								Expression lb_expr  = ((RangeExpression)expr).getLB();
								Expression ub_expr  = ((RangeExpression)expr).getUB();
								if (lb_expr instanceof InfExpression)
									lb_expr = new IntegerLiteral(0);
								if (ub_expr instanceof InfExpression)
								{
									List<ArraySepcifier> specifiers = symbol.getArraySpecifiers();
									assert specifiers.size() == 1 : "number of specifiers should be 1";
									ArraySpecifier spec = spcifiers.get(0);
									ub_expr = (Expression)spec.getDimension(i).clone();
								}
							}
							dimension_count++;
						}
					}
				}
			}
		}
	}
*/

	public void display()
	{
		if (debug_level < 5) return;

		for ( int i=0; i<cfg.size(); i++)
		{
			DFANode node = cfg.getNode(i);

			if ( isBarrierNode(node) )
			{
				PrintTools.println("\n" + node.toDot("tag,ir", 1), 2);

				Section.MAP live_out = (Section.MAP)node.getData("live_out");
				if (live_out != null) PrintTools.println("    live_out"+live_out, 5);

				Section.MAP ueuse = (Section.MAP)node.getData("ueuse");
				if (ueuse != null) PrintTools.println("    ueuse"+ueuse, 5);
			}
		}

		PrintTools.println(cfg.toDot("tag,ir,ueuse", 3), 2);
	}

}
