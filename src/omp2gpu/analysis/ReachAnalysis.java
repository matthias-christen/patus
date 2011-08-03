package omp2gpu.analysis; 

import java.util.*;
import cetus.hir.*;
import cetus.exec.*;
import cetus.analysis.*;
import omp2gpu.analysis.*;

/**
 * @author Seyong Lee <lee222@purdue.edu>
 *         ParaMount Group 
 *         School of ECE, Purdue University
 *
 * ReachingDef performs may & must reaching-definition analysis of the program
 *
 * Input  : CFGraph cfg
 * Output : Must and May ReachDEF set for each node in cfg
 *
 * ReachDEF(entry-node) = {}	: only intra-procedural analysis
 *
 * for ( node m : predecessor nodes of node n )
 * 	ReachDEF(n) = ^ { DEDef(m) v (ReachDEF(m) ^ ~Killed(m)) }  : Must Reaching Def
 * 	ReachDEF(n) = v { DEDef(m) v (ReachDEF(m) ^ ~Killed(m)) }  : May Reaching Def
 * where,
 *   DEDef(m) is a Downward-Exposed-Def set of node m
 *   ~Killed(m) is a set of variables not defined_vars in node m
 *   
 *  Additionally, this analysis calculates whether variables of interest  are 
 *  modified by CPU or GPU, but to use this information, input procedure (icfg) 
 *  should be called by CPU.
 *
 */
public class ReachAnalysis
{
	private int debug_tab=0;
	private int debug_level;

	private CFGraph cfg;
	private Traversable input_code;
	private Set<Symbol> shared_vars;

	public ReachAnalysis(Traversable icode, CFGraph icfg, Set<Symbol> shvars)
	{
		input_code = icode;
		cfg = icfg;
		shared_vars = shvars;
		debug_level = PrintTools.getVerbosity();
	}

	public String getPassName()
	{
		return new String("[ReachAnalysis]");
	}
	
	public void run()
	{
		ReachingDef();
		
		display();
	}

	private boolean hasChanged(Section.MAP prev, Section.MAP curr)
	{
		if ( prev == null || curr == null || !prev.equals(curr) )
			return true;
		else
			return false;
	}
	
	private boolean hasChangedRM(AnalysisTools.REGIONMAP prev, AnalysisTools.REGIONMAP curr)
	{
		if ( prev == null || curr == null || !prev.equals(curr) )
			return true;
		else
			return false;
	}

	private Set<Symbol> getDefinedVariables()
	{
		Set<Symbol> defined_vars = DataFlowTools.getDefSymbol(input_code);
		List<Symbol> remove_vars = new ArrayList<Symbol>();
		
		for(Symbol s : defined_vars) {
			String sname = s.getSymbolName();
			if( sname.startsWith("sh__") || sname.startsWith("red__") || sname.startsWith("_bid") || 
					sname.startsWith("_gtid") || sname.startsWith("_ti_100") || sname.startsWith("row_temp_")
					|| sname.endsWith("__extended") ) {
				remove_vars.add(s);
			}				
		}

		for(Symbol s : remove_vars)
			defined_vars.remove(s);

		return defined_vars;
	}

	private void ReachingDef()
	{
		PrintTools.println("[ReachingDef] strt *****************************", 3);

		Set<Symbol> defined_vars = getDefinedVariables();
		PrintTools.print("              shared variables in the input: ", 3);
		PrintTools.println("{" + PrintTools.collectionToString(shared_vars, ",") + "}", 3);
		PrintTools.print("              defined variables in the input: ", 3);
		PrintTools.println("{" + PrintTools.collectionToString(defined_vars, ",") + "}", 3);
		TreeMap work_list = new TreeMap();

		// Enter the entry node in the work_list
		DFANode entry = cfg.getNodeWith("stmt", "ENTRY");
		entry.putData("may_def_in", new Section.MAP());
		entry.putData("must_def_in", new Section.MAP());
		entry.putData("may_def_inRM", new AnalysisTools.REGIONMAP());
		entry.putData("must_def_inRM", new AnalysisTools.REGIONMAP());
		work_list.put(entry.getData("top-order"), entry);
		
		////////////////////////////////////////////////////////////////////////////
		// [CAUTION] This analysis assumes that the procedure of interest is      //
		// called by CPU, even though it can contain kernel function calls.       //
		////////////////////////////////////////////////////////////////////////////
		String currentRegion = new String("CPU");

		// Do iterative steps
		while ( !work_list.isEmpty() )
		{
			DFANode node = (DFANode)work_list.remove(work_list.firstKey());
			RangeDomain rd = (RangeDomain)node.getData("range");
			String tag = (String)node.getData("tag");
			// Check whether the node is in the kernel region or not.
			if( tag != null && tag.equals("barrier") ) {
				String type = (String)node.getData("type");
				if( type != null ) {
					if( type.equals("S2P") ) {
						currentRegion = new String("GPU");
					} else if( type.equals("P2S") ) {
						currentRegion = new String("CPU");
					}
				}
			}

			PrintTools.println("\nnode = " + node.getData("ir"), 4);

			Section.MAP may_def_in = null;
			Section.MAP must_def_in = null;
			AnalysisTools.REGIONMAP may_def_inRM = null;
			AnalysisTools.REGIONMAP must_def_inRM = null;

			DFANode temp = (DFANode)node.getData("back-edge-from");
			for ( DFANode pred : node.getPreds() )
			{
				Section.MAP pred_may_def_out = (Section.MAP)pred.getData("may_def_out");
				Section.MAP pred_must_def_out = (Section.MAP)pred.getData("must_def_out");
				AnalysisTools.REGIONMAP pred_may_def_outRM = (AnalysisTools.REGIONMAP)pred.getData("may_def_outRM");
				AnalysisTools.REGIONMAP pred_must_def_outRM = (AnalysisTools.REGIONMAP)pred.getData("must_def_outRM");

				PrintTools.println("  pred must_def_out = " + pred_must_def_out, 5);

				if ( may_def_in == null ) {
					may_def_in = (Section.MAP)pred_may_def_out.clone();
				} else {
					may_def_in = may_def_in.unionWith(pred_may_def_out, rd);
				}

				if ( must_def_in == null ) {
					must_def_in = (Section.MAP)pred_must_def_out.clone();
				} else {
					if (temp != null && temp == pred)
					{
						PrintTools.println("  back-edge: rd = " + rd, 5);
						// this data is from a back-edge, union it with the current data
						must_def_in = must_def_in.unionWith(pred_must_def_out, rd);
					}
					else
					{
						PrintTools.println("  branch: rd = " + rd, 5);
						// this is an if-else branch, thus intersect it with the current data
						must_def_in = must_def_in.intersectWith(pred_must_def_out, rd);
					}
				}
				
				if ( may_def_inRM == null ) {
					may_def_inRM = (AnalysisTools.REGIONMAP)pred_may_def_outRM.clone();
				} else {
					may_def_inRM = may_def_inRM.unionWith(pred_may_def_outRM);
				}

				if ( must_def_inRM == null ) {
					must_def_inRM = (AnalysisTools.REGIONMAP)pred_must_def_outRM.clone();
				} else {
					if (temp != null && temp == pred)
					{
						// this data is from a back-edge, union it with the current data
						must_def_inRM = must_def_inRM.unionWith(pred_must_def_outRM);
					}
					else
					{
						// this is an if-else branch, thus intersect it with the current data
						must_def_inRM = must_def_inRM.intersectWith(pred_must_def_outRM);
					}
				}
			}

			PrintTools.println("  curr must_def_in = " + must_def_in, 5);
			PrintTools.println("  curr must_def_inRM = " + must_def_inRM, 4);
			

			// previous may_def_in and previous must_def_in
			Section.MAP p_may_def_in = (Section.MAP)node.getData("may_def_in");
			Section.MAP p_must_def_in = (Section.MAP)node.getData("must_def_in");
			AnalysisTools.REGIONMAP p_may_def_inRM = (AnalysisTools.REGIONMAP)node.getData("may_def_inRM");
			AnalysisTools.REGIONMAP p_must_def_inRM = (AnalysisTools.REGIONMAP)node.getData("must_def_inRM");

			if ( hasChanged(p_may_def_in, may_def_in) || hasChanged(p_must_def_in, must_def_in) 
				|| hasChangedRM(p_may_def_inRM, may_def_inRM) || hasChangedRM(p_must_def_inRM, must_def_inRM))
			{
				node.putData("may_def_in", may_def_in);
				node.putData("must_def_in", must_def_in);
				node.putData("may_def_inRM", may_def_inRM);
				node.putData("must_def_inRM", must_def_inRM);

				// Handles data kill, union, etc.
				Section.MAP may_def_out = computeOutDef(node, may_def_in, defined_vars);
				node.putData("may_def_out", may_def_out);

				Section.MAP must_def_out = computeOutDef(node, must_def_in, defined_vars);
				node.putData("must_def_out", must_def_out);
				
				AnalysisTools.REGIONMAP may_def_outRM = 
					computeOutDefRM(node, may_def_inRM, defined_vars, currentRegion);
				node.putData("may_def_outRM", may_def_outRM);
				
				AnalysisTools.REGIONMAP must_def_outRM = 
					computeOutDefRM(node, must_def_inRM, defined_vars, currentRegion);
				node.putData("must_def_outRM", must_def_outRM);
				
				PrintTools.println("  curr must_def_out = " + must_def_out, 5);
				PrintTools.println("  curr must_def_outRM = " + must_def_outRM, 4);

				for ( DFANode succ : node.getSuccs() )
					work_list.put(succ.getData("top-order"), succ);
			}
			
		}

		PrintTools.println("[ReachingDef] done *****************************", 3);
	}

	private Section.MAP computeOutDef(DFANode node, Section.MAP in, Set<Symbol> defined_vars)
	{
		Section.MAP out = null;
		RangeDomain rd = (RangeDomain)node.getData("range");
		Set<Symbol> modified_vars = new HashSet<Symbol>();

		if (in == null) in = new Section.MAP();

		out = new Section.MAP();

		Object o = CFGraph.getIR(node);

		if ( o instanceof Traversable )
		{
			Traversable tr = (Traversable)o;

			for ( Expression e : DataFlowTools.getDefSet(tr) )
			{
				Symbol def_symbol = SymbolTools.getSymbolOf(e);
				if (def_symbol != null && shared_vars.contains(def_symbol) )
				{
					out = out.unionWith(DataFlowTools.getDefSectionMap(e, rd, defined_vars), rd);
				}
			}

			modified_vars.addAll(DataFlowTools.getDefSymbol(tr));

			// if functioncall, kill DEF section containing globals and actual params.
			in.removeSideAffected(tr);
		}

		// if symbols in a Section (subscript expression) are redefined, remove such DEF Sections
		in.removeAffected(modified_vars);

		// Since there is no control divergence within a node, both may and must analysis
		// need to perform UNION operation between in and out
		out = in.unionWith(out, rd);

		return out;
	}
	
	private AnalysisTools.REGIONMAP computeOutDefRM(DFANode node, AnalysisTools.REGIONMAP in, 
			Set<Symbol> defined_vars, String region)
	{
		AnalysisTools.REGIONMAP out = null;

		if (in == null) in = new AnalysisTools.REGIONMAP();

		out = new AnalysisTools.REGIONMAP();

		Object o = CFGraph.getIR(node);

		if ( o instanceof Traversable )
		{
			Traversable tr = (Traversable)o;

			for ( Expression e : DataFlowTools.getDefSet(tr) )
			{
				Symbol def_symbol = SymbolTools.getSymbolOf(e);
				if (def_symbol != null && shared_vars.contains(def_symbol) )
				{
					out.put(def_symbol, region);
				}
			}

			// if functioncall, kill DEF variables being globals or actual params.
			in.removeSideAffected(tr);
		}
		//PrintTools.println("[computeOutDefRM] curr in = " + in, 4);
		//out = in.unionWith(out);
		out = out.overwritingUnionWith(in); 
		//PrintTools.println("[computeOutDefRM] curr out = " + out, 4);

		return out;
	}

	private boolean isBarrierNode(DFANode node)
	{
		String tag = (String)node.getData("tag");
		if (tag != null && tag.equals("barrier"))
		{
			return true;
		}
		return false;
	}

	public void display()
	{
		if (debug_level < 5) return;

		for ( int i=0; i<cfg.size(); i++)
		{
			DFANode node = cfg.getNode(i);

			if ( (isBarrierNode(node) && debug_level >= 5) || debug_level >= 8 )
			{
				PrintTools.println("\n" + node.toDot("tag,ir", 1), 5);

				Section.MAP may_def_in = (Section.MAP)node.getData("may_def_in");
				if (may_def_in != null) PrintTools.println("    may_def_in" + may_def_in, 9);

				Section.MAP must_def_in = (Section.MAP)node.getData("must_def_in");
				if (must_def_in != null) PrintTools.println("    must_def_in" + must_def_in, 9);

				Section.MAP may_def_out = (Section.MAP)node.getData("may_def_out");
				if (may_def_out != null) PrintTools.println("    may_def_out" + may_def_out, 5);

				Section.MAP must_def_out = (Section.MAP)node.getData("must_def_out");
				if (must_def_out != null) PrintTools.println("    must_def_out" + must_def_out, 5);
			}
		}

		PrintTools.println(cfg.toDot("tag,ir,use_set", 5), 5);
	}

}
