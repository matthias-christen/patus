package cetus.analysis;

import java.io.*;
import java.util.*;
import cetus.hir.*;
import cetus.exec.*;
import cetus.analysis.PointsToDomain.Universe;

/**
 * Alias analysis is used to obtain alias sets i.e. sets of 
 * variable names that reference the same memory locations, in 
 * other words, aliases. Alias information can be flow and/or 
 * context sensitive or neither. The implementation of this analysis in 
 * Cetus provides two different levels.
 * <p>
 * Switch off alias analysis intentionally using option level 
 * 0 on the command line.
 * <p>
 * The default for alias analysis is to provide alias information 
 * using advanced interprocedural points-to analysis. See 
 * {@link IPPointsToAnalysis}. Points-to analysis in Cetus is 
 * flow as well as context sensitive i.e. the analysis provides 
 * pointer relationships at every program point using 
 * limited context information as well. Alias analysis can use the 
 * sets of pointer relationships at the relevant program 
 * point to create accurate alias sets. Interprocedural 
 * points-to analysis further helps crucial disambiguation, 
 * especially in the case of arrays.
 * <p>
 * The advanced interprocedural analyzer is supplemented by a 
 * simple array-argument disambiguation analysis, 
 * {@link ArrayParameterAnalysis}.
 * 
 * @see PointsToAnalysis
 * @see IPPointsToAnalysis
 * @see ArrayParameterAnalysis
 */

public class AliasAnalysis extends AnalysisPass
{
	private int debug_level;
	private boolean no_alias = false;

	private int alias_level;
	private static final int NO_ALIAS = 0;
	private static final int ADVANCED_INTERPROC = 1;

	// Specifies "*", which means that a symbol 
	// has alias set = "*" i.e. it is aliased to 
	// everything
	private HashSet all_set;

	// Array parameter analysis for disambiguation over 
	// and above the results obtained using interprocedural
	// points to information
	ArrayParameterAnalysis array_analysis;

	public AliasAnalysis(Program program)
	{
		super(program);
		debug_level = Integer.valueOf(Driver.getOptionValue("verbosity")).intValue();
		all_set = new HashSet();
		all_set.add("*");

		alias_level = Integer.valueOf(Driver.getOptionValue("alias")).intValue();
		if ( alias_level == NO_ALIAS)
			no_alias = true;
		
		array_analysis = new ArrayParameterAnalysis(program);
	}

	public String getPassName() { return new String("[AliasAnalysis]"); }

	/**
	 * This provides the high-level driver for Alias analysis. The default is 
	 * to run interprocedural points-to analysis and use that information, hence 
	 * a majority of this procedure is not executed in that case.
	 * <p>
	 * For other global alias analyses, this procedure traverses the IR 
	 * and collects whole program alias sets. These can be supported 
	 * by user-provided information regarding interprocedural assumptions.
	 */
	public void start()
	{
		if (no_alias) return;

		if (debug_level > 1) System.out.println("[AliasAnalysis] Start");
		
		// if points to analysis is requested, run the 
		// supplementary ArrayParameterAnalysis for more precise 
		// alias information. Interprocedural points-to analysis 
		// will be run on demand, if required.
		if (alias_level == ADVANCED_INTERPROC)
		{
			array_analysis.start();
			return;
		}

		if (debug_level > 1) System.out.println("[AliasAnalysis] Done");

	}	/* end of start function */

	/**
	 * Returns the alias set for the given input symbol, which includes the 
	 * input symbol itself.
	 * This forms the interface API to other passes, for requested alias 
	 * information. 
	 * <p>
	 * For advanced analysis, it uses the interface to interprocedural 
	 * points-to analysis. Alias sets are created from the map 
	 * returned by the points-to analyzer.
	 * 
	 * @param cur_stmt Statement used to request flow-sensitive alias 
	 * 					information using advanced points-to analysis
	 * @param symbol The variable symbol for which alias sets are being 
	 * 					requested.
	 * 
	 * @return alias set for symbol, which includes the input symbol itself.
	 */
	public Set get_alias_set(Statement cur_stmt, Symbol symbol)
	{
		if (no_alias)
			return null;
		else if (alias_level == ADVANCED_INTERPROC)
		{
			if (PointsToAnalysis.isPointer(symbol))
			{
				if (cur_stmt == null)
					return all_set;
				
				Domain points_to = 
					IPPointsToAnalysis.getPointsToRelations(cur_stmt);
				if (points_to == null)
					return null;
				
				Map stmt_alias_map = getAliasesFromPointsToDomain(points_to);
				if (stmt_alias_map.containsKey("universe"))
					return ((Set)stmt_alias_map.get("universe"));
				else
				{
					Set aliased_symbols = (Set)stmt_alias_map.get(symbol);
					if (aliased_symbols != null)
						aliased_symbols = 
							arrayParameterFilter(symbol, aliased_symbols, cur_stmt);
					return (aliased_symbols);
				}
			}
			else
				return (new HashSet());
		}
		else
			return all_set;
	}

	/**
	 * Prints to standard out a pretty version of the global alias map.
   * @deprecated
	 */
	@Deprecated public void displayAliasMap()
	{
		if (debug_level < 2) return;

		System.out.println("****** alias_map ******");
		System.out.println("Deprecated");
	}

	/**
	 * Returns true if Symbol a is aliased to Symbol b at the given 
	 * statement.
	 */
	public boolean isAliased(Statement stmt, Symbol a, Symbol b)
	{
		if (no_alias)
			return false;
		else if (alias_level == ADVANCED_INTERPROC)
		{
			if (PointsToAnalysis.isPointer(a))
			{
				if (stmt == null)
					return true;
				
				Domain points_to = 
					IPPointsToAnalysis.getPointsToRelations(stmt);
				if (points_to == null)
					return false;
				
				Map stmt_alias_map = getAliasesFromPointsToDomain(points_to);
				if (stmt_alias_map.containsKey("universe"))
					return true;
				else
				{
					Set aliased_symbols = (Set)stmt_alias_map.get(a);
					if (aliased_symbols != null)
					{
						aliased_symbols = 
							arrayParameterFilter(a, aliased_symbols, stmt);
						if (aliased_symbols.contains(b))
							return true;
						else
							return false;
					}
					else
						return false;
				}
			}
			else
				return false;
		}
		else
			return true;
	}

	/**
	 * Returns true if the given Symbol, a, is aliased to any symbol in the
	 * set of Symbols, bset.
	 */
	public boolean isAliased(Statement stmt, Symbol a, Set<Symbol> bset)
	{
		if (no_alias)
			return false;
		else
		{
			for (Symbol sym : bset)
			{
				if (isAliased(stmt, a, sym)) return true; 
			}
		}
		return false;
	}

	// ========================================================================
	// Points-to analysis based alias information API
	// ========================================================================
	private void addAliases(Symbol lsymbol, Symbol rsymbol, Map alias_map)
	{
		HashSet lset = (HashSet)alias_map.get(lsymbol);
		HashSet rset = (HashSet)alias_map.get(rsymbol);

		HashSet merged_set = new HashSet();
		if (lset != null) merged_set.addAll(lset);
		if (rset != null) merged_set.addAll(rset);

		merged_set.add(lsymbol);
		merged_set.add(rsymbol);

		for (Symbol s : (HashSet<Symbol>)merged_set)
		{
			alias_map.remove(s);
			alias_map.put(s, merged_set);
		}
	}

	private Map getAliasesFromPointsToDomain(Domain points_to_info)
	{
		Map alias_map = new HashMap();
		if (points_to_info instanceof PointsToDomain)
		{
			for (Symbol s : ((PointsToDomain)points_to_info).keySet())
			{
				HashSet<PointsToRel> rel_set = 
					((PointsToDomain)points_to_info).get(s);
				for (PointsToRel p_rel : rel_set)
					addAliases(s, p_rel.getPointedToSymbol(), alias_map);
			}
		}
		else if (points_to_info instanceof Universe)
		{
			alias_map.put("universe", all_set);
		}

		return alias_map;
	}
	
	private Set arrayParameterFilter(Symbol source, Set aliased, Statement stmt)
	{
		Set filtered_aliases = new HashSet<Symbol>();
		Procedure proc = stmt.getProcedure();
		
		for (Symbol alias : (Set<Symbol>)aliased)
		{
			if ( !array_analysis.isDisjoint(source, alias, proc) )
				filtered_aliases.add(alias);
		}
		return filtered_aliases;
	}
}


