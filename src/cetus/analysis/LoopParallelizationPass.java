package cetus.analysis;

import java.util.*;

import cetus.hir.*;

import cetus.exec.*;

/**
 * Whole program analysis that uses data-dependence information to 
 * internally annotate loops that are parallel
 */
public class LoopParallelizationPass extends AnalysisPass
{
	// The level of parallelization requested from this pass
	private static final int PARALLELIZE_LOOP_NEST = 1;
	private static final int PARALLELIZE_DISABLE_NESTED = 2;

	// Store the level of parallelization required
	private int parallelization_level;
	// Enable/disable nested loop parallelism
	private boolean nested_parallelism;

	public LoopParallelizationPass(Program program)
	{
		super(program);
		parallelization_level = Integer.valueOf(Driver.getOptionValue("parallelize-loops")).intValue();
	}

	/**
	 * Get Pass name
	 */
	public String getPassName()
	{
		return new String("[LOOP-PARALLELIZATION]");
	}

	/**
	 * Start whole program loop parallelization analysis
	 */
	public void start()
	{
		/* Implemented nested or non-nested parallelism as per user request */
		switch (parallelization_level) {
		case PARALLELIZE_LOOP_NEST:
			nested_parallelism = true;
			parallelizeAllNests();
			break;
		case PARALLELIZE_DISABLE_NESTED:
			nested_parallelism = false;
			parallelizeAllNests();
			break;
		}
		reportParallelization();
	}

	public void parallelizeAllNests()
	{	
		LinkedList<Loop> tested_loops = new LinkedList<Loop>();
		DepthFirstIterator iter = new DepthFirstIterator(program);
		iter.pruneOn(Loop.class);
		for(;;)
		{
			Loop l = null;
			try {
				l = (Loop)iter.next(Loop.class);
			}
			catch (NoSuchElementException e) {
				break;
			}
			parallelizeLoopNest(l);
		}
	}

	private void addCetusAnnotation(Loop loop, boolean parallel)
	{
		if (parallel)
		{
			CetusAnnotation note = new CetusAnnotation();
			note.put("parallel", "true");
			((Annotatable)loop).annotate(note);
		}
	}
	/**
	 * Check if a specific loop in the program is parallel, irrespective of the effects
	 * of parallelizing or serializing enclosing loops
	 * @param loop the for loop to check parallelism for based only on dependence analysis
	 * @return
	 */
	private boolean checkParallel(Loop loop)
	{
		boolean is_parallel = false;
		boolean nest_eligible = false;
		DDGraph loop_graph = null;
		DDGraph pdg = program.getDDGraph();
		// Check eligibility of enclosed nest for dependence testing
		List<Loop> entire_nest = LoopTools.calculateInnerLoopNest(loop);
		for (Loop l : entire_nest)
		{
			nest_eligible = LoopTools.checkDataDependenceEligibility(l);
			if (nest_eligible == false)
				break;
		}
		if (nest_eligible==true) {
			// Check if scalar dependences might exist
			if(LoopTools.scalarDependencePossible(loop)==true)
				is_parallel=false;
			// Check if early exit break statement might exist
			else if (LoopTools.containsBreakStatement(loop)==true)
				is_parallel=false;
			// check if array loop carried dependences exist
			else if (pdg.checkLoopCarriedDependence(loop)==true) {
				// Also check if loop carried dependences might be
				// because of private or reduction or induction variables
				loop_graph = pdg.getSubGraph(loop);
				ArrayList<DDGraph.Arc> loop_carried_deps = loop_graph.getLoopCarriedDependencesForGraph();
				for (DDGraph.Arc dep : loop_carried_deps)
				{
					if (dep.isCarried(loop))
					{
						ArrayAccess dep_access = dep.getSource().getArrayAccess();
						Symbol dep_symbol = SymbolTools.getSymbolOf((Expression)dep_access);
						// Check if loop carried dependence is for private variable
						if (LoopTools.isPrivate(dep_symbol, loop))
							is_parallel = true;
						// Check if loop carried dependence is for reduction variable
						else if (LoopTools.isReduction(dep_symbol, loop))
							is_parallel = true;
						//else if (LoopTools.isInductionVariable(dep_symbol, l))
						//	is_parallel = true;
						else
						{
							is_parallel = false;
							break;
						}
					}
					else
						is_parallel = true;
				}
			}
			// No scalar or array dependences
			else
				is_parallel=true;
		}
		else
			is_parallel=false;

		return is_parallel;
	}

	/**
	 * Using dependence information, parallelize the entire loop nest covered by the enclosing
	 * loop. If an outer loop is found to be serial, serialize it and eliminate all loop
	 * carried dependences originating from it, this will in turn expose inner parallelism
	 * @param enclosing_loop the loop which encloses the nest to be parallelized
	 */
	private void parallelizeLoopNest(Loop enclosing_loop)
	{
		boolean is_parallel;
		DDGraph dependence_graph = program.getDDGraph();
		List<Loop> eligible_loops = 
			LoopTools.extractOutermostDependenceTestEligibleLoops(
					(Traversable)enclosing_loop);
		for (Loop outer_loop : eligible_loops)
		{
			DDGraph nest_ddgraph =
				dependence_graph.getSubGraph(outer_loop);
			LinkedList<Loop> contained_nest = 
				LoopTools.calculateInnerLoopNest(outer_loop);

			/* Move inwards from outer loop to inner loop to check for parallelism
			 * and account for covered dependences due to serialization
			 */
			 for (Loop l : contained_nest)
			 {
				 is_parallel = true;
				 if (LoopTools.scalarDependencePossible(l)==true)
				 {
					 is_parallel = false;
				 }
				 else if (LoopTools.containsBreakStatement(l)==true)
				 {
					 is_parallel = false;
				 }
				 // Else check for array dependences
				 else
				 {
					 /* Check if this loop is parallel */
					 Iterator<DDGraph.Arc> iter = nest_ddgraph.getAllArcs().iterator();
					 while (iter.hasNext())
					 {
						 DDGraph.Arc row = iter.next();
						 // If direction is loop carried
						 if ((row.getDependenceVector().getDirectionVector()).containsKey(l) &&
								 (row.getDependenceVector().getDirection(l)
										 != DependenceVector.equal) &&
										 (row.getDependenceVector().getDirection(l)
												 != DependenceVector.nil))
						 {
							 ArrayAccess src_access = row.getSource().getArrayAccess();
							 Symbol src_symbol = SymbolTools.getSymbolOf((Expression)src_access);
							 ArrayAccess sink_access = row.getSink().getArrayAccess();
							 Symbol sink_symbol = SymbolTools.getSymbolOf((Expression)sink_access);

							 // Check if loop carried dependence is for private variable
							 // Check if loop carried dependence is for reduction variable
							 // If not, must serialize this loop
							 boolean serialize = true;
							 if ( LoopTools.isPrivate(src_symbol, l) || 
									 LoopTools.isReduction(src_symbol, l) )
							 {
								 if ( LoopTools.isPrivate(sink_symbol, l) ||
										 LoopTools.isReduction(sink_symbol, l) )
									 serialize = false;
							 }
							 else
							 {
								 serialize = true;
							 }

							 if (serialize)
							 {
								 is_parallel = false;
								 /* Remove this dependence vector as serializing this loop
								  * will remove covered dependences (this direction will be
								  * < if the enclosing_loop passed into this loop is at the
								  * outermost level in the program. If not, it might be a >
								  * direction, but is assumed to be covered by an outer <
								  * direction and hence, the dependence vector can be deleted
								  * If the direction is any, there can be an equal direction
								  * as well and hence the row should not be deleted
								  */
								 if (row.getDependenceVector().getDirection(l)
										 != DependenceVector.any)
									 iter.remove();
							 }
						 }
					 }
				 }
				 /* If parallel, add annotation */
				 addCetusAnnotation(l, is_parallel);
				 /* If nested_parallelism disabled, exit parallelization as soon as first available parallel
				  * loop is found */
				 if ((is_parallel) && (nested_parallelism == false))
					 break;
			 }
		}
	}

	/**
	 * Prints summary of loop parallelization pass.
	 */
	private void reportParallelization()
	{
		if ( PrintTools.getVerbosity() < 1 )
			return;

		LoopTools.addLoopName(program);

		String tag = "[PARALLEL REPORT] ";
		String separator = "";
		for ( int i=0; i<80-tag.length(); i++ ) separator += ":";

		StringBuilder legend = new StringBuilder(300);
		legend.append(tag+separator+"\n");
		legend.append(tag+"InputParallel: loop is parallel in the input program\n");
		legend.append(tag+"CetusParallel: loop is auto-parallelized\n");
		legend.append(tag+"NonCanonical : loop is not canonical\n");
		legend.append(tag+"NonPerfect   : loop is not perfect nest\n");
		legend.append(tag+"ControlFlow  : loop may exit prematually\n");
		legend.append(tag+"SymbolicStep : loop step is symbolic\n");
		legend.append(tag+"FunctionCall : loop contains function calls\n");
		legend.append(tag+"I/O          : loop contains I/O calls\n");
		legend.append(tag+separator+"\n");
		System.out.print(legend);

		String loop_name = "-";
		boolean omp_found = false, cetus_parallel_found = false;

		DepthFirstIterator iter = new DepthFirstIterator(program);
		while ( iter.hasNext() )
		{
			Object o = iter.next();

			if ( o instanceof ForLoop )
			{
				ForLoop for_loop = (ForLoop)o;
				StringBuilder out = new StringBuilder(80);
				out.append(LoopTools.getLoopName(for_loop));
				if ( for_loop.containsAnnotation(OmpAnnotation.class, "for") )
					out.append(", InputParallel");
				if ( for_loop.containsAnnotation(CetusAnnotation.class, "parallel") )
					out.append(", CetusParallel");
				if ( !LoopTools.isCanonical(for_loop) )
					out.append(", NonCanonical");
				if ( !LoopTools.isPerfectNest(for_loop) )
					out.append(", NonPerfect");
				if ( LoopTools.containsControlFlowModifier(for_loop) )
					out.append(", ControlFlow");
				if ( !LoopTools.isIncrementEligible(for_loop) )
					out.append(", SymbolicStep");
				if ( IRTools.containsClass(for_loop, FunctionCall.class) )
					out.append(", FunctionCall");
				System.out.println(tag+out);
			}
		}
	}

}
