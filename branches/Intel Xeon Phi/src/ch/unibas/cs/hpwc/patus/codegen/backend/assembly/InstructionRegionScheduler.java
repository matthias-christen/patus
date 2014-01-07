package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.log4j.Logger;

import ch.unibas.cs.hpwc.patus.arch.IArchitectureDescription;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.analyze.DAGraph;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.analyze.DAGraph.Vertex;
import ch.unibas.cs.hpwc.patus.graph.IVertex;
import ch.unibas.cs.hpwc.patus.graph.algorithm.GraphUtil;
import ch.unibas.cs.hpwc.patus.util.MathUtil;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

public class InstructionRegionScheduler extends AbstractInstructionScheduler
{
	///////////////////////////////////////////////////////////////////
	// Constants
	
	private Logger LOGGER = Logger.getLogger (InstructionRegionScheduler.class);

	
	///////////////////////////////////////////////////////////////////
	// Member Variables
	
	public static boolean DEBUG = false;
	
	
	///////////////////////////////////////////////////////////////////
	// Implementation

	public InstructionRegionScheduler (DAGraph graph, IArchitectureDescription arch)
	{
		super (graph, arch);		
	}
	
	@Override
	protected int doSchedule (InstructionList ilOut)
	{
		// simplify the subgraph before solving the linear program
		removeRedundantEdges ();
		linearizeRegions ();

		computeInitialScheduleBounds ();
		
		InstructionList ilOutLastFromILP = null;
		boolean bAllOneCycleBounds = false;
		//InstructionRegionSchedulerILPSolver ilpsolver = new InstructionRegionSchedulerILPSolver (getGraph (), getIssueRate ());
		InstructionRegionSchedulerILPSolver3 ilpsolver = new InstructionRegionSchedulerILPSolver3 (getGraph(), getArchitectureDescription ());
		
		// iteratively reduce the schedule length until the lower bound is reached or
		// the scheduling becomes infeasible
		int nCurrentScheduleLength = getUpperScheduleLengthBound () - 1;
		int nUpperBoundDecrease = 1;
		
		for ( ; nCurrentScheduleLength >= getLowerScheduleLengthBound (); nCurrentScheduleLength--)
		{
			LOGGER.info (StringUtil.concat ("Trying to find schedule of length ", nCurrentScheduleLength, " (lower bound is ", getLowerScheduleLengthBound (), ")"));
			
			bAllOneCycleBounds = false;
			nUpperBoundDecrease = 1;
			
			if (!tightenScheduleBoundsIteratively ())
				break;
			commitBounds ();

			if (!allOneCycleBounds ())
			{
				if (!probeInstructions ())
					break;
				
				if (DEBUG)
					getGraph ().graphviz ();
				
				if (!allOneCycleBounds ())
				{
					InstructionList ilTmp = new InstructionList ();
					int nNewScheduleLength = ilpsolver.solve (nCurrentScheduleLength, ilTmp);
					
					if (nNewScheduleLength == -1)
					{
						// ILP could not be solved successfully
						break;
					}
					
					nUpperBoundDecrease = nCurrentScheduleLength - nNewScheduleLength;
					nCurrentScheduleLength = nNewScheduleLength;
					ilOutLastFromILP = ilTmp;
				}
				else
					bAllOneCycleBounds = true;
			}
			
			reduceCurrentScheduleLength (nUpperBoundDecrease);
		}
				
		// build the output instruction list
		if (ilOutLastFromILP != null)
			ilOut.addInstructions (ilOutLastFromILP);
		else if (bAllOneCycleBounds)
			ilpsolver.reconstructInstructionList (ilOut);
		else
		{
			// no feasible ILP schedule found, and the bounds are too loose:
			// use the default critical path schedule
			getCriticalPathSchedule (ilOut);
		}
		
		return nCurrentScheduleLength;
	}
	
	/**
	 * Determines whether the schedule bounds of all vertices are one-cycle
	 * bounds (i.e., if lower bound = upper bound for each vertex in the graph).
	 * 
	 * @return <code>true</code> iff all vertices have one-cycle schedule bounds
	 */
	protected boolean allOneCycleBounds ()
	{
		for (DAGraph.Vertex v : getGraph ().getVertices ())
			if (v.getLowerScheduleBound () != v.getUpperScheduleBound ())
				return false;
		return true;
	}

	/**
	 * Compute the lower and upper scheduling bounds for each vertex in the graph based on the critical path analysis.
	 */
	protected void computeInitialScheduleBounds ()
	{
		DAGraph graph = getGraph ();
		Iterable<DAGraph.Vertex> itRoots = GraphUtil.getRootVertices (graph);
		Iterable<DAGraph.Vertex> itLeaves = GraphUtil.getLeafVertices (graph);
		
		int nIssueRate = getIssueRate ();
		int nMinExecUnits = getMinExecUnits ();
		
		for (DAGraph.Vertex v : graph.getVertices ())
		{
			// get the critical path distance from the roots to v
			int nCritPathDistFromRoots = 0;
			for (DAGraph.Vertex vertRoot : itRoots)
				nCritPathDistFromRoots = MathUtil.max (nCritPathDistFromRoots, getCriticalPathLengthCalculator ().getCriticalPathDistance (vertRoot, v));
			
			// get the critical path distance from v to the leaves
			int nCritPathDistToLeaves = 0;
			for (DAGraph.Vertex vertLeaf : itLeaves)
				nCritPathDistToLeaves = MathUtil.max (nCritPathDistToLeaves, getCriticalPathLengthCalculator ().getCriticalPathDistance (v, vertLeaf));
			
			// count predecessors and successors and get the minimum latencies from/to the predecessors/successors
			int nPredecessorsCount = graph.getIncomingEdges (v).size ();
			int nSuccessorsCount = graph.getOutgoingEdges (v).size ();
			
			int nPredMinLatency = Integer.MAX_VALUE;
			for (DAGraph.Edge edge : graph.getIncomingEdges (v))
				nPredMinLatency = nPredMinLatency > edge.getLatency () ? edge.getLatency () : nPredMinLatency;
			if (nPredMinLatency == Integer.MAX_VALUE)
				nPredMinLatency = 0;
			
			int nSuccMinLatency = Integer.MAX_VALUE;
			for (DAGraph.Edge edge : graph.getOutgoingEdges (v))
				nSuccMinLatency = nSuccMinLatency > edge.getLatency () ? edge.getLatency () : nSuccMinLatency;
			if (nSuccMinLatency == Integer.MAX_VALUE)
				nSuccMinLatency = 1;
			
			v.setInitialScheduleBounds (
				/*1 +*/ MathUtil.max (
					nCritPathDistFromRoots,
					MathUtil.divCeil (1 + nPredecessorsCount, nIssueRate) - 1,
					nPredecessorsCount / nIssueRate + nPredMinLatency),
				getUpperScheduleLengthBound () - MathUtil.max (
					nCritPathDistToLeaves,
					MathUtil.divCeil (1 + nSuccessorsCount, nMinExecUnits) - 1,
					nSuccessorsCount / nMinExecUnits + nSuccMinLatency) /**/ -1
			);
		}
	}

	/**
	 * 
	 */
	protected void removeRedundantEdges ()
	{
		DAGraph graph = getGraph ();
		IVertex[] rgVerticesInTopologicalOrder = GraphUtil.getTopologicalSort (graph);
		
		for (int i = 0; i < rgVerticesInTopologicalOrder.length; i++)
		{
			List<DAGraph.Edge> listEdgesToRemove = new LinkedList<> ();
			
			for (DAGraph.Edge edgeIV : graph.getOutgoingEdges ((DAGraph.Vertex) rgVerticesInTopologicalOrder[i]))
				for (DAGraph.Edge edgeIW : graph.getOutgoingEdges ((DAGraph.Vertex) rgVerticesInTopologicalOrder[i]))
					if (edgeIV != edgeIW)
						if (edgeIW.getLatency () + getCriticalPathLengthCalculator ().getCriticalPathDistance (edgeIW.getHeadVertex (), edgeIV.getHeadVertex ()) >= edgeIV.getLatency ())
							listEdgesToRemove.add (edgeIV);
			
			for (DAGraph.Edge e : listEdgesToRemove)
				graph.removeEdge (e);
		}
	}

	/**
	 * 
	 */
	protected void linearizeRegions ()
	{
		// TODO
	}
	
	/**
	 * Iteratively tightens the schedule bounds. If the schedule is infeasible,
	 * the method returns <code>false</code>.
	 * 
	 * @return <code>false</code> iff tightening the schedule bounds shows that
	 *         the scheduling is infeasible
	 */
	protected boolean tightenScheduleBoundsIteratively ()
	{
		DAGraph graph = getGraph ();
		Map<Integer, Integer> mapScheduleAt = new HashMap<> ();
	
		boolean bTighteningApplied = false;
		do
		{
			bTighteningApplied = false;
			mapScheduleAt.clear ();
		
			// count the number of instructions which have to be scheduled at a specific cycle
			for (DAGraph.Vertex v : graph.getVertices ())
			{
				if (v.getLowerScheduleBound () > v.getUpperScheduleBound ())
				{
					// empty range: infeasible scheduling
					return false;
				}
				
				if (v.getLowerScheduleBound () == v.getUpperScheduleBound ())
				{
					Integer nCount = mapScheduleAt.get (v.getLowerScheduleBound ());
					mapScheduleAt.put (v.getLowerScheduleBound (), nCount == null ? 1 : nCount + 1);
				}
			}
			
			// check whether schedule bounds can be adjusted
			for (Integer nCycleNum : mapScheduleAt.keySet ())
			{
				int nInstructionsCount = mapScheduleAt.get (nCycleNum);
				
				if (nInstructionsCount > getIssueRate ())
				{
					// not all of the required instructions can be scheduled in this cycle:
					// infeasible scheduling
					return false;
				}
				
				if (nInstructionsCount == getIssueRate ())
				{
					// no more instructions can be scheduled in this cycle:
					// adjust the bounds if of non-single schedule range vertices
					
					for (DAGraph.Vertex v : graph.getVertices ())
					{
						if (v.getLowerScheduleBound () != v.getUpperScheduleBound ())
						{
							if (v.getLowerScheduleBound () == nCycleNum)
							{
								v.setScheduleBounds (nCycleNum + 1, v.getUpperScheduleBound ());
								bTighteningApplied = true;
							}
							if (v.getUpperScheduleBound () == nCycleNum)
							{
								v.setScheduleBounds (v.getLowerScheduleBound (), nCycleNum - 1);
								bTighteningApplied = true;
							}
							
							// adjust predecessors and successors
							if (bTighteningApplied)
								if (!adjustBounds (v))
									return false;
						}
					}
				}
			}
		} while (bTighteningApplied);
		
		return true;
	}
	
	/**
	 * Makes the lower and upper schedule bounds consistent throughout the
	 * graph.
	 * 
	 * @param v
	 *            The vertex whose bounds were modified
	 * @return <code>false</code> iff the schedule turns out to be infeasible
	 *         after making the bounds consistent
	 */
	protected boolean adjustBounds (DAGraph.Vertex v)
	{
		DAGraph graph = getGraph ();
		Set<DAGraph.Vertex> setTightened = new HashSet<> ();
		
		// tighten predecessors
		for (DAGraph.Edge edge : graph.getIncomingEdges (v))
		{
			DAGraph.Vertex vPred = edge.getTailVertex ();
			
			int nMinUpperBnd = vPred.getUpperScheduleBound ();
			for (DAGraph.Edge edgeSucc : graph.getOutgoingEdges (vPred))
				nMinUpperBnd = Math.min (nMinUpperBnd, edgeSucc.getHeadVertex ().getUpperScheduleBound () - edgeSucc.getLatency ());
			
			if (nMinUpperBnd != vPred.getUpperScheduleBound ())
			{
				// check whether the range has become empty
				if (vPred.getLowerScheduleBound () > nMinUpperBnd)
					return false;
				
				setTightened.add (vPred);
				vPred.setScheduleBounds (vPred.getLowerScheduleBound (), nMinUpperBnd);
			}
		}
		
		// tighten successors
		for (DAGraph.Edge edge : graph.getOutgoingEdges (v))
		{
			DAGraph.Vertex vSucc = edge.getHeadVertex ();
			
			int nMaxLowerBnd = vSucc.getLowerScheduleBound ();
			for (DAGraph.Edge edgePred : graph.getIncomingEdges (vSucc))
				nMaxLowerBnd = Math.max (nMaxLowerBnd, edgePred.getTailVertex ().getLowerScheduleBound () + edgePred.getLatency ());
			
			if (nMaxLowerBnd != vSucc.getLowerScheduleBound ())
			{
				// check whether the range has become empty
				if (nMaxLowerBnd > vSucc.getUpperScheduleBound ())
					return false;
				
				setTightened.add (vSucc);
				vSucc.setScheduleBounds (nMaxLowerBnd, vSucc.getUpperScheduleBound ());
			}			
		}
		
		// recursively adjust predecessors/successors of vertices that have been adjusted
		for (DAGraph.Vertex vertTightened : setTightened)
			if (!adjustBounds (vertTightened))
				return false;
		
		return true;
	}
	
	/**
	 * <p>Performs instruction probing:</p>
	 * <p>For each vertex in the graph, the lower and upper instruction bounds are
	 * temporarily set to the same value to simulate scheduling of the
	 * instruction at (a) the lower and (b) the upper bound. If the resulting
	 * schedule turns out to be infeasible, the corresponding bound is discarded
	 * and, if the bounds were set to the lower bound, the lower bound is
	 * incremented (since the original lower bound results in an infeasible
	 * schedule), and, vice versa, if the bounds were set to the upper bound,
	 * the upper bound is decremented.</p>
	 * 
	 * @return <code>false</code> iff instruction probing shows that the
	 *         scheduling is infeasible
	 */
	protected boolean probeInstructions ()
	{
		boolean bBoundsModified = false;
		boolean bNewBoundsCommitted = false;
				
		do
		{
			bBoundsModified = false;
			
			for (DAGraph.Vertex v : getGraph ().getVertices ())
			{
				LOGGER.info (StringUtil.concat ("Probing instructions for ", v.toString (), "..."));
				
				// temporarily fix the upper bound to the lower bound and check whether the problem becomes infeasible
				int nPrevLower = 0;
				int nPrevUpper = v.getUpperScheduleBound ();
	
				do
				{
					bNewBoundsCommitted = false;
					nPrevLower = v.getLowerScheduleBound ();
					v.setScheduleBounds (nPrevLower, nPrevLower);
					
					if (!tightenScheduleBoundsIteratively ())
					{
						discardBounds ();
						v.setScheduleBounds (nPrevLower + 1, nPrevUpper);
						if (!adjustBounds (v))
							return false;
						if (!tightenScheduleBoundsIteratively ())
							return false;
						commitBounds ();
						bNewBoundsCommitted = true;
						bBoundsModified = true;
					}
					else
						discardBounds ();
				} while (bNewBoundsCommitted);
				
				// fix the lower bound to the upper bound to tighten the upper bounds
				nPrevLower = v.getLowerScheduleBound ();
				do
				{
					bNewBoundsCommitted = false;
					nPrevUpper = v.getUpperScheduleBound ();
					v.setScheduleBounds (nPrevUpper, nPrevUpper);
					
					if (!tightenScheduleBoundsIteratively ())
					{
						discardBounds ();
						v.setScheduleBounds (nPrevLower, nPrevUpper - 1);
						if (!adjustBounds (v))
							return false;
						if (!tightenScheduleBoundsIteratively ())
							return false;
						commitBounds ();
						bNewBoundsCommitted = true;
						bBoundsModified = true;
					}
					else
						discardBounds ();
				} while (bNewBoundsCommitted);
			}
		} while (bBoundsModified);
		
		return true;
	}
	
	/**
	 * Decrements the initial upper schedule bounds of all the vertices in the
	 * graph.
	 */
	protected void reduceCurrentScheduleLength (int nUpperBoundDecrease)
	{
		for (DAGraph.Vertex v : getGraph ().getVertices ())
			v.setInitialScheduleBounds (v.getInitialLowerScheduleBound (), v.getInitialUpperScheduleBound () - nUpperBoundDecrease);
	}
	
	/**
	 * Commits all the temporarily set bounds, i.e., calls
	 * {@link Vertex#commitBounds()} for each vertex in the graph.
	 */
	protected void commitBounds ()
	{
		for (DAGraph.Vertex v : getGraph ().getVertices ())
			v.commitBounds ();
	}
	
	/**
	 * Discards all the temporarily set bounds, i.e., calls
	 * {@link Vertex#discardBounds()} for each vertex in the graph.
	 */
	protected void discardBounds ()
	{
		for (DAGraph.Vertex v : getGraph ().getVertices ())
			v.discardBounds ();
	}	
}
