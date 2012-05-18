package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import lpsolve.LpSolve;
import lpsolve.LpSolveException;

import ch.unibas.cs.hpwc.patus.arch.IArchitectureDescription;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.analyze.DAGraph;
import ch.unibas.cs.hpwc.patus.graph.IVertex;
import ch.unibas.cs.hpwc.patus.graph.algorithm.GraphUtil;
import ch.unibas.cs.hpwc.patus.util.MathUtil;

public class InstructionRegionScheduler extends AbstractInstructionScheduler
{
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
		for (int nCurrentScheduleLength = getUpperScheduleLengthBound () - 1; nCurrentScheduleLength >= getLowerScheduleLengthBound (); nCurrentScheduleLength--)
		{
			if (!tightenScheduleBoundsIteratively ())
				break;
			commitBounds ();
			
			if (!probeInstructions ())
				break;
			
			if (DEBUG)
				getGraph ().graphviz ();
			
			if (!solveILP (nCurrentScheduleLength))
				break;
			
			reduceCurrentScheduleLength ();
		}
		
		return 0;
	}

	/**
	 * Compute the lower and upper scheduling bounds for each vertex in the graph based on the critical path analysis.
	 */
	protected void computeInitialScheduleBounds ()
	{
		DAGraph graph = getGraph ();
		Iterable<DAGraph.Vertex> itRoots = GraphUtil.getRootVertices (graph);
		Iterable<DAGraph.Vertex> itLeaves = GraphUtil.getLeafVertices (graph);
		
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
					MathUtil.divCeil (1 + nPredecessorsCount, getIssueRate ()) - 1,
					nPredecessorsCount / getIssueRate () + nPredMinLatency),
				getUpperScheduleLengthBound () - MathUtil.max (
					nCritPathDistToLeaves,
					MathUtil.divCeil (1 + nSuccessorsCount, getIssueRate ()) - 1,
					nSuccessorsCount / getIssueRate () + nSuccMinLatency) /**/ -1
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
	 * 
	 * @param v
	 * @return
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
	 * Performs instruction probing.
	 * @return <code>false</code> iff instruction probing shows that the scheduling is infeasible
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
				} while (bNewBoundsCommitted);
			}
		} while (bBoundsModified);
		
		return true;
	}
	
	/**
	 * 
	 */
	protected void reduceCurrentScheduleLength ()
	{
		for (DAGraph.Vertex v : getGraph ().getVertices ())
			v.setInitialScheduleBounds (v.getInitialLowerScheduleBound (), v.getInitialUpperScheduleBound () - 1);
	}
	
	protected void commitBounds ()
	{
		for (DAGraph.Vertex v : getGraph ().getVertices ())
			v.commitBounds ();
	}
	
	protected void discardBounds ()
	{
		for (DAGraph.Vertex v : getGraph ().getVertices ())
			v.discardBounds ();
	}

	/**
	 * Builds the ILP formulation from the graph and tries to solve the ILP.
	 * 
	 * @param graph
	 *            The (simplified) dependence analysis graph for which to build
	 *            the ILP
	 * @return <code>true</code> if the solver was able to solve the ILP,
	 *         <code>false</code> if no solution was found or the problem is
	 *         infeasible
	 */
	@SuppressWarnings("static-method")
	protected boolean solveILP (int nUpperSchedulingLengthBound)
	{
		int nVarsCount = 1 + getGraph ().getVerticesCount () * nUpperSchedulingLengthBound;
		int nConstraintsCount = 2 * getGraph ().getVerticesCount () + nUpperSchedulingLengthBound;
		
		try
		{
			LpSolve solver = LpSolve.makeLp (nConstraintsCount, nVarsCount);
			
			// set variable bounds
			for (int k = 1; k < nVarsCount; k++)
			{
				solver.setLowbo (k, 0);
				solver.setUpbo (k, 1);
			}
			
			// add constraints
			
			// must-schedule constraints
			for (int i = 0; i < getGraph ().getVerticesCount (); i++)
			{
				double[] rgCoeffs = new double[nVarsCount];
				for (int j = 0; j < nUpperSchedulingLengthBound; j++)
					rgCoeffs[1 + i + j * nUpperSchedulingLengthBound] = 1;
				solver.addConstraint (null, LpSolve.EQ, 1);
			}
			
			// issue constraints
			for (int j = 0; j < nUpperSchedulingLengthBound; j++)
			{
				double[] rgCoeffs = new double[nVarsCount];
				for (int i = 0; i < getGraph ().getVerticesCount (); i++)
					rgCoeffs[1 + i + j * nUpperSchedulingLengthBound] = 1;
				solver.addConstraint (null, LpSolve.LE, getIssueRate ());
			}
			
			// dependence constraints
			
			// time constraints
			for (int i = 0; i < getGraph ().getVerticesCount (); i++)
			{
				double[] rgCoeffs = new double[nVarsCount];
				for (int j = 0; j < nUpperSchedulingLengthBound; j++)
					rgCoeffs[1 + i + j * nUpperSchedulingLengthBound] = j + 1;
				solver.addConstraint (null, LpSolve.LE, 0);
			}
			
			// set objective
			double[] rgObj = new double[nVarsCount];
			rgObj[0] = 1;
			solver.setObjFn (rgObj);
			
			solver.solve ();
			
			// get solution
			solver.getPtrVariables ();
			
			solver.deleteLp ();
		}
		catch (LpSolveException e)
		{
			e.printStackTrace();
		}
		
		return true;
	}
}
