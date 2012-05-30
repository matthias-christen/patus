package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
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
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.analyze.DAGraph.Vertex;
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
		
		InstructionList ilOutLastFromILP = null;
		boolean bAllOneCycleBounds = false;
		
		// iteratively reduce the schedule length until the lower bound is reached or the scheduling
		// becomes infeasible
		int nCurrentScheduleLength = getUpperScheduleLengthBound ();// - 1;
solveILP (nCurrentScheduleLength, ilOut);		
		
		for ( ; nCurrentScheduleLength >= getLowerScheduleLengthBound (); nCurrentScheduleLength--)
		{
			bAllOneCycleBounds = false;
			
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
					InstructionList ilOutTmp = new InstructionList ();
					if (!solveILP (nCurrentScheduleLength, ilOutTmp))
						break;
					
					// the ILP schedule is feasible; memorize it
					ilOutLastFromILP = ilOutTmp;
				}
				else
					bAllOneCycleBounds = true;
			}
			
			reduceCurrentScheduleLength ();
		}
		
		// build the output instruction list
		if (ilOutLastFromILP != null)
			ilOut.addInstructions (ilOutLastFromILP);
		else if (bAllOneCycleBounds)
			reconstructInstructionList (ilOut);
		else
		{
			// no feasible ILP schedule found, and the bounds are too loose:
			// use the default critical path schedule
			getCriticalPathSchedule (ilOut);
		}
		
		return nCurrentScheduleLength;
	}
	
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
	protected boolean solveILP (final int nCyclesCount, InstructionList ilOut)
	{
		// problem:
		//     min T  (schedule time)
		// st. \sum_{j=1}^m j x_i^j  <=  T  for all i   ("time" constraint)
		//     \sum_{j=1}^m x_i^j  =  1     for all i   ("must schedule" constraint)
		//     \sum_{i=1}^n  <=  r          for all j   ("issue" constraint, r is issue rate)
		//     \sum_{j=1}^m j x_k^j + L_{ki}  <=  \sum_{j=1}^m j x_i^j   for all  edges (i,k)   ("dependence" constraint)
		//
		// note:
		// - m is the number of cycles
		// - n is the number of instructions
		// - the decision variable x_i^j is 1 if the instruction i is scheduled in cycle j, and 0 otherwise
		// - \sum_{j=1}^m j x_i^j is the cycle in which instruction i is scheduled
		
		// variables:
		// - T
		// - x_i^j  (1 <= i <= n, 1 <= j <= m)
		//          i: instruction index
		//          j: cycle index
		//          memory layout: x_1^1 x_1^2 ... x_1^m  x_2^1 x_2^2 ... x_2^m  ...  x_n^1 x_n^2 ... x_n^m
		//          (first cycles, then instructions)
		
		// constraints:
		// - n time constraints
		// - n must schedule constraints
		// - m issue constraints
		// - #edges dependence constraints
		
		int nResult = -1;
		
		final int nVerticesCount = getGraph ().getVerticesCount ();
		int nVarsCount = 1 + nVerticesCount * nCyclesCount;
//		int nConstraintsCount = 2 * nVerticesCount + nCyclesCount + getGraph ().getEdgesCount ();
		
		// count redundant constraints
		// - a must-schedule constraint is redundant if the instruction has a one-cycle range
		// - a dependence constraint between instructions j,k is redundant if L_{jk} + ubnd(j) <= lbnd(k)
//		for (DAGraph.Vertex v : getGraph ().getVertices ())
//			if (v.getLowerScheduleBound () == v.getUpperScheduleBound ())
//				nConstraintsCount--;
//		for (DAGraph.Edge e : getGraph ().getEdges ())
//			if (e.getLatency () + e.getTailVertex ().getUpperScheduleBound () <= e.getHeadVertex ().getLowerScheduleBound ())
//				nConstraintsCount--;
		
		
		// make indexing easier...
		class X
		{
			public int idx (int i, int j)
			{
				return 1 + (j - 1) + (i - 1) * nCyclesCount + 1;
			}
		}
		X x = new X ();

		try
		{
			LpSolve solver = LpSolve.makeLp (0, nVarsCount);
						
			// add constraints
			
			// time constraints
			for (int i = 1; i <= nVerticesCount; i++)
			{
				double[] rgCoeffs = new double[nVarsCount + 1];
				rgCoeffs[1] = -1;
				for (int j = 1; j <= nCyclesCount; j++)
					rgCoeffs[x.idx (i, j)] = j;
				solver.addConstraint (rgCoeffs, LpSolve.LE, 0);
			}
			
			// must-schedule constraints
			int i = 1;
			for (DAGraph.Vertex v : getGraph ().getVertices ())
			{
				// the constraint is redundant if the instruction has a one-cycle scheduling range
				//if (v.getLowerScheduleBound () != v.getUpperScheduleBound ())
				{
					double[] rgCoeffs = new double[nVarsCount + 1];
					for (int j = 1; j <= nCyclesCount; j++)
						rgCoeffs[x.idx (i, j)] = 1;
					solver.addConstraint (rgCoeffs, LpSolve.EQ, 1);
				}
				i++;
			}
			
			// issue constraints
			for (int j = 1; j <= nCyclesCount; j++)
			{
				double[] rgCoeffs = new double[nVarsCount + 1];
				for (i = 1; i <= nVerticesCount; i++)
					rgCoeffs[x.idx (i, j)] = 1;
				solver.addConstraint (rgCoeffs, LpSolve.LE, getIssueRate ());
			}
			
			// dependence constraints
			for (DAGraph.Edge e : getGraph ().getEdges ())
			{
				final int b = e.getTailVertex ().getUpperScheduleBound ();
				final int c = e.getHeadVertex ().getLowerScheduleBound ();
				final int L = e.getLatency ();
				
				// the dependence constraint between instructions j,k is redundant if L_{jk} + ubnd(j) <= lbnd(k)
				if (L + b > c)
				{
					final int M = b + L - c;
					
					// find the indices of the tail and head vertices
					int k = -1;
					i = -1;
					int j = 1;
					for (DAGraph.Vertex v : getGraph ().getVertices ())
					{
						if (v == e.getTailVertex ())
							k = j;
						if (v == e.getHeadVertex ())
							i = j;
						if (k > -1 && i > -1)
							break;
						j++;
					}

					// fill in the coefficients
					double[] rgCoeffs = new double[nVarsCount + 1];
					for (j = c - L + 1; j <= b; j++)
						rgCoeffs[x.idx (k, j)] = j + L - c;
					for (j = c; j <= b + L - 1; j++)
						rgCoeffs[x.idx (i, j)] = M - j + c;
					
					solver.addConstraint (rgCoeffs, LpSolve.LE, M);
				}
			}
			
			// set variable bounds for x
			solver.setLowbo (1, 0);
			solver.setUpbo (1, nCyclesCount);
			for (i = 2; i <= nVarsCount; i++)
				solver.setBinary (i, true);

			// set objective
			double[] rgObj = new double[nVarsCount + 1];
			rgObj[1] = 1;
			for (i = 2; i <= nVarsCount; i++)
				rgObj[i] = 0;
			solver.setObjFn (rgObj);
			
			// print the matrix
			if (DEBUG)
			{
				for (int $y = 1; $y <= solver.getNrows (); $y++)
				{
					for (int $x = 1; $x <= solver.getNcolumns (); $x++)
						System.out.printf ("%.0f ", solver.getMat ($y, $x));
					System.out.println ();
				}
			}
			
			// solve the ILP
			nResult = solver.solve ();
			
			// get solution
			double[] y = solver.getPtrVariables ();
			if (DEBUG)
			{
				solver.printObjective ();
				System.out.println (Arrays.toString (y));
			}
			
			i = 1;
			for (DAGraph.Vertex v : getGraph ().getVertices ())
			{
				int nCycle = 0;
				for (int j = 1; i <= nCyclesCount; j++)
					if (y[x.idx (i, j)] > 0)
					{
						nCycle = j;
						break;
					}
				v.setScheduleBounds (nCycle, nCycle);
				i++;
			}
			reconstructInstructionList (ilOut);
			discardBounds ();
			
			solver.deleteLp ();
		}
		catch (LpSolveException e)
		{
			e.printStackTrace();
		}
		
		return nResult == LpSolve.OPTIMAL || nResult == LpSolve.SUBOPTIMAL;
	}

	/**
	 * Reconstruct the instruction list from the DAG if all instructions
	 * have one-cycle schedule bounds.
	 * @param ilOut The output instruction list
	 */
	private void reconstructInstructionList (InstructionList ilOut)
	{
		List<DAGraph.Vertex> listVertices = new ArrayList<> ();
		for (DAGraph.Vertex v : getGraph ().getVertices ())
			listVertices.add (v);
		
		// sort the vertices according to their lower schedule bounds
		Collections.sort (listVertices, new Comparator<DAGraph.Vertex> ()
		{
			@Override
			public int compare (Vertex v1, Vertex v2)
			{
				return v1.getLowerScheduleBound () - v2.getLowerScheduleBound ();
			}
		});
		
		// create the instruction list
		for (DAGraph.Vertex v : listVertices)
			ilOut.addInstruction (v.getInstruction ());
	}
}
