package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.log4j.Logger;

import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.analyze.DAGraph;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.analyze.DAGraph.Vertex;
import ch.unibas.cs.hpwc.patus.ilp.ILPModel;
import ch.unibas.cs.hpwc.patus.ilp.ILPSolution;
import ch.unibas.cs.hpwc.patus.ilp.ILPSolver;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 * <p>
 * Builds the ILP formulation from the graph and tries to solve the ILP.
 * </p>
 * 
 * <p>
 * The problem:
 * </p>
 * min <i>T</i> (schedule time)<br/>
 * such that
 * <ul>
 * <li>&sum;<sub>j=1</sub><sup>m</sup> j x<sub>i</sub><sup>j</sup> &le; T
 * for all i ("time" constraint)</li>
 * <li>&sum;<sub>j=1</sub><sup>m</sup> x<sub>i</sub><sup>j</sup> = 1 for all
 * i ("must schedule" constraint)</li>
 * <li>&sum;<sub>i=1</sub><sup>n</sup> &le; r for all j ("issue" constraint,
 * r is issue rate)</li>
 * <li>&sum;<sub>j=1</sub><sup>m</sup> j x<sub>k</sub><sup>j</sup> +
 * L<sub>ki</sub> &le; &sum;<sub>j=1</sub><sup>m</sup> j
 * x<sub>i</sub><sup>j</sup> for all edges (i,k) ("dependence" constraint)</li>
 * </ul>
 * where
 * <ul>
 * <li>m is the number of cycles</li>
 * <li>n is the number of instructions</li>
 * <li>the decision variable x<sub>i</sub><sup>j</sup> is 1 if the
 * instruction i is scheduled in cycle j, and 0 otherwise
 * <li>&sum;<sub>j=1</sub><sup>m</sup> j x<sub>i</sub><sup>j</sup> is the
 * cycle in which instruction i is scheduled.</li>
 * </ul>
 * The variables of the model are:
 * <ul>
 * <li>T (the schedule time)</li>
 * <li>x<sub>i</sub><sup>j</sup> (for 1 &le; i &le; n, 1 &le; j &le; m);
 * <ul>
 * <li>i: instruction index</li>
 * <li>j: cycle index</li>
 * <li>memory layout: x<sub>1</sub><sup>1</sup> x<sub>1</sub><sup>2</sup>
 * ... x<sub>1</sub><sup>m</sup> &nbsp;&nbsp; x<sub>2</sub><sup>1</sup>
 * x<sub>2</sub><sup>2</sup> ... x<sub>2</sub><sup>m</sup> &nbsp;&nbsp; ...
 * &nbsp;&nbsp; x<sub>n</sub><sup>1</sup> x<sub>n</sub><sup>2</sup> ...
 * x<sub>n</sub><sup>m</sup><br/>
 * (first cycles, then instructions)</li>
 * </ul>
 * </li>
 * </ul>
 * The constraints of the model are:
 * <ul>
 * <li>n time constraints</li>
 * <li>n must schedule constraints</li>
 * <li>m issue constraints</li>
 * <li>#edges dependence constraints</li>
 * </ul>
 */
public class InstructionRegionSchedulerILPSolver
{
	///////////////////////////////////////////////////////////////////
	// Constants

	private final static boolean DEBUG = false;
	
	private final static Logger LOGGER = Logger.getLogger (InstructionRegionSchedulerILPSolver.class);
	
	
	///////////////////////////////////////////////////////////////////
	// Inner Types

	// make indexing easier...
	private static class IndexCalc
	{
		private int m_nCyclesCount;
		
		public IndexCalc (int nCyclesCount)
		{
			m_nCyclesCount = nCyclesCount;
		}
		
		public int idx (int nInstructionIdx, int nCycleIdx)
		{
			return 1 + (nCycleIdx - 1) + (nInstructionIdx - 1) * m_nCyclesCount;
		}
	}
	
	/**
	 * <p>The problem to solve:</p>
	 * <pre>
	 * 	min T  (schedule time)
	 * 	st. \sum_{j=1}^m j x_i^j  <=  T  for all i   ("time" constraint)
	 * 		\sum_{j=1}^m x_i^j  =  1     for all i   ("must schedule" constraint)
	 * 		\sum_{i=1}^n  <=  r          for all j   ("issue" constraint, r is issue rate)
	 * 		\sum_{j=1}^m j x_k^j + L_{ki}  <=  \sum_{j=1}^m j x_i^j   for all  edges (i,k)   ("dependence" constraint)
	 * </pre>
	 * <b>Notes:</b>
	 * <ul>
	 * 	<li>m is the number of cycles</li>
	 * 	<li>n is the number of instructions</li>
	 * 	<li>the decision variable x_i^j is 1 if the instruction i is scheduled in cycle j, and 0 otherwise</li>
	 * 	<li>\sum_{j=1}^m j x_i^j is the cycle in which instruction i is scheduled</li>
	 * </ul>
	 * <b>Variables:</b>
	 * <ul>
	 * 	<li>T</li>
	 * 	<li>x_i^j  (1 &lt;= i &lt;= n, 1 &lt;= j &lt;= m)
	 * 		<ul>
	 * 			<li>i: instruction index</li>
	 * 			<li>j: cycle index</li>
	 * 			<li>memory layout: x_1^1 x_1^2 ... x_1^m  x_2^1 x_2^2 ... x_2^m  ...  x_n^1 x_n^2 ... x_n^m<br/>
	 * 				(first cycles, then instructions)</li>
	 * 		</ul>
	 *	</li>
	 * </ul>
	 * <b>Constraints:</b>
	 * <ul>
	 * 	<li>n time constraints</li>
	 * 	<li>n must schedule constraints</li>
	 * 	<li>m issue constraints</li>
	 * 	<li>#edges dependence constraints</li>
	 * </ul>
	 */
	private class Solver
	{
		private final static boolean SOLVE_FULL = false;
		
		private IndexCalc x;

		private int m_nVarsCount;
		
		private int m_nVerticesCount;
		private int m_nCyclesCount;
		
		
		public Solver (int nCyclesCount)
		{
			m_nCyclesCount = nCyclesCount;
			
			m_nVerticesCount = m_graph.getVerticesCount ();
			m_nVarsCount = 1 + m_nVerticesCount * m_nCyclesCount;
			
			x = new IndexCalc (m_nCyclesCount);
		}
		
		/**
		 * 
		 * @param model
		 */
		protected void addTimeConstraints (ILPModel model)
		{
			for (int i = 1; i <= m_nVerticesCount; i++)
			{
				double[] rgCoeffs = new double[m_nVarsCount];
				rgCoeffs[0] = -1;
				for (int j = 1; j <= m_nCyclesCount; j++)
					rgCoeffs[x.idx (i, j)] = j;
				model.addConstraint (rgCoeffs, ILPModel.EOperator.LE, 0);
			}
		}
		
		/**
		 * 
		 * @param model
		 */
		protected void addMustScheduleConstraints (ILPModel model)
		{
			int i = 1;
			for (DAGraph.Vertex v : m_graph.getVertices ())
			{
				// the constraint is redundant if the instruction has a one-cycle scheduling range
				if (SOLVE_FULL || v.getLowerScheduleBound () != v.getUpperScheduleBound ())
				{
					double[] rgCoeffs = new double[m_nVarsCount];
					for (int j = 1; j <= m_nCyclesCount; j++)
						rgCoeffs[x.idx (i, j)] = 1;
					model.addConstraint (rgCoeffs, ILPModel.EOperator.EQ, 1);
				}
				i++;
			}
		}
		
		/**
		 * 
		 * @param model
		 */
		protected void addIssueConstraints (ILPModel model)
		{
			for (int j = 1; j <= m_nCyclesCount; j++)
			{
				double[] rgCoeffs = new double[m_nVarsCount];
				for (int i = 1; i <= m_nVerticesCount; i++)
					rgCoeffs[x.idx (i, j)] = 1;
				model.addConstraint (rgCoeffs, ILPModel.EOperator.LE, m_nIssueRate);
			}
		}
		
		protected void addReducedDependenceConstraints (ILPModel model)
		{
			for (DAGraph.Edge e : m_graph.getEdges ())
			{
				final int b = Math.min (e.getTailVertex ().getUpperScheduleBound (), m_nCyclesCount);
				final int c = e.getHeadVertex ().getLowerScheduleBound ();
				final int L = e.getLatency ();
				
				// the dependence constraint between instructions j,k is redundant if L_{jk} + ubnd(j) <= lbnd(k)
				if (L + b > c)
				{
					final int M = b + L - c;
					
					// find the indices of the tail and head vertices
					int k = m_mapVertexToILPIdx.get (e.getTailVertex ());
					int i = m_mapVertexToILPIdx.get (e.getHeadVertex ());
					
					if (DEBUG)
						System.out.println (StringUtil.concat ("Dep Constr between ", e.getTailVertex ().toString (), " [Idx ", k, "] and ", e.getHeadVertex ().toString (), " [Idx ", i, "]"));

					// fill in the coefficients
					double[] rgCoeffs = new double[m_nVarsCount];
					for (int j = c - L + 1; j <= b; j++)
						rgCoeffs[x.idx (k, j)] = j + L - c;
					for (int j = c; j <= Math.min (b + L - 1, m_nCyclesCount); j++)
						rgCoeffs[x.idx (i, j)] = M - j + c;
					
					model.addConstraint (rgCoeffs, ILPModel.EOperator.LE, M);
				}
				else if (DEBUG)
					System.out.println (StringUtil.concat ("Omitted Dep Constr between ", e.getTailVertex ().toString (), " and ", e.getHeadVertex ().toString ()));
			}
		}
		
		/**
		 * 
		 * @param model
		 */
		protected void addFullDependenceConstraints (ILPModel model)
		{
			for (DAGraph.Edge e : m_graph.getEdges ())
			{
				double[] rgCoeffs = new double[m_nVarsCount];
				for (int j = 1; j <= m_nCyclesCount; j++)
				{
					rgCoeffs[x.idx (m_mapVertexToILPIdx.get (e.getTailVertex ()), j)] = j;
					rgCoeffs[x.idx (m_mapVertexToILPIdx.get (e.getHeadVertex ()), j)] = -j;
				}
				
				model.addConstraint (rgCoeffs, ILPModel.EOperator.LE, -e.getLatency ());
			}
		}
		
		/**
		 * Instructions must be scheduled within the predetermined upper and lower bounds on the cycle numbers.
		 * @param model
		 */
		protected void addBoundsConstraints (ILPModel model)
		{
			for (int i = 1; i <= m_nVerticesCount; i++)
			{
				double[] rgCoeffs = new double[m_nVarsCount];
				
				int nLbnd = m_rgILPIdxToVertex[i].getLowerScheduleBound ();
				int nUbnd = Math.min (m_rgILPIdxToVertex[i].getUpperScheduleBound (), m_nCyclesCount);
				
				// two equivalent formulations:
				// - the sum of all vars outside the bounds must be 0
				// - the sum of all vars within the bounds must be 1
				
				// pick the one which generates "more sparsity"
				if (nLbnd - 1 + m_nCyclesCount - nUbnd - 1 < nUbnd - nLbnd + 1)
				{
					for (int j = 1; j < nLbnd; j++)
						rgCoeffs[x.idx (i, j)] = 1;
					for (int j = nUbnd + 1; j <= m_nCyclesCount; j++)
						rgCoeffs[x.idx (i, j)] = 1;
	
					model.addConstraint (rgCoeffs, ILPModel.EOperator.EQ, 0);
				}
				else
				{
					for (int j = nLbnd; j <= nUbnd; j++)
						rgCoeffs[x.idx (i, j)] = 1;
					model.addConstraint (rgCoeffs, ILPModel.EOperator.EQ, 1);
				}
			}
		}
		
		/**
		 * 
		 * @param model
		 * @param solution
		 */
		protected void addDependenceCuts (ILPModel model, ILPSolution solution)
		{
			for (DAGraph.Edge e : m_graph.getEdges ())
			{
				int j = m_mapVertexToILPIdx.get (e.getTailVertex ());
				int k = m_mapVertexToILPIdx.get (e.getHeadVertex ());
	
				double[] rgCoeffs = new double[m_nVarsCount + 1];
				
				// find c, the last cycle in which the head vertex k is scheduled
				int c = 0;
				for (int i = 1; i <= m_nCyclesCount; i++)
					if (solution.getSolution ()[x.idx (k, i) - 1] > 0)
						c = i;
				if (c == 0)
					continue;
				
				boolean bAddConstraint = false;
				
				for (int i = e.getTailVertex ().getLowerScheduleBound (); i <= c && i <= m_nCyclesCount; i++)
				{
					rgCoeffs[x.idx (j, i)] = 1;
					bAddConstraint = true;
				}
				for (int i = e.getHeadVertex ().getLowerScheduleBound (); i <= c + e.getLatency () && i <= m_nCyclesCount; i++)
				{
					rgCoeffs[x.idx (k, i)] = -1;
					bAddConstraint = true;
				}
				
				if (bAddConstraint)
					model.addConstraint (rgCoeffs, ILPModel.EOperator.GE, 0);
			}
		}
		
		/**
		 * 
		 * @param model
		 * @param solution
		 */
		protected void addSpreadingCuts (ILPModel model, ILPSolution solution)
		{
			for (DAGraph.Vertex v : m_graph.getVertices ())
			{
				int k = m_mapVertexToILPIdx.get (v);
				
				// find the maximum latency
				int l = 0;
				for (DAGraph.Edge e : m_graph.getIncomingEdges (v))
					l = Math.max (l, e.getLatency ());
				if (l == 0)
					continue;
				
				double[] rgCoeffs = new double[m_nVarsCount + 1];
				
				// find c, the lowest cycle in which a fraction of k is scheduled
				int c = 0;
				for (int i = 1; i <= m_nCyclesCount; i++)
					if (solution.getSolution ()[x.idx (k, i) - 1] > 0)
					{
						c = i;
						break;
					}
				if (c == 0)
					continue;
				
				for (int i = c - l + 1; i <= c; i++)
					rgCoeffs[x.idx (k, i)] = 1;
				for (DAGraph.Edge e : m_graph.getIncomingEdges (v))
					if (c - l + 1 >= 1 && c - l + 1 <= m_nCyclesCount)
						rgCoeffs[x.idx (m_mapVertexToILPIdx.get (e.getTailVertex ()), c - l + 1)] = 1;
				
				model.addConstraint (rgCoeffs, ILPModel.EOperator.LE, 1);
			}
		}
		
		/**
		 * 
		 * @param model
		 */
		protected void setVariableBounds (ILPModel model)
		{
			model.setVariableType (0, 0.0, new Double (m_nCyclesCount), false);
			for (int i = 1; i < m_nVarsCount; i++)
				model.setVariableBinary (i);
		}
		
		/**
		 * 
		 * @param model
		 */
		protected void setObjective (ILPModel model)
		{
			double[] rgObj = new double[m_nVarsCount];
			rgObj[0] = 1;
			for (int i = 1; i < m_nVarsCount; i++)
				rgObj[i] = 0;
			model.setObjective (rgObj);			
		}
		
		/**
		 * 
		 * @return
		 */
		public int solve (InstructionList il)
		{
			ILPModel model = new ILPModel (m_nVarsCount);

			// build the model
			
			addTimeConstraints (model);
			addMustScheduleConstraints (model);
			addIssueConstraints (model);
			
			if (SOLVE_FULL)
				addFullDependenceConstraints (model);
			else
			{
				addReducedDependenceConstraints (model);
				addBoundsConstraints (model);
			}
			
			setVariableBounds (model);
			setObjective (model);
			
			
			// solve the problem
			
			LOGGER.info (StringUtil.concat ("Solving ILP (has ", model.getVariablesCount (), " variables and ", model.getConstraintsCount (), " constraints)..."));
			if (DEBUG)
				model.writeMPS (StringUtil.concat ("scheduling-cbc_", new Date ().toString ().replaceAll (":", ".").replaceAll (" ", "_"), ".mps"));

			// invoke the external solver
			ILPSolution solution = ILPSolver.getInstance ().solve (model, 30);
			boolean bOptimalSolutionFound = solution.getStatus ().equals (ILPSolution.ESolutionStatus.OPTIMAL);
			
			if (bOptimalSolutionFound)
			{
				if (DEBUG)
					printSolution (solution);
				
				buildSchedule (solution, il);
				
				if (DEBUG)
					System.out.println (il);
			}
			
			model.delete ();
			
			return bOptimalSolutionFound ? (int) solution.getObjective () : -1;
		}
		
		/**
		 * 
		 * @param solution
		 */
		public void printSolution (ILPSolution solution)
		{
			System.out.println (solution.getObjective ());
			System.out.println (Arrays.toString (solution.getSolution ()));
			
			System.out.println ("Decision variables != 0:");
			long nTotalPossibilities = 1;
			for (int i = 1; i <= m_nVerticesCount; i++)
			{
				int nPossibilities = 0;
				for (int j = 1; j <= m_nCyclesCount; j++)
					if (solution.getSolution ()[x.idx (i, j)] > 0)
					{
						boolean bInBounds = m_rgILPIdxToVertex[i].getLowerScheduleBound () <= j && j <= m_rgILPIdxToVertex[i].getUpperScheduleBound ();
						if (bInBounds)
							nPossibilities++;
						System.out.println (StringUtil.concat ((bInBounds ? "* " : "  "),
							"x[instr=", i, ", cycle=", j, ", y=", solution.getSolution ()[x.idx (i, j)], "] :: ", m_rgILPIdxToVertex[i]));
					}
				
				if (nPossibilities == 0)
					System.out.println (StringUtil.concat ("! ", m_rgILPIdxToVertex[i], " cannot be scheduled within its range!"));
				nTotalPossibilities *= nPossibilities;
			}
			
			System.out.println (StringUtil.concat (nTotalPossibilities, " possibilities."));
		}
		
		/**
		 * 
		 * @param solution
		 * @param ilOut
		 */
		public void buildSchedule (ILPSolution solution, InstructionList ilOut)
		{
			// build the schedule from the solution of the ILP
			for (int i = 1; i <= m_nVerticesCount; i++)
				for (int j = 1; j <= m_nCyclesCount; j++)
					if (solution.getSolution ()[x.idx (i, j)] > 0)
						m_rgILPIdxToVertex[i].setScheduleBounds (j, j);

			reconstructInstructionList (ilOut);

			// discard all bounds
			for (DAGraph.Vertex v : m_graph.getVertices ())
				v.discardBounds ();
		}
	}

	
	///////////////////////////////////////////////////////////////////
	// Member Variables
	
	private DAGraph m_graph;
	private int m_nIssueRate;

	private Map<DAGraph.Vertex, Integer> m_mapVertexToILPIdx;
	private DAGraph.Vertex[] m_rgILPIdxToVertex;

	
	///////////////////////////////////////////////////////////////////
	// Implementation
	
	public InstructionRegionSchedulerILPSolver (DAGraph graph, int nIssueRate)
	{
		m_graph = graph;
		m_nIssueRate = nIssueRate;

		m_mapVertexToILPIdx = new HashMap<> ();
		m_rgILPIdxToVertex = new DAGraph.Vertex[graph.getVerticesCount () + 1];
		int i = 1;
		for (DAGraph.Vertex v : graph.getVertices ())
		{
			m_mapVertexToILPIdx.put (v, i);
			m_rgILPIdxToVertex[i] = v;
			i++;
		}
	}

	protected int solve (final int nCyclesCount, InstructionList ilOut)
	{
		Solver solver = new Solver (nCyclesCount);
		return solver.solve (ilOut);
	}

	/**
	 * Reconstruct the instruction list from the DAG if all instructions have
	 * one-cycle schedule bounds.
	 * 
	 * @param ilOut
	 *            The output instruction list
	 */
	public void reconstructInstructionList (InstructionList ilOut)
	{
		List<DAGraph.Vertex> listVertices = new ArrayList<> ();
		for (DAGraph.Vertex v : m_graph.getVertices ())
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
