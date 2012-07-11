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

import ch.unibas.cs.hpwc.patus.arch.IArchitectureDescription;
import ch.unibas.cs.hpwc.patus.arch.TypeExecUnitType;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.analyze.DAGraph;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.analyze.DAGraph.Edge;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.analyze.DAGraph.Vertex;
import ch.unibas.cs.hpwc.patus.ilp.ILPModel;
import ch.unibas.cs.hpwc.patus.ilp.ILPSolution;
import ch.unibas.cs.hpwc.patus.ilp.ILPSolver;
import ch.unibas.cs.hpwc.patus.util.IParallelOperation;
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
 * (first execution units, then cycles, then instructions)</li>
 * </ul>
 * </li>
 * </ul>
 * The constraints of the model are:
 * <ul>
 * <li>n time constraints</li>
 * <li>n must schedule constraints</li>
 * <li> resource constraints</li>
 * <li>m issue constraints</li>
 * <li>#edges dependence constraints</li>
 * </ul>
 */
public class InstructionRegionSchedulerILPSolver2
{
	///////////////////////////////////////////////////////////////////
	// Constants

	private final static boolean DEBUG = false;
	
	private final static Logger LOGGER = Logger.getLogger (InstructionRegionSchedulerILPSolver2.class);
	
	
	///////////////////////////////////////////////////////////////////
	// Inner Types

	// make indexing easier...
	private static class IndexCalc
	{
		private int m_nExecUnitsCount;
		private int m_nCyclesCount;
		private int m_nTotalVarsCount;
		
		public IndexCalc (int nCyclesCount, int nExecUnitsCount, int nTotalVarsCount)
		{
			m_nCyclesCount = nCyclesCount;
			m_nExecUnitsCount = nExecUnitsCount;
			m_nTotalVarsCount = nTotalVarsCount;
		}
		
		public int idx (int nInstructionIdx, int nCycleIdx, int nExecUnitIndex)
		{
			int nIdx = 1 + (nExecUnitIndex - 1) + m_nExecUnitsCount * ((nCycleIdx - 1) + m_nCyclesCount * (nInstructionIdx - 1));
			
			if (nIdx >= m_nTotalVarsCount)
				throw new RuntimeException (StringUtil.concat ("Index out of bounds: instr=", nInstructionIdx, ", cycle=", nCycleIdx, ", execunit=", nExecUnitIndex));
			
			return nIdx;
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
		/**
		 * Determine whether to use bounds constraints (i.e., add constraints
		 * that constrain the lower and upper scheduling bounds of
		 * instructions).
		 * If no bounds constraints are added, the solution needs to be checked
		 * whether it is a valid solution, but if bounds constraints are added,
		 * the ILP solving time increases.
		 */
		private static final boolean USE_BOUNDS_CONSTRAINTS = false;

		
		private IndexCalc x;

		private int m_nVarsCount;
		
		private int m_nVerticesCount;
		private int m_nCyclesCount;
		private int m_nExecUnitsCount;
		
		
		public Solver (int nCyclesCount, int nExecUnitsCount)
		{
			m_nCyclesCount = nCyclesCount;
			m_nExecUnitsCount = nExecUnitsCount;
			
			m_nVerticesCount = m_graph.getVerticesCount ();
			m_nVarsCount = 1 + m_nVerticesCount * m_nCyclesCount * m_nExecUnitsCount;
			
			x = new IndexCalc (m_nCyclesCount, m_nExecUnitsCount, m_nVarsCount);
		}
		
		/**
		 * 
		 * @param model The ILP model to which the constraints are added
		 */
		protected void addTimeConstraints (final ILPModel model)
		{
			m_graph.forAllVertices (new IParallelOperation<DAGraph.Vertex> ()
			{
				public void perform (DAGraph.Vertex v)
				{
					int i = m_mapVertexToILPIdx.get (v);
					
					double[] rgCoeffs = new double[m_nVarsCount];
					rgCoeffs[0] = -1;
					
					for (int j = 1; j <= m_nCyclesCount; j++)
						for (int k = 1; k <= m_nExecUnitsCount; k++)
							rgCoeffs[x.idx (i, j, k)] = j;
					
					model.addConstraint (rgCoeffs, ILPModel.EOperator.LE, 0);
				}
			});
		}
		
		/**
		 * 
		 * @param model The ILP model to which the constraints are added
		 */
		protected void addMustScheduleConstraints (final ILPModel model)
		{
			m_graph.forAllVertices (new IParallelOperation<DAGraph.Vertex> ()
			{
				public void perform (DAGraph.Vertex v)
				{
					if (v.getExecUnitTypes () == null)
						throw new RuntimeException (StringUtil.concat ("No execution unit type for vertex ", v.toString ()));

					int i = m_mapVertexToILPIdx.get (v);
					
					double[] rgCoeffs = new double[m_nVarsCount];
					int nExecUnitTypesCode = 0;
					boolean bHasMultipleExecUnitTypes = v.getExecUnitTypes ().size () > 1;
					
					for (TypeExecUnitType eu : v.getExecUnitTypes ())
					{
						Integer k = m_mapExecUnitTypeToILPIdx.get (eu);
						if (k == null)
							throw new RuntimeException (StringUtil.concat ("No execution unit type for type ", eu.getName (), " in vertex ", v.toString ()));
							
						for (int j = v.getLowerScheduleBound (); j <= v.getUpperScheduleBound (); j++)
							rgCoeffs[x.idx (i, j, k)] = bHasMultipleExecUnitTypes ? 1 << (k - 1) : 1;
						
						nExecUnitTypesCode += (1 << (k - 1));
					}
										
					model.addConstraint (rgCoeffs, ILPModel.EOperator.EQ, bHasMultipleExecUnitTypes ? nExecUnitTypesCode : 1);					
				}
			});			
		}
		
		/**
		 * 
		 * @param model The ILP model to which the constraints are added
		 */
		protected void addResourceConstraints (final ILPModel model)
		{
			for (int k = 1; k <= m_nExecUnitsCount; k++)
			{
				for (int j = 1; j <= m_nCyclesCount; j++)
				{
					double[] rgCoeffs = new double[m_nVarsCount];
					
					for (DAGraph.Vertex v : m_graph.getVertices ())
					{
						if (v.getExecUnitTypes ().contains (m_rgILPIdxToExecUnitType[k]))
							rgCoeffs[x.idx (m_mapVertexToILPIdx.get (v), j, k)] = 1;
					}
					
					model.addConstraint (rgCoeffs, ILPModel.EOperator.LE, m_rgILPIdxToExecUnitType[k].getQuantity ().intValue ());
				}
			}
		}
		
		/**
		 * 
		 * @param model The ILP model to which the constraints are added
		 */
		protected void addIssueConstraints (final ILPModel model)
		{
			for (int j = 1; j <= m_nCyclesCount; j++)
			{
				double[] rgCoeffs = new double[m_nVarsCount];

				for (int k = 1; k <= m_nExecUnitsCount; k++)
					for (int i = 1; i <= m_nVerticesCount; i++)
						rgCoeffs[x.idx (i, j, k)] = 1;
				
				model.addConstraint (rgCoeffs, ILPModel.EOperator.LE, m_nIssueRate);
			}
		}
		
		/**
		 * 
		 * @param model The ILP model to which the constraints are added
		 */
		protected void addDependenceConstraints (final ILPModel model)
		{
			m_graph.forAllEdges (new IParallelOperation<DAGraph.Edge> ()
			{
				@Override
				public void perform (Edge e)
				{
					if (e.getTailVertex ().getUpperScheduleBound () + e.getLatency () - 1 < e.getHeadVertex ().getLowerScheduleBound ())
						return;
					
					final int t_min = Math.max (e.getTailVertex ().getLowerScheduleBound () + e.getLatency () - 1, e.getHeadVertex ().getLowerScheduleBound ());
					final int t_max = Math.min (e.getTailVertex ().getUpperScheduleBound () + e.getLatency () - 1, e.getHeadVertex ().getUpperScheduleBound ());
					
					for (int t = t_min; t <= t_max; t++)
					{
						double[] rgCoeffs = new double[m_nVarsCount];
						
						for (int k = 1; k <= m_nExecUnitsCount; k++)
						{
							for (int tn = e.getHeadVertex ().getLowerScheduleBound (); tn <= Math.min (t, e.getHeadVertex ().getUpperScheduleBound ()); tn++)
								rgCoeffs[x.idx (m_mapVertexToILPIdx.get (e.getHeadVertex ()), tn, k)] = 1;
							for (int tm = t - e.getLatency () + 1; tm <= e.getTailVertex ().getUpperScheduleBound (); tm++)
								rgCoeffs[x.idx (m_mapVertexToILPIdx.get (e.getTailVertex ()), tm, k)] = 1;
						}
						
						model.addConstraint (rgCoeffs, ILPModel.EOperator.LE, 1);
					}
				}
			});
		}
				
		/**
		 * Instructions must be scheduled within the predetermined upper and
		 * lower bounds on the cycle numbers.
		 * 
		 * @param model
		 *            The ILP model to which the constraints are added
		 */
		protected void addBoundsConstraints (final ILPModel model)
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
					for (int k = 1; k <= m_nExecUnitsCount; k++)
					{
						for (int j = 1; j < nLbnd; j++)
							rgCoeffs[x.idx (i, j, k)] = 1;
						for (int j = nUbnd + 1; j <= m_nCyclesCount; j++)
							rgCoeffs[x.idx (i, j, k)] = 1;
					}
					
					model.addConstraint (rgCoeffs, ILPModel.EOperator.EQ, 0);
				}
				else
				{
					for (int j = nLbnd; j <= nUbnd; j++)
						for (int k = 1; k <= m_nExecUnitsCount; k++)
							rgCoeffs[x.idx (i, j, k)] = 1;
					model.addConstraint (rgCoeffs, ILPModel.EOperator.EQ, 1);
				}
			}
		}		
		
		/**
		 * 
		 * @param model
		 */
		protected void setVariableBounds (final ILPModel model)
		{
			model.setVariableType (0, 0.0, new Double (m_nCyclesCount), false);
			for (int i = 1; i < m_nVarsCount; i++)
				model.setVariableBinary (i);
		}
		
		/**
		 * 
		 * @param model
		 */
		protected void setObjective (final ILPModel model)
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
			if (DEBUG)
				m_graph.graphviz ();
			
			ILPModel model = new ILPModel (m_nVarsCount);

			// build the model
			
			addTimeConstraints (model);
			addMustScheduleConstraints (model);
			addResourceConstraints (model);
			addIssueConstraints (model);
			addDependenceConstraints (model);
			
			if (USE_BOUNDS_CONSTRAINTS)
				addBoundsConstraints (model);
			
			setVariableBounds (model);
			setObjective (model);
			
			
			// solve the problem
			
			LOGGER.info (StringUtil.concat ("Solving ILP (has ", model.getVariablesCount (), " variables and ", model.getConstraintsCount (), " constraints)..."));
			if (DEBUG)
				model.writeMPS (StringUtil.concat ("scheduling-cbc_", new Date ().toString ().replaceAll (":", ".").replaceAll (" ", "_"), ".mps"));

			// invoke the external solver
			ILPSolution solution = ILPSolver.getInstance ().solve (model, 30);
			boolean bOptimalSolutionFound = solution.getStatus ().equals (ILPSolution.ESolutionStatus.OPTIMAL);
			
			if (!USE_BOUNDS_CONSTRAINTS)
				bOptimalSolutionFound &= checkSolution (solution);
			
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
		 * Checks whether the solution is valid, i.e., all instructions are
		 * scheduled within their lower and upper bounds.
		 * 
		 * @param solution
		 *            The solution to check
		 * @return <code>true</code> iff the solution is valid
		 */
		private boolean checkSolution (ILPSolution solution)
		{
			for (int i = 1; i <= m_nVerticesCount; i++)
			{
				for (int j = 1; j <= m_nCyclesCount; j++)
				{
					for (int k = 1; k <= m_nExecUnitsCount; k++)
						if (solution.getSolution ()[x.idx (i, j, k)] > 0)
						{
							boolean bInBounds = m_rgILPIdxToVertex[i].getLowerScheduleBound () <= j && j <= m_rgILPIdxToVertex[i].getUpperScheduleBound ();
							if (!bInBounds)
								return false;
						}
				}
			}
			
			return true;
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
				{
					for (int k = 1; k <= m_nExecUnitsCount; k++)
						if (solution.getSolution ()[x.idx (i, j, k)] > 0)
						{
							boolean bInBounds = m_rgILPIdxToVertex[i].getLowerScheduleBound () <= j && j <= m_rgILPIdxToVertex[i].getUpperScheduleBound ();
							if (bInBounds)
								nPossibilities++;
							System.out.println (StringUtil.concat ((bInBounds ? "* " : "  "),
								"x[instr=", i, ", cycle=", j, ", unit=", k, ", y=", solution.getSolution ()[x.idx (i, j, k)], "] :: ", m_rgILPIdxToVertex[i]));
						}
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
					for (int k = 1; k <= m_nExecUnitsCount; k++)
						if (solution.getSolution ()[x.idx (i, j, k)] > 0)
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
	private IArchitectureDescription m_arch;
	
	private int m_nIssueRate;

	private Map<DAGraph.Vertex, Integer> m_mapVertexToILPIdx;
	private DAGraph.Vertex[] m_rgILPIdxToVertex;
	
	private Map<TypeExecUnitType, Integer> m_mapExecUnitTypeToILPIdx;
	private TypeExecUnitType[] m_rgILPIdxToExecUnitType;

	
	///////////////////////////////////////////////////////////////////
	// Implementation
	
	public InstructionRegionSchedulerILPSolver2 (DAGraph graph, IArchitectureDescription arch)
	{
		m_graph = graph;
		m_arch = arch;
		m_nIssueRate = m_arch.getIssueRate ();

		m_mapVertexToILPIdx = new HashMap<> ();
		m_rgILPIdxToVertex = new DAGraph.Vertex[graph.getVerticesCount () + 1];
		int i = 1;
		for (DAGraph.Vertex v : graph.getVertices ())
		{
			m_mapVertexToILPIdx.put (v, i);
			m_rgILPIdxToVertex[i] = v;
			i++;
		}
		
		m_mapExecUnitTypeToILPIdx = new HashMap<> ();
		m_rgILPIdxToExecUnitType = new TypeExecUnitType[m_arch.getExecutionUnitTypesCount () + 1];
		if (m_arch.getAssemblySpec () != null)
		{
			i = 1;
			for (TypeExecUnitType type : m_arch.getAssemblySpec ().getExecUnitTypes ().getExecUnitType ())
			{
				m_mapExecUnitTypeToILPIdx.put (type, i);
				m_rgILPIdxToExecUnitType[i] = type;
				i++;
			}
		}
	}

	protected int solve (final int nCyclesCount, InstructionList ilOut)
	{
		int nExecUnitsCount = 1;
		if (m_arch.getAssemblySpec () != null && m_arch.getAssemblySpec ().getExecUnitTypes () != null)
			nExecUnitsCount = m_arch.getAssemblySpec ().getExecUnitTypes ().getExecUnitType ().size ();
			
		Solver solver = new Solver (nCyclesCount, nExecUnitsCount);
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
