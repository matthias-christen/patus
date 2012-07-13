package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import java.util.Collection;
import java.util.LinkedList;
import java.util.List;

import ch.unibas.cs.hpwc.patus.arch.IArchitectureDescription;
import ch.unibas.cs.hpwc.patus.arch.TypeArchitectureType.Intrinsics.Intrinsic;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.analyze.DAGraph;
import ch.unibas.cs.hpwc.patus.graph.algorithm.CriticalPathLengthCalculator;
import ch.unibas.cs.hpwc.patus.graph.algorithm.GraphUtil;
import ch.unibas.cs.hpwc.patus.util.MathUtil;

/**
 * 
 * @author Matthias-M. Christen
 */
public abstract class AbstractInstructionScheduler
{
	///////////////////////////////////////////////////////////////////
	// Member Variables
	
	/**
	 * The architecture description
	 */
	private IArchitectureDescription m_arch;

	/**
	 * The analysis graph
	 */
	private DAGraph m_graph;
	
	private CriticalPathLengthCalculator<DAGraph.Vertex, DAGraph.Edge, Integer> m_cpcalc;
	
	private CriticalPathInstructionScheduler m_cpsched;
	
	/**
	 * The processor's issue rate as defined in the architecture description
	 */
	private int m_nIssueRate;
	
	private int m_nMinExecUnits;
	
	/**
	 * The lower bound <i>L</i> for the schedule
	 * <dl>
	 * 	<di>L = 1 + max { <i>c</i>, ceil(<i>n</i>/<i>r</i>) - 1 },</di>
	 * </dl>
	 * where <i>c</i> is the length of the DAG's critical path,
	 * <i>n</i> is the number of instructions, and <i>r</i> is the processor's
	 * issue rate.
	 */
	private int m_nLowerScheduleLengthBound;
	
	/**
	 * An upper bound for the length of the schedule, obtained by critical path
	 * scheduling
	 */
	private int m_nUpperScheduleLengthBound;

	private InstructionList m_ilScheduled;
	private int m_nScheduleLength;
	
	
	///////////////////////////////////////////////////////////////////
	// Implementation

	public AbstractInstructionScheduler (DAGraph graph, IArchitectureDescription arch)
	{
		this (graph, null, arch);
	}

	protected AbstractInstructionScheduler (DAGraph graph, CriticalPathLengthCalculator<DAGraph.Vertex, DAGraph.Edge, Integer> cpcalc, IArchitectureDescription arch)
	{
		m_graph = graph;
		m_arch = arch;
		m_nIssueRate = m_arch.getIssueRate ();
		m_nMinExecUnits = m_arch.getMinimumNumberOfExecutionUnitsPerType (AbstractInstructionScheduler.collectIntrinsics (graph, arch));

		m_cpcalc = cpcalc == null ? new CriticalPathLengthCalculator<> (m_graph, Integer.class) : cpcalc;
		m_cpsched = null;

		m_nLowerScheduleLengthBound = -1;
		m_nUpperScheduleLengthBound = -1;
		
		m_ilScheduled = null;
		m_nScheduleLength = -1;
	}
	
	public static Iterable<Intrinsic> collectIntrinsics (DAGraph graph, IArchitectureDescription arch)
	{
		List<Intrinsic> listIntrinsics = new LinkedList<> ();
		for (DAGraph.Vertex v : graph.getVertices ())
		{
			if (v.getInstruction () instanceof Instruction)
			{
				Collection<Intrinsic> intrinsics = arch.getIntrinsicsByIntrinsicName (((Instruction) v.getInstruction ()).getInstructionName ());
				if (intrinsics != null)
					listIntrinsics.addAll (intrinsics);
			}
		}
		
		return listIntrinsics;
	}
	
	/**
	 * Computes the lower bound <i>L</i> on the length of the schedule, which is given by
	 * <dl>
	 * 	<di>L = 1 + max { <i>c</i>, ceil(<i>n</i>/<i>r</i>) - 1 },</di>
	 * </dl>
	 * where <i>c</i> is the length of the DAG's critical path,
	 * <i>n</i> is the number of instructions, and <i>r</i> is the processor's
	 * issue rate.
	 * 
	 * @return The lower bound on the schedule length
	 */
	protected int computeScheduleLengthLowerBound ()
	{
		// compute the lower bound on the schedule length
		int nCriticalPathLength = 0;
		
		Iterable<DAGraph.Vertex> itRoots = GraphUtil.getRootVertices (m_graph);
		Iterable<DAGraph.Vertex> itLeaves = GraphUtil.getLeafVertices (m_graph);
		
		for (DAGraph.Vertex vertRoot : itRoots)
			for (DAGraph.Vertex vertLeaf : itLeaves)
				nCriticalPathLength = Math.max (nCriticalPathLength, m_cpcalc.getCriticalPathDistance (vertRoot, vertLeaf));
		
		return 1 + Math.max (nCriticalPathLength, MathUtil.divCeil (m_graph.getVerticesCount (), m_nIssueRate) - 1);
	}

	/**
	 * Computes an upper bound by performing a critical path list scheduling.
	 * @return An upper bound on the schedule length 
	 */
	protected int computeScheduleLengthUpperBound ()
	{
		if (m_cpsched == null)
		{
			m_cpsched = new CriticalPathInstructionScheduler (m_graph, m_cpcalc, m_arch);
			m_cpsched.schedule ();
		}
		
		return m_cpsched.getScheduleLength ();
	}
	
	public void getCriticalPathSchedule (InstructionList ilOut)
	{
		ilOut.addInstructions (m_cpsched.getSchedule ());
	}
	
	public IArchitectureDescription getArchitectureDescription ()
	{
		return m_arch;
	}
	
	public DAGraph getGraph ()
	{
		return m_graph;
	}
	
	public CriticalPathLengthCalculator<DAGraph.Vertex, DAGraph.Edge, Integer> getCriticalPathLengthCalculator ()
	{
		return m_cpcalc;
	}
	
	public int getIssueRate ()
	{
		return m_nIssueRate;
	}
	
	public int getMinExecUnits ()
	{
		return m_nMinExecUnits;
	}
	
	public int getLowerScheduleLengthBound ()
	{
		if (m_nLowerScheduleLengthBound == -1)
			m_nLowerScheduleLengthBound = computeScheduleLengthLowerBound ();
		return m_nLowerScheduleLengthBound;
	}
	
	public int getUpperScheduleLengthBound ()
	{
		if (m_nUpperScheduleLengthBound == -1)
			m_nUpperScheduleLengthBound = computeScheduleLengthUpperBound ();
		return m_nUpperScheduleLengthBound;
	}
	
	public InstructionList schedule ()
	{
		if (m_ilScheduled == null)
		{
			m_ilScheduled = new InstructionList ();
			m_nScheduleLength = doSchedule (m_ilScheduled);
		}
		
		return m_ilScheduled;
	}
	
	public InstructionList getSchedule ()
	{
		return m_ilScheduled;
	}
	
	/**
	 * Get the length of the schedule in number of cycles.
	 * @return
	 */
	public int getScheduleLength ()
	{
		if (m_nScheduleLength == -1)
			schedule ();
		return m_nScheduleLength;
	}
	
	/**
	 * Performs the scheduling.
	 * 
	 * @param ilOutput
	 *            The instruction list to which the scheduled instructions
	 *            (corresponding to the input {@link DAGraph}) are added by the
	 *            concrete scheduler implementation
	 * @return The length in cycles of the generated schedule
	 */
	protected abstract int doSchedule (InstructionList ilOutput);
}
