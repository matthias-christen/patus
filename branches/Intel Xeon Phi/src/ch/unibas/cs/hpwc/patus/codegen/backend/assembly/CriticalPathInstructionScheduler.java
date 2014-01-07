package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

import ch.unibas.cs.hpwc.patus.arch.IArchitectureDescription;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.analyze.DAGraph;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.analyze.DAGraph.Vertex;
import ch.unibas.cs.hpwc.patus.graph.algorithm.CriticalPathLengthCalculator;
import ch.unibas.cs.hpwc.patus.graph.algorithm.GraphUtil;

public class CriticalPathInstructionScheduler extends AbstractInstructionScheduler
{
	///////////////////////////////////////////////////////////////////
	// Member Variables

	
	///////////////////////////////////////////////////////////////////
	// Implementation

	public CriticalPathInstructionScheduler (DAGraph graph, CriticalPathLengthCalculator<DAGraph.Vertex, DAGraph.Edge, Integer> cpcalc, IArchitectureDescription arch)
	{
		super (graph, cpcalc, arch);
	}

	/**
	 * <pre>
	 * cycle = 0
	 * ready = leaves of dependence graph G
	 * active = empty
	 * while (ready union active != empty)
	 *   if available remove an instruction from ready based on priority
	 *     add instruction to active
	 *   for each instruction in active
	 *     if completed remove from active
	 *     for each successor of instruction
	 *       if successors operand ready then add to ready
	 * </pre>
	 */
	@Override
	protected int doSchedule (InstructionList ilOut)
	{
		int nCyclesCount = 0;
		
		List<DAGraph.Vertex> listActive = new LinkedList<> ();
		List<DAGraph.Vertex> listToRemoveFromActive = new LinkedList<> ();
		Map<DAGraph.Vertex, Integer> mapInstructionStartCycles = new HashMap<> ();
		
		final Collection<DAGraph.Vertex> collLeaves = GraphUtil.getLeafVertices (getGraph ());

		// create a priority queue of vertices (instructions), prioritizing vertices on the critical path
		PriorityQueue<DAGraph.Vertex> quReady = new PriorityQueue<DAGraph.Vertex> (10, new Comparator<DAGraph.Vertex> ()
		{
			@Override
			public int compare (DAGraph.Vertex v1, DAGraph.Vertex v2)
			{
				// first prioritize by critical path distance
				CriticalPathLengthCalculator<DAGraph.Vertex, DAGraph.Edge, Integer> calc = getCriticalPathLengthCalculator ();
				int nMaxDist1 = 0;
				int nMaxDist2 = 0;
				
				for (DAGraph.Vertex vertLeaf : collLeaves)
				{
					nMaxDist1 = Math.max (nMaxDist1, calc.getCriticalPathDistance (v1, vertLeaf));
					nMaxDist2 = Math.max (nMaxDist2, calc.getCriticalPathDistance (v2, vertLeaf));
				}

				if (nMaxDist1 != nMaxDist2)
					return nMaxDist1 - nMaxDist2;	// check this!

				// prioritize by maximum latency
				return getMaximumLatency (v1) - getMaximumLatency (v2);
			}
		})
		{
			private static final long serialVersionUID = 1L;
			
			@Override
			public boolean offer (Vertex v)
			{
				if (contains (v))
					return false;
				return super.offer (v);
			}
		};
		
		// add leaves to the priority queue
		quReady.addAll (GraphUtil.getRootVertices (getGraph ()));
		
		while (listActive.size () + quReady.size () > 0)
		{
			// remove the next available instruction from the priority queue and schedule it

			// TODO: account for execution unit types
//			for (int i = 0; i < getIssueRate (); i++)
			for (int i = 0; i < getMinExecUnits (); i++)
			{
				DAGraph.Vertex v = quReady.poll ();
				if (v != null)
				{
					listActive.add (v);
					mapInstructionStartCycles.put (v, nCyclesCount);
					ilOut.addInstruction (v.getInstruction ());
				}
				else
					break;
			}
			nCyclesCount++;

			// add successors of any instruction that has completed to the priority queue
			listToRemoveFromActive.clear ();
			for (DAGraph.Vertex w : listActive)
			{
				if (isCompleted (w, nCyclesCount, mapInstructionStartCycles))
				{
					listToRemoveFromActive.add (w);
					
					// add the instruction if it hasn't been issued yet
					// (i.e., if it isn't contained in mapInstructionStartCycles;
					// note that the queue also does not accept duplicate entries)
					for (DAGraph.Edge edge : getGraph ().getOutgoingEdges (w))
						if (!mapInstructionStartCycles.containsKey (edge.getHeadVertex ()))
						{
							// check whether all instructions which this one depends on have completed
							if (allDependentCompleted (edge.getHeadVertex (), nCyclesCount, mapInstructionStartCycles))
								quReady.offer (edge.getHeadVertex ());
						}
				}
			}
			
			// remove completed instructions from the list of active instructions
			for (DAGraph.Vertex w : listToRemoveFromActive)
				listActive.remove (w);
		}

		return nCyclesCount;
	}
	
	/**
	 * Determines whether the instruction in vertex <code>v</code> has already
	 * completed.
	 * 
	 * @param v
	 *            The vertex representing the instruction
	 * @param nCurrentCycle
	 *            The current cycle
	 * @param mapInstructionStartCycles
	 *            A map mapping scheduled instructions to their starting cycle
	 * @return <code>true</code> if the instruction has completed
	 */
	private boolean isCompleted (DAGraph.Vertex v, int nCurrentCycle, Map<DAGraph.Vertex, Integer> mapInstructionStartCycles)
	{
		Integer nStartCycle = mapInstructionStartCycles.get (v);
		if (nStartCycle == null)
			return false;
		
		return nCurrentCycle >= nStartCycle + getMaximumLatency (v);
	}
	
	private boolean allDependentCompleted (DAGraph.Vertex v, int nCurrentCycle, Map<DAGraph.Vertex, Integer> mapInstructionStartCycles)
	{
		for (DAGraph.Edge e : getGraph ().getIncomingEdges (v))
			if (!isCompleted (e.getTailVertex (), nCurrentCycle, mapInstructionStartCycles))
				return false;
		return true;
	}
	
	private int getMaximumLatency (DAGraph.Vertex v)
	{
		int nMaxLatency = 0;
		for (DAGraph.Edge e : getGraph ().getOutgoingEdges (v))
			nMaxLatency = Math.max (nMaxLatency, e.getLatency ());

		return nMaxLatency;
	}
}
