package cetus.analysis;

import cetus.hir.*;
import java.util.*;

/**
 * Data-dependence Graph to store the result of dependence testing
 *
 */
public class DDGraph 
{
	public static final boolean summarize = true;
	public static final boolean not_summarize = false;
	
	private boolean summarized_status;
	
	private ArrayList<Arc> depArcs;
	
	static public class Arc
	{
		/* source node */
		private DDArrayAccessInfo source;
		/* sink node */
		private DDArrayAccessInfo sink;
		/* byte depType
		 * 1 - Flow (True) Dependence
		 * 2 - Anti Dependence
		 * 3 - Output Dependence
		 * 4 - Input Dependence
		 */
		private byte depType;
		/* single direction vector from source to sink */
		DependenceVector depVector;
		
		/**
		 * Creates a dependence arc from source expr1 to sink expr2 with the relevant direction vector
		 * @param expr1 - Contains all information related to source array access
		 * @param expr2 - Contains all information related to sink array access
		 * @param depVector
		 */
		public Arc(DDArrayAccessInfo expr1, DDArrayAccessInfo expr2, DependenceVector depVector)
		{
			
			if (depVector.plausibleVector())
			{
				this.source = expr1;
				this.sink = expr2;
				this.depVector = depVector;
				setDependenceType(expr1.getAccessType(), expr2.getAccessType());
			}
			else
			{
				this.source = expr2;
				this.sink = expr1;
				this.depVector = depVector.reverseVector();
				setDependenceType(expr2.getAccessType(), expr1.getAccessType());
			}
		}
		
		public Arc(Arc a)
		{
			this.source = a.source;
			this.sink = a.sink;
			this.depVector = new DependenceVector(a.depVector);
			this.depType = a.depType;
		}
		
		public DDArrayAccessInfo getSource()
		{
			return this.source;
		}
		
		public DDArrayAccessInfo getSink()
		{
			return this.sink;
		}
		
		public Statement getSourceStatement()
		{
			return this.source.getParentStatement();
		}
		
		public Statement getSinkStatement()
		{
			return this.sink.getParentStatement();
		}
		
		public byte getDependenceType()
		{
			return this.depType;
		}
		
		public DependenceVector getDependenceVector()
		{
			return this.depVector;
		}
		
		public void setDependenceType(int source_type, int sink_type)
		{
			if (source_type == DDArrayAccessInfo.write_type)
			{
				/* Output Dependence */
				if (sink_type == DDArrayAccessInfo.write_type)
					this.depType = 3;
				/* Flow Dependence */
				else if (sink_type == DDArrayAccessInfo.read_type)
					this.depType = 1;
			}
			else if(source_type == DDArrayAccessInfo.read_type)
			{
				/* Anti Dependence */
				if(sink_type == DDArrayAccessInfo.write_type)
					this.depType = 2;
				/* Don't test for Input Dependence */
			}
		}
		
		public boolean belongsToLoop(Loop loop)
		{
			boolean ret_val = false;
			boolean contains_source = false;
			boolean contains_sink = false;
			
			LinkedList<Loop> source_loop_nest = LoopTools.calculateLoopNest(this.source.getAccessLoop());
			LinkedList<Loop> sink_loop_nest = LoopTools.calculateLoopNest(this.sink.getAccessLoop());
			
			for (Loop l : source_loop_nest)
			{
				if (l.equals(loop))
					contains_source = true;
			}
			
			for (Loop l : sink_loop_nest)
			{
				if (l.equals(loop))
					contains_sink = true;
			}
			
			if ((contains_source == true) && (contains_sink ==true))
				ret_val = true;
			
			return ret_val;
		}
		
		public boolean isCarried(Loop l)
		{
			boolean is_carried_for_loop = false;
			DependenceVector dv = this.getDependenceVector();
			
			if ((dv.getDirectionVector()).containsKey(l))
			{
				int direction_for_l = dv.getDirection(l);
				if ((direction_for_l == DependenceVector.any) ||
						(direction_for_l == DependenceVector.greater) ||
						(direction_for_l == DependenceVector.less))
					is_carried_for_loop = true;
			}
			return is_carried_for_loop;
		}
		
		public boolean containsLoopCarriedDependence()
		{
			boolean loop_carried = false;
			DependenceVector dv = this.getDependenceVector();

			if ((dv.getDirectionVector()).containsValue(DependenceVector.less) ||
					(dv.getDirectionVector()).containsValue(DependenceVector.greater) ||
					(dv.getDirectionVector()).containsValue(DependenceVector.any))
				loop_carried = true;
			
			return loop_carried;
		}
		
		public String toString()
		{
			return ("ArcSource: " + this.source + " ArcSink: " + this.sink +
					" depType: " + Byte.valueOf(this.depType).toString() +
					" depVector: " + this.depVector.VectorToString()
					);
		}

	}
	
	public DDGraph ()
	{
		/* Initialize list of arcs in this graph */
		depArcs = new ArrayList<Arc>();
		/* Arcs are not summarized, all dependences are explicit by default */
		summarized_status = DDGraph.not_summarize;
	}

	public DDGraph (ArrayList<Arc> dependence_arcs)
	{
		/* Create a list of arcs for this graph */
		depArcs = new ArrayList<Arc>();
		depArcs.addAll(dependence_arcs);
		/* Arcs are not summarized, all dependences are explicit by default */
		summarized_status = DDGraph.not_summarize;
	}
	
	public void addArc(Arc arc_to_add)
	{
		boolean this_is_redundant = false;

		if (arc_to_add.getDependenceVector().isValid() == true) {
			for (Arc arc : depArcs) {
				if ((arc.getSource() == arc_to_add.getSource()) &&
					(arc.getSink() == arc_to_add.getSink()) &&
					(arc.getDependenceType() == arc_to_add.getDependenceType()) &&
					((arc.getDependenceVector().getDirectionVector()).equals(
						arc_to_add.getDependenceVector().getDirectionVector())))
					this_is_redundant = true;
			}
			if ( !this_is_redundant )
				depArcs.add(arc_to_add);
		}
	}

	public void deleteArc(Arc arc)
	{
		depArcs.remove(arc);
	}
	
	public ArrayList<Arc> getAllArcs()
	{
		return depArcs;
	}
	
	public void addAllArcs(ArrayList<Arc> arcs)
	{
		for (Arc a : arcs)
		{
			this.addArc(a);
		}
	}
	
	/**
	 * Removes arcs with directions:
	 * (.) --> containing '.' = invalid merged direction
	 */
	public void filterUnwantedArcs()
	{
		Iterator<Arc> iter = depArcs.iterator();
		while (iter.hasNext())
		{
			Arc arc = iter.next();
			DependenceVector dv = arc.getDependenceVector();

//			PrintTools.print("Stuck in filterUnwantedArcs", 0);
			if ( !dv.isValid() )
				//this.deleteArc(arc);
				// Cannot use this.deleteArc due to ArrayList
				// Iterator synchronization issues
				iter.remove();
			//else if ((dv.getDependenceVector()).containsValue(DependenceVector.any))
				//this.deleteArc(arc);
				// Cannot use this.deleteArc due to ArrayList
				// Iterator synchronization issues
				//iter.remove();
			else
			{
			}
		}
	}
	
	/**
	 * Filter out duplicate and unwanted arcs from the graph
	 */
	public void removeDuplicateArcs()
	{
		/* First remove invalid arcs if any, then proceed to check for duplicate arcs */
		this.filterUnwantedArcs();
		
		ListIterator<Arc> arc_list_1 = depArcs.listIterator();
		while (arc_list_1.hasNext())
		{
			Arc arc1 = arc_list_1.next();
			int index = depArcs.indexOf(arc1);

			if ( arc1.getDependenceVector().isValid() )
			{
				ListIterator<Arc> arc_list_2 = depArcs.listIterator(++index);
				while (arc_list_2.hasNext())
				{
					Arc arc2 = arc_list_2.next();
					if ( arc2.getDependenceVector().isValid() )
					{
						if ((arc1.getSource() == arc2.getSource()) &&
								(arc1.getSink() == arc2.getSink()) &&
								(arc1.getDependenceType() == arc2.getDependenceType()) &&
								((arc1.getDependenceVector().getDirectionVector()).equals(arc2.getDependenceVector().getDirectionVector())))
							arc2.getDependenceVector().setValid(false);
					}
				}
			}
		}
		/* Remove arcs marked invalid by the duplicate check */ 
		this.filterUnwantedArcs();
	}

	/**
	 * Summarize the direction vectors between nodes of this graph
	 */
	public void summarizeGraph()
	{
		this.summarized_status = DDGraph.summarize;
		/* Create a new set of arcs that must be added to the graph */
		ArrayList<Arc> newArcsForGraph = new ArrayList<Arc>();
		
		Iterator<Arc> arc_list = depArcs.iterator();
		
		while (arc_list.hasNext())
		{
			Arc arc = arc_list.next();
			
			if ( arc.getDependenceVector().isValid() )
			{
				Set<Loop> loopsInVector = ((arc.getDependenceVector())).getLoops();
				ArrayList<Arc> arcsToBeSummarized = new ArrayList<Arc>();
				arcsToBeSummarized = getDependenceArcsFromTo(arc.getSource().getArrayAccess(),
						arc.getSink().getArrayAccess());

				ArrayList<DependenceVector> dvsToBeSummarized = new ArrayList<DependenceVector>();

				for (Arc arc_iter : arcsToBeSummarized)
				{
					dvsToBeSummarized.add(arc_iter.getDependenceVector());
				}

				for (Loop l : loopsInVector)
				{
					int equal_dir_cnt = 0;
					int less_dir_cnt = 0;
					int great_dir_cnt = 0;
					for (DependenceVector v : dvsToBeSummarized)
					{
						switch (v.getDirection(l))
						{
						case DependenceVector.equal:
							equal_dir_cnt++;
							break;
						case DependenceVector.less:
							less_dir_cnt++;
							break;
						case DependenceVector.greater:
							great_dir_cnt++;
							break;
						case DependenceVector.any:
							equal_dir_cnt++;
							less_dir_cnt++;
							great_dir_cnt++;
							break;
						default:
							break;
						}
					}
					/* Check if all directions are present for that loop, then we can summarize */
					if ((equal_dir_cnt > 0) &&
							(less_dir_cnt > 0) &&
							(great_dir_cnt > 0))
					{
						for (Arc a :  arcsToBeSummarized)
						{
							Arc summarized_arc = new Arc(a);
							(summarized_arc.getDependenceVector()).setDirection(l, DependenceVector.any);
							newArcsForGraph.add(summarized_arc);
							/* Invalidate the old arc, it will be deleted by the remove duplicates routine */
							a.getDependenceVector().setValid(false);
						}
					}
				}
			}
		}
		
		/* Add all new arcs to the graph */
		for (Arc arc : newArcsForGraph)
		{
			depArcs.add(arc);
		}
		
		/* Remove duplicate arcs that might have been added as part of the summarization process */
		this.removeDuplicateArcs();
	}
	
	/**
	 * Returns true if there exists a loop carried dependence for ANY loop in the
	 * nest represented by this dependence graph
	 * @return true if it does, false otherwise
	 */
	public boolean checkLoopCarriedDependenceForGraph()
	{
		boolean contains_loop_carried_deps = false;
		ArrayList<Arc> loop_carried_deps = getLoopCarriedDependencesForGraph();
		
		if(loop_carried_deps.isEmpty())
			contains_loop_carried_deps = false;
		else
			contains_loop_carried_deps = true;
		
		return contains_loop_carried_deps;
	}

	/**
	 * Returns a list of all dependences (arcs) that are loop carried with respect
	 * to ANY of the loops in the nest represented by this dependence graph
	 * @return the list of dependence arcs.
	 */
	public ArrayList<Arc> getLoopCarriedDependencesForGraph()
	{
		ArrayList<Arc> loop_carried_deps = new ArrayList<Arc>();
		Iterator<Arc> iter = depArcs.iterator();
		
		while (iter.hasNext())
		{
			Arc arc = iter.next();
			if (arc.containsLoopCarriedDependence())
				loop_carried_deps.add(arc);
		}
		
		return loop_carried_deps;
	}
	
	/**
	 * Check if the dependence direction for the input loop is equal in all dependences
	 * in the graph
	 * @param loop the loop for which equal dependence direction must be checked
	 * @return true if it is, false otherwise
	 */
	public boolean checkEqualDependences(Loop loop)
	{
		Iterator<Arc> iter = depArcs.iterator();
		
		while (iter.hasNext())
		{
			Arc arc = iter.next();
			DependenceVector dv = arc.getDependenceVector();
			
			if (((dv.getDirectionVector()).get(loop)) != DependenceVector.equal)
				return false;
		}
		return true;
	}
	
	/**
	 * Check if the dependence direction for the input loop is loop-carried for any of
	 * the dependences in the graph
	 * @param l the loop for which loop-carried dependence existence must be checked
	 * @return true if it is, false otherwise
	 */
	public boolean checkLoopCarriedDependence(Loop l)
	{
		boolean ret_val = false;
		
		for (Arc dep : depArcs)
		{
			if (dep.isCarried(l))
			{
				ret_val = true;
				return ret_val;
			}
		}
		return ret_val;
	}

	/**
	 * Obtain all possible dependence information from expr1 to expr2 in a given loop
	 * @param expr1 - ArrayAccess
	 * @param expr2 - ArrayAccess
	 * @return arcSet - List of all existing dependence arcs from expr1 to expr2
	 */
	public ArrayList<Arc> getDependenceArcsFromTo(ArrayAccess expr1, ArrayAccess expr2)
	{
		ArrayList<Arc> arcSet = new ArrayList<Arc>();
		
		for(Arc arc: depArcs)
		{
			if (arc.getSource().getArrayAccess() == expr1)
			{
				if (arc.getSink().getArrayAccess() == expr2)
					arcSet.add(arc);
			}
		}
		return (arcSet);
	}
	
	/**
	 * Obtain all possible dependence information between a pair of array accesses in a given loop
	 * @param expr1 - ArrayAccess
	 * @param expr2 - ArrayAccess
	 * @return arcSet - List of all existing dependence arcs between the two accesses
	 */
	public ArrayList<Arc> getDependences(Expression expr1, Expression expr2)
	{
		ArrayList<Arc> arcSet = new ArrayList<Arc>();

		arcSet.addAll(getDependenceArcsFromTo((ArrayAccess)expr1, (ArrayAccess)expr2));

		arcSet.addAll(getDependenceArcsFromTo((ArrayAccess)expr2, (ArrayAccess)expr1));

		return (arcSet);
	}
	
	private boolean checkDependenceType(Expression e1, Expression e2, byte depType)
	{
		boolean ret_val = false;
		ArrayList<Arc> arcs_e1_e2 = getDependences(e1, e2);
		for (Arc a : arcs_e1_e2)
		{
			if (a.getDependenceType() == depType)
			{
				ret_val = true;
				break;
			}
		}
		return ret_val;
	}
	
	/**
	 * Check if there is flow dependence from e1 to e2
	 * @param e1 source of dependence
	 * @param e2 sink of dependence
	 * @return true if it is, false otherwise
	 */
	public boolean checkFlowDependence(Expression e1, Expression e2)
	{
		return (checkDependenceType((ArrayAccess)e1, (ArrayAccess)e2, (byte)1));
	}
	
	/**
	 * Check if there is anti dependence from e1 to e2
	 * @param e1 source of dependence
	 * @param e2 sink of dependence
	 * @return true if it is, false otherwise
	 */
	public boolean checkAntiDependence(Expression e1, Expression e2)
	{
		return (checkDependenceType((ArrayAccess)e1, (ArrayAccess)e2, (byte)2));
	}
	
	/**
	 * Check if there is an output dependence from e1 to e2
	 * @param e1 source of dependence
	 * @param e2 sink of dependence
	 * @return true if it is, false otherwise
	 */
	public boolean checkOutputDependence(Expression e1, Expression e2)
	{
		return (checkDependenceType((ArrayAccess)e1, (ArrayAccess)e2, (byte)3));
	}
	
	/**
	 * Obtain all possible dependence information between a pair of statements in a given loop
	 * @param stmt1 - Statement
	 * @param stmt2 - Statement
	 * @return arcSet - List of all existing dependence arcs between the two statements
	 */
	public ArrayList<Arc> getDependences(Statement stmt1, Statement stmt2)
	{
		ArrayList<Arc> arcSet = new ArrayList<Arc>();
		
		for(Arc arc: depArcs)
		{
			if (arc.getSourceStatement() == stmt1)
			{
				if (arc.getSinkStatement() == stmt2)
					arcSet.add(arc);
			}
			else if (arc.getSourceStatement() == stmt2)
			{
				if (arc.getSinkStatement() == stmt1)
					arcSet.add(arc);
			}
		}
		return (arcSet);
	}
	
	/**
	 * From the current dependence graph, extract a subgraph containing dependences only
	 * for the specified loop and its inner nest
	 * @param loop the loop and its inner nest for which dependences are to be extracted into a
	 * separate graph
	 * @return the sub-{@code DDGraph} that belongs to {@code loop}
	 */
	public DDGraph getSubGraph(Loop loop)
	{
		DDGraph loop_graph = new DDGraph();
		
		for (Arc a : depArcs)
		{
			if (a.belongsToLoop(loop))
				loop_graph.addArc(a);
		}
		
		loop_graph.summarized_status = this.summarized_status;
		
		return loop_graph;
	}
	
	/**
	 * Print function to print entire dependence graph information
	 */
	public String toString()
	{
		String temp = "Arc Info for this DD graph\n";
		
		for (Arc arc: depArcs)
		{
			temp += (arc.toString() + "\n");
		}
		return temp;
	}
	
	/**
	 * Obtain a matrix representation of direction vectors for the dependences
	 * contained within this dependence graph
	 * @param nest the loop nest for which the matrix needs to be obtained
	 * @return the list that encodes the matrix in terms of list of
   * {@code DependenceVector}
	 */
	public ArrayList<DependenceVector> getDirectionMatrix(LinkedList<Loop> nest)
	{
		/*
		 * dir_matrix has the following representation:
		 * - Each row of the matrix is a DependenceVector that maps
		 * 	 every loop to its dependence direction
		 * - There are 'n' rows. Each row represents a dependence arc contained
		 * 	 within this loop nest
		 * - There are 'd' columns where d is the depth of the loop nest
		 * 	 represented by this direction vector matrix
		 */
		ArrayList<DependenceVector> dir_matrix = 
					new ArrayList<DependenceVector>();
		for (Arc dep : depArcs)
		{
			DependenceVector dv = dep.getDependenceVector();
			DependenceVector row = new DependenceVector(nest);
			for (Loop l : nest)
			{
				/* Get the direction for loop l from dv and insert
				 * it for loop l in the dir_matrix */
				if (dv.getDirectionVector().containsKey(l))
				{
					int direction = dv.getDirection(l);
					row.setDirection(l, direction);
				}
				else
				{
					int direction = DependenceVector.nil;
					row.setDirection(l, direction);
				}
			}
			dir_matrix.add(row);
		}
		return dir_matrix;
	}
}
