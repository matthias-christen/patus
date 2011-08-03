package omp2gpu.transforms;

import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import cetus.transforms.TransformPass;
import cetus.analysis.Reduction;
import cetus.exec.Driver;
import cetus.hir.*;
import omp2gpu.analysis.AnalysisTools;

/**
 * @author Seyong Lee <lee222@purdue.edu>
 *         ParaMount Group 
 *         School of ECE, Purdue University
 */
public class SplitOmpPRegion extends TransformPass {
	
	private HashSet<String> visitedProcs;
	private boolean enableCritical2ReductionConv = true;

	public SplitOmpPRegion( Program program ) {
		super(program);
	}
	
	@Override
	public String getPassName() {
		return new String("[Split Omp Parallel Regions]");
	}

	@Override
	public void start() {
		String value = Driver.getOptionValue("disableCritical2ReductionConv");
		if( value != null ) {
			enableCritical2ReductionConv = false;
		}
		/////////////////////////////////////////////////////////////////////////
		// DEBUG: Below method should be commented out since empty clauses are //
		// still needed. Methods updating omp-clauses are invoked only for     //
		// existing clauses; omp-private and firstprivate clauses may be empty //
		// before updated for functions called in a parallel region.           //
		/////////////////////////////////////////////////////////////////////////
		//cleanEmptyOmpClauses(program);
		mark_intervals();
		mark_additionalIntervals();
		if( enableCritical2ReductionConv ) {
			convertCritical2Reduction();
		}
		splitParallelRegions();
		/////////////////////////////////////////////////////////////////////////////////
		// DEBUG: convertCritical2Reduction() method does not check reduction patterns //
		// existing in a function called in a parallel region. This method is called   //
		// again after kernel splitting, since kernel splitting may create a parallel  //
		// region inside the called function.                                          //
		/////////////////////////////////////////////////////////////////////////////////
		if( enableCritical2ReductionConv ) {
			convertCritical2Reduction();
		}
		cleanDummyBarriers();
		updateReductionClause();
		updatePrivateClause();
		cleanEmptyOmpClauses(program);
	}

	/**
	 * Create a OpenMP barrier statement where the barrier annotation
	 * contains input String, type, as a value. 
	 * [CAUTION] this new barrier annotation points to its enclosing statement;
	 * old version of this method had additional input parameter that the barrier
	 * annotation will point to, but this parameter was deleted in the new version.
	 * @param type is a String that will be stored in the barrier annotation.
	 * @return AnnotationStatement enclosing the newly created barrier annotation.
	 */
	public static AnnotationStatement insertBarrier(String type)
	{
		OmpAnnotation annot = new OmpAnnotation("barrier", type);	
		AnnotationStatement annot_stmt = new AnnotationStatement(annot);
		return annot_stmt;
	}

	/**
		*	Implicit barrier
		* 	- at the end of the parallel construct
		* 	- at the end of the worksharing construct (check an existence of nowait clause)
		* 	- at the end of the sections construct (check an existence of nowait clause)
		* 	- at the end of the single construct (check an existence of nowait clause)
		* 
		*   Additional barrier 
		*   - at the beginning of the parallel construct for analysis purpose.
		*
		*/

	public void mark_intervals()
	{
		PrintTools.println("[mark_intervals] strt", 5);

		DepthFirstIterator iter = new DepthFirstIterator(program);

		while ( iter.hasNext() )
		{
			Object obj = iter.next();

			if( (obj instanceof Annotatable) && (obj instanceof Statement) )
			{
				Annotatable at = (Annotatable)obj;
				Statement atstmt = (Statement)obj;
				Traversable parent = atstmt.getParent();
				CompoundStatement parent_stmt = null;
				if( parent instanceof CompoundStatement ) {
					parent_stmt = (CompoundStatement)parent;
				} else {
					continue;
				}

				if ( at.containsAnnotation(OmpAnnotation.class, "parallel") &&
						at.containsAnnotation(OmpAnnotation.class, "for") )
				{
					parent_stmt.addStatementBefore(atstmt, insertBarrier("S2P"));
					if ( at.containsAnnotation(OmpAnnotation.class, "nowait") == false )
					{
						parent_stmt.addStatementAfter(atstmt, insertBarrier("P2S"));
					}
				}
				else if ( at.containsAnnotation(OmpAnnotation.class, "parallel") )
				{
					parent_stmt.addStatementBefore(atstmt, insertBarrier("S2P"));
					parent_stmt.addStatementAfter(atstmt, insertBarrier("P2S"));

					PrintTools.println("[mark] parent_stmt", 10);
					PrintTools.println(parent_stmt.toString(), 10);
				}
				else if ( at.containsAnnotation(OmpAnnotation.class, "for") ||
						at.containsAnnotation(OmpAnnotation.class, "sections") ||
						at.containsAnnotation(OmpAnnotation.class, "single") )
				{	
					if ( at.containsAnnotation(OmpAnnotation.class, "nowait") == false )
					{
						parent_stmt.addStatementAfter(atstmt, insertBarrier("P2P"));
					}
				}
			}
		}

		PrintTools.println("[mark_intervals] done", 5);
	}

	/**
	 * Insert additional barriers at the following points: 
	 *     - at the entry and exit of sections and single, even if they have nowait clauses
	 *     - at the entry and exit of master
	 * These additional barriers are used to identify sub-parallel regions that are eligible for
	 * kernel extraction; these are unnecessary for OpenMP programs, but useful for 
	 * valid OpenMP-to-CUDA translation (omp sections, single, and master are handled by CPU.).
	 * 
	 */
	public void mark_additionalIntervals() {
		PrintTools.println("[mark_additionalIntervals] strt", 5);
		
		DepthFirstIterator iter = new DepthFirstIterator(program);
		AnnotationStatement barrStmt = null;

		while ( iter.hasNext() )
		{
			Object obj = iter.next();

			if( (obj instanceof Annotatable) && (obj instanceof Statement) )
			{
				Annotatable at = (Annotatable)obj;
				Statement atstmt = (Statement)obj;
				Traversable parent = atstmt.getParent();
				CompoundStatement parent_stmt = null;
				if( parent instanceof CompoundStatement ) {
					parent_stmt = (CompoundStatement)parent;
				} else {
					continue;
				}

				List<Traversable> children = parent_stmt.getChildren();
				int Index = 0;
				if ( at.containsAnnotation(OmpAnnotation.class, "section") ||
						at.containsAnnotation(OmpAnnotation.class, "single") )
				{
					Index = Tools.indexByReference(children, atstmt);
					Traversable prev_t = null;
					if( Index > 0 ) {
						prev_t = children.get(Index-1);
					}
					// Insert a barrier statement before the annotatable statement.
					barrStmt = insertBarrier("dummy");
					parent_stmt.addStatementBefore(atstmt, barrStmt);
					if ( at.containsAnnotation(OmpAnnotation.class, "nowait") )
					{
						//Insert a barrier even if omp single or sections has nowait clause.
						//parent_stmt.addStatementAfter(attached_stmt, insertBarrier(attached_stmt, "true"));
						barrStmt = insertBarrier("dummy");
						parent_stmt.addStatementAfter(atstmt, barrStmt);
					}
				}
				else if ( at.containsAnnotation(OmpAnnotation.class, "master") )
				{	
					Index = Tools.indexByReference(children, atstmt);
					Traversable prev_t = null;
					if( Index > 0 ) {
						prev_t = children.get(Index-1);
					}
					barrStmt = insertBarrier("dummy");
					parent_stmt.addStatementBefore(atstmt, barrStmt);
					barrStmt = insertBarrier("dummy");
					parent_stmt.addStatementAfter(atstmt, barrStmt);
				}
			}
		}

		PrintTools.println("[mark_additionalIntervals] done", 5);
	}
	
	/**
	 * Split omp parallel regions in the program at every explicit/implicit barrier.
	 * This split may break the correctness of the program if private data written in 
	 * one split parallel subregion should be read in the other split parallel subregion.
	 * [CAUTION] New parallel regions split by this method will contain accurate 
	 * annotations for each parallel region, but omp-for annotations in parallel regions 
	 * may not have accurate information. 
	 * 
	 */
	public void splitParallelRegions()
	{
		PrintTools.println("[splitParallelRegions] strt", 5);

		visitedProcs = new HashSet<String>();
		List<OmpAnnotation> pRegionAnnots = new LinkedList<OmpAnnotation>();
		DepthFirstIterator iter = new DepthFirstIterator(program);

		while ( iter.hasNext() )
		{
			Object obj = iter.next();

			if( (obj instanceof Annotatable) && (obj instanceof Statement) )
			{
				Annotatable at = (Annotatable)obj;
				Statement atstmt = (Statement)obj;

				if ( at.containsAnnotation(OmpAnnotation.class, "parallel") && 
						at.containsAnnotation(OmpAnnotation.class, "for") )
				{
					if( atstmt instanceof ForLoop ) {
						ForLoop floop = (ForLoop)atstmt;
						Statement body = floop.getBody();
						List annot_list = IRTools.collectPragmas(body, OmpAnnotation.class, "barrier");
						if( annot_list.size() != 0 ) {
							Tools.exit("Error in splitParallelRegions(): omp-for loop can not be split!");
						}
					} else {
						PrintTools.println("Omp-for annotation is included in the wrong annotatable container("
								+ atstmt + ")", 0);
					}
				}
				else if ( at.containsAnnotation(OmpAnnotation.class, "parallel") && 
						at.containsAnnotation(OmpAnnotation.class, "section") )
				{
					if( atstmt instanceof CompoundStatement ) {
						List annot_list = IRTools.collectPragmas(atstmt, OmpAnnotation.class, "barrier");
						if( annot_list.size() != 0 ) {
							Tools.exit("Error in splitParallelRegions(): omp parallel sections can not be split!");
						}
					} else {
						PrintTools.println("Omp Section annotation is included in the wrong annotatable container("
								+ atstmt + ")", 0);
					}
				}
				else if ( at.containsAnnotation(OmpAnnotation.class, "parallel") )
				{
					if( atstmt instanceof CompoundStatement ) {
						if( AnalysisTools.ipaContainsBarrierInRegion(atstmt) ) { //found a parallel region that may be split.
							//List annot_list = IRTools.collectPragmas(atstmt, OmpAnnotation.class, "for");
							List annot_list = AnalysisTools.ipCollectPragmas(atstmt, OmpAnnotation.class, "for");
							////////////////////////////////////////////////////////////////////////////////
							// If the parallel region does not contain any omp-for loop, we don't have to //
							// split this region.                                                         //
							////////////////////////////////////////////////////////////////////////////////
							if( annot_list.size() > 0 ) {
								pRegionAnnots.add(at.getAnnotation(OmpAnnotation.class, "parallel"));
							}
						}
					} else {
						PrintTools.println("Omp Parallel annotation is included in the wrong annotatable container("
								+ atstmt + ")", 0);
					}	
				}
			}
		}
		if( pRegionAnnots.size() == 0 ) {
			PrintTools.println("[splitParallelRegions] No split operation is conducted.", 0);
		} else {
			for( OmpAnnotation omp_annot : pRegionAnnots ) {
				Statement atstmt = (Statement)omp_annot.getAnnotatable();

				ipCreateSubRegions(omp_annot, atstmt);

				// When splitting occurs, old parallel region should be removed.
				// To do so, old omp pragma is replaced with comment statement.
				CommentAnnotation comment = new CommentAnnotation(omp_annot.toString());
				AnnotationStatement comment_stmt = new AnnotationStatement(comment);
				CompoundStatement parent = (CompoundStatement)atstmt.getParent();
				parent.addStatementBefore(atstmt, comment_stmt);
				atstmt.removeAnnotations(OmpAnnotation.class);
			}
			PrintTools.println("[splitParallelRegions] done", 5);
		}
	}
	
	/**
	 * Split parallel region into two sub regions at every Barrier point.
	 * This method is called once for each main parallel region containing barriers, 
	 * and if the parallel region contains function calls, this method is called 
	 * again for each called function.
	 * 
	 * @param barrier_list List of Barriers
	 * @param old_map Omp HashMap of OmpAnnotation that refers to the enclosing parallel region
	 */
	private void createSubRegions(List<OmpAnnotation> barrier_list, HashMap old_map) {
		int barIndex = 0;
		int pBarIndex = 0;
		int lastDeclIndex_plus1 = 0;
		int list_size = 0;
		for( OmpAnnotation barrier_annot : barrier_list ) {
			Statement barrier_stmt = (Statement)barrier_annot.getAnnotatable();
			Traversable t = barrier_stmt.getParent();
			if( t instanceof CompoundStatement ) {
				CompoundStatement parent = (CompoundStatement)t;
				List<Traversable> children = parent.getChildren();
				LinkedList<Traversable> temp_list = new LinkedList<Traversable>();
				Statement lastDeclStmt = IRTools.getLastDeclarationStatement(parent);
				if( lastDeclStmt != null ) {
					lastDeclIndex_plus1 = 1 + Tools.indexByReference(children, lastDeclStmt);
				} else {
					lastDeclIndex_plus1 = 0;
				}
				barIndex = Tools.indexByReference(children, barrier_stmt);
				pBarIndex = 0;
				list_size = 0;
				/*
				 * Check statements between current barrier and previous barrier.
				 * If there is no statement that contains a barrier inside, below splitting will
				 * create one sub-egion, which is a parallel region.
				 * Otherwise, the sub-region enclosed by both current barrier and previous barrier
				 * should be split further at each statement containing a barrier internally.
				 */
				while( barIndex > lastDeclIndex_plus1 ) {
					pBarIndex = lastDeclIndex_plus1;
					Statement cur_barrier = (Statement)children.get(barIndex);
					for(int i=barIndex-1; i>=lastDeclIndex_plus1; i--) {
						//Traversable child = children.remove(i);
						Traversable child = children.get(i);
						if( AnalysisTools.ipaContainsBarrierInRegion(child) ) {
							//children.add(i, child);
							if( !(child instanceof AnnotationStatement) ) {
								pBarIndex = i;
							}
							break;
						} else {
							temp_list.add(child);
						}
					}
					list_size = temp_list.size();
					if( list_size > 0 ) {
						//////////////////////////////////////////////////////////////////
						// Check whether temp_list contains any computation statement.  //
						// If not, temp_list contains only declaration or annotation,   //
						// and this sub-region should not be extracted as a parallel    //
						// region.                                                      //
						//////////////////////////////////////////////////////////////////
						boolean foundCompStmt = false;
						for( Traversable temp : temp_list) {
							if( !(temp instanceof AnnotationStatement) && 
									!(temp instanceof DeclarationStatement) ) {
								foundCompStmt = true;
								break;
							}
						}
						if( foundCompStmt ) {
							CompoundStatement pRegion = new CompoundStatement();
							for( int i=0; i<list_size; i++ ) {
								Traversable child = temp_list.removeLast();
								//child.setParent(null);
								////////////////////////////////////////////////////////////
								// Child is really removed from the parent at this point. //
								////////////////////////////////////////////////////////////
								parent.removeChild(child);
								if( child instanceof DeclarationStatement ) {
									Declaration decl = ((DeclarationStatement)child).getDeclaration();
									decl.setParent(null);
									pRegion.addDeclaration(decl);
								} else {
									pRegion.addStatement((Statement)child);
								}
							}
							OmpAnnotation omp_annot = new OmpAnnotation();
							omp_annot.putAll(old_map);
							pRegion.annotate(omp_annot);
							parent.addStatementBefore(cur_barrier, pRegion);
							if( !(cur_barrier instanceof AnnotationStatement) ) {
								parent.addStatementBefore(cur_barrier, insertBarrier("dummy"));
							}
						} 
						/* else {
							for( int i=0; i<list_size; i++ ) {
								children.add(tailIndex, temp_list.removeFirst());
							}
						} */
					}
					barIndex = pBarIndex;
				} 
				/*
				 * There can be a sub-region after the last barrier; 
				 * this last sub-region is handled here.
				 */
				barIndex = children.size()-1;
				temp_list.clear();
				Traversable lastchild = children.get(barIndex); 
				boolean lastSubRHandled = false;
				if( AnalysisTools.ipaContainsBarrierInRegion(lastchild) ) {
					if( lastchild instanceof AnnotationStatement ) {
						lastSubRHandled = true;
					}
				} else { //Insert a barrier at the end of this compound statement.
					parent.addStatement(insertBarrier("dummy"));
					barIndex = barIndex + 1; //point to the last, newly-inserted barrier.
				}
				if( !lastSubRHandled ) {
					while( barIndex > lastDeclIndex_plus1 ) {
						pBarIndex = lastDeclIndex_plus1;
						Statement cur_barrier = (Statement)children.get(barIndex);
						for(int i=barIndex-1; i>=lastDeclIndex_plus1; i--) {
							//Traversable child = children.remove(i);
							Traversable child = children.get(i);
							if( AnalysisTools.ipaContainsBarrierInRegion(child) ) {
								//children.add(i, child);
								if( !(child instanceof AnnotationStatement) ) {
									pBarIndex = i;
								}
								break;
							} else {
								temp_list.add(child);
							}
						}
						list_size = temp_list.size();
						if( list_size > 0 ) {
							//////////////////////////////////////////////////////////////////
							// Check whether temp_list contains any computation statement.  //
							// If not, temp_list contains only declaration or annotation,   //
							// and this sub-region should not be extracted as a parallel    //
							// region.                                                      //
							//////////////////////////////////////////////////////////////////
							boolean foundCompStmt = false;
							for( Traversable temp : temp_list) {
								if( !(temp instanceof AnnotationStatement) && 
										!(temp instanceof DeclarationStatement) ) {
									foundCompStmt = true;
									break;
								}
							}
							if( foundCompStmt ) {
								CompoundStatement pRegion = new CompoundStatement();
								for( int i=0; i<list_size; i++ ) {
									Traversable child = temp_list.removeLast();
									//child.setParent(null);
									////////////////////////////////////////////////////////////
									// Child is really removed from the parent at this point. //
									////////////////////////////////////////////////////////////
									parent.removeChild(child);
									if( child instanceof DeclarationStatement ) {
										Declaration decl = ((DeclarationStatement)child).getDeclaration();
										decl.setParent(null);
										pRegion.addDeclaration(decl);
									} else {
										pRegion.addStatement((Statement)child);
									}
								}
								OmpAnnotation omp_annot = new OmpAnnotation();
								omp_annot.putAll(old_map);
								pRegion.annotate(omp_annot);
								parent.addStatementBefore(cur_barrier, pRegion);
								if( !(cur_barrier instanceof AnnotationStatement) ) {
									parent.addStatementBefore(cur_barrier, insertBarrier("dummy"));
								}
							} 
							/* else {
								for( int i=0; i<list_size; i++ ) {
									children.add(tailIndex, temp_list.removeFirst());
								}
							} */
						}
						barIndex = pBarIndex;
					}
				}
			}
		}
	}
	
	public void ipCreateSubRegions(HashMap old_map, Statement region) {
		//////////////////////////////////////////////////////////
		// If current region does not have any barrier, return. //
		//////////////////////////////////////////////////////////
		if( !AnalysisTools.ipaContainsBarrierInRegion(region) ) {
			return;
		}
		List<OmpAnnotation> barrier_list = IRTools.collectPragmas(region, OmpAnnotation.class, "barrier");
		int num_barriers = barrier_list.size();
		createSubRegions(barrier_list, old_map);
		// Split parallel regions in called functions
		List<FunctionCall> calledFuncs = IRTools.getFunctionCalls(region);
		for( FunctionCall call : calledFuncs ) {
			Procedure called_procedure = call.getProcedure();
			/*
			 * If called function is a system call, parse may not be able to find corresponding
			 * function body, and in this case, call.getProcedure() will return null.
			 */
			if( (called_procedure == null) || 
					visitedProcs.contains(called_procedure.getSymbolName()) ) {
				continue;
			} else {
				visitedProcs.add(called_procedure.getSymbolName());
				CompoundStatement body = called_procedure.getBody();
				//barrier_list = IRTools.collectPragmas(body, OmpAnnotation.class, "barrier");
				HashMap new_map = updateOmpMapForCalledFunc(old_map, 
						(List<Expression>)call.getArguments(), called_procedure );
				//createSubRegions(barrier_list, new_map);
				ipCreateSubRegions(new_map, body);
				/////////////////////////////////////////////////////////////////////////////////
				// DEBUG: If called_procecure does not have any barrier, the above call does   //
				// not update Omp annotations using the new_map data, and thus Omp annotations //
				// in the procedure still contains old hashmap data. Therefore, the next       //
				// method, updateOmpAnnotationsInRegion(), can not update annotation accurately//
				// (More specifically, the next call can not handle cases where arguments of   //
				// the called procedure are shared variables.)                                 //
				// ==> In a procedure called in a parallel region, omp-for annotations may not //
				//     have accurate information, missing shared data passed as parameters.    //
				//     ==> updateOmpMapForCalledFunc() updates omp-for in the called function. //
				/////////////////////////////////////////////////////////////////////////////////
				//Update OmpAnnotations
				TransformTools.updateAnnotationsInRegion(called_procedure);
			}
		}

		/*
		 * If the parallel region contains barriers only in the functions called within this
		 * region, createSubRegions() will not do any thing for the main parallel region.
		 * Thus, the below section splits the main parallel region at every function call 
		 * containing barriers.
		 */
		if( num_barriers == 0 ) {
			int barIndex = 0;
			int pBarIndex = 0;
			int list_size = 0;
			int lastDeclIndex_plus1 = 0;
			CompoundStatement parent = (CompoundStatement)region;
			List<Traversable> children = parent.getChildren();
			LinkedList<Traversable> temp_list = new LinkedList<Traversable>();
			barIndex = children.size()-1;
			Statement lastDeclStmt = IRTools.getLastDeclarationStatement(parent);
			if( lastDeclStmt != null ) {
				lastDeclIndex_plus1 = 1 + Tools.indexByReference(children, lastDeclStmt);
			} else {
				lastDeclIndex_plus1 = 0;
			}

			Traversable lastchild = children.get(barIndex); 
			if( !AnalysisTools.ipaContainsBarrierInRegion(lastchild) ) {
				//Insert a barrier at the end of this compound statement.
				parent.addStatement(insertBarrier("dummy"));
				barIndex = barIndex + 1; //point to the last, newly-inserted barrier.
			}
			while( barIndex > lastDeclIndex_plus1 ) {
				pBarIndex = lastDeclIndex_plus1;
				Statement cur_barrier = (Statement)children.get(barIndex);
				for(int i=barIndex-1; i>=lastDeclIndex_plus1; i--) {
					//Traversable child = children.remove(i);
					Traversable child = children.get(i);
					if( AnalysisTools.ipaContainsBarrierInRegion(child) ) {
						//children.add(i, child);
						if( !(child instanceof AnnotationStatement) ) {
							pBarIndex = i;
						}
						break;
					} else {
						temp_list.add(child);
					}
				}
				list_size = temp_list.size();
				if( list_size > 0 ) {
					//////////////////////////////////////////////////////////////////
					// Check whether temp_list contains any computation statement.  //
					// If not, temp_list contains only declaration or annotation,   //
					// and this sub-region should not be extracted as a parallel    //
					// region.                                                      //
					//////////////////////////////////////////////////////////////////
					boolean foundCompStmt = false;
					for( Traversable temp : temp_list) {
						if( !(temp instanceof AnnotationStatement) && 
								!(temp instanceof DeclarationStatement) ) {
							foundCompStmt = true;
							break;
						}
					}
					if( foundCompStmt ) {
						CompoundStatement pRegion = new CompoundStatement();
						for( int i=0; i<list_size; i++ ) {
							Traversable child = temp_list.removeLast();
							//child.setParent(null);
							////////////////////////////////////////////////////////////
							// Child is really removed from the parent at this point. //
							////////////////////////////////////////////////////////////
							parent.removeChild(child);
							if( child instanceof DeclarationStatement ) {
								Declaration decl = ((DeclarationStatement)child).getDeclaration();
								decl.setParent(null);
								pRegion.addDeclaration(decl);
							} else {
								pRegion.addStatement((Statement)child);
							}
						}
						OmpAnnotation new_annot = new OmpAnnotation();
						new_annot.putAll(old_map);
						pRegion.annotate(new_annot);
						parent.addStatementBefore(cur_barrier, pRegion);
						if( !(cur_barrier instanceof AnnotationStatement) ) {
							parent.addStatementBefore(cur_barrier, insertBarrier("dummy"));
						}
					}
				}
				barIndex = pBarIndex;
			}

		}
		//Update OmpAnnotations
		TransformTools.updateAnnotationsInRegion(region);
	}
	
	/**
	 * [Convert critical sections into reduction sections]
	 * For each critical section in a parallel region,
	 *     if the critical section is a kind of reduction form, necessary reduction 
	 *     clause is added to the annotation of the enclosing parallel region, and 
	 *     the original critical construct is commented out.
	 * A critical section is considered as a reduction form if reduction variables recognized
	 * by Reduction.analyzeStatement2() are the only shared variables modified in the
	 * critical section.
	 * [CAUTION] Cetus compiler can recognize array reduction, but the array reduction 
	 * is not supported by standard OpenMP compilers. Therefore, below conversion may
	 * not be handled correctly by other OpenMP compilers.
	 * [FIXME] Reduction.analyzeStatement2() returns a set of reduction variables as expressions,
	 * but this method converts them into a set of symbols. This conversion loses some information
	 * and thus complex reduction expressions such as a[0][i] and a[i].b can not be handled properly;
	 * current translator supports only simple scalar or array variables.
	 */
	public void convertCritical2Reduction()
	{
		List<OmpAnnotation> ompPAnnots = IRTools.collectPragmas(program, OmpAnnotation.class, "parallel");
		Reduction redAnalysis = new Reduction(program);
		for (OmpAnnotation omp_annot : ompPAnnots)
		{
			Statement pstmt = (Statement)omp_annot.getAnnotatable();
			HashSet<Symbol> shared_set = (HashSet<Symbol>)omp_annot.get("shared");
			HashMap pRedMap = (HashMap)omp_annot.get("reduction");
			List<OmpAnnotation> ompCAnnots = IRTools.collectPragmas(pstmt, OmpAnnotation.class, "critical");
			for (OmpAnnotation cannot : ompCAnnots)
			{
				boolean foundError = false;
				Statement cstmt = (Statement)cannot.getAnnotatable();
				Set<Symbol> definedSymbols = DataFlowTools.getDefSymbol(cstmt);
				HashSet<Symbol> shared_subset = new HashSet<Symbol>();
				shared_subset.addAll(shared_set);
				Map<String, Set<Expression>> reduce_map = redAnalysis.analyzeStatement2(cstmt);
				Map<String, Set<Symbol>> reduce_map2 = new HashMap<String, Set<Symbol>>();
				if (!reduce_map.isEmpty())
				{
					// Remove reduction variables from shared_subset.
					for (String ikey : (Set<String>)(reduce_map.keySet())) {
						if( foundError ) {
							break;
						}
						Set<Expression> tmp_set = (Set<Expression>)reduce_map.get(ikey);
						HashSet<Symbol> redSet = new HashSet<Symbol>();
						for (Expression iexp : tmp_set) {
							//Symbol redSym = findsSymbol(shared_set, iexp.toString());
							Symbol redSym = SymbolTools.getSymbolOf(iexp);
							if( redSym != null ) {
								if( redSym instanceof VariableDeclarator ) {
									shared_subset.remove(redSym);
									redSet.add(redSym);
								} else {
									PrintTools.println("[INFO in convertCritical2Reduction()] the following expression has reduction pattern" +
											" but not handled by current translator: " + iexp, 1);
									//Skip current critical section.
									foundError = true;
									break;
									
								}
							} else {
								PrintTools.println("[WARNING in convertCritical2Reduction()] found unrecognizable reduction expression (" +
										iexp+")", 0);
								//Skip current critical section.
								foundError = true;
								break;
							}
						}
						reduce_map2.put(ikey, redSet);
					}
					//If error is found, skip current critical section.
					if( foundError ) {
						continue;
					}
					//////////////////////////////////////////////////////////////////////
					// If shared_subset and definedSymbols are disjoint,                //
					// it means that reduction variables are the only shared variables  //
					// defined in the critical section.                                 //
					//////////////////////////////////////////////////////////////////////
					if( Collections.disjoint(shared_subset, definedSymbols) ) {
						if( pRedMap == null ) {
							pRedMap = new HashMap();
							omp_annot.put("reduction", pRedMap);
						}
						for (String ikey : (Set<String>)(reduce_map2.keySet())) {
							Set<Symbol> tmp_set = (Set<Symbol>)reduce_map2.get(ikey);
							HashSet<Symbol> redSet = (HashSet<Symbol>)pRedMap.get(ikey);
							if( redSet == null ) {
								redSet = new HashSet<Symbol>();
								pRedMap.put(ikey, redSet);
							}
							redSet.addAll(tmp_set);
						}
						// Remove omp critical annotation and add comment annotation. 
						CommentAnnotation comment = new CommentAnnotation(cannot.toString());
						AnnotationStatement comment_stmt = new AnnotationStatement(comment);
						CompoundStatement parent = (CompoundStatement)cstmt.getParent();
						parent.addStatementBefore(cstmt, comment_stmt);
						cstmt.removeAnnotations(OmpAnnotation.class);
					}
				}
			}
		}
	}
	
	/**
	 * Delete dummy barriers that were used to help kernel splitting.
	 */
	private void cleanDummyBarriers( ) {
		List<OmpAnnotation> barrList = new LinkedList<OmpAnnotation>();
		DepthFirstIterator iter = new DepthFirstIterator(program);
		while ( iter.hasNext() )
		{
			Object o = iter.next();
			if ( o instanceof Annotatable )
			{
				Annotatable at = (Annotatable)o;
				OmpAnnotation omp_annot = at.getAnnotation(OmpAnnotation.class, "barrier");
				if ( (omp_annot != null) && ((String)omp_annot.get("barrier")).equals("dummy") )
					barrList.add(omp_annot);
			}
		}
		PrintTools.println("Number of dummy barriers to be removed = " + barrList.size(), 5);
		for( OmpAnnotation o_annot : barrList ) {
			Statement astmt = (Statement)o_annot.getAnnotatable();
			if( astmt != null ) {
				Traversable parent = astmt.getParent();
				if( parent != null )
					parent.removeChild(astmt);
				else
					PrintTools.println("[Error in cleanAdditionalBarriers()] parent is null!", 1);
			}
		}
	}
	
	/**
	 * Remove empty Omp clauses.
	 */
	public static void cleanEmptyOmpClauses(Program prog) {
		List<OmpAnnotation> omp_annots = null;
		DepthFirstIterator iter = new DepthFirstIterator(prog);
		while ( iter.hasNext() )
		{
			Object o = iter.next();
			if ( o instanceof Annotatable )
			{
				Annotatable at = (Annotatable)o;
				omp_annots = at.getAnnotations(OmpAnnotation.class);
				if ( (omp_annots != null) && (omp_annots.size() > 0) ) {
					OmpAnnotation annot = omp_annots.get(0);
					Set hSet = annot.get("shared");
					if( (hSet != null) && (hSet.size() == 0) ) {
						annot.remove("shared");
					}
					hSet = annot.get("private");
					if( (hSet != null) && (hSet.size() == 0) ) {
						annot.remove("private");
					}
					hSet = annot.get("firstprivate");
					if( (hSet != null) && (hSet.size() == 0) ) {
						annot.remove("firstprivate");
					}
					hSet = annot.get("threadprivate");
					if( (hSet != null) && (hSet.size() == 0) ) {
						annot.remove("threadprivate");
					}
					hSet = annot.get("copyin");
					if( (hSet != null) && (hSet.size() == 0) ) {
						annot.remove("copyin");
					}
					Map hMap = annot.get("reduction");
					if( (hMap != null) && (hMap.size() == 0) ) {
						annot.remove("reduction");
					}
				}
			}
		}
	}
	
	/**
	 * If a parallel region contains an omp-for loop with reduction clause , 
	 * and reduction variables are used only in the omp-for loop, the reduction
	 * clause can be moved up to the enclosing parallel region.
	 */
	private void updateReductionClause() {
		List<OmpAnnotation> annotList = IRTools.collectPragmas(program, 
				OmpAnnotation.class, "parallel");
		for( OmpAnnotation annot : annotList ) {
			Annotatable at = annot.getAnnotatable();
			if( at instanceof CompoundStatement ) {
				CompoundStatement cStmt = (CompoundStatement)at;
				boolean ompForLoopOnly = true;
				ForLoop ompForLoop = null;
				for( Traversable t : cStmt.getChildren() ) {
					if( t instanceof AnnotationStatement ) {
						continue;
					} else if( t instanceof ForLoop ) {
						ForLoop floop = (ForLoop)t;
						if( floop.containsAnnotation(OmpAnnotation.class, "for") &&
								floop.containsAnnotation(OmpAnnotation.class, "reduction")) {
							ompForLoop = floop;
						} else {
							ompForLoopOnly = false;
						}
					} else {
						ompForLoopOnly = false;
					}
				}
				// Move reduction clause to the annotation of the enclosing parallel region.
				if( ompForLoop != null) {
					OmpAnnotation o_annot = ompForLoop.getAnnotation(OmpAnnotation.class, 
							"reduction");
					HashMap reduction_map1 = o_annot.get("reduction");
					HashMap reduction_map2 = annot.get("reduction");
					Collection<Symbol> sharedSet = annot.get("shared");
					Collection<Symbol> tCollect1 = null;
					Collection<Symbol> tCollect2 = null;
					if( ompForLoopOnly ) {
						// The enclosing parallel region has only this omp-for loop.
						if( reduction_map2 == null ) {
							annot.put("reduction", reduction_map1);
							for (String ikey : (Set<String>)(reduction_map1.keySet())) {
								tCollect1 = (Collection<Symbol>)reduction_map1.get(ikey);
								if( sharedSet != null ) {
									sharedSet.removeAll(tCollect1);
								}
							}
						} else {
							for (String ikey : (Set<String>)(reduction_map1.keySet())) {
								tCollect1 = (Collection<Symbol>)reduction_map1.get(ikey);
								if( sharedSet != null ) {
									sharedSet.removeAll(tCollect1);
								}
								if( reduction_map2.keySet().contains(ikey) ) {
									tCollect2 = (Collection<Symbol>)reduction_map2.get(ikey);
									tCollect2.addAll(tCollect1);
								} else {
									reduction_map2.put(ikey, tCollect1);
								}
							}
						}
						o_annot.remove("reduction");
					} else {
						/////////////////////////////////////////////////////////////////////////////
						//If reduction variables are used only in this omp-for loop, the reduction //
						//clause can be moved to the enclosing parallel region too.                //
						/////////////////////////////////////////////////////////////////////////////
						if( reduction_map2 == null ) {
							reduction_map2 = new HashMap<String, Collection<Symbol>>(); 
							annot.put("reduction", reduction_map2);
						}
						Set<String> removeSet = new HashSet<String>();
						for (String ikey : (Set<String>)(reduction_map1.keySet())) {
							tCollect1 = (Collection<Symbol>)reduction_map1.get(ikey);
							Collection<Symbol> tCollect3 = new HashSet<Symbol>(tCollect1);
							//////////////////////////////////////////////////////////////////////
							//Check whether reduction variables are used in other statements.   //
							//If not, the reduction variables can be moved up to the annotation //
							//of the enclosing parallel region.                                 //      
							//////////////////////////////////////////////////////////////////////
							for( Symbol sym : tCollect1 ) {
								for( Traversable t : cStmt.getChildren() ) {
									if( t.equals(ompForLoop) ) {
										continue;
									}
									if( IRTools.containsSymbol(t, sym) ) {
										tCollect3.remove(sym);
									}
								}
							}
							if( tCollect3.size() > 0 ) {
								if( sharedSet != null ) {
									sharedSet.removeAll(tCollect3);
								}
								tCollect1.removeAll(tCollect3);
								if( tCollect1.size() == 0 ) {
									//reduction_map1.remove(ikey);
									removeSet.add(ikey);
								}
								if( reduction_map2.keySet().contains(ikey) ) {
									tCollect2 = (Collection<Symbol>)reduction_map2.get(ikey);
									tCollect2.addAll(tCollect3);
								} else {
									reduction_map2.put(ikey, tCollect3);
								}
							}
						}
						for( String iKey : removeSet ) {
							reduction_map1.remove(iKey);
						}
						if( reduction_map1.size() == 0 ) {
							o_annot.remove("reduction");
						}
						if( reduction_map2.size() == 0 ) {
							annot.remove("reduction");
						}
					}
				}
			}
		}
	}
	
	/**
	 * If a shared variable is used only as a private variable in a parallel region,
	 * remove it from the shared clause and add into private clause.
	 * FIXME: firstprivate variable should be checked too.
	 */
	private void updatePrivateClause() {
		List<OmpAnnotation> annotList = IRTools.collectPragmas(program, 
				OmpAnnotation.class, "parallel");
		for( OmpAnnotation annot : annotList ) {
			Annotatable at = annot.getAnnotatable();
			if( at instanceof CompoundStatement ) {
				CompoundStatement cStmt = (CompoundStatement)at;
				Collection<Symbol> sharedSet = annot.get("shared");
				if( (sharedSet == null) || (sharedSet.size() == 0) ) {
					continue;
				}
				Set<Symbol> ipAccessedSymbols = new HashSet<Symbol>();
				List<FunctionCall> calledFuncs = IRTools.getFunctionCalls(cStmt);
				for( FunctionCall call : calledFuncs ) {
					Procedure called_procedure = call.getProcedure();
					if( called_procedure != null ) {
						CompoundStatement body = called_procedure.getBody();
						Set<Symbol> procAccessedSymbols = AnalysisTools.getIpAccessedGlobalorStaticSymbols(body);
						ipAccessedSymbols.addAll(procAccessedSymbols);
					}
				}
				Collection<Symbol> privateSet = annot.get("private");
				if( privateSet == null ) {
					privateSet = new HashSet<Symbol>();
					annot.put("private", privateSet);
				}
				Collection<Symbol> tPrivSet = null;
				Set<Symbol> removeSet = new HashSet<Symbol>();
				for( Symbol sym : sharedSet ) {
					boolean usedAsPrivateOnly = true;
					if( ipAccessedSymbols.contains(sym) ) {
						/////////////////////////////////////////////////////////////////////
						// FIXME: If a variable in omp-shared set is accessed in any of    //
						// functions called in the current parallel region, conservatively //
						// assume that it is accessed as shared variable.                  //
						// For better accuracy, each called function should be examined.   //
						/////////////////////////////////////////////////////////////////////
						usedAsPrivateOnly = false;
					} else {
						for( Traversable t : cStmt.getChildren() ) {
							if( t instanceof AnnotationStatement ) {
								continue;
							} else if( IRTools.containsSymbol(t, sym) ) {
								if( t instanceof Annotatable ) {
									Annotatable c_at = (Annotatable)t;
									OmpAnnotation o_annot = c_at.getAnnotation(OmpAnnotation.class, "private");
									if( o_annot == null ) {
										usedAsPrivateOnly = false;
										break;
									} else {
										tPrivSet = o_annot.get("private");
										if( !tPrivSet.contains(sym) ) {
											usedAsPrivateOnly = false;
											break;
										}
									}
								} else {
									usedAsPrivateOnly = false;
									break;
								}
							}
						}
					}
					if( usedAsPrivateOnly ) {
						//sharedSet.remove(sym);
						removeSet.add(sym);
						privateSet.add(sym);
						if( tPrivSet != null ) {
							tPrivSet.remove(sym);
						}
					}
				}
				sharedSet.removeAll(removeSet);
				if( privateSet.size() == 0 ) {
					annot.remove("private");
				}
				if( sharedSet.size() == 0 ) {
					annot.remove("shared");
				}
			}
		}
	}
	
	/**
	 * Delete extra OpenMP clauses, which were added by omp2gpu.analysis.OmpAnalysis,
	 * and delete barriers inserted by mark_interval().
	 * This method should be called to make the output code conform to OpenMP specification.
	 * @param prog input program
	 */
	static public void cleanExtraOmpClauses(Program prog) {
		List<OmpAnnotation> barrList = new LinkedList<OmpAnnotation>();
		DepthFirstIterator iter = new DepthFirstIterator(prog);
		while ( iter.hasNext() )
		{
			Object o = iter.next();
			if ( o instanceof Annotatable )
			{
				Annotatable at = (Annotatable)o;
				OmpAnnotation omp_annot = at.getAnnotation(OmpAnnotation.class, "barrier");
				if ( omp_annot != null ) {
					if( ((String)(omp_annot.get("barrier"))).equals("S2P") ||
							((String)(omp_annot.get("barrier"))).equals("P2S") ) {
						barrList.add(omp_annot);
					}
					if( ((String)(omp_annot.get("barrier"))).equals("P2P") ||
							((String)(omp_annot.get("barrier"))).equals("S2S") ) {
						barrList.add(omp_annot);
					}
				}
				omp_annot = at.getAnnotation(OmpAnnotation.class, "parallel");
				if( omp_annot != null ) {
					omp_annot.remove("threadprivate");
				}
				omp_annot = at.getAnnotation(OmpAnnotation.class, "for");
				if( omp_annot != null ) {
					omp_annot.remove("threadprivate");
					if( !omp_annot.containsKey("parallel") ) {
						omp_annot.remove("shared");
					}
				}
			}
		}
		PrintTools.println("Number of extra barriers to be removed = " + barrList.size(), 5);
		for( OmpAnnotation o_annot : barrList ) {
			Statement astmt = (Statement)o_annot.getAnnotatable();
			if( astmt != null ) {
				Traversable parent = astmt.getParent();
				if( parent != null )
					parent.removeChild(astmt);
				else
					PrintTools.println("[Error in cleanExtraOmpClauses()] parent is null!", 1);
			}
		}
	}
	
	static public void cleanExtraBarriers(Program prog, boolean deleteAllBarriers) {
		List<OmpAnnotation> barrList = new LinkedList<OmpAnnotation>();
		DepthFirstIterator iter = new DepthFirstIterator(prog);
		while ( iter.hasNext() )
		{
			Object o = iter.next();
			if ( o instanceof Annotatable )
			{
				Annotatable at = (Annotatable)o;
				OmpAnnotation omp_annot = at.getAnnotation(OmpAnnotation.class, "barrier");
				if ( omp_annot != null ) {
					if( ((String)(omp_annot.get("barrier"))).equals("S2P") ||
							((String)(omp_annot.get("barrier"))).equals("P2S") ) {
						barrList.add(omp_annot);
					}
					if( deleteAllBarriers && ( ((String)(omp_annot.get("barrier"))).equals("P2P") ||
							((String)(omp_annot.get("barrier"))).equals("S2S") ) ) {
						barrList.add(omp_annot);
					}
				}
			}
		}
		PrintTools.println("Number of extra barriers to be removed = " + barrList.size(), 5);
		for( OmpAnnotation o_annot : barrList ) {
			Statement astmt = (Statement)o_annot.getAnnotatable();
			if( astmt != null ) {
				Traversable parent = astmt.getParent();
				if( parent != null )
					parent.removeChild(astmt);
				else
					PrintTools.println("[Error in cleanExtraBarriers()] parent is null!", 1);
			}
		}
	}
	
	/**
	 * Add nowait clause to omp-for loops existing in omp paralllel region.
	 * This addition is needed to avoid unnecessary splitting when an input program, 
	 * which was split by SplitOmpPRegion, is fed to the O2G translator again.
	 * This method should be called only after SplitOmpPRegion.splitParallelRegions() 
	 * is called.
	 * 
	 * @param prog input program
	 */
	static public void addNowaitToOmpForLoops(Program prog) {
		DepthFirstIterator iter = new DepthFirstIterator(prog);
		while ( iter.hasNext() )
		{
			Object o = iter.next();
			if ( o instanceof Annotatable )
			{
				Annotatable at = (Annotatable)o;
				OmpAnnotation omp_annot = at.getAnnotation(OmpAnnotation.class, "for");
				if( omp_annot != null ) {
					if( !omp_annot.containsKey("parallel") && !omp_annot.containsKey("nowait") ) {
						omp_annot.put("nowait", "true");
					}
				}
			}
		}
	}
	
	/**
	 * Create a HashMap which contains updated shared, reduction, private, and threadprivate data sets
	 * for the function called in a Omp parallel region. Depending on the sharing attributes of
	 * the actual arguments of the called function, corresponding formal parameters are 
	 * assigned to one of HashSets (shared, reduction, private, and threadprivate sets) in the HashMap.
	 * In addition, shared data that are accessed in the called function, but not passed 
	 * as parameters are added into the new shared set, and all local variables are added to 
	 * the new private set.
	 * 
	 * @param par_map HashMap of an enclosing parallel region.
	 * @param argList List of actual arguments passed into the function proc.
	 * @param proc Procedure that is called in a parallel region.
	 * @return New HashMap that contains updated shared, private, and threadprivate data sets.
	 */
	static private HashMap updateOmpMapForCalledFunc(HashMap par_map, List<Expression> argList, Procedure proc) {
		HashSet<Symbol> old_set = null;
		HashSet<Symbol> new_set = null;
		HashSet<Symbol> argSyms = new HashSet<Symbol>();
		HashMap new_map = new HashMap();
		// Copy all hash mapping except for shared, private, firstprivate, and threadprivate data sets
		new_map.putAll(par_map); 
		new_map.remove("shared");
		new_map.remove("reduction");
		new_map.remove("private");
		new_map.remove("firstprivate");
		new_map.remove("threadprivate");
		
		Set<Symbol> accessedSymbols = SymbolTools.getAccessedSymbols(proc.getBody());
		
		List paramList = proc.getParameters();
		int list_size = paramList.size();
		for(int i=0; i<list_size; i++) {
			Expression arg = argList.get(i);
			Set<Expression> UseSet = DataFlowTools.getUseSet(arg);
			for( Expression exp : UseSet) {
				if( exp instanceof Identifier ) {
					Symbol sm = SymbolTools.getSymbolOf(exp);
					argSyms.add(sm);
				}
			}
		}
		if( par_map.keySet().contains("shared") ) {
			old_set = (HashSet<Symbol>)par_map.get("shared");
			new_set = new HashSet<Symbol>();
			// If actual argument is shared, put corresponding parameter into the new shared set.
			for(int i=0; i<list_size; i++) {
				Expression arg = argList.get(i);
				Set<Expression> UseSet = DataFlowTools.getUseSet(arg);
				for( Expression exp : UseSet) {
					Symbol sm = SymbolTools.getSymbolOf(exp);
					if(old_set.contains(sm)) {
						Object obj = paramList.get(i);
						if( obj instanceof VariableDeclaration ) {
							VariableDeclarator vdecl = 
								(VariableDeclarator)((VariableDeclaration)obj).getDeclarator(0);
							new_set.add(vdecl);
						} 
						break;
					}
				}
			}
			// Put other shared variables in the old_set, which are accessed 
			// in the called function, into the new set.
			for( Symbol ssm : old_set ) {
				if( accessedSymbols.contains(ssm) ) {
					new_set.add(ssm);
				}
			}
			new_map.put("shared", new_set);
		}
		
		if( par_map.keySet().contains("reduction") ) {
			HashMap reduction_map = (HashMap)par_map.get("reduction");
			HashMap newreduction_map = new HashMap(4);
			HashSet<Symbol> allItemsSet = new HashSet<Symbol>();
			for (String ikey : (Set<String>)(reduction_map.keySet())) {
				HashSet<Symbol> o_set = (HashSet<Symbol>)reduction_map.get(ikey);
				HashSet<Symbol> n_set = new HashSet<Symbol>();
				// If actual argument is a reduction variable, put corresponding 
				// parameter into the new reduction set.
				for(int i=0; i<list_size; i++) {
					Expression arg = argList.get(i);
					Set<Expression> UseSet = DataFlowTools.getUseSet(arg);
					for( Expression exp : UseSet) {
						Symbol sm = SymbolTools.getSymbolOf(exp);
						if(o_set.contains(sm)) {
							Object obj = paramList.get(i);
							if( obj instanceof VariableDeclaration ) {
								VariableDeclarator vdecl = 
									(VariableDeclarator)((VariableDeclaration)obj).getDeclarator(0);
								n_set.add(vdecl);
							} 
							break;
						}
					}
				}
				// Put other reduction variables in the o_set, which are accessed 
				// in the called function, into the n_set.
				for( Symbol ssm : o_set ) {
					if( accessedSymbols.contains(ssm) ) {
						n_set.add(ssm);
					}
				}
				newreduction_map.put(ikey, n_set);
			}
			new_map.put("reduction", newreduction_map);
		}
		/*
		 * FIXME: What if a private variable is passed as a reference?
		 *        What if an argument consists of both shared and private variables?
		 */
		if( par_map.keySet().contains("private") ) {
			old_set = (HashSet<Symbol>)par_map.get("private");
			new_set = new HashSet<Symbol>();
			// If actual argument is private, put corresponding parameter into the new private set.
			for(int i=0; i<list_size; i++) {
				Expression arg = argList.get(i);
				Set<Expression> UseSet = DataFlowTools.getUseSet(arg);
				for( Expression exp : UseSet) {
					Symbol sm = SymbolTools.getSymbolOf(exp);
					if(old_set.contains(sm)) {
						Object obj = paramList.get(i);
						if( obj instanceof VariableDeclaration ) {
							VariableDeclarator vdecl = (VariableDeclarator)((VariableDeclaration)obj).getDeclarator(0);
							if( SymbolTools.isScalar(vdecl) && !SymbolTools.isPointer(vdecl) ) {
								new_set.add(vdecl);
							} else {
								PrintTools.println("[WARNING] private variable, "+vdecl.getSymbolName()+", " +
										"is passed as a reference in procedure, " +  proc.getSymbolName() + 
										"(); splitting parallel region in "+proc.getSymbolName()+"() may result in "
										+ "incorrect output codes if "+vdecl.getSymbolName()+" upwardly exposed " +
										"in "+proc.getSymbolName()+"().", 0);
								new_set.add(vdecl);
							}
						} 
						break;
					}
				}
			}
			// Put other private variables in the old_set, which are accessed 
			// in the called function, into the new set.
			for( Symbol ssm : old_set ) {
				if( accessedSymbols.contains(ssm) ) {
					new_set.add(ssm);
				}
			}
			// Put other private variables that are declared within this function call.
			Set<Symbol> localSymbols = new HashSet<Symbol>();
			DepthFirstIterator iter = new DepthFirstIterator(proc.getBody());
			while(iter.hasNext())
			{
				Object obj = iter.next();	
				if( obj instanceof SymbolTable ) {
					localSymbols.addAll(SymbolTools.getVariableSymbols((SymbolTable)obj));
				}
			}
			Set<Symbol> StaticLocalSet = getStaticVariables(localSymbols);
			new_set.addAll(localSymbols);
			new_set.removeAll(StaticLocalSet);
			new_map.put("private", new_set);
			////////////////////////////////////////////////////////////////////////////////
			//If shared variable is used as a private variable, it should be removed from //
			// the shared set.                                                            //
			////////////////////////////////////////////////////////////////////////////////
			old_set = (HashSet<Symbol>) new_map.get("shared");
			old_set.removeAll(new_set);
		}
		
		if( par_map.keySet().contains("firstprivate") ) {
			old_set = (HashSet<Symbol>)par_map.get("firstprivate");
			new_set = new HashSet<Symbol>();
			// If actual argument is firstprivate, put corresponding parameter into the new firstprivate set.
			for(int i=0; i<list_size; i++) {
				Expression arg = argList.get(i);
				Set<Expression> UseSet = DataFlowTools.getUseSet(arg);
				for( Expression exp : UseSet) {
					Symbol sm = SymbolTools.getSymbolOf(exp);
					if(old_set.contains(sm)) {
						Object obj = paramList.get(i);
						if( obj instanceof VariableDeclaration ) {
							VariableDeclarator vdecl = (VariableDeclarator)((VariableDeclaration)obj).getDeclarator(0);
							if( SymbolTools.isScalar(vdecl) && !SymbolTools.isPointer(vdecl) ) {
								new_set.add(vdecl);
							} else {
								PrintTools.println("[WARNING] firstprivate variable, "+vdecl.getSymbolName()+", " +
										"is passed as a reference in procedure, " +  proc.getSymbolName() + 
										"(); splitting parallel region in "+proc.getSymbolName()+"() may result in "
										+ "incorrect output codes if "+vdecl.getSymbolName()+" upwardly exposed " +
										"in "+proc.getSymbolName()+"().", 0);
								new_set.add(vdecl);
							}
						} 
						break;
					}
				}
			}
			// Put other firstprivate variables in the old_set, which are accessed 
			// in the called function, into the new set.
			for( Symbol ssm : old_set ) {
				if( accessedSymbols.contains(ssm) ) {
					new_set.add(ssm);
				}
			}
			new_map.put("firstprivate", new_set);
			////////////////////////////////////////////////////////////////////////////////
			//If shared variable is used as a private variable, it should be removed from //
			// the shared set.                                                            //
			////////////////////////////////////////////////////////////////////////////////
			old_set = (HashSet<Symbol>) new_map.get("shared");
			old_set.removeAll(new_set);
		}
		
		if( par_map.keySet().contains("threadprivate") ) {
			old_set = (HashSet<Symbol>)par_map.get("threadprivate");
			new_set = new HashSet<Symbol>();
			for(int i=0; i<list_size; i++) {
				Expression arg = argList.get(i);
				Set<Expression> UseSet = DataFlowTools.getUseSet(arg);
				for( Expression exp : UseSet) {
					Symbol sm = SymbolTools.getSymbolOf(exp);
					if(old_set.contains(sm)) {
						Object obj = paramList.get(i);
						if( obj instanceof VariableDeclaration ) {
							VariableDeclarator vdecl = (VariableDeclarator)((VariableDeclaration)obj).getDeclarator(0);
							new_set.add(vdecl);
						} 
						break;
					}
				}
			}
			new_map.put("threadprivate", new_set);
		}
		
		
		// Update annotations of omp-for loops enclosed by the called procedure, proc.
		List<OmpAnnotation> ompfor_annotList = 
			IRTools.collectPragmas(proc.getBody(), OmpAnnotation.class, "for");
		for( OmpAnnotation ompfor_annot : ompfor_annotList ) {
			Statement atstmt = (Statement)ompfor_annot.getAnnotatable();
			accessedSymbols = SymbolTools.getAccessedSymbols(atstmt);
			if( ompfor_annot.keySet().contains("shared") ) {
				HashSet<Symbol> o_set = (HashSet<Symbol>)ompfor_annot.remove("shared");
				HashSet<Symbol> n_set = new HashSet<Symbol>();
				new_set = (HashSet<Symbol>)new_map.get("shared");
				for( Symbol sm : new_set ) {
					if( accessedSymbols.contains(sm)) {
						n_set.add(sm);
					}
				}
				for( Symbol sm : o_set ) {
					if( !n_set.contains(sm)) {
						n_set.add(sm);
					}
				}
				ompfor_annot.put("shared", n_set);
			}
		}
		return new_map;
	}

	static private HashSet<Symbol> getStaticVariables(Set<Symbol> iset)
	{
		HashSet<Symbol> ret = new HashSet<Symbol> ();
		for (Symbol symbol : iset)
		{
			List<Specifier> type_specs = symbol.getTypeSpecifiers();
			for (Specifier spec : type_specs)
			{
				if ( spec.toString().compareTo("static")==0 )
				{
					ret.add(symbol);
					break;
				}
			}
		}
		return ret;
	}

}
