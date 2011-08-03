/**
 * 
 */
package omp2gpu.analysis;

import omp2gpu.hir.CudaAnnotation;
import omp2gpu.transforms.SplitOmpPRegion;
import omp2gpu.transforms.TransformTools;
import cetus.analysis.AnalysisPass;
import cetus.analysis.CFGraph;
import cetus.analysis.DFANode;
import cetus.exec.Driver;
import cetus.hir.Annotatable;
import cetus.hir.AnnotationStatement;
import cetus.hir.BreadthFirstIterator;
import cetus.hir.CommentAnnotation;
import cetus.hir.CompoundStatement;
import cetus.hir.Declaration;
import cetus.hir.Declarator;
import cetus.hir.DepthFirstIterator;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.FunctionCall;
import cetus.hir.IDExpression;
import cetus.hir.Identifier;
import cetus.hir.NameID;
import cetus.hir.OmpAnnotation;
import cetus.hir.Procedure;
import cetus.hir.ProcedureDeclarator;
import cetus.hir.Program;
import cetus.hir.Specifier;
import cetus.hir.Statement;
import cetus.hir.StandardLibrary;
import cetus.hir.SymbolTools;
import cetus.hir.Tools;
import cetus.hir.DataFlowTools;
import cetus.hir.PrintTools;
import cetus.hir.IRTools;
import cetus.hir.Symbol;
import cetus.hir.TranslationUnit;
import cetus.hir.Traversable;
import cetus.hir.VariableDeclaration;
import cetus.hir.VariableDeclarator;

import java.util.HashSet;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Set;
import java.util.Stack;
import java.util.TreeMap;
import java.util.Iterator;

/**
 * @author Seyong Lee <lee222@purdue.edu>
 *         ParaMount Group 
 *         School of ECE, Purdue University
 *         
 * Inter-procedural, forward data-flow analysis to compute OpenMP shared variables 
 * residing in the GPU global memory.
 * 
 * Input  : input program 
 * Output : set of CUDA clauses(nocudamalloc, nocudafree, and noc2gmemtr) annotated 
 * to each kernel region. These clauses are calculated based on gResidentGVars and 
 * gMallocVars;
 * 	at a barrier node just before each kernel region
 *     - if a shared variable exists in gResidentGVars_in set,
 *     		- add the variable into noc2gmemtr clause for the kernel region.
 *     - if a shared variable exists in gMallocVars_in set,
 *     		- add the variable into nocudamalloc clause and nocudafree clause 
 *            for the kernel region. 
 * 	at a barrier node just after each kernel region
 *     - if a shared variable exists in gMallocVars_in set,
 *     		- add the variable into the nocudafree clause 
 *            for the kernel region. 
 *
 * gResidentGVars_in(program entry-node) = {}
 *
 * for ( node m : predecessor nodes of node n )
 * 	gResidentGVars_in(n)  ^= gResidentGVars_out(m) // ^ : intersection
 * 
 * gResidentGVars_out(n) = gResidentGVars_in(n) + GEN(n) - KILL(n) // + : union
 *  where,
 *   GEN(n) = set of shared variables whose GPU variables are globally allocated
 *                - if n is a barrier node after a kernel region.
 *            ()  - otherwise 
 *   KILL(n) = set of reduction variables in a kernel region
 *                 - if n is a barrier node after a kernel region.
 *             set of shared variables modified in a CPU region
 *                 - if n represents a node in a CPU region.
 *             set of R/O shared scalar variables in a kernel region
 *             		- if the variables do not exist in gResidentGVars_in set
 *                    and if shrdSclrCachingOnSM option is on
 *                    and if n is a barrier node after a kernel region.
 *             ()  - otherwise
 *             
 * gMallocVars(program entry-node) = {}
 *
 * for ( node m : predecessor nodes of node n )
 * 	gMallocVars_in(n)  ^= gMallocVars_out(m) // ^ : intersection
 * 
 * gMallocVars_out(n) = gMallocVars_in(n) + GEN(n) // + : union
 *  where,
 *   GEN(n) = set of shared variables whose GPU variables are globally allocated
 *                - if n is a barrier node after a kernel region.
 *            ()  - otherwise 
 *   KILL(n)= set of R/O shared scalar variables in a kernel region
 *           	  - if the variables do not exist in gMallocVars_in set
 *                  and if shrdSclrCachingOnSM option is on
 *                  and if n is a barrier node after a kernel region.
 * 
 * For each kernel region, gResidentGVars_in set and gMallocVars_in set are stored in 
 * a barrier just before the kernel region.
 * This analysis is context-sensitive; if the same procedure is called with different
 * context, the procedure is cloned and the function call is swapped with the new one 
 * that calls the cloned procedure.
 */
public class IpResidentGVariableAnalysis extends AnalysisPass {
	private boolean assumeNonZeroTripLoops;
	private boolean showGResidentGVars;
	private boolean shrdSclrCachingOnSM;
	private HashMap<Symbol, Symbol> l2gGVMap;
	private Stack<HashMap<Symbol, Symbol>> l2gGVMapStack;
	private HashMap<Procedure, HashSet<Symbol>> visitedProcs;
	private String currentRegion;
	private Procedure main;

	/**
	 * @param program
	 */
	public IpResidentGVariableAnalysis(Program program) {
		super(program);
	}

	/* (non-Javadoc)
	 * @see cetus.analysis.AnalysisPass#getPassName()
	 */
	@Override
	public String getPassName() {
		return new String("[ipResidentGVariableAnalysis]");
	}

	/* (non-Javadoc)
	 * @see cetus.analysis.AnalysisPass#start()
	 */
	@Override
	public void start() {
		main = null;
		l2gGVMapStack = new Stack<HashMap<Symbol, Symbol>>();
		String value = Driver.getOptionValue("useGlobalGMalloc");
		if( value == null ) {
			PrintTools.println("[WARNING in IpResidentGVariableAnalysis()] " +
					"to run this analysis, useGlobalGMalloc option should be on; " +
					"ignore this analysis!", 0);
			return;
		}
		value = Driver.getOptionValue("showGResidentGVars");
		if( value == null ) {
			showGResidentGVars = false;
		} else {
			showGResidentGVars = true;
		}
		value = Driver.getOptionValue("shrdSclrCachingOnSM");
		if( value == null ) {
			shrdSclrCachingOnSM = false;
		} else {
			shrdSclrCachingOnSM = true;
		}
		for ( Traversable tu : program.getChildren() ) {
			if( main == null ) {
				BreadthFirstIterator iter = new BreadthFirstIterator(tu);
				iter.pruneOn(Procedure.class);

				for (;;)
				{
					Procedure proc = null;

					try {
						proc = (Procedure)iter.next(Procedure.class);
					} catch (NoSuchElementException e) {
						break;
					}

					String name = proc.getName().toString();

					/* f2c code uses MAIN__ */
					if (name.equals("main") || name.equals("MAIN__")) {
						main = proc;
						break;
					}
				}
			} else {
				break;
			}
		}
		if( main == null ) {
			Tools.exit("[ERROR in ipResidentGVariableAnalysis] can't find a main()");
		}
		AnalysisTools.markIntervalForKernelRegions(program);
		assumeNonZeroTripLoops = false;
		value = Driver.getOptionValue("assumeNonZeroTripLoops");
		if( value != null ) {
			assumeNonZeroTripLoops = true;
		}
		// Initialize currentRegion.
		currentRegion = new String("CPU");
		visitedProcs = new HashMap<Procedure, HashSet<Symbol>>();
		HashSet<Symbol> dummySet1 = new HashSet<Symbol>();
		HashSet<Symbol> dummySet2 = new HashSet<Symbol>();
		// Start interprocedural analysis from main() procedure.
		gResidentGVAnalysis(main, dummySet1, dummySet2, null);
		SplitOmpPRegion.cleanExtraBarriers(program, false);

	}
	
	private boolean gResidentGVAnalysis(Procedure proc, HashSet<Symbol> gRGVSet, 
			HashSet<Symbol> gMallocSet, FunctionCall funcCall) {
		boolean containsStaticData = false;
		Procedure clonedProc = null;
		VariableDeclaration clonedProcDecl = null;
		FunctionCall orgFCall = funcCall;
		FunctionCall newFCall = null;
		boolean AnnotationAdded = false;
		l2gGVMap = new HashMap<Symbol, Symbol>();
		if( visitedProcs.containsKey(proc) ) {
			HashSet<Symbol> prevContext = visitedProcs.get(proc);
			/////////////////////////////////////////////////////////////
			// If the same procedure is called with different context, //
			// create a new procedure by cloning.                      //
			/////////////////////////////////////////////////////////////
			if( !prevContext.equals(gRGVSet) ) {
				boolean cloneProcedure = false;
				int k = 0;
				/////////////////////////////////////////////////////////////////
				// Check whether this procedure is clonable; if this procedure //
				// contains static variables, it can not be cloned.            //
				/////////////////////////////////////////////////////////////////
				Set<Symbol> staticSyms = AnalysisTools.findStaticSymbols(proc.getBody());
				if( staticSyms.size() > 0 ) {
					containsStaticData = true;
					prevContext.retainAll(gRGVSet);
					// Delete Comments containing GResidentGPUVariables.
					DepthFirstIterator itr = new DepthFirstIterator(proc);
					while(itr.hasNext())
					{
						Object obj = itr.next();

						if ( (obj instanceof Annotatable) && (obj instanceof Statement) )
						{
							Annotatable at = (Annotatable)obj;
							List<CommentAnnotation> aList = at.getAnnotations(CommentAnnotation.class);
							if( aList != null ) {
								List<CommentAnnotation> newList = new LinkedList<CommentAnnotation>();
								for( CommentAnnotation cAnnot : aList ) {
									String comment = cAnnot.get("comment");
									if( !comment.startsWith("GResidentGPUVariables") ) {
										newList.add(cAnnot);
									}
									at.removeAnnotations(CommentAnnotation.class);
									if( newList.size() > 0 ) {
										for( CommentAnnotation newAnnot : newList ) {
											at.annotate(newAnnot);
										}
									} else if( obj instanceof AnnotationStatement ) {
										Traversable p = ((Traversable)obj).getParent();
										p.removeChild((Traversable)obj);
									}
								}
							}
						}
					}
/*					Tools.exit("[ERROR in gResidentGVAnalysis()] procedure (" +
							proc.getSymbolName() + ") can not be cloned, since it " +
							"contains static symbols (" + AnalysisTools.symbolsToString(staticSyms, ",") +
							"); either turn off globalGMallocOpt option, or remove the static variables " +
					"by promoting them as global variables.");*/
					PrintTools.println("[WARNING in gResidentGVAnalysis()] procedure (" +
							proc.getSymbolName() + ") can not be cloned, since it " +
							"contains static symbols (" + AnalysisTools.symbolsToString(staticSyms, ",") +
							"); a conservative analysis will be conducted. For more accurate analysis, " +
							"remove the static variables by promoting them as global variables, and run " +
							"this analysis again.", 0);
				} else {
					//////////////////////////////////////////////////////////////
					// Find existing cloned procedure with the same context, or //
					// create a new clone procedure.                            //
					//////////////////////////////////////////////////////////////
					Set<Procedure> procSet = visitedProcs.keySet();
					HashMap<String, Procedure> procMap = new HashMap<String, Procedure>();
					for( Procedure tProc : procSet ) {
						procMap.put(tProc.getSymbolName(), tProc);
					}
					String new_proc_name = proc.getSymbolName() + "_cloned" + k++;
					while(true) {
						Procedure tProc = procMap.get(new_proc_name);
						if( tProc == null ) {
							cloneProcedure = true;
							break;
						} else {
							prevContext = visitedProcs.get(tProc);
							if( prevContext.equals(gRGVSet) ) {
								//found the cloned procedure with the same context. 
								proc = tProc;
								cloneProcedure = false;
								break;
							} else {
								//check another cloned procedure.
								new_proc_name = proc.getSymbolName() + "_cloned" + k++;
							}
						}
					}
					if( cloneProcedure ) {
						List<Specifier> return_types = proc.getReturnType();
						List<VariableDeclaration> oldParamList = 
							(List<VariableDeclaration>)proc.getParameters();
						CompoundStatement body = (CompoundStatement)proc.getBody().clone();
						Procedure new_proc = new Procedure(return_types,
								new ProcedureDeclarator(new NameID(new_proc_name),
										new LinkedList()), body);	
						if( oldParamList != null ) {
							for( VariableDeclaration param : oldParamList ) {
								VariableDeclarator param_declarator = (VariableDeclarator)param.getDeclarator(0);
								VariableDeclaration cloned_decl = (VariableDeclaration)param.clone();
								///////////////////////
								// DEBUG: deprecated //
								///////////////////////
								//IDExpression paramID = param_declarator.getSymbol();
								//IDExpression cloned_ID = cloned_decl.getDeclarator(0).getSymbol();
								//cloned_ID.setSymbol((VariableDeclarator)cloned_decl.getDeclarator(0));
								Identifier paramID = new Identifier(param_declarator);
								Identifier cloned_ID = new Identifier((Symbol)cloned_decl.getDeclarator(0));
								new_proc.addDeclaration(cloned_decl);
								IRTools.replaceAll((Traversable) body, paramID, cloned_ID);
							}
						}
						TranslationUnit tu = (TranslationUnit)proc.getParent();
						////////////////////////////
						// Add the new procedure. //
						////////////////////////////
						/////////////////////////////////////////////////////////////////////
						// DEBUG: the following two commented blocks can't find function   //
						// declaration statements; it seems that 1) TranslationUnit symbol //
						// table contains a symbol of procedure, but not of procedure      //
						// declaration, and 2) ProcedureDeclarators used in a Procedure    //
						// and a procedure delaration are not identical.                   //
						/////////////////////////////////////////////////////////////////////
						/*					Traversable t = proc.getDeclarator().getParent();
					if( t != null ) {
						tu.addDeclarationAfter((Declaration)t, new_proc);
					} else {
						tu.addDeclarationAfter(proc, new_proc);
					}*/
						/*					Declaration procDecl = tu.findSymbol(proc.getName());
					tu.addDeclarationAfter(procDecl, new_proc);*/
						//////////////////////////////////////////////////////////////////
						//If declaration statement exists for the original procedure,   //
						//create a new declaration statement for the new procedure too. //
						//////////////////////////////////////////////////////////////////
						BreadthFirstIterator iter = new BreadthFirstIterator(tu);
						iter.pruneOn(ProcedureDeclarator.class);
						for (;;)
						{
							ProcedureDeclarator procDeclr = null;

							try {
								procDeclr = (ProcedureDeclarator)iter.next(ProcedureDeclarator.class);
							} catch (NoSuchElementException e) {
								break;
							}
							if( procDeclr.getID().equals(proc.getName()) ) {
								//Found function declaration.
								Declaration procDecl = (Declaration)procDeclr.getParent();
								//Create a new function declaration.
								VariableDeclaration newProcDecl = 
									new VariableDeclaration(new_proc.getReturnType(), new_proc.getDeclarator().clone());
								//Insert the new function declaration.
								tu.addDeclarationAfter(procDecl, newProcDecl);
								clonedProcDecl = newProcDecl;
								break;
							}
						}
						tu.addDeclarationAfter(proc, new_proc);
						clonedProc = new_proc;
						/////////////////////////////////////////////////////////////////////////
						// Update the newly cloned procedure:                                  //
						//     1) Update symbols in the new procedure, including symbols       //
						//        in OmpAnnoations.                                            //
						//     2) Delete CudaAnnotations previously inserted by this analysis. //
						/////////////////////////////////////////////////////////////////////////
						SymbolTools.linkSymbol(new_proc);
						TransformTools.updateAnnotationsInRegion(new_proc);
						DepthFirstIterator itr = new DepthFirstIterator(new_proc);
						while(itr.hasNext())
						{
							Object obj = itr.next();

							if ( (obj instanceof Annotatable) && (obj instanceof Statement) )
							{
								Annotatable at = (Annotatable)obj;
								at.removeAnnotations(CudaAnnotation.class);
								List<CommentAnnotation> aList = at.getAnnotations(CommentAnnotation.class);
								if( aList != null ) {
									List<CommentAnnotation> newList = new LinkedList<CommentAnnotation>();
									for( CommentAnnotation cAnnot : aList ) {
										String comment = cAnnot.get("comment");
										if( !comment.startsWith("GResidentGPUVariables") ) {
											newList.add(cAnnot);
										}
										at.removeAnnotations(CommentAnnotation.class);
										if( newList.size() > 0 ) {
											for( CommentAnnotation newAnnot : newList ) {
												at.annotate(newAnnot);
											}
										} else if( obj instanceof AnnotationStatement ) {
											Traversable p = ((Traversable)obj).getParent();
											p.removeChild((Traversable)obj);
										}
									}
								}
							}
						}

						proc = new_proc;
					}
					//////////////////////////////////////////////////////////
					// Create a new function call for the cloned procedure. //
					//////////////////////////////////////////////////////////
					if( funcCall != null ) {
						FunctionCall new_funcCall = new FunctionCall(new NameID(new_proc_name));
						List<Expression> argList = (List<Expression>)funcCall.getArguments();
						if( argList != null ) {
							for( Expression exp : argList ) {
								new_funcCall.addArgument(exp.clone());
							}
						}
						funcCall.swapWith(new_funcCall);
						newFCall = new_funcCall;
					}
					visitedProcs.put(proc, (HashSet<Symbol>)gRGVSet.clone());
				}
			}
		} else {
			// Delete Comments containing GResidentGPUVariables.
			DepthFirstIterator itr = new DepthFirstIterator(proc);
			while(itr.hasNext())
			{
				Object obj = itr.next();

				if ( (obj instanceof Annotatable) && (obj instanceof Statement) )
				{
					Annotatable at = (Annotatable)obj;
					List<CommentAnnotation> aList = at.getAnnotations(CommentAnnotation.class);
					if( aList != null ) {
						List<CommentAnnotation> newList = new LinkedList<CommentAnnotation>();
						for( CommentAnnotation cAnnot : aList ) {
							String comment = cAnnot.get("comment");
							if( !comment.startsWith("GResidentGPUVariables") && 
									!comment.startsWith("\nGResidentGPUVariables") ) {
								newList.add(cAnnot);
							}
							at.removeAnnotations(CommentAnnotation.class);
							if( newList.size() > 0 ) {
								for( CommentAnnotation newAnnot : newList ) {
									at.annotate(newAnnot);
								}
							} else if( obj instanceof AnnotationStatement ) {
								Traversable p = ((Traversable)obj).getParent();
								p.removeChild((Traversable)obj);
							}
						}
					}
				}
			}
			visitedProcs.put(proc, (HashSet<Symbol>)gRGVSet.clone());
		}
		
		PrintTools.println("[gResidentGVAnalysis] analyze " + proc.getSymbolName(), 1);
		
		OCFGraph.setNonZeroTripLoops(assumeNonZeroTripLoops);
		CFGraph cfg = new OCFGraph(proc, null);
		
		// sort the control flow graph
		cfg.topologicalSort(cfg.getNodeWith("stmt", "ENTRY"));
		
		// Annotate barriers enclosing kernel regions.
		AnalysisTools.annotateBarriers(proc, cfg);
		
		TreeMap work_list = new TreeMap();
		
		// Enter the entry node in the work_list
		DFANode entry = cfg.getNodeWith("stmt", "ENTRY");
		HashSet<Symbol> gResidentGVars_in = new HashSet<Symbol>();
		HashSet<Symbol> gResidentGVars_out = new HashSet<Symbol>();
		gResidentGVars_in.addAll(gRGVSet);
		gResidentGVars_out.addAll(gRGVSet);
		entry.putData("gResidentGVars_in", gResidentGVars_in);
		entry.putData("gResidentGVars_out", gResidentGVars_out);
		HashSet<Symbol> gMallocVars_in = new HashSet<Symbol>();
		HashSet<Symbol> gMallocVars_out = new HashSet<Symbol>();
		gMallocVars_in.addAll(gMallocSet);
		gMallocVars_out.addAll(gMallocSet);
		entry.putData("gMallocVars_in", gMallocVars_in);
		entry.putData("gMallocVars_out", gMallocVars_out);
		//work_list.put(entry.getData("top-order"), entry);
		// work_list contains all nodes except for the entry node.
		for ( DFANode succ : entry.getSuccs() ) {
			work_list.put(succ.getData("top-order"), succ);
		}
		
		// Do iterative steps
		while ( !work_list.isEmpty() )
		{
			DFANode node = (DFANode)work_list.remove(work_list.firstKey());
			
			String tag = (String)node.getData("tag");
			// Check whether the node is in the kernel region or not.
			if( tag != null && tag.equals("barrier") ) {
				String type = (String)node.getData("type");
				if( type != null ) {
					if( type.equals("S2P") ) {
						currentRegion = new String("GPU");
					} else if( type.equals("P2S") ) {
						currentRegion = new String("CPU");
					}
				}
			}
	
			gResidentGVars_in = null;
			gMallocVars_in = null;
			
	
			for ( DFANode pred : node.getPreds() )
			{
				Set<Symbol> pred_gResidentGVars_out = (Set<Symbol>)pred.getData("gResidentGVars_out");
				if ( gResidentGVars_in == null ) {
					if ( pred_gResidentGVars_out != null ) {
						gResidentGVars_in = new HashSet<Symbol>();
						gResidentGVars_in.addAll(pred_gResidentGVars_out);
					}
				} else {
					// Calculate intersection of previous nodes.
					if ( pred_gResidentGVars_out != null ) {
						gResidentGVars_in.retainAll(pred_gResidentGVars_out);
					} /* else {
						//This is the first visit to this node; ignore it
						//gResidentGVars_in.clear();
					} */
				}
				
				Set<Symbol> pred_gMallocVars_out = (Set<Symbol>)pred.getData("gMallocVars_out");
				if ( gMallocVars_in == null ) {
					if ( pred_gMallocVars_out != null ) {
						gMallocVars_in = new HashSet<Symbol>();
						gMallocVars_in.addAll(pred_gMallocVars_out);
					}
				} else {
					// Calculate intersection of previous nodes.
					if ( pred_gMallocVars_out != null ) {
						gMallocVars_in.retainAll(pred_gMallocVars_out);
					} /* else {
						//This is the first visit to this node; ignore it
						//gMallocVars_in.clear();
					} */
				}
			}
	
			// previous gResidentGVars_in
			Set<Symbol> p_gResidentGVars_in = (Set<Symbol>)node.getData("gResidentGVars_in");
			// previous gMallocVars_in
			Set<Symbol> p_gMallocVars_in = (Set<Symbol>)node.getData("gMallocVars_in");
	
			if ( (gResidentGVars_in == null) || (p_gResidentGVars_in == null) || !gResidentGVars_in.equals(p_gResidentGVars_in) ) {
				node.putData("gResidentGVars_in", gResidentGVars_in);
				node.putData("gMallocVars_in", gMallocVars_in);
				
				// compute gResidentGVars_out, a set of GPU variables residing  
				// in the GPU global memory.
				gResidentGVars_out = new HashSet<Symbol>();
				if( gResidentGVars_in != null ) {
					gResidentGVars_out.addAll(gResidentGVars_in);
				}
				// compute gMallocVars_out, a set of GPU variables allocated  
				// in the GPU global memory.
				gMallocVars_out = new HashSet<Symbol>();
				if( gMallocVars_in != null ) {
					gMallocVars_out.addAll(gMallocVars_in);
				}
				
				/////////////////////
				// Handle GEN set. //
				/////////////////////
				// Check whether the node contains "pKernelRegion" key, which is stored in a barrier
				// just after a kernel region.
				Statement stmt = node.getData("pKernelRegion");
				if( stmt != null ) {
					OmpAnnotation annot = stmt.getAnnotation(OmpAnnotation.class, "parallel");
					if( annot != null ) {
						Set<Symbol> sharedVars = (Set<Symbol>)annot.get("shared");
						if( sharedVars != null ) {
							Set<Symbol> redSyms = AnalysisTools.findReductionSymbols(stmt);
							Set<Symbol> defSyms = DataFlowTools.getDefSymbol(stmt);
							for( Symbol sym : sharedVars ) {
								Symbol gSym = null;
								if( l2gGVMap.containsKey(sym) ) {
									gSym = l2gGVMap.get(sym);
								} else {
									List symInfo = AnalysisTools.findOrgSymbol(sym, proc);
									if( symInfo.size() == 2 ) {
										gSym = (Symbol)symInfo.get(0);
										l2gGVMap.put(sym, gSym);
									} 
								}
								if( gSym != null ) {
									////////////////////////////////////////////////////////////////
									// If shrdSclrCachingOnSM option is on, R/O shared scalar     //
									// variables are not globally malloced in this kernel region. //
									////////////////////////////////////////////////////////////////
									if( SymbolTools.isScalar(sym) && !defSyms.contains(sym) &&
											shrdSclrCachingOnSM ) {
										continue;
									}
									/////////////////////////////////////////////////////////////
									// gResidentGVars contains GPU variables that are globally // 
									// allocated, but not used as reduction variables.         //
									/////////////////////////////////////////////////////////////
									if( !redSyms.contains(sym) ) {
										gResidentGVars_out.add(gSym);
									}
									gMallocVars_out.add(gSym);
								}
							}
						}
					} else {
						Tools.exit("[ERROR in gResidentGVariableAnalysis] Incorrect tag in a node: " + node);
					}
				}
				//////////////////////
				// Handle KILL set. //
				//////////////////////
				if( currentRegion.equals("CPU") ) {
					// Handle function calls interprocedurally.
					Traversable ir = node.getData("ir");
					if( (ir != null) && (ir instanceof ExpressionStatement) ) {
						ExpressionStatement estmt = (ExpressionStatement)ir;
						Expression expr = estmt.getExpression();
						List<FunctionCall> fcalls = IRTools.getFunctionCalls(expr);
						if( fcalls !=null ) {
							for( FunctionCall funCall : fcalls ) {
								if( !StandardLibrary.contains(funCall) ) {
									Procedure calledProc = funCall.getProcedure();
									if( calledProc != null ) {
										if( showGResidentGVars ) {
											StringBuilder str = new StringBuilder(100);
											str.append("GResidentGPUVariables: ");
											str.append(AnalysisTools.symbolsToString(gResidentGVars_out, ","));
											str.append("\nGMallocedGPUVariables: ");
											str.append(AnalysisTools.symbolsToString(gMallocVars_out, ","));
											CommentAnnotation cAnnot = new CommentAnnotation(str.toString());
											estmt.annotateBefore(cAnnot);
										}
										l2gGVMapStack.push(l2gGVMap);
										if( gResidentGVAnalysis(calledProc, gResidentGVars_out, gMallocVars_out, funCall)) {
											AnnotationAdded = true;
										}
										l2gGVMap = l2gGVMapStack.pop();
										if( showGResidentGVars ) {
											StringBuilder str = new StringBuilder(100);
											str.append("GResidentGPUVariables: ");
											str.append(AnalysisTools.symbolsToString(gResidentGVars_out, ","));
											str.append("\nGMallocedGPUVariables: ");
											str.append(AnalysisTools.symbolsToString(gMallocVars_out, ","));
											CommentAnnotation cAnnot = new CommentAnnotation(str.toString());
											estmt.annotateAfter(cAnnot);
										}
									}
								}
							}
						}
					}
					///////////////////////////////////////////////////////////////
					// If shared variables are modified by CPU, remove them from //
					// gResidentGVars_out set.                                    //
					///////////////////////////////////////////////////////////////
					if( ir != null ) {
						Set<Symbol> defSet = DataFlowTools.getDefSymbol(ir);
						if( defSet != null ) {
							for( Symbol sym: defSet ) {
								Symbol gSym = null;
								if( l2gGVMap.containsKey(sym) ) {
									gSym = l2gGVMap.get(sym);
								} else {
									List symInfo = AnalysisTools.findOrgSymbol(sym, proc);
									if( symInfo.size() == 2 ) {
										gSym = (Symbol)symInfo.get(0);
										l2gGVMap.put(sym, gSym);
									} 
								}
								if( (gSym != null) && gResidentGVars_out.contains(gSym) ) {
									gResidentGVars_out.remove(gSym);
								}
							}
						}
					}
				}
					
				node.putData("gResidentGVars_out", gResidentGVars_out);
				node.putData("gMallocVars_out", gMallocVars_out);
	
				for ( DFANode succ : node.getSuccs() ) {
					work_list.put(succ.getData("top-order"), succ);
				}
			}
		}
		// Create a new globalResidentGVar set at the end of this procedure execution.
		gRGVSet.clear();
		gMallocSet.clear();
		List<DFANode> exit_nodes = cfg.getExitNodes();
		boolean firstNode = true;
		// If multiple exit nodes exist, intersect gResidentGVars_out sets.
		for( DFANode exit_node : exit_nodes ) {
			if( firstNode ) {
				gRGVSet.addAll((Set<Symbol>)exit_node.getData("gResidentGVars_out"));
				gMallocSet.addAll((Set<Symbol>)exit_node.getData("gMallocVars_out"));
				firstNode = false;
			} else {
				gRGVSet.retainAll((Set<Symbol>)exit_node.getData("gResidentGVars_out"));
				gMallocSet.retainAll((Set<Symbol>)exit_node.getData("gMallocVars_out"));
			}
		}
		
		/////////////////////////////////////////////////////////////////////////
		// Annotate kernel regions with CUDA annotations such as nocudamalloc, //
		// nocudafree, and noc2gmemtr.                                         //
		/////////////////////////////////////////////////////////////////////////
		Iterator<DFANode> iter = cfg.iterator();
		while ( iter.hasNext() )
		{
			DFANode node = iter.next();
			Statement stmt = node.getData("kernelRegion");
			if( stmt != null ) {
				OmpAnnotation annot = stmt.getAnnotation(OmpAnnotation.class, "parallel");
				if( annot != null ) {
					Set<Symbol> sharedVars = (Set<Symbol>)annot.get("shared");
					if( sharedVars != null ) {
						HashSet<String> noC2GMemTrSet = null;
						HashSet<String> cudaNoC2GMemTrSet = new HashSet<String>();
						CudaAnnotation noC2GAnnot = null;
						HashSet<String> noCudaFreeSet = null;
						HashSet<String> cudaNoCudaFreeSet = new HashSet<String>();
						CudaAnnotation noCudaFreeAnnot = null;
						HashSet<String> noCudaMallocSet = null;
						HashSet<String> cudaNoCudaMallocSet = new HashSet<String>();
						CudaAnnotation noCudaMallocAnnot = null;
						List<CudaAnnotation> cudaAnnots = stmt.getAnnotations(CudaAnnotation.class);
						if( cudaAnnots != null ) {
							for( CudaAnnotation cannot : cudaAnnots ) {
								HashSet<String> dataSet = (HashSet<String>)cannot.get("noc2gmemtr");
								if( dataSet != null ) {
									noC2GMemTrSet = dataSet;
									noC2GAnnot = cannot;
								}
								dataSet = (HashSet<String>)cannot.get("nocudamalloc");
								if( dataSet != null ) {
									noCudaMallocSet = dataSet;
									noCudaMallocAnnot = cannot;
								}
								dataSet = (HashSet<String>)cannot.get("nocudafree");
								if( dataSet != null ) {
									noCudaFreeSet = dataSet;
									noCudaFreeAnnot = cannot;
								}
							}
						}
						Set<Symbol> gResidentGVars = node.getData("gResidentGVars_in");
						Set<Symbol> gMallocVars = node.getData("gMallocVars_in");
						Symbol gSym = null;
						for( Symbol sym : sharedVars ) {
							gSym = l2gGVMap.get(sym);
							if( gResidentGVars.contains(gSym) ) {
								cudaNoC2GMemTrSet.add(sym.getSymbolName());
							}
							if( gMallocVars.contains(gSym) ) {
								cudaNoCudaMallocSet.add(sym.getSymbolName());
								cudaNoCudaFreeSet.add(sym.getSymbolName());
							}
						}
						if( cudaNoC2GMemTrSet.size() > 0 ) {
							if( containsStaticData ) {
								//Compute intersection set.
								if( noC2GAnnot != null ) {
									noC2GMemTrSet.retainAll(cudaNoC2GMemTrSet);
								}
							} else {
								//Compute union set.
								if( noC2GAnnot == null ) {
									AnnotationAdded = true;
									noC2GAnnot = new CudaAnnotation("gpurun", "true");
									noC2GAnnot.put("noc2gmemtr", cudaNoC2GMemTrSet);
									stmt.annotate(noC2GAnnot);
								} else {
									if( !noC2GMemTrSet.containsAll(cudaNoC2GMemTrSet) ) {
										AnnotationAdded = true;
										noC2GMemTrSet.addAll(cudaNoC2GMemTrSet);
									}
								}
							}
						} else if( containsStaticData ) {
							if( noC2GAnnot != null ) {
								noC2GAnnot.remove("noc2gmemtr");
							}
						}
						if( cudaNoCudaMallocSet.size() > 0 ) {
							if( containsStaticData ) {
								//Compute intersection set.
								if( noCudaMallocAnnot != null ) {
									noCudaMallocSet.retainAll(cudaNoCudaMallocSet);
								}
							} else {
								if( noCudaMallocAnnot == null ) {
									AnnotationAdded = true;
									noCudaMallocAnnot = new CudaAnnotation("gpurun", "true");
									noCudaMallocAnnot.put("nocudamalloc", cudaNoCudaMallocSet);
									stmt.annotate(noCudaMallocAnnot);
								} else {
									if( !noCudaMallocSet.containsAll(cudaNoCudaMallocSet) ) {
										AnnotationAdded = true;
										noCudaMallocSet.addAll(cudaNoCudaMallocSet);
									}
								}
							}
						} else if( containsStaticData ) {
							if( noCudaMallocAnnot != null ) {
								noCudaMallocAnnot.remove("nocudamalloc");
							}
						}
						if( cudaNoCudaFreeSet.size() > 0 ) {
							if( containsStaticData ) {
								//Compute intersection set.
								if( noCudaFreeAnnot != null ) {
									noCudaFreeSet.retainAll(cudaNoCudaFreeSet);
								}
							} else {
								if( noCudaFreeAnnot == null ) {
									AnnotationAdded = true;
									noCudaFreeAnnot = new CudaAnnotation("gpurun", "true");
									noCudaFreeAnnot.put("nocudafree", cudaNoCudaFreeSet);
									stmt.annotate(noCudaFreeAnnot);
								} else {
									if( !noCudaFreeSet.containsAll(cudaNoCudaFreeSet) ) {
										AnnotationAdded = true;
										noCudaFreeSet.addAll(cudaNoCudaFreeSet);
									}
								}
							}
						} else if( containsStaticData ) {
							if( noCudaFreeAnnot != null ) {
								noCudaFreeAnnot.remove("nocudafree");
							}
						}
						if( showGResidentGVars  && !containsStaticData ) {
							StringBuilder str = new StringBuilder(100);
							str.append("GResidentGPUVariables: ");
							str.append(AnalysisTools.symbolsToString(gResidentGVars, ","));
							str.append("\nGMallocedGPUVariables: ");
							str.append(AnalysisTools.symbolsToString(gMallocVars, ","));
							CommentAnnotation cAnnot = new CommentAnnotation(str.toString());
							AnnotationStatement cAStmt = new AnnotationStatement(cAnnot);
							CompoundStatement pStmt = (CompoundStatement)stmt.getParent();
							pStmt.addStatementBefore(stmt, cAStmt);
						}
					}
				}
			}
			////////////////////////////////////////////////////////////////////////
			// When GPU variables are globally allocated, no cudaFree() calls are //
			// inserted.                                                          //
			////////////////////////////////////////////////////////////////////////
			stmt = node.getData("pKernelRegion");
			if( stmt != null ) {
				OmpAnnotation annot = stmt.getAnnotation(OmpAnnotation.class, "parallel");
				if( annot != null ) {
					Set<Symbol> sharedVars = (Set<Symbol>)annot.get("shared");
					if( sharedVars != null ) {
						HashSet<String> noCudaFreeSet = null;
						HashSet<String> cudaNoCudaFreeSet = new HashSet<String>();
						CudaAnnotation noCudaFreeAnnot = null;
						List<CudaAnnotation> cudaAnnots = stmt.getAnnotations(CudaAnnotation.class);
						if( cudaAnnots != null ) {
							for( CudaAnnotation cannot : cudaAnnots ) {
								HashSet<String> dataSet = (HashSet<String>)cannot.get("nocudafree");
								if( dataSet != null ) {
									noCudaFreeSet = dataSet;
									noCudaFreeAnnot = cannot;
								}
							}
						}
						Set<Symbol> gResidentGVars = node.getData("gResidentGVars_out");
						Set<Symbol> gMallocVars = node.getData("gMallocVars_out");
						Symbol gSym = null;
						for( Symbol sym : sharedVars ) {
							gSym = l2gGVMap.get(sym);
							////////////////////////////////////////////////////////////
							//Any globally allocatable GPU variables should not be in //
							//cudafree set.                                           //
							////////////////////////////////////////////////////////////
							//if( gMallocVars.contains(gSym) ) {
							if( gSym != null ) {
								cudaNoCudaFreeSet.add(sym.getSymbolName());
							}
						}
						if( cudaNoCudaFreeSet.size() > 0 ) {
							if( containsStaticData ) {
								//Compute intersection set.
								if( noCudaFreeAnnot != null ) {
									noCudaFreeSet.retainAll(cudaNoCudaFreeSet);
								}
							} else {
								if( noCudaFreeAnnot == null ) {
									AnnotationAdded = true;
									noCudaFreeAnnot = new CudaAnnotation("gpurun", "true");
									noCudaFreeAnnot.put("nocudafree", cudaNoCudaFreeSet);
									stmt.annotate(noCudaFreeAnnot);
								} else {
									if( !noCudaFreeSet.containsAll(cudaNoCudaFreeSet) ) {
										AnnotationAdded = true;
										noCudaFreeSet.addAll(cudaNoCudaFreeSet);
									}
								}
							}
						} else if( containsStaticData ) {
							if( noCudaFreeAnnot != null ) {
								noCudaFreeAnnot.remove("nocudafree");
							}
						}
						if( showGResidentGVars  && !containsStaticData ) {
							StringBuilder str = new StringBuilder(100);
							str.append("GResidentGPUVariables: ");
							str.append(AnalysisTools.symbolsToString(gResidentGVars, ","));
							str.append("\nGMallocedGPUVariables: ");
							str.append(AnalysisTools.symbolsToString(gMallocVars, ","));
							CommentAnnotation cAnnot = new CommentAnnotation(str.toString());
							// DEBUG: below way causes an error.
							//AnnotationStatement cAStmt = new AnnotationStatement(cAnnot);
							//CompoundStatement pStmt = (CompoundStatement)stmt.getParent();
							//pStmt.addStatementAfter(stmt, cAStmt);
							stmt.annotateAfter(cAnnot);
						}
					}
				}
			}
		}
		
		/////////////////////////////////////////////////////////////////////////////////////
		// If procedure is cloned, but no new annotation is added to the cloned procedure, //
		// cloning is not needed; revert to the original procedure.                        //
		/////////////////////////////////////////////////////////////////////////////////////
		if( (clonedProc != null) && (!AnnotationAdded) ) {
			PrintTools.println("[gResidentGVAnalysis] delete cloned procedure: " + clonedProc.getSymbolName(), 1);
			Traversable tu = clonedProc.getParent();
			//Delete the cloned procedure.
			tu.removeChild(clonedProc);
			//Swap the new function call with the original function call.
			newFCall.swapWith(orgFCall);
			//Delete the cloned procedure declaration.
			if( clonedProcDecl != null ) {
				//tu.removeChild(clonedProcDecl);
				TransformTools.removeChild(tu, clonedProcDecl);
			}
			visitedProcs.remove(clonedProc);
		}
		
		PrintTools.println("[gResidentGVAnalysis] analysis of " + proc.getSymbolName() + " ended", 1);
		return AnnotationAdded;
	}
}
