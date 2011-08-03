package omp2gpu.analysis;

import java.util.*;

import cetus.exec.Driver;
import cetus.hir.*;
import cetus.analysis.*;
import omp2gpu.hir.*;

/**
 * @author Seyong Lee <lee222@purdue.edu>
 *         ParaMount Group 
 *         School of ECE, Purdue University
 *
 */
public class UEPrivateAnalysis extends AnalysisPass {
	
	private int debug_level;
	// post-order traversal of Procedures in the Program
	private List<Procedure> procedureList;

	/**
	 * @param program
	 */
	public UEPrivateAnalysis(Program program) {
		super(program);
		debug_level = PrintTools.getVerbosity();
	}

	/* (non-Javadoc)
	 * @see cetus.analysis.AnalysisPass#getPassName()
	 */
	@Override
	public String getPassName() {
		return new String("[UEPrivateAnalysis]");
	}

	/* (non-Javadoc)
	 * @see cetus.analysis.AnalysisPass#start()
	 */
	@Override
	public void start() {
		// generate a list of procedures in post-order traversal
		CallGraph callgraph = new CallGraph(program);
		// procedureList contains Procedure in ascending order; the last one is main
		List<Procedure> procedureList = callgraph.getTopologicalCallList();

		///////////////////////
		// DEBUG: deprecated //
		///////////////////////
		//RangeAnalysis range = new RangeAnalysis(program);
		
		boolean assumeNonZeroTripLoops = false;
		String value = Driver.getOptionValue("assumeNonZeroTripLoops");
		if( value != null ) {
			//assumeNonZeroTripLoops = Boolean.valueOf(value).booleanValue();
			assumeNonZeroTripLoops = true;
		}

		/* drive the engine; visit every procedure */
		for (Procedure proc : procedureList)
		{
			////////////////////////////////////////////////////////////////////////
			// This analysis is conducted on kernel functions and other functions // 
			// containing omp parallel region.                                    //
			////////////////////////////////////////////////////////////////////////
			List returnTypes = proc.getTypeSpecifiers();
			PrintTools.println("Procedure is "+returnTypes+" "+proc.getName(), 3);
			boolean is_Kernel_Func = false;
			if( returnTypes.contains(CUDASpecifier.CUDA_GLOBAL) ) is_Kernel_Func = true;
			List<OmpAnnotation> pRegion_annots = (List<OmpAnnotation>)
			IRTools.collectPragmas(proc.getBody(), OmpAnnotation.class, "parallel");
			if( (!is_Kernel_Func) && (pRegion_annots.size() == 0) ) {
				PrintTools.println("The procedure, " + proc.getName() + ", is skipped.", 3);
				continue;
			}
		
			// Find a set of private variables in the kernel function.
			Set<Symbol> private_vars = SymbolTools.getLocalSymbols(proc.getBody());
			/////////////////////////////////////////////////////////////////////////////////
			// FIXME: If proc is not a kernel function, and if a scalar function parameter //
			// of the procedure is used as a private variable, the above private_vars set  //
			// will not include the private variable. However, this omission is OK as long //
			// as this procedure is executed sequentially by CPU; if this procedure is     //
			// executed by multiple OpenMP threads, the function parameter variable should //
			// be included to the private_vars set to check whether it is upwardly exposed.//
			/////////////////////////////////////////////////////////////////////////////////
			PrintTools.print("Private variable symbols in a fucntion " + proc.getName() + " = ", 5);
			PrintTools.println("{" + PrintTools.collectionToString(private_vars, ",") + "}", 5);

			// get the range map
			Map<Statement, RangeDomain> rmap = RangeAnalysis.getRanges(proc);

			OCFGraph.setNonZeroTripLoops(assumeNonZeroTripLoops);
			CFGraph cfg = new OCFGraph(proc, null);
			//CFGraph cfg = new CFGraph(proc, null);
			// get the parallel version of control flow graph
			//PCFGraph cfg = new PCFGraph(proc, null);

			// sort the control flow graph
			cfg.topologicalSort(cfg.getNodeWith("stmt", "ENTRY"));

			// attach the range information to the control flow graph
			AnalysisTools.addRangeDomainToCFG(cfg, rmap);

			// perform reaching-definition analysis (It should come before LiveAnalysis)
			ReachAnalysis reach_analysis = new ReachAnalysis(proc, cfg, private_vars);
			reach_analysis.run();

			// perform live-out analysis (It should come after ReachAnalysis)
			LiveAnalysis live_analysis = new LiveAnalysis(proc, cfg, private_vars);
			live_analysis.run();
			
			Section.MAP ueuse = null;
			Set<Symbol> ueuse_set = null;
			if( is_Kernel_Func ) {
				// Enter the entry node in the work_list
				ueuse_set = new HashSet<Symbol>();
				List<DFANode> entry_nodes = cfg.getEntryNodes();
				if (entry_nodes.size() > 1)
				{
					PrintTools.println("[WARNING in UEPrivateAnalysis()] multiple entries in the kernel funcion, " 
							+ proc.getName(), 2);
				}

				for ( DFANode entry_node : entry_nodes ) {
					ueuse = (Section.MAP)entry_node.getData("ueuse");
					for( Symbol sym : ueuse.keySet() ) {
						String sname = sym.getSymbolName();
						if( !sname.startsWith("_gtid") && 
							!sname.startsWith("_bid") &&
							!sname.startsWith("_ti_100_") &&
							!sname.startsWith("row_temp_") &&
							!sname.endsWith("__extended") &&
							!sname.startsWith("gpu__") &&
							!sname.startsWith("param__") &&
							!sname.startsWith("sh__") ) {
							ueuse_set.add(sym);
						}		
					}
				}
				if( ueuse_set.size() > 0 ) {
					StringBuilder str = new StringBuilder(512);
					str.append("///////////////////////////////////////////////////////////////////////\n");	
					str.append("// [WARNING] Upward-Exposed Use of private variables in the following\n");
					str.append("// kernel function: " + proc.getName() + "\n");	
					str.append("// The following private variables seem to be used before written. \n");
					str.append("// UEUSE: " + ueuse_set + "\n");
					str.append("// This prboblem may be caused by incorrect kernel region splitting.\n");
					str.append("// To solve this problem, remove unnecessary synchronizations \n");
					str.append("// such as removing unnecessary barriers or adding nowait clause to \n");
					str.append("// omp-for-loop annotations if applicable. \n"); 
					str.append("// CF1: current UEPrivateAnalysis conducts an intraprocedural analysis; \n");
					str.append("//      if a function is called inside of the target procedure, the \n"); 
					str.append("//      analysis conservatively assumes all variables of interest are\n");
					str.append("//      accessed in the called function, and thus analysis may result in \n");
					str.append("//      overly estimated, false outputs. \n");
					str.append("// CF2: private arrays can be falsely included to the UEUSE set; even if \n");
					str.append("//      they are initialized in for-loops, compiler may not be sure of \n");
					str.append("//     their initialization due to the possibility of zero-trip loops. \n");
					str.append("// CF3: current UEPrivateAnalysis handles scalar and array expressions, \n");
					str.append("//      but not pointer expressions. Therefore, if a memory region is  \n");
					str.append("//      accessed both by a pointer expression and by an array expression, \n ");
					str.append("//      the analysis may not be able to return accurate results. \n");
					str.append("///////////////////////////////////////////////////////////////////////\n");	
					PrintTools.println(str.toString(), 0);
				}
			} else {
				PrintTools.println("Number of cetus parallel annotations in this procedure: "
						+ pRegion_annots.size() , 3);
				HashSet<Statement> pRegions = new HashSet<Statement>();
				HashMap<Statement, Set<Symbol>> pRMap = new HashMap<Statement, Set<Symbol>>();
				HashMap<Statement, Annotation> pAMap = new HashMap<Statement, Annotation>();
				for( OmpAnnotation omp_annot : pRegion_annots ) {
					Statement pstmt = (Statement)omp_annot.getAnnotatable();
					Set<Symbol> accessedSymbols = SymbolTools.getAccessedSymbols(pstmt);
					if( pstmt instanceof ForLoop) {
						pRegions.add(pstmt);
						pRMap.put(pstmt, accessedSymbols);
						pAMap.put(pstmt, omp_annot);
					} else if( pstmt instanceof CompoundStatement ) {
						List<Traversable> childList = pstmt.getChildren();
						for( Traversable child : childList ) {
							//Find the first statement that is neither AnnotationStatement
							//and nor DeclarationStatement.
							if( !(child instanceof DeclarationStatement) &&
									!(child instanceof AnnotationStatement) ) {
								pRegions.add((Statement)child);
								pRMap.put((Statement)child, accessedSymbols);
								pAMap.put((Statement)child, omp_annot);
								break;
							}
						}
					} else if( (pstmt instanceof ExpressionStatement) ) {
						ExpressionStatement estmt = (ExpressionStatement)pstmt;
						////////////////////////////////////////////////////////////////////////////////
						// KernelFunctionCall has OmpAnnotation of the original parallel region.      //
						// For this function call, corresponding KernelFunction body will be checked. //
						////////////////////////////////////////////////////////////////////////////////
						if( !(estmt.getExpression() instanceof KernelFunctionCall) ) {
							Tools.exit("[Error in UEPrivateAnalysis.start()] Unexpected statement attached " +
								"to an annotation");
						}
					} else {
						Tools.exit("[Error in UEPrivateAnalysis.start()] Unexpected statement attached " +
						"to an annotation");
					}
				}
				Iterator<DFANode> iter = cfg.iterator();
				while ( iter.hasNext() )
				{
					DFANode node = iter.next();
					if( pRegions.size() == 0 ) {
						// All parallel regions of interest are searched.
						break;
					}
					Statement IRStmt = null;
					Object obj = node.getData("ir");
					if( obj instanceof Statement ) {
						IRStmt = (Statement)obj;
					} else {
						continue;
					}

					boolean found_pRegion = false;
					Statement foundStmt = null;
					for( Statement stmt : pRegions ) {
						if( stmt.equals(IRStmt) ) {
							found_pRegion = true;
							foundStmt = stmt;
							break;
						}
					}
					if( found_pRegion ) {
						PrintTools.println("Found parallel region of interest!", 3);
						PrintTools.println("====> " + foundStmt, 3);
						ueuse_set = new HashSet<Symbol>();
						ueuse = (Section.MAP)node.getData("ueuse");
						Set<Symbol> accessedSymbols = pRMap.get(foundStmt);
						for( Symbol sym : ueuse.keySet() ) {
							String sname = sym.getSymbolName();
							if( !sname.startsWith("_gtid") && 
									!sname.startsWith("gpu__") ) {
								if( accessedSymbols.contains(sym) ) {
									ueuse_set.add(sym);
								}
							}		
						}
						if( ueuse_set.size() > 0 ) {
							StringBuilder str = new StringBuilder(512);
							str.append("///////////////////////////////////////////////////////////////////////\n");	
							str.append("// [WARNING] Upward-Exposed Use of local variables in the following\n");
							str.append("// function: " + proc.getName() + "\n");	
							str.append("// The following local variables seem to be used before written in  \n");
							str.append("// the parallel region annotated by this cetus annotation: \n");
							str.append("// Annotation: "+pAMap.get(foundStmt) + "\n");
							str.append("// UEUSE: " + ueuse_set + "\n");
							str.append("// This prboblem may be caused by incorrect kernel region splitting.\n");
							str.append("// To solve this problem, remove unnecessary synchronizations \n");
							str.append("// such as removing unnecessary barriers or adding nowait clause to \n");
							str.append("// omp-for-loop annotations if applicable. \n");
							str.append("// CF1: current UEPrivateAnalysis conducts an intraprocedural analysis; \n");
							str.append("//      if a function is called inside of the target procedure, the \n"); 
							str.append("//      analysis conservatively assumes all variables of interest are\n");
							str.append("//      accessed in the called function, and thus analysis may result in \n");
							str.append("//      overly estimated, false outputs. \n");
							str.append("// CF2: private arrays can be falsely included to the UEUSE set; even if \n");
							str.append("//     they are initialized in for-loops, compiler may not be sure of \n");
							str.append("//     their initialization due to the possibility of zero-trip loops. \n");
							str.append("// CF3: current UEPrivateAnalysis handles scalar and array expressions, \n");
							str.append("//      but not pointer expressions. Therefore, if a memory region is  \n");
							str.append("//      accessed both by a pointer expression and by an array expression, \n ");
							str.append("//      the analysis may not be able to return accurate results. \n");
							str.append("///////////////////////////////////////////////////////////////////////\n");	
							PrintTools.println(str.toString(), 0);
						}
						pRegions.remove(foundStmt);
						pRMap.remove(foundStmt);
						pAMap.remove(foundStmt);
					}
				}
			}

			AnalysisTools.displayCFG(cfg, debug_level);
		}
	}
	


}
