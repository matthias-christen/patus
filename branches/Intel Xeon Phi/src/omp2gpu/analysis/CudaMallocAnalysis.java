package omp2gpu.analysis;

import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.HashSet;

import omp2gpu.hir.CudaAnnotation;
import omp2gpu.transforms.SplitOmpPRegion;

import cetus.analysis.AnalysisPass;
import cetus.analysis.CFGraph;
import cetus.analysis.CallGraph;
import cetus.analysis.DFANode;
import cetus.analysis.RangeAnalysis;
import cetus.analysis.RangeDomain;
import cetus.analysis.Section;
import cetus.exec.Driver;
import cetus.hir.CompoundStatement;
import cetus.hir.OmpAnnotation;
import cetus.hir.Procedure;
import cetus.hir.Program;
import cetus.hir.Symbol;
import cetus.hir.Tools;
import cetus.hir.PrintTools;
import cetus.hir.IRTools;
import cetus.hir.Traversable;
import cetus.hir.Statement;

/**
 * @author Seyong Lee <lee222@purdue.edu>
 *         ParaMount Group 
 *         School of ECE, Purdue University
 *         
 * Intra-procedural analysis to optimize cudaMalloc and cudaFree.
 * As a result of this analysis, CudaAnnotations with nocudamalloc, 
 * nocudafree, and cudafree clauses are added to each kernel region.         
 */
public class CudaMallocAnalysis extends AnalysisPass {

	private int debug_level;
	private boolean assumeNonZeroTripLoops;
	/**
	 * @param program
	 */
	public CudaMallocAnalysis(Program program) {
		super(program);
		debug_level = PrintTools.getVerbosity();
	}

	/* (non-Javadoc)
	 * @see cetus.analysis.AnalysisPass#getPassName()
	 */
	@Override
	public String getPassName() {
		return new String("[CudaMallocAnalysis]");
	}

	/* (non-Javadoc)
	 * @see cetus.analysis.AnalysisPass#start()
	 */
	@Override
	public void start() {
		String value = Driver.getOptionValue("cudaMallocOptLevel");
		int cudaMallocOptLevel = 0;
		if( value != null ) {
			cudaMallocOptLevel = Integer.valueOf(value).intValue();
		}
		value = Driver.getOptionValue("assumeNonZeroTripLoops");
		assumeNonZeroTripLoops = false;
		if( value != null ) {
			//assumeNonZeroTripLoops = Boolean.valueOf(value).booleanValue();
			assumeNonZeroTripLoops = true;
		}
		if( cudaMallocOptLevel > 0 ) {
			AnalysisTools.markIntervalForKernelRegions(program);
			cudaMallocOpt1();
			SplitOmpPRegion.cleanExtraBarriers(program, false);
		}
	}
	
	/**
	 * FIXME: 1) This analysis can not properly handle a function call if other kernel
	 * regions exist in the function call.
	 */
	private void cudaMallocOpt1() {
		// generate a list of procedures in post-order traversal
		CallGraph callgraph = new CallGraph(program);
		// procedureList contains Procedure in ascending order; the last one is main
		List<Procedure> procedureList = callgraph.getTopologicalCallList();

		/* drive the engine; visit every procedure */
		for (Procedure proc : procedureList)
		{
			/////////////////////////////////////////////
			// This analysis is conducted on functions // 
			// containing omp parallel region.         //
			/////////////////////////////////////////////
			List<OmpAnnotation> pRegion_annots = (List<OmpAnnotation>)
			IRTools.collectPragmas(proc.getBody(), OmpAnnotation.class, "parallel");
			if( pRegion_annots.size() == 0 ) {
				PrintTools.println("The procedure, " + proc.getName() + ", is skipped.", 3);
				continue;
			}

			// Find a set of shared variables in the kernel function.
			Set<Symbol> shared_vars = AnalysisTools.getOmpSharedVariables(proc);
			PrintTools.print("Shared variable symbols in a fucntion " + proc.getName() + " = ", 5);
			PrintTools.println("{" + PrintTools.collectionToString(shared_vars, ",") + "}", 5);

			OCFGraph.setNonZeroTripLoops(assumeNonZeroTripLoops);
			CFGraph cfg = new OCFGraph(proc, null);

			// sort the control flow graph
			cfg.topologicalSort(cfg.getNodeWith("stmt", "ENTRY"));
			
			// For a barrier node with type = "S2P", put ("kernelRegion", pStmt)
			// mapping, where pStmt is a parallel region next to the barrier.
			// This mapping is needed for CFG-based analysis to identify kernel regions.
			HashSet<Statement> bBarriers = new HashSet<Statement>();
			HashMap<Statement, Statement> pRegions = new HashMap<Statement, Statement>();
			HashMap<Statement, DFANode> bNodes = new HashMap<Statement, DFANode>();
			List<OmpAnnotation> bBarrier_annots = (List<OmpAnnotation>)
			IRTools.collectPragmas(proc.getBody(), OmpAnnotation.class, "barrier");
			for( OmpAnnotation omp_annot : bBarrier_annots ) {
				String type = (String)omp_annot.get("barrier");
				Statement bstmt = null;
				Statement pstmt = null;
				if( type.equals("S2P") ) {
					bstmt = (Statement)omp_annot.getAnnotatable();
					pstmt = AnalysisTools.getStatementAfter((CompoundStatement)bstmt.getParent(), 
							bstmt);
					bBarriers.add(bstmt);
					pRegions.put(bstmt, pstmt);
				} else {
					continue;
				}
			}
			Iterator<DFANode> iter = cfg.iterator();
			while ( iter.hasNext() )
			{
				DFANode node = iter.next();
				Statement IRStmt = null;
				Object obj = node.getData("tag");
				if( obj instanceof String ) {
					String tag = (String)obj;
					if( !tag.equals("barrier") ) {
						continue;
					}
				} else {
					continue;
				}
				obj = node.getData("stmt");
				if( obj instanceof Statement ) {
					IRStmt = (Statement)obj;
				} else {
					continue;
				}

				boolean found_bBarrier = false;
				Statement foundStmt = null;
				for( Statement stmt : bBarriers ) {
					if( stmt.equals(IRStmt) ) {
						found_bBarrier = true;
						foundStmt = stmt;
						break;
					}
				}
				if( found_bBarrier ) {
					Statement pStmt = pRegions.get(foundStmt);
					node.putData("kernelRegion", pStmt);
					bNodes.put(foundStmt, node);
				}
			}	
			
			// Backward data-flow analysis to compute liveG_out, a set of live GPU variables, 
	 		// which are accessed in later nodes. 
			AnalysisTools.liveGVariableAnalysis(cfg, false);
			// Forward data-flow analysis to compute reachingGMalloc_in, a set of GPU variables 
			// mallocated in the previous nodes.
			AnalysisTools.reachingGMallocAnalysis(cfg);
			// Advanced live GPU variable analysis.
			AnalysisTools.advLiveGVariableAnalysis(cfg, false);
			
			for( Statement barStmt: bBarriers )
			{
				Statement pStmt = pRegions.get(barStmt);
				DFANode pNode = bNodes.get(barStmt);
				if( pStmt != null ) {
					Set<Symbol> sharedVars = null;
					OmpAnnotation annot = pStmt.getAnnotation(OmpAnnotation.class, "parallel");
					if( annot != null ) {
						sharedVars = (Set<Symbol>)annot.get("shared");
						if( sharedVars == null ) {
							sharedVars = new HashSet<Symbol>();
						}
					} else {
						Tools.exit("[Error1 in cudaMallocOpt1] kernel region w/o parallel region: " + pStmt);
					}
					Set<String> noCudaMallocSet = null;
					Set<String> noCudaFreeSet = null;
					Set<String> noCudaFreeSetAll = new HashSet<String>();
					Set<String> CudaFreeSet = null;
					Set<String> CudaFreeSetAll = new HashSet<String>();
					Set<String> noCudaMallocSetNew = new HashSet<String>();
					Set<String> noCudaFreeSetNew = new HashSet<String>();
					Set<String> CudaFreeSetNew = new HashSet<String>();
					CudaAnnotation noCudaMallocAnnot = null;
					CudaAnnotation noCudaFreeAnnot = null;
					CudaAnnotation CudaFreeAnnot = null;
					List<CudaAnnotation> cudaAnnots = pStmt.getAnnotations(CudaAnnotation.class);
					if( cudaAnnots != null ) {
						for( CudaAnnotation cannot : cudaAnnots ) {
							HashSet<String> dataSet = (HashSet<String>)cannot.get("nocudamalloc");
							if( dataSet != null ) {
								noCudaMallocSet = dataSet;
								noCudaMallocAnnot = cannot;
							}
							dataSet = (HashSet<String>)cannot.get("nocudafree");
							if( dataSet != null ) {
								noCudaFreeSet = dataSet;
								noCudaFreeAnnot = cannot;
								noCudaFreeSetAll.addAll(dataSet);
							}
							dataSet = (HashSet<String>)cannot.get("cudafree");
							if( dataSet != null ) {
								CudaFreeSet = dataSet;
								CudaFreeAnnot = cannot;
								CudaFreeSetAll.addAll(dataSet);
							}
						}
					}
					Set<Symbol> GMalloc_in = (Set<Symbol>)pNode.getData("reachingGMalloc_in");
					Set<Symbol> advLiveG_out = (Set<Symbol>)pNode.getData("advLiveG_out");
					if( GMalloc_in == null ) {
						PrintTools.println("==> Parallel region: " + pStmt, 0);
						Tools.exit("[Error in cudaMallocOpt1()] reachingGMalloc_in does not exist; " +
								"run reachingGMallocAnalysis() before this analysis.");
					}
					if( advLiveG_out == null ) {
						PrintTools.println("==> Parallel region: " + pStmt, 0);
						Tools.exit("[Error in cudaMallocOpt1()] advLiveG_out does not exist; " +
								"run advLiveGVariableAnalysis() before this analysis.");
					}
					for( Symbol sVar: sharedVars ) {
						String symName = sVar.getSymbolName();
						if( (GMalloc_in != null) && GMalloc_in.contains(sVar) ) {
							noCudaMallocSetNew.add(symName);
						}
						if( (advLiveG_out != null) && advLiveG_out.contains(sVar) ) {
							if( !CudaFreeSetAll.contains(symName) ) {
								noCudaFreeSetNew.add(symName);
							}
						} else {
							if( !noCudaFreeSetAll.contains(symName) ) {
								CudaFreeSetNew.add(symName);
							}
						}
					}
					if( noCudaMallocSetNew.size() > 0 ) {
						if( noCudaMallocAnnot == null ) {
							noCudaMallocAnnot = new CudaAnnotation("gpurun", "true");
							noCudaMallocAnnot.put("nocudamalloc", noCudaMallocSetNew);
							pStmt.annotate(noCudaMallocAnnot);
						} else {
							noCudaMallocSet.addAll(noCudaMallocSetNew);
						}
					}
					if( noCudaFreeSetNew.size() > 0 ) {
						if( noCudaFreeAnnot == null ) {
							noCudaFreeAnnot = new CudaAnnotation("gpurun", "true");
							noCudaFreeAnnot.put("nocudafree", noCudaFreeSetNew);
							pStmt.annotate(noCudaFreeAnnot);
						} else {
							noCudaFreeSet.addAll(noCudaFreeSetNew);
						}
					}
					if( CudaFreeSetNew.size() > 0 ) {
						if( CudaFreeAnnot == null ) {
							CudaFreeAnnot = new CudaAnnotation("gpurun", "true");
							CudaFreeAnnot.put("cudafree", CudaFreeSetNew);
							pStmt.annotate(CudaFreeAnnot);
						} else {
							CudaFreeSet.addAll(CudaFreeSetNew);
						}
					}
				} else {
					Tools.exit("[Error2 in cudaMallocOpt1] can't find a kernel region");
				}
			}
		}
	}
}
