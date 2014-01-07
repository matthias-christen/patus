package omp2gpu.analysis;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import omp2gpu.hir.CudaAnnotation;
import omp2gpu.transforms.SplitOmpPRegion;
import cetus.analysis.AnalysisPass;
import cetus.analysis.CallGraph;
import cetus.exec.Driver;
import cetus.hir.ArraySpecifier;
import cetus.hir.CompoundStatement;
import cetus.hir.Expression;
import cetus.hir.OmpAnnotation;
import cetus.hir.Procedure;
import cetus.hir.Program;
import cetus.hir.Statement;
import cetus.hir.Symbol;
import cetus.hir.SymbolTools;
import cetus.hir.DataFlowTools;
import cetus.hir.IRTools;
import cetus.hir.PrintTools;
import cetus.hir.ArrayAccess;

/**
 * @author Seyong Lee <lee222@purdue.edu>
 *         ParaMount Group 
 *         School of ECE, Purdue University
 *
 * Locality Analysis
 * If shrdSclrCachingOnReg == true,
 *     - if R/O shared scalar variables have locality, 
 *         add them into cuda registerRO clause. 
 *     - if R/W shared scalar variables have locality,
 *         add them into cuda registerRW clause. 
 * If shrdSclrCachingOnSM == true,
 *     - if R/O shared scalar variables exist, 
 *         add them into cuda sharedRO clause. 
 *     - if R/W shared scalar variables have locality,
 *         add them into cuda sharedRW clause. 
 *         (not yet implemented). 
 * If both shrdSclrCachingOnReg and shrdSclrCachingOnSM are on,        
 *     - R/O shared scalar variables with locality will be put
 *       into sharedRO clause.
 *     - R/W shared scalar variables with locality will be put
 *       into registerRW clause.
 * If shrdArryElmtCachingOnReg == true,
 *     - if R/O shared array elements have locality, 
 *         add them into cuda registerRO clause. 
 *     - if R/W shared array elements have locality,
 *         add them into cuda registerRW clause. 
 * If prvtArryCachingOnSM == true,
 *     - add private array variables into sharedRW clause.
 * If shrdArryCachingOnTM == true,
 *     - If one-dimensional, R/O shared arrays have locality,
 *         add them into cuda texture clause.
 *
 */
public class LocalityAnalysis extends AnalysisPass {
	private boolean shrdSclrCachingOnReg;
	private boolean shrdSclrCachingOnSM;
	private boolean shrdArryElmtCachingOnReg;
	private boolean prvtArryCachingOnSM;
	private boolean shrdArryCachingOnTM;
	private boolean extractTuningParameters;

	/**
	 * @param program
	 */
	public LocalityAnalysis(Program program) {
		super(program);
	}

	/* (non-Javadoc)
	 * @see cetus.analysis.AnalysisPass#getPassName()
	 */
	@Override
	public String getPassName() {
		return new String("[LocalityAnalysis]");
	}

	/* (non-Javadoc)
	 * @see cetus.analysis.AnalysisPass#start()
	 */
	@Override
	public void start() {
		shrdSclrCachingOnReg = false;
		String value = Driver.getOptionValue("shrdSclrCachingOnReg");
		if( value != null ) {
			shrdSclrCachingOnReg = true;
		}
		shrdArryElmtCachingOnReg = false;
		value = Driver.getOptionValue("shrdArryElmtCachingOnReg");
		if( value != null ) {
			shrdArryElmtCachingOnReg = true;
		}
		shrdSclrCachingOnSM = false;
		value = Driver.getOptionValue("shrdSclrCachingOnSM");
		if( value != null ) {
			shrdSclrCachingOnSM = true;
		}
		prvtArryCachingOnSM = false;
		value = Driver.getOptionValue("prvtArryCachingOnSM");
		if( value != null ) {
			prvtArryCachingOnSM = true;
		}
		shrdArryCachingOnTM = false;
		value = Driver.getOptionValue("shrdArryCachingOnTM");
		if( value != null ) {
			shrdArryCachingOnTM = true;
		}
		extractTuningParameters = false;
		value = Driver.getOptionValue("extractTuningParameters");
		if( value != null ) {
			extractTuningParameters = true;
		}
		
		AnalysisTools.markIntervalForKernelRegions(program);
		
		// generate a list of procedures in post-order traversal
		CallGraph callgraph = new CallGraph(program);
		// procedureList contains Procedure in ascending order; the last one is main
		List<Procedure> procedureList = callgraph.getTopologicalCallList();

		/* drive the engine; visit every procedure */
		for (Procedure proc : procedureList)
		{
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
				} else {
					continue;
				}
				OmpAnnotation annot = pstmt.getAnnotation(OmpAnnotation.class, "shared");
				if( annot != null ) {
					Set<Symbol> sharedVars = annot.get("shared");
					Set<Symbol> privVars = annot.get("private");
					Map<Expression, Set<Integer>> useExpMap = DataFlowTools.getUseMap(pstmt);
					Map<Expression, Set<Integer>> defExpMap = DataFlowTools.getDefMap(pstmt);
					Map<Symbol, Set<Integer>> useSymMap = DataFlowTools.convertExprMap2SymbolMap(useExpMap);
					Map<Symbol, Set<Integer>> defSymMap = DataFlowTools.convertExprMap2SymbolMap(defExpMap);
					Set<Symbol> useSymSet = useSymMap.keySet();
					Set<Symbol> defSymSet = defSymMap.keySet();
					Set<Expression> useExpSet = useExpMap.keySet();
					Set<Expression> defExpSet = defExpMap.keySet();
					HashSet<String> regROSet = null;
					HashSet<String> cudaRegROSet = new HashSet<String>();
					CudaAnnotation regROAnnot = null;
					HashSet<String> regRWSet = null;
					HashSet<String> cudaRegRWSet = new HashSet<String>();
					HashSet<String> noRegSet = new HashSet<String>();
					CudaAnnotation regRWAnnot = null;
					HashSet<String> sharedROSet = null;
					HashSet<String> cudaSharedROSet = new HashSet<String>();
					CudaAnnotation sharedROAnnot = null;
					HashSet<String> sharedRWSet = null;
					HashSet<String> cudaSharedRWSet = new HashSet<String>();
					HashSet<String> noSharedSet = new HashSet<String>();
					CudaAnnotation sharedRWAnnot = null;
					HashSet<String> textureSet = null;
					HashSet<String> cudaTextureSet = new HashSet<String>();
					HashSet<String> noTextureSet = new HashSet<String>();
					CudaAnnotation textureAnnot = null;
					////////////////////////////
					// Tunable parameter sets //
					////////////////////////////
					HashSet<String> tRegisterROSet = new HashSet<String>();
					HashSet<String> tRegisterRWSet = new HashSet<String>();
					HashSet<String> tSharedROSet = new HashSet<String>();
					HashSet<String> tSharedRWSet = new HashSet<String>();
					HashSet<String> tTextureSet = new HashSet<String>();
					HashSet<String> tROShSclrNL = new HashSet<String>();
					HashSet<String> tROShSclr = new HashSet<String>();
					HashSet<String> tRWShSclr = new HashSet<String>();
					HashSet<String> tROShArEl = new HashSet<String>();
					HashSet<String> tRWShArEl = new HashSet<String>();
					HashSet<String> tRO1DShAr = new HashSet<String>();
					HashSet<String> tPrvAr = new HashSet<String>();
					CudaAnnotation aInfoAnnot = pstmt.getAnnotation(CudaAnnotation.class, "ainfo");
					List<CudaAnnotation> cudaAnnots = pstmt.getAnnotations(CudaAnnotation.class);
					if( cudaAnnots != null ) {
						for( CudaAnnotation cannot : cudaAnnots ) {
							HashSet<String> dataSet = (HashSet<String>)cannot.get("registerRO");
							if( dataSet != null ) {
								regROSet = dataSet;
								regROAnnot = cannot;
								continue;
							}
							dataSet = (HashSet<String>)cannot.get("registerRW");
							if( dataSet != null ) {
								regRWSet = dataSet;
								regRWAnnot = cannot;
								continue;
							}
							dataSet = (HashSet<String>)cannot.get("noregister");
							if( dataSet != null ) {
								noRegSet.addAll(dataSet);
								continue;
							}
							dataSet = (HashSet<String>)cannot.get("sharedRO");
							if( dataSet != null ) {
								sharedROSet = dataSet;
								sharedROAnnot = cannot;
								continue;
							}
							dataSet = (HashSet<String>)cannot.get("sharedRW");
							if( dataSet != null ) {
								sharedRWSet = dataSet;
								sharedRWAnnot = cannot;
								continue;
							}
							dataSet = (HashSet<String>)cannot.get("noshared");
							if( dataSet != null ) {
								noSharedSet.addAll(dataSet);
								continue;
							}
							dataSet = (HashSet<String>)cannot.get("texture");
							if( dataSet != null ) {
								textureSet = dataSet;
								textureAnnot = cannot;
								continue;
							}
							dataSet = (HashSet<String>)cannot.get("notexture");
							if( dataSet != null ) {
								noTextureSet.addAll(dataSet);
								continue;
							}
						}
					}
					int useCnt = 0;
					int defCnt = 0;
					for( Symbol sym: sharedVars ) {
						useCnt = 0;
						defCnt = 0;
						if( useSymSet.contains(sym) ) {
							useCnt = useSymMap.get(sym).size();
						}
						if( defSymSet.contains(sym) ) {
							defCnt = defSymMap.get(sym).size();
						}
						if( (useCnt <= 1) && (defCnt <= 1) ) {
							////////////////////////
							//No locality exists. //
							/////////////////////////////////////////////////////
							//Even if there is no locality, passing R/O shared //
							// scalar variable as kernel parameter can save    //
							// GPU global memory access.                       //
							/////////////////////////////////////////////////////
							if( (defCnt == 0) && SymbolTools.isScalar(sym) ) {
								tSharedROSet.add(sym.getSymbolName());
								tROShSclrNL.add(sym.getSymbolName());
								if( shrdSclrCachingOnSM ) {
									cudaSharedROSet.add(sym.getSymbolName());
								} else {
									continue;
								}
							} else {
								continue;
							}
						} else if ( defCnt == 0 ) {
							//R/O shared variable
							if( SymbolTools.isScalar(sym) ) {
								////////////////////////////////////////
								// For R/O shared scalar variables,   //
								// caching on SM is preferred method. //
								////////////////////////////////////////
								if( shrdSclrCachingOnSM ) {
									cudaSharedROSet.add(sym.getSymbolName());
								} else if( shrdSclrCachingOnReg ) {
									cudaRegROSet.add(sym.getSymbolName());
								}
								tSharedROSet.add(sym.getSymbolName());
								tRegisterROSet.add(sym.getSymbolName());
								tROShSclr.add(sym.getSymbolName());
							} else if( SymbolTools.isArray(sym) ) {
								List aspecs = sym.getArraySpecifiers();
								ArraySpecifier aspec = (ArraySpecifier)aspecs.get(0);
								int dimsize = aspec.getNumDimensions();
								if( dimsize == 1 ) {
									if( shrdArryCachingOnTM ) {
										cudaTextureSet.add(sym.getSymbolName());
									}
									tTextureSet.add(sym.getSymbolName());
									tRO1DShAr.add(sym.getSymbolName());
								}
								for( Expression exp : useExpSet ) {
									if( exp instanceof ArrayAccess ) {
										ArrayAccess aa = (ArrayAccess)exp;
										if( IRTools.containsSymbol(aa.getArrayName(), sym) ) {
											if( useExpMap.get(exp).size() > 1 ) {
												if( shrdArryElmtCachingOnReg && !cudaTextureSet.contains(sym.getSymbolName()) ) {
													cudaRegROSet.add(aa.toString());
												}
												tRegisterROSet.add(aa.toString());
												tROShArEl.add(aa.toString());
											}
										}
									}
								}
							}
						} else {
							//R/W shared variable
							if( SymbolTools.isScalar(sym) ) {
								///////////////////////////////////////
								// For R/W shared scalar variables,  //
								// caching on Register is preferred. //
								///////////////////////////////////////
								if( shrdSclrCachingOnReg ) {
									cudaRegRWSet.add(sym.getSymbolName());
								} else if( shrdSclrCachingOnSM ) {
									cudaSharedRWSet.add(sym.getSymbolName());
								}
								tRegisterRWSet.add(sym.getSymbolName());
								tSharedRWSet.add(sym.getSymbolName());
								tRWShSclr.add(sym.getSymbolName());
							} else if( SymbolTools.isArray(sym) ) {
								for( Expression exp : defExpSet ) {
									if( exp instanceof ArrayAccess ) {
										ArrayAccess aa = (ArrayAccess)exp;
										if( IRTools.containsSymbol(aa.getArrayName(), sym) ) {
											if( defExpMap.get(exp).size() > 1 ) {
												if( shrdArryElmtCachingOnReg ) {
													cudaRegRWSet.add(aa.toString());
												}
												tRegisterRWSet.add(aa.toString());
												tRWShArEl.add(aa.toString());
											}
										}
									}
								}
							}
						}
					}
					for( Symbol sym: privVars ) {
						if( SymbolTools.isArray(sym) ) {
							if( prvtArryCachingOnSM ) {
								cudaSharedRWSet.add(sym.getSymbolName());
							}
							tSharedRWSet.add(sym.getSymbolName());
							tPrvAr.add(sym.getSymbolName());
						}
					}
					cudaRegROSet.removeAll(noRegSet);
					if( cudaRegROSet.size() > 0 ) {
						if( regROAnnot == null ) {
							regROAnnot = new CudaAnnotation("gpurun", "true");
							regROAnnot.put("registerRO", cudaRegROSet);
							pstmt.annotate(regROAnnot);
						} else {
							regROSet.addAll(cudaRegROSet);
						}
					}
					cudaRegRWSet.removeAll(noRegSet);
					if( cudaRegRWSet.size() > 0 ) {
						if( regRWAnnot == null ) {
							regRWAnnot = new CudaAnnotation("gpurun", "true");
							regRWAnnot.put("registerRW", cudaRegRWSet);
							pstmt.annotate(regRWAnnot);
						} else {
							regRWSet.addAll(cudaRegRWSet);
						}
					}
					cudaSharedROSet.removeAll(noSharedSet);
					if( cudaSharedROSet.size() > 0 ) {
						if( sharedROAnnot == null ) {
							sharedROAnnot = new CudaAnnotation("gpurun", "true");
							sharedROAnnot.put("sharedRO", cudaSharedROSet);
							pstmt.annotate(sharedROAnnot);
						} else {
							sharedROSet.addAll(cudaSharedROSet);
						}
					}
					cudaSharedRWSet.removeAll(noSharedSet);
					if( cudaSharedRWSet.size() > 0 ) {
						if( sharedRWAnnot == null ) {
							sharedRWAnnot = new CudaAnnotation("gpurun", "true");
							sharedRWAnnot.put("sharedRW", cudaSharedRWSet);
							pstmt.annotate(sharedRWAnnot);
						} else {
							sharedRWSet.addAll(cudaSharedRWSet);
						}
					}
					cudaTextureSet.removeAll(noTextureSet);
					if( cudaTextureSet.size() > 0 ) {
						if( textureAnnot == null ) {
							textureAnnot = new CudaAnnotation("gpurun", "true");
							textureAnnot.put("texture", cudaTextureSet);
							pstmt.annotate(textureAnnot);
						} else {
							textureSet.addAll(cudaTextureSet);
						}
					}
					if( extractTuningParameters && (aInfoAnnot != null) ) {
						//"tuningparameter" clause is used only internally .
						CudaAnnotation tAnnot = pstmt.getAnnotation(CudaAnnotation.class, "tuningparameters");
						if( tAnnot == null ) {
							tAnnot = new CudaAnnotation("tuningparameters", "true");
							pstmt.annotate(tAnnot);
						}
						if( tRegisterROSet.size() > 0 ) {
							tAnnot.put("registerRO", tRegisterROSet);
						}
						if( tRegisterRWSet.size() > 0 ) {
							tAnnot.put("registerRW", tRegisterRWSet);
						}
						if( tSharedROSet.size() > 0 ) {
							tAnnot.put("sharedRO", tSharedROSet);
						}
						if( tSharedRWSet.size() > 0 ) {
							tAnnot.put("sharedRW", tSharedRWSet);
						}
						if( tTextureSet.size() > 0 ) {
							tAnnot.put("texture", tTextureSet);
						}
						if( tROShSclrNL.size() > 0 ) {
							tAnnot.put("ROShSclrNL", tROShSclrNL);
						}
						if( tROShSclr.size() > 0 ) {
							tAnnot.put("ROShSclr", tROShSclr);
						}
						if( tRWShSclr.size() > 0 ) {
							tAnnot.put("RWShSclr", tRWShSclr);
						}
						if( tROShArEl.size() > 0 ) {
							tAnnot.put("ROShArEl", tROShArEl);
						}
						if( tRWShArEl.size() > 0 ) {
							tAnnot.put("RWShArEl", tRWShArEl);
						}
						if( tRO1DShAr.size() > 0 ) {
							tAnnot.put("RO1DShAr", tRO1DShAr);
						}
						if( tPrvAr.size() > 0 ) {
							tAnnot.put("PrvAr", tPrvAr);
						}
					}
				}
			}
		}
		
		SplitOmpPRegion.cleanExtraBarriers(program, false);

	}

}
