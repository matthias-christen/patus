package omp2gpu.transforms;

import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;

import omp2gpu.analysis.AnalysisTools;
import omp2gpu.analysis.OmpAnalysis;
import omp2gpu.hir.CudaAnnotation;

import cetus.analysis.LoopTools;
import cetus.hir.ArrayAccess;
import cetus.hir.ArraySpecifier;
import cetus.hir.ExpressionStatement;
import cetus.hir.NotAChildException;
import cetus.hir.NotAnOrphanException;
import cetus.hir.Symbolic;
import cetus.hir.Annotatable;
import cetus.hir.AssignmentExpression;
import cetus.hir.AssignmentOperator;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.ChainedList;
import cetus.hir.CompoundStatement;
import cetus.hir.DeclarationStatement;
import cetus.hir.Declaration;
import cetus.hir.DepthFirstIterator;
import cetus.hir.DoLoop;
import cetus.hir.Expression;
import cetus.hir.FloatLiteral;
import cetus.hir.ForLoop;
import cetus.hir.Identifier;
import cetus.hir.NameID;
import cetus.hir.IntegerLiteral;
import cetus.hir.OmpAnnotation;
import cetus.hir.Specifier;
import cetus.hir.Statement;
import cetus.hir.Symbol;
import cetus.hir.SymbolTable;
import cetus.hir.Tools;
import cetus.hir.SymbolTools;
import cetus.hir.IRTools;
import cetus.hir.Traversable;
import cetus.hir.UnaryExpression;
import cetus.hir.UnaryOperator;
import cetus.hir.VariableDeclaration;
import cetus.hir.VariableDeclarator;
import cetus.hir.WhileLoop;

/**
 * @author Seyong Lee <lee222@purdue.edu>
 *         ParaMount Group 
 *         School of ECE, Purdue University
 */
public abstract class TransformTools {
	
	/**
	 * Java doesn't allow a class to be both abstract and final,
	 * so this private constructor prevents any derivations.
	 */
	private TransformTools()
	{
	}

	/**
	 * Get a temporary integer variable that can be used as a loop index variable 
	 * or other temporary data holder. The name of the variable is decided by
	 * using the trailer value, and if the variable with the given name exists
	 * in a region, where, this function returns the existing variable.
	 * Otherwise, this function create a new variable with the given name.
	 * This function differs from Tools.getTemp() in two ways; first if the 
	 * temporary variable exits in the region, this function returns existing
	 * one, but Tools.getTemp() creates another new one. 
	 * Second, if the temporary variable does not exist in the region, this 
	 * function creates the new variable, but Tools.getTemp() searches parents
	 * of the region and creates the new variable only if none of parents contains
	 * the temporary variable.
	 * 
	 * @param where code region from where temporary variable is searched or 
	 *        created. 
	 * @param trailer integer trailer that is used to create/search a variable name
	 * @return
	 */
	public static Identifier getTempIndex(Traversable where, int trailer) {
	    Traversable t = where;
	    while ( !(t instanceof SymbolTable) )
	      t = t.getParent();
	    // Traverse to the parent of a loop statement
	    if (t instanceof ForLoop || t instanceof DoLoop || t instanceof WhileLoop) {
	      t = t.getParent();
	      while ( !(t instanceof SymbolTable) )
	        t = t.getParent();
	    }
	    SymbolTable st = (SymbolTable)t;
	    String header = "_ti_100";
	    String name = header+"_"+trailer;
	    Identifier ret = null;
	    ///////////////////////////////////////////////////////////////////////////
	    // SymbolTable.findSymbol(IDExpression name) can not be used here, since //
	    // it will search parent tables too.                                     //
	    ///////////////////////////////////////////////////////////////////////////
	    Set<String> symNames = AnalysisTools.symbolsToStringSet(st.getSymbols());
	    if( symNames.contains(name) ) {
	    	VariableDeclaration decl = (VariableDeclaration)st.findSymbol(new NameID(name));
	    	ret = new Identifier((VariableDeclarator)decl.getDeclarator(0));
	    } else {
	    	//ret = SymbolTools.getTemp(t, Specifier.INT, header);
	    	///////////////////////////////////////////////////////////////////
	    	//SymbolTools.getTemp() may cause a problem if parent symbol tables    //
	    	//contain a variable whose name is the same as the one of ret.   //
	    	//To avoid this problem, a new temp variable is created directly //
	    	//here without using SymbolTools.getTemp().                           //
	    	///////////////////////////////////////////////////////////////////
	    	VariableDeclarator declarator = new VariableDeclarator(new NameID(name));
	        VariableDeclaration decl = new VariableDeclaration(Specifier.INT, declarator);
	        st.addDeclaration(decl);
	    	ret = new Identifier(declarator);
	    }
	    return ret;
	}

	/**
	 * Get a new temporary integer variable, which has not been created 
	 * by getTempIndex() method.
	 * 
	 * @param where
	 * @return
	 */
	public static Identifier getNewTempIndex(Traversable where) {
	    Traversable t = where;
	    while ( !(t instanceof SymbolTable) )
	      t = t.getParent();
	    // Traverse to the parent of a loop statement
	    if (t instanceof ForLoop || t instanceof DoLoop || t instanceof WhileLoop) {
	      t = t.getParent();
	      while ( !(t instanceof SymbolTable) )
	        t = t.getParent();
	    }
	    SymbolTable st = (SymbolTable)t;
	    String header = "_ti_100";
	    int trailer = 0;
		Identifier ret = null;
	   	Set<String> symNames = AnalysisTools.symbolsToStringSet(st.getSymbols());
	    while ( true ) {
	    	String name = header+"_"+trailer;
	    	if( symNames.contains(name) ) {
	    		trailer++;
	    	} else {
	    		//ret = SymbolTools.getTemp(t, Specifier.INT, header);
	    		///////////////////////////////////////////////////////////////////
	    		//SymbolTools.getTemp() may cause a problem if parent symbol tables    //
	    		//contain a variable whose name is the same as the one of ret.   //
	    		//To avoid this problem, a new temp variable is created directly //
	    		//here without using SymbolTools.getTemp().                            //
	    		///////////////////////////////////////////////////////////////////
	    		VariableDeclarator declarator = new VariableDeclarator(new NameID(name));
	    		VariableDeclaration decl = new VariableDeclaration(Specifier.INT, declarator);
	    		st.addDeclaration(decl);
	    		ret = new Identifier(declarator);
	    		break;
	    	}
	    }
	    return ret;
	}

	/**
	 * If shared variable in a parallel region is used as private/firstprivate variable 
	 * in a omp-for loop in the parallel region, the shared variable is privatized 
	 * in the omp-for loop and initialization statement is added if the shared variable
	 * is used as a firstprivate variable.
	 * 
	 * @param map
	 * @param region
	 */
	public static void privatizeSharedData(HashMap map, CompoundStatement region) {
		HashSet<Symbol> OmpSharedSet = null;
		if (map.keySet().contains("shared"))
			OmpSharedSet = (HashSet<Symbol>) map.get("shared");
		
		List<OmpAnnotation> omp_annots = IRTools.collectPragmas(region, OmpAnnotation.class, "for");
		for ( OmpAnnotation fannot : omp_annots ) {
			Statement target_stmt = (Statement)fannot.getAnnotatable();
			HashSet<Symbol> ForPrivSet = null; 
			HashSet<Symbol> ForFirstPrivSet = null; 
			HashSet<Symbol> PrivSet = new HashSet<Symbol>();
			if (fannot.keySet().contains("private") || fannot.keySet().contains("firstprivate")) {
				ForPrivSet = (HashSet<Symbol>) fannot.get("private");
				if( ForPrivSet != null ) {
					PrivSet.addAll(ForPrivSet);
				}
				ForFirstPrivSet = (HashSet<Symbol>) fannot.get("firstprivate");
				if( ForFirstPrivSet != null ) {
					PrivSet.addAll(ForFirstPrivSet);
				}
				for( Symbol privSym : PrivSet ) {
					if( AnalysisTools.containsSymbol(OmpSharedSet, privSym.getSymbolName()) ) {
						/* 
						 * Create a new temporary variable for the shared variable.
						 */
						VariableDeclaration decl = (VariableDeclaration)
							((VariableDeclarator)privSym).getParent();
						VariableDeclarator cloned_declarator = 
							(VariableDeclarator)((VariableDeclarator)privSym).clone();
						/////////////////////////////////////////////////////////////////////////////////
						// __device__ and __global__ functions can not declare static variables inside //
						// their body.                                                                 //
						/////////////////////////////////////////////////////////////////////////////////
						List<Specifier> clonedspecs = new ChainedList<Specifier>();
						clonedspecs.addAll(decl.getSpecifiers());
						clonedspecs.remove(Specifier.STATIC);
						Identifier cloned_ID = SymbolTools.getArrayTemp(region, clonedspecs, 
								cloned_declarator.getArraySpecifiers(), privSym.getSymbolName());
						/////////////////////////////////////////////////////////////////////////////
						// Replace the symbol pointer of the shared variable with this new symbol. //
						/////////////////////////////////////////////////////////////////////////////
						IRTools.replaceAll(target_stmt, new Identifier((VariableDeclarator)privSym), cloned_ID);
						
						/////////////////////////////////////////////////////////////////////
						// Load the value of shared variable to the firstprivate variable. //
						/////////////////////////////////////////////////////////////////////
						if( (ForFirstPrivSet != null) && ForFirstPrivSet.contains(privSym) ) {
							Symbol sharedSym = AnalysisTools.findsSymbol(OmpSharedSet, privSym.getSymbolName());
							Identifier shared_ID = new Identifier(sharedSym);
							CompoundStatement parentStmt = (CompoundStatement)target_stmt.getParent();
							if( SymbolTools.isScalar(sharedSym) && !SymbolTools.isPointer(sharedSym) ) {
								Statement estmt = new ExpressionStatement(new AssignmentExpression(cloned_ID.clone(), 
										AssignmentOperator.NORMAL, shared_ID.clone()));
								parentStmt.addStatementBefore(target_stmt,estmt);
							} else if( SymbolTools.isArray(sharedSym) ) {
								List aspecs = sharedSym.getArraySpecifiers();
								ArraySpecifier aspec = (ArraySpecifier)aspecs.get(0);
								int dimsize = aspec.getNumDimensions();
								//////////////////////////////
								// Sample loading statement //
								///////////////////////////////////////////////////////
								// Ex: for(i=0; i<SIZE1; i++) {                      //
								//         for(k=0; k<SIZE2; k++) {                  //
								//             fpriv_var[i][k] = shared_var[i][k];   //
								//         }                                         //
								//      }                                            //
								///////////////////////////////////////////////////////
								//////////////////////////////////////// //////
								// Create or find temporary index variables. // 
								//////////////////////////////////////// //////
								List<Identifier> index_vars = new LinkedList<Identifier>();
								for( int i=0; i<dimsize; i++ ) {
									index_vars.add(TransformTools.getTempIndex(region, i));
								}
								Identifier index_var = null;
								Expression assignex = null;
								Statement loop_init = null;
								Expression condition = null;
								Expression step = null;
								CompoundStatement loop_body = null;
								ForLoop innerLoop = null;
								for( int i=dimsize-1; i>=0; i-- ) {
									index_var = index_vars.get(i);
									assignex = new AssignmentExpression((Identifier)index_var.clone(),
											AssignmentOperator.NORMAL, new IntegerLiteral(0));
									loop_init = new ExpressionStatement(assignex);
									condition = new BinaryExpression(index_var.clone(),
											BinaryOperator.COMPARE_LT, aspec.getDimension(i).clone());
									step = new UnaryExpression(UnaryOperator.POST_INCREMENT, 
											(Identifier)index_var.clone());
									loop_body = new CompoundStatement();
									if( i == (dimsize-1) ) {
										List<Expression> indices1 = new LinkedList<Expression>();
										List<Expression> indices2 = new LinkedList<Expression>();
										for( int k=0; k<dimsize; k++ ) {
											indices1.add((Expression)index_vars.get(k).clone());
											indices2.add((Expression)index_vars.get(k).clone());
										}
										assignex = new AssignmentExpression(new ArrayAccess(
												cloned_ID.clone(), indices1), 
												AssignmentOperator.NORMAL, 
												new ArrayAccess(shared_ID.clone(), indices2)); 
										loop_body.addStatement(new ExpressionStatement(assignex));
									} else {
										loop_body.addStatement(innerLoop);
									}
									innerLoop = new ForLoop(loop_init, condition, step, loop_body);
								}	
								parentStmt.addStatementBefore(target_stmt,innerLoop);
							}
						}
							
					}
				}
			}
		}
	}

	/**
	 * Calculate the iteration space sizes of omp-for loops existing in a parallel region.
	 * This calculation should be done before the parallel region is transformed into a
	 * kernel function.
	 * 
	 * @param region parallel region to be searched
	 * @param map annotation map beloning to the parallel region, region
	 * @return set of symbols used in calculating iteration space
	 */
	public static Set<Symbol> calcLoopItrSize(Statement region, HashMap map) {
		ForLoop ploop = null;
		Expression iterspace = null;
		Set<Symbol> usedSymbols = new HashSet<Symbol>();
		if( region instanceof ForLoop ) {
			ploop = (ForLoop)region;
			// check for a canonical loop
			if ( !LoopTools.isCanonical(ploop) ) {
				Tools.exit("[Error in calcLoopItrSize()] Parallel Loop is not " +
						"a canonical loop; compiler can not determine iteration space of " +
						"the following loop: \n" +  ploop);
			}
			// identify the loop index variable 
			Expression ivar = LoopTools.getIndexVariable(ploop);
			Expression lb = LoopTools.getLowerBoundExpression(ploop);
			Expression ub = LoopTools.getUpperBoundExpression(ploop);
			iterspace = Symbolic.add(Symbolic.subtract(ub,lb),new IntegerLiteral(1));
			// Insert the calculated iteration size into the annotation map.
			map.put("iterspace", iterspace);
			usedSymbols.addAll(SymbolTools.getAccessedSymbols(iterspace));
		} else if( region instanceof CompoundStatement ){
			List<OmpAnnotation>
			omp_annots = IRTools.collectPragmas(region, OmpAnnotation.class, "for");
			for ( OmpAnnotation annot : omp_annots ) {
				Statement target_stmt = (Statement)annot.getAnnotatable();
				if( target_stmt instanceof ForLoop ) {
					ploop = (ForLoop)target_stmt;
					if ( !LoopTools.isCanonical(ploop) ) {
						Tools.exit("[Error in calLoopItrSize()] Parallel Loop is not " +
								"a canonical loop; compiler can not determine iteration space of the " +
								"following loop: \n" +  ploop);
					}
					Expression ivar = LoopTools.getIndexVariable(ploop);
					Expression lb = LoopTools.getLowerBoundExpression(ploop);
					Expression ub = LoopTools.getUpperBoundExpression(ploop);
					iterspace = Symbolic.add(Symbolic.subtract(ub,lb),new IntegerLiteral(1));
					annot.put("iterspace", iterspace);
					usedSymbols.addAll(SymbolTools.getAccessedSymbols(iterspace));
				}
			}
		}
		return usedSymbols;
	}

	/**
	 * Find appropriate initialization value for a given reduction operator
	 * and variable type.
	 * @param redOp reductioin operator
	 * @param specList list containing type specifiers of the reduction variable
	 * @return initialization value for the reduction variable
	 */
	public static Expression getRInitValue(BinaryOperator redOp, List specList) {
		///////////////////////////////////////////////////////
		// Operator		Initialization value                 //
		///////////////////////////////////////////////////////
		//	+			0
		//	*			1
		//	-			0
		//	&			~0
		//	|			0
		//	^			0
		//	&&			1
		//	||			0
		///////////////////////////////////////////////////////
		Expression initValue = null;
		if( redOp.equals(BinaryOperator.ADD) || redOp.equals(BinaryOperator.SUBTRACT) ) {
			if(specList.contains(Specifier.FLOAT) || specList.contains(Specifier.DOUBLE)) {
				initValue = new FloatLiteral(0.0f, "F");
			} else {
				initValue = new IntegerLiteral(0);
			}
		} else if( redOp.equals(BinaryOperator.BITWISE_INCLUSIVE_OR)
				|| redOp.equals(BinaryOperator.BITWISE_EXCLUSIVE_OR)
				|| redOp.equals(BinaryOperator.LOGICAL_OR) ) {
			initValue = new IntegerLiteral(0);
		} else if( redOp.equals(BinaryOperator.MULTIPLY) ) {
			if(specList.contains(Specifier.FLOAT) || specList.contains(Specifier.DOUBLE)) {
				initValue = new FloatLiteral(1.0f, "F");
			} else {
				initValue = new IntegerLiteral(1);
			}
		} else if( redOp.equals(BinaryOperator.LOGICAL_AND) ) {
			initValue = new IntegerLiteral(1);
		} else if( redOp.equals(BinaryOperator.BITWISE_AND) ) {
			initValue = new UnaryExpression(UnaryOperator.BITWISE_COMPLEMENT, 
					new IntegerLiteral(0));
		}
		return initValue;
	}

	/**
	 * Create a reduction assignment expression for the given reduction operator.
	 * This function is used to perform both in-block partial reduction and across-
	 * block final reduction.
	 * [CAUTION] the partial results of a subtraction reduction are added to form the 
	 * final value.
	 * 
	 * @param RedExp expression of reduction variable/array
	 * @param redOp reduction operator
	 * @param Rexp right-hand-side expression
	 * @return reduction assignment expression
	 */
	public static AssignmentExpression RedExpression(Expression RedExp, BinaryOperator redOp,
			Expression Rexp) {
		AssignmentExpression assignExp = null;
		if( redOp.equals(BinaryOperator.ADD) ) {
			assignExp = new AssignmentExpression( RedExp, AssignmentOperator.ADD,
					Rexp);
		}else if( redOp.equals(BinaryOperator.SUBTRACT) ) {
			assignExp = new AssignmentExpression( RedExp, AssignmentOperator.ADD,
					Rexp);
		}else if( redOp.equals(BinaryOperator.BITWISE_INCLUSIVE_OR) ) {
			assignExp = new AssignmentExpression( RedExp, AssignmentOperator.BITWISE_INCLUSIVE_OR,
					Rexp);
		}else if( redOp.equals(BinaryOperator.BITWISE_EXCLUSIVE_OR) ) {
			assignExp = new AssignmentExpression( RedExp, AssignmentOperator.BITWISE_EXCLUSIVE_OR,
					Rexp);
		}else if( redOp.equals(BinaryOperator.MULTIPLY) ) {
			assignExp = new AssignmentExpression( RedExp, AssignmentOperator.MULTIPLY,
					Rexp);
		}else if( redOp.equals(BinaryOperator.BITWISE_AND) ) {
			assignExp = new AssignmentExpression( RedExp, AssignmentOperator.BITWISE_AND,
					Rexp);
		}else if( redOp.equals(BinaryOperator.LOGICAL_AND) ) {
			assignExp = new AssignmentExpression( RedExp, AssignmentOperator.NORMAL,
					new BinaryExpression((Expression)RedExp.clone(), redOp, Rexp));
		}else if( redOp.equals(BinaryOperator.LOGICAL_OR) ) {
			assignExp = new AssignmentExpression( RedExp, AssignmentOperator.NORMAL,
					new BinaryExpression((Expression)RedExp.clone(), redOp, Rexp));
		}
		return assignExp;
			
	}

	/**
	 * Update information of OmpAnnotations contained in the region. 
	 * If input region is a cloned one, cloning of the original OmpAnnotations
	 * will do a shallow copy of the HashMap instance: the keys and values themselves 
	 * are not cloned. 
	 * For each OmpAnnotation in the input region,
	 *     - shared, private, reduction, and threadprivate data sets are updated.
	 * 
	 * [CAUTION] this method does not update symbol pointers of Identifiers
	 * in the input region. 
	 *     
	 * @param region code region where OmpAnnotations will be updated.
	 */
	static public void updateAnnotationsInRegion( Traversable region ) {
		HashSet<Symbol> old_set = null;
		HashSet<Symbol> new_set = null;
	
		/* iterate over everything, with particular attention to annotations */
		DepthFirstIterator iter = new DepthFirstIterator(region);
	
		while(iter.hasNext())
		{
			Object obj = iter.next();
	
			if ( (obj instanceof Annotatable) && (obj instanceof Statement) )
			{
				Annotatable at = (Annotatable)obj;
				Statement atstmt = (Statement)obj;
				OmpAnnotation omp_annot = null;
				Collection tCollect = null;
				/////////////////////////////////////////////////////////////////////////
				// Update omp shared, private, reduction, and threadprivate data sets. //
				/////////////////////////////////////////////////////////////////////////
				omp_annot = at.getAnnotation(OmpAnnotation.class, "shared");
				if( omp_annot != null ) {
					old_set = new HashSet<Symbol>();
					tCollect = (Collection)omp_annot.remove("shared");
					old_set.addAll(tCollect);
					new_set = new HashSet<Symbol>();
					AnalysisTools.updateSymbols((Traversable)obj, old_set, new_set, true);
					omp_annot.put("shared", new_set);
				}
				omp_annot = at.getAnnotation(OmpAnnotation.class, "private");
				if( omp_annot != null ) {
					old_set = new HashSet<Symbol>();
					tCollect = (Collection)omp_annot.remove("private");
					old_set.addAll(tCollect);
					new_set = new HashSet<Symbol>();
					AnalysisTools.updateSymbols((Traversable)obj, old_set, new_set, false);
					omp_annot.put("private", new_set);
					//////////////////////////////////////////////////////////
					// If a shared variable is included in the private set, //
					// remove the variable from the shared set.             //
					//////////////////////////////////////////////////////////
					if( omp_annot.keySet().contains("shared") ) {
						old_set = (HashSet<Symbol>)omp_annot.get("shared");
						old_set.removeAll(new_set);
					}
				}
				omp_annot = at.getAnnotation(OmpAnnotation.class, "firstprivate");
				if( omp_annot != null ) {
					old_set = new HashSet<Symbol>();
					tCollect = (Collection)omp_annot.remove("firstprivate");
					old_set.addAll(tCollect);
					new_set = new HashSet<Symbol>();
					AnalysisTools.updateSymbols((Traversable)obj, old_set, new_set, false);
					omp_annot.put("firstprivate", new_set);
					//////////////////////////////////////////////////////////
					// If a shared variable is included in the private set, //
					// remove the variable from the shared set.             //
					//////////////////////////////////////////////////////////
					if( omp_annot.keySet().contains("shared") ) {
						old_set = (HashSet<Symbol>)omp_annot.get("shared");
						old_set.removeAll(new_set);
					}
				}
				omp_annot = at.getAnnotation(OmpAnnotation.class, "threadprivate");
				if( omp_annot != null ) {
					old_set = new HashSet<Symbol>();
					tCollect = (Collection)omp_annot.remove("threadprivate");
					old_set.addAll(tCollect);
					new_set = new HashSet<Symbol>();
					AnalysisTools.updateSymbols((Traversable)obj, old_set, new_set, false);
					omp_annot.put("threadprivate", new_set);
				}
				omp_annot = at.getAnnotation(OmpAnnotation.class, "reduction");
				if( omp_annot != null ) {
					OmpAnalysis.updateReductionClause((Traversable)obj, omp_annot);
				}
				////////////////////////////////////////////////////////
				// Update CudaAnnotations so that each annotation     //
				// contains HashSets as values; this update is needed //
				// only if CudaAnnotations are cloned.                //
				// FIXME: if Annotation.clone() allows HashSet as     //
				// values, we don't need this update.                 //
				////////////////////////////////////////////////////////
				List<CudaAnnotation> cuda_annots = at.getAnnotations(CudaAnnotation.class);
				if( (cuda_annots != null) && (cuda_annots.size() > 0) ) {
					for( CudaAnnotation cannot : cuda_annots ) {
						Collection<String> dataSet = (Collection<String>)cannot.remove("c2gmemtr");
						HashSet<String> newSet = new HashSet<String>();
						if( dataSet != null ) {
							newSet.addAll(dataSet);
							cannot.put("c2gmemtr", newSet);
						}
						dataSet = (Collection<String>)cannot.get("noc2gmemtr");
						newSet = new HashSet<String>();
						if( dataSet != null ) {
							newSet.addAll(dataSet);
							cannot.put("noc2gmemtr", newSet);
						}
						dataSet = (Collection<String>)cannot.get("g2cmemtr");
						newSet = new HashSet<String>();
						if( dataSet != null ) {
							newSet.addAll(dataSet);
							cannot.put("g2cmemtr", newSet);
						}
						dataSet = (Collection<String>)cannot.get("nog2cmemtr");
						newSet = new HashSet<String>();
						if( dataSet != null ) {
							newSet.addAll(dataSet);
							cannot.put("nog2cmemtr", newSet);
						}
						dataSet = (Collection<String>)cannot.get("registerRO");
						newSet = new HashSet<String>();
						if( dataSet != null ) {
							newSet.addAll(dataSet);
							cannot.put("registerRO", newSet);
						}
						dataSet = (Collection<String>)cannot.get("registerRW");
						newSet = new HashSet<String>();
						if( dataSet != null ) {
							newSet.addAll(dataSet);
							cannot.put("registerRW", newSet);
						}
						dataSet = (Collection<String>)cannot.get("noregister");
						newSet = new HashSet<String>();
						if( dataSet != null ) {
							newSet.addAll(dataSet);
							cannot.put("noregister", newSet);
						}
						dataSet = (Collection<String>)cannot.get("sharedRO");
						newSet = new HashSet<String>();
						if( dataSet != null ) {
							newSet.addAll(dataSet);
							cannot.put("sharedRO", newSet);
						}
						dataSet = (Collection<String>)cannot.get("sharedRW");
						newSet = new HashSet<String>();
						if( dataSet != null ) {
							newSet.addAll(dataSet);
							cannot.put("sharedRW", newSet);
						}
						dataSet = (Collection<String>)cannot.get("noshared");
						newSet = new HashSet<String>();
						if( dataSet != null ) {
							newSet.addAll(dataSet);
							cannot.put("noshared", newSet);
						}
						dataSet = (Collection<String>)cannot.get("texture");
						newSet = new HashSet<String>();
						if( dataSet != null ) {
							newSet.addAll(dataSet);
							cannot.put("texture", newSet);
						}
						dataSet = (Collection<String>)cannot.get("notexture");
						newSet = new HashSet<String>();
						if( dataSet != null ) {
							newSet.addAll(dataSet);
							cannot.put("notexture", newSet);
						}
						dataSet = (Collection<String>)cannot.get("constant");
						newSet = new HashSet<String>();
						if( dataSet != null ) {
							newSet.addAll(dataSet);
							cannot.put("constant", newSet);
						}
						dataSet = (Collection<String>)cannot.get("noconstant");
						newSet = new HashSet<String>();
						if( dataSet != null ) {
							newSet.addAll(dataSet);
							cannot.put("noconstant", newSet);
						}
						dataSet = (Collection<String>)cannot.get("noreductionunroll");
						newSet = new HashSet<String>();
						if( dataSet != null ) {
							newSet.addAll(dataSet);
							cannot.put("noreductionunroll", newSet);
						}
						dataSet = (Collection<String>)cannot.get("nocudamalloc");
						newSet = new HashSet<String>();
						if( dataSet != null ) {
							newSet.addAll(dataSet);
							cannot.put("nocudamalloc", newSet);
						}
						dataSet = (Collection<String>)cannot.get("nocudafree");
						newSet = new HashSet<String>();
						if( dataSet != null ) {
							newSet.addAll(dataSet);
							cannot.put("nocudafree", newSet);
						}
						dataSet = (Collection<String>)cannot.get("cudafree");
						newSet = new HashSet<String>();
						if( dataSet != null ) {
							newSet.addAll(dataSet);
							cannot.put("cudafree", newSet);
						}
					}
				}
			} 
		}
	}
	
	/**
	 * Add a statement before the ref_stmt in the parent CompoundStatement.
	 * This method can be used to insert declaration statement before the ref_stmt,
	 * which is not allowed in CompoundStatement.addStatementBefore() method.
	 * 
	 * @param parent parent CompoundStatement containing the ref_stmt as a child
	 * @param ref_stmt reference statement
	 * @param new_stmt new statement to be added
	 */
	public static void addStatementBefore(CompoundStatement parent, Statement ref_stmt, Statement new_stmt) {
		List<Traversable> children = parent.getChildren();
		int index = Tools.indexByReference(children, ref_stmt);
		if (index == -1)
			throw new IllegalArgumentException();
		if (new_stmt.getParent() != null)
			throw new NotAnOrphanException();
		children.add(index, new_stmt);
		new_stmt.setParent(parent);
		if( new_stmt instanceof DeclarationStatement ) {
			Declaration decl = ((DeclarationStatement)new_stmt).getDeclaration();
			SymbolTools.addSymbols(parent, decl);
		}
	}

	/**
	 * Add a statement after the ref_stmt in the parent CompoundStatement.
	 * This method can be used to insert declaration statement after the ref_stmt,
	 * which is not allowed in CompoundStatement.addStatementAfter() method.
	 * 
	 * @param parent parent CompoundStatement containing the ref_stmt as a child
	 * @param ref_stmt reference statement
	 * @param new_stmt new statement to be added
	 */
	public static void addStatementAfter(CompoundStatement parent, Statement ref_stmt, Statement new_stmt) {
		List<Traversable> children = parent.getChildren();
		int index = Tools.indexByReference(children, ref_stmt);
		if (index == -1)
			throw new IllegalArgumentException();
		if (new_stmt.getParent() != null)
			throw new NotAnOrphanException();
		children.add(index+1, new_stmt);
		new_stmt.setParent(parent);
		if( new_stmt instanceof DeclarationStatement ) {
			Declaration decl = ((DeclarationStatement)new_stmt).getDeclaration();
			SymbolTools.addSymbols(parent, decl);
		}
	}
	
	/**
	 * Remove a child from a parent; this method is used to delete ProcedureDeclaration
	 * when both Procedure and ProcedureDeclaration need to be deleted. TranslationUnit
	 * symbol table contains only one entry for both, and thus TranslationUnit.removeChild()
	 * complains an error when trying to delete both of them. 
	 * 
	 * 
	 * @param parent parent traversable containing the child
	 * @param child child traversable to be removed
	 */
	public static void removeChild(Traversable parent, Traversable child)
	{
		List<Traversable> children = parent.getChildren();
		int index = Tools.indexByReference(children, child);

		if (index == -1)
			throw new NotAChildException();

		child.setParent(null);
		children.remove(index);
	}

}
