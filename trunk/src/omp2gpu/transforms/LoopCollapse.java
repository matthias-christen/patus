package omp2gpu.transforms;

import java.util.*;

import omp2gpu.hir.CUDASpecifier;
import omp2gpu.analysis.AnalysisTools;

import cetus.hir.*;
import cetus.transforms.TransformPass;
import cetus.analysis.*;

/**
 * @author Seyong Lee <lee222@purdue.edu>
 *         ParaMount Group 
 *         School of ECE, Purdue University
 *
 * This class is a helper class, which is called in
 * O2GTranslator 
 */
public class LoopCollapse {
	
	private Program program;
	private Procedure main_proc = null;
	private TranslationUnit main_TU = null;
	private HashMap<Symbol, VariableDeclarator> rowIDMap 
		= new HashMap<Symbol, VariableDeclarator>();
	private VariableDeclarator gpu_rowid_declarator = null;
	private VariableDeclaration rowid_decl = null;
	//private boolean rowID_Initialized = false;
	
	public LoopCollapse(Program prog) {
		program = prog;
		findMain();
	}

	private void findMain() {
		////////////////////////////////
		// Find the main() procedure. //
		////////////////////////////////
		boolean found_main = false;
		for ( Traversable tu : program.getChildren() )
		{
			/* find main()procedure */
			if( found_main ) {
				break;
			} else {
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
						main_proc = proc;
						main_TU = (TranslationUnit)tu;
						found_main = true;
						break;
					}
				}
			}
		}
	}
	
	/**
	 * Recognize SMVP and generate a reduction communication.
	 * [WARN] Current implementation assumes that there exists only one rowptr array 
	 * per input parallel region, parallelRegion.
	 * 
	 * Current implementation recognizes only the following patterns:
	 *	[SMVP pattern 1]
	 *	for (j = 0; j < N; j++) {
	 *     sum = 0.0;
	 *     for (k = rowstr[j]; k < rowstr[j+1]; k++) {
	 *       sum = sum + a[k]*p[colidx[k]];
	 *     }
	 *     w[j] = sum;
	 *   }
	 *   
	 *	[SMVP Pattern 2 (from CG)]
	 *	for (j = 1; j <= lastrow-firstrow+1; j++) {
	 *		d = 0.0f;
	 *		for (k = rowstr[j]; k <= rowstr[j+1]-1; k++) {
	 *			d = d + a[k]*z[colidx[k]];
	 *		}
	 *		w[j] = d;
	 *	}
	 *
	 *	[SMVP Pattern 3 (from CG)]
	 *	for (j = 1; j <= lastrow-firstrow+1; j++) {
	 *     sum = 0.0f;
	 *     for (k = rowstr[j]; k < rowstr[j+1]; k++) {
	 *       sum = sum + a[k]*p[colidx[k]];
	 *     }
	 *     w[j] = sum;
	 *   }
	 *   
	 *	[SMVP pattern 4 (from spmul)]
	 *	for( j=0; j<SIZE2; j++ ) { 
	 *		y[j] = 0.0f;
	 *		for( k=0; k<(rowstr[1+j]-rowstr[j]); k++ ) { 
	 *		  y[j] = y[j] + a[rowstr[j]+k-1]*p[colind[rowstr[j]+k-1]-1];
	 *		}   
	 *	}
	 *
	 * 
	 * @param parallelRegion input parallel region to be transformed into a kernel function.
	 * @param analysisOnly if true, this method recognizes SMVP patterns but not conducts transformation.
	 * @return true if SMVP pattern is recognized
	 */
	public boolean handleSMVP(Statement parallelRegion, boolean analysisOnly)
	{
		PrintTools.println("[handleSMVP] strt", 2);

		boolean isSMVP = false;
		boolean isInitialized = false;
		Procedure parentProc = null;
		Statement parentInMain = null;
		Identifier eout_arrayID = null;
		Identifier rowArrayID = null;
		Identifier srowID = null;
		Identifier erowID = null;
		Identifier rowsizeID = null;
		gpu_rowid_declarator = null;
		rowid_decl = null;
		List<OmpAnnotation> annot_lists = IRTools.collectPragmas(parallelRegion, OmpAnnotation.class, "for");
		
		Reduction reduce_pass = new Reduction(program);
		
		CompoundStatement targetRegion = null;
		if( parallelRegion instanceof ForLoop ) {
			targetRegion = (CompoundStatement)((ForLoop)parallelRegion).getBody();
		} else {
			targetRegion = (CompoundStatement)parallelRegion;
		}

		/////////////////////////////////////////////////////////////////////
		// Check whether the input parallel region contains SPMV patterns. //
		/////////////////////////////////////////////////////////////////////
		for (OmpAnnotation annot : annot_lists)
		{
			ForLoop par_loop = (ForLoop)annot.getAnnotatable();
			CompoundStatement loop_body = (CompoundStatement)par_loop.getBody();
			Expression LB = LoopTools.getLowerBoundExpression(par_loop);
			Expression UB = LoopTools.getUpperBoundExpression(par_loop);
			Identifier ivar1 = (Identifier)LoopTools.getIndexVariable(par_loop);
			Identifier ivar2 = null;
			NameID gpu_rowid = null;
			Identifier out_arrayID = null;
			BinaryExpression productExp = null;
			Statement sumInitStmt = null;
			boolean lsumIsUsed = false;
			int iLoopPattern = 0;
			
			PrintTools.println("par_loop: " + par_loop + "\n", 4);

			/*	[SMVP pattern 1]
			 *	for (j = 0; j < N; j++) {
			 *     sum = 0.0;
			 *     for (k = rowstr[j]; k < rowstr[j+1]; k++) {
			 *       sum = sum + a[k]*p[colidx[k]];
			 *     }
			 *     w[j] = sum;
			 *   }
			 *   
    		 *	[SMVP Pattern 2 (from CG)]
    		 *	for (j = 1; j <= lastrow-firstrow+1; j++) {
    		 *		d = 0.0f;
    		 *		for (k = rowstr[j]; k <= rowstr[j+1]-1; k++) {
             *			d = d + a[k]*z[colidx[k]];
    		 *		}
    		 *		w[j] = d;
    		 *	}
    		 *
    		 *	[SMVP Pattern 3 (from CG)]
			 *	for (j = 1; j <= lastrow-firstrow+1; j++) {
			 *     sum = 0.0f;
			 *     for (k = rowstr[j]; k < rowstr[j+1]; k++) {
			 *       sum = sum + a[k]*p[colidx[k]];
			 *     }
			 *     w[j] = sum;
			 *   }
			 *   
			 *	[SMVP pattern 4 (from spmul)]
			 *	for( j=0; j<SIZE2; j++ ) { 
      		 *		y[j] = 0.0f;
      		 *		for( k=0; k<(rowstr[1+j]-rowstr[j]); k++ ) { 
        	 *		  y[j] = y[j] + a[rowstr[j]+k-1]*p[colind[rowstr[j]+k-1]-1];
      		 *		}   
    		 *	}
    		 *
    		 *	[Not SMVP, but LoopCollapsable Pattern 1 (from CG)]
    		 *  (current implementation does not handle this.)
    		 *	for (j = 1; j <= lastrow - firstrow + 1; j++) {
    		 *		for (k = rowstr[j]; k < rowstr[j+1]; k++) {
             *			colidx[k] = colidx[k] - firstcol + 1; 
    		 *		}    
    		 *	}      
    		 *
			 */

			if ( !(loop_body instanceof CompoundStatement) )
				continue;
			
			////////////////////////////////////////////////////////
			// Check the increment of the outer loop is 1 or not. //
			////////////////////////////////////////////////////////
			Expression incrExp = LoopTools.getIncrementExpression(par_loop);
			if( incrExp instanceof IntegerLiteral ) {
				if( ((IntegerLiteral)incrExp).getValue() != 1 ) {
					continue;
				}
			} else {
				continue;
			}

			FlatIterator iter = new FlatIterator(loop_body);
			List stmt_list = iter.getList(Statement.class);

			if (stmt_list.size() == 3) {
				lsumIsUsed = true;
			} else if (stmt_list.size() == 2) {
				lsumIsUsed = false;	
			} else {
				continue;
			}

			Statement stmt = (Statement)stmt_list.get(0);

			Identifier sum = null;
			ArrayAccess row_array = null, col_array = null, mat_array = null, vec_array = null, out_array = null;
			Symbol rowptrSym = null;
			Identifier row_arrayID = null;

			boolean test_passed = false;

			//////////////////////////////////////////////////
			// match pattern: "sum = 0.0;" or "y[j] = 0.0;" //
			//////////////////////////////////////////////////
			if (stmt instanceof ExpressionStatement)
			{
				Expression expr = ((ExpressionStatement)stmt).getExpression();
				if ( expr instanceof AssignmentExpression)
				{
					AssignmentExpression assign_expr = (AssignmentExpression)expr;
					Expression lhs_expr = assign_expr.getLHS();
					if (lhs_expr instanceof Identifier) {
						sum = (Identifier)lhs_expr;
					} else if (lhs_expr instanceof ArrayAccess) {
						out_array = (ArrayAccess)lhs_expr;
						out_arrayID = (Identifier)out_array.getArrayName();
					} else {
						continue;
					}

					Expression rhs_expr = assign_expr.getRHS();
					if (rhs_expr instanceof FloatLiteral)
					{
						String strExp = ((FloatLiteral)rhs_expr).toString();
						if ( strExp.equals("0.0") || strExp.equals("0.0f") 
								|| strExp.equals("0.0F") )
						{
							test_passed = true;
							sumInitStmt = stmt;
						}
					}
				}
			}
			if ( !test_passed ) continue;
			//////////////////////////////////////////////////////////////////////////////////
			// Check initial statement of the inner loop.                                   //
			//////////////////////////////////////////////////////////////////////////////////
			// match pattern1: "k=rowstr[j]" in "for (k = rowstr[j]; k < rowstr[j+1]; k++)" //
			// match pattern2: "k=0" in "for (k = 0; k < rowstr[j+1]-rowstr[j]; k++)"       //
			//////////////////////////////////////////////////////////////////////////////////
			test_passed = false;
			stmt = (Statement)stmt_list.get(1);
			if ( stmt instanceof ForLoop )
			{
				ForLoop forloop = (ForLoop)stmt;
				ivar2 = (Identifier)LoopTools.getIndexVariable(forloop);
				Statement init_stmt = forloop.getInitialStatement();
				if ( init_stmt instanceof ExpressionStatement)
				{
					Expression expr = ((ExpressionStatement)init_stmt).getExpression();
					if (expr instanceof AssignmentExpression)
					{
						Expression rhs_expr = ((AssignmentExpression)expr).getRHS();
						row_array = AnalysisTools.getArrayAccess(rhs_expr);
						if ( row_array != null )
						{
							if (row_array.getNumIndices() == 1) // 1-dim array
							{
								test_passed = true;
								iLoopPattern = 1;
							}
						} else if ( rhs_expr instanceof IntegerLiteral ) {
							if( ((IntegerLiteral)rhs_expr).getValue() == 0 ) {
								test_passed = true;
								iLoopPattern = 2;
							}
						}
					}
				}
			}

			if ( !test_passed ) continue;
			test_passed = false;
			
			////////////////////////////////////////////////////////
			// Check the increment of the inner loop is 1 or not. //
			////////////////////////////////////////////////////////
			incrExp = LoopTools.getIncrementExpression((ForLoop)stmt);
			if( incrExp instanceof IntegerLiteral ) {
				if( ((IntegerLiteral)incrExp).getValue() != 1 ) {
					continue;
				}
			} else {
				continue;
			}

			////////////////////////////////////////////////////////////////////////////////////////////////
			// Check condition expression of the inner loop.                                              //
			////////////////////////////////////////////////////////////////////////////////////////////////
			// match pattern1: "k<rowstr[j+1]" in "for (k = rowstr[j]; k < rowstr[j+1]; k++)"             //
			// match pattern2: "k<rowstr[j+1]-rowstr[j]" in "for (k = 0; k < rowstr[j+1]-rowstr[j]; k++)" //
			////////////////////////////////////////////////////////////////////////////////////////////////
			Expression cond_expr = ((ForLoop)stmt).getCondition();
			if ( cond_expr instanceof BinaryExpression )
			{
				Expression rhs_expr = ((BinaryExpression)cond_expr).getRHS();
				if( iLoopPattern == 1 ) {
					ArrayAccess rhsArray = AnalysisTools.getArrayAccess(rhs_expr);
					if ( rhsArray != null )
					{
						Expression rhs_array_name = rhsArray.getArrayName();
						if ( row_array.getArrayName().toString().equals(rhs_array_name.toString()) )
						{
							test_passed = true;
						}
					}
				} else if( iLoopPattern == 2 ) {
					List<ArrayAccess> aList = AnalysisTools.getArrayAccesses(rhs_expr);
					if( aList.size() == 2 ) {
						ArrayAccess rhsArray1 = aList.get(0);
						ArrayAccess rhsArray2 = aList.get(1);
						if( rhsArray1.getArrayName().equals(rhsArray2.getArrayName()) ) {
							row_array = rhsArray2;
							test_passed = true;
						}
					}
				}
			}

			if ( !test_passed ) continue;
			test_passed = false;

			FlatIterator inner_loop_iter = new FlatIterator( ((ForLoop)stmt).getBody() );
			List inner_loop_stmt_list = inner_loop_iter.getList(Statement.class);

			/////////////////////////////////////////////////////////////////////////////////
			// Check reduction statement.                                                  //
			/////////////////////////////////////////////////////////////////////////////////
			//match pattern1: "sum = sum + a[k]*p[colidx[k]];"                             //
        	//match pattern2: "y[j] = y[j] + a[rowptr[j]+k-1]*p[colind[rowptr[j]+k-1]-1];" //
			/////////////////////////////////////////////////////////////////////////////////
			if (inner_loop_stmt_list.size() == 1)
			{
				Statement reduce_stmt = (Statement)inner_loop_stmt_list.get(0);
				Map<String, Set<Expression>> reduce_map = reduce_pass.analyzeStatement(reduce_stmt);
				LinkedList<Expression> reduce_list = new LinkedList<Expression>(reduce_map.get("+"));
				//////////////////////////////////////////////////////////////////////////////////////
				//DEBUG: it seems that HashSet.contains() compares hashCode values instead of using //
				//overridden equals() method. Therefore, below commented line does not work.        //
				//Java Spec says that whenever equals method is overridden, hashCode method should  //
				//be overridden so that equal objects must have equal hash codes, but current Cetus //
				//implementation does not follow this rule.                                         //
				//////////////////////////////////////////////////////////////////////////////////////
				//if ( reduce_map.get("+").contains( sum ) || reduce_map.get("+").contains( out_array ) )
				if ( reduce_list.contains( sum ) || reduce_list.contains( out_array ) )
				{
					if ( reduce_stmt instanceof ExpressionStatement)
					{
						Expression expr = ((ExpressionStatement)reduce_stmt).getExpression();
						if (expr instanceof AssignmentExpression)
						{
							Expression rhs_expr = ((AssignmentExpression)expr).getRHS();
							if ( rhs_expr instanceof BinaryExpression )
							{
								if ( ((BinaryExpression)rhs_expr).getOperator() == BinaryOperator.ADD )
								{
									Expression term_expr = ((BinaryExpression)rhs_expr).getRHS();
									if ( term_expr instanceof BinaryExpression )
									{
										if ( ((BinaryExpression)term_expr).getOperator() == BinaryOperator.MULTIPLY )
										{
											Expression l_expr = ((BinaryExpression)term_expr).getLHS();
											Expression r_expr = ((BinaryExpression)term_expr).getRHS();
											if( iLoopPattern == 1 ) {
												productExp = (BinaryExpression)term_expr.clone();
												if (l_expr instanceof ArrayAccess)
												{
													mat_array = (ArrayAccess)l_expr;
													if (mat_array.getNumIndices() == 1) // 1-dim array
													{
														if (r_expr instanceof ArrayAccess)
														{
															vec_array = (ArrayAccess)r_expr;
															if (vec_array.getNumIndices() == 1) // 1-dim array
															{
																Expression subscript_expr = vec_array.getIndex(0);
																col_array = AnalysisTools.getArrayAccess(subscript_expr);
																if (col_array != null)
																{
																	if (col_array.getNumIndices() == 1) // 1-dim array
																	{
																		test_passed = true;
																	}
																}
															}
														}
													}
												}
											} else if( iLoopPattern == 2 ) {
												if (l_expr instanceof ArrayAccess)
												{
													mat_array = (ArrayAccess)l_expr.clone();
													if (mat_array.getNumIndices() == 1) // 1-dim array
													{
														Expression subscript_expr = mat_array.getIndex(0);
														if( IRTools.containsExpression(subscript_expr, row_array) ) {
															IRTools.replaceAll(subscript_expr, row_array, new IntegerLiteral(0));
															Expression simple_subExpr = Symbolic.simplify(subscript_expr);
															mat_array.setIndex(0, simple_subExpr);
														} else {
															continue;
														}
														if (r_expr instanceof ArrayAccess)
														{
															vec_array = (ArrayAccess)r_expr.clone();
															if (vec_array.getNumIndices() == 1) // 1-dim array
															{
																subscript_expr = vec_array.getIndex(0);
																col_array = AnalysisTools.getArrayAccess(subscript_expr);
																if( (col_array != null) && (col_array.getNumIndices() == 1) ) {// 1-dim array
																	subscript_expr = col_array.getIndex(0);
																	if( IRTools.containsExpression(subscript_expr, row_array) ) {
																		IRTools.replaceAll(subscript_expr, row_array, new IntegerLiteral(0));
																		Expression simple_subExpr = Symbolic.simplify(subscript_expr);
																		col_array.setIndex(0, simple_subExpr);
																		productExp = new BinaryExpression(mat_array, 
																				BinaryOperator.MULTIPLY, vec_array);
																		test_passed = true;
																	}
																}
															}
														}
													}
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
			if ( !test_passed ) continue;
			
			if( lsumIsUsed ) {
				test_passed = false;

				stmt = (Statement)stmt_list.get(2);

				/* match pattern "w[j] = sum;" */
				if (stmt instanceof ExpressionStatement)
				{
					Expression expr = ((ExpressionStatement)stmt).getExpression();
					if ( expr instanceof AssignmentExpression)
					{
						AssignmentExpression assign_expr = (AssignmentExpression)expr;
						Expression lhs_expr = assign_expr.getLHS();
						if ( lhs_expr instanceof ArrayAccess )
						{
							out_array = (ArrayAccess)lhs_expr;
							out_arrayID = (Identifier)out_array.getArrayName();
							Expression rhs_expr = assign_expr.getRHS();
							if ( rhs_expr instanceof Identifier )
							{
								Identifier id = (Identifier)rhs_expr;
								if ( id.getName().equals( sum.getName() ) )
								{
									test_passed = true;
								}
							}
						}
					}
				}

				if ( !test_passed ) continue;
			}
			

			///////////////////////////
			// Found a SPMV pattern! //
			///////////////////////////
			PrintTools.println("[INFO in LoopCollapse()] Found a SPMV pattern!", 0);
			PrintTools.println(par_loop + "\n", 2);
			isSMVP = true;
			if( lsumIsUsed ) {
				PrintTools.println("sum : " + sum, 2);
			}
			PrintTools.println("row_array: " + row_array, 3);
			PrintTools.println("col_array: " + col_array, 3);
			PrintTools.println("mat_array: " + mat_array, 3);
			PrintTools.println("vec_array: " + vec_array, 3);
			PrintTools.println("out_array: " + out_array, 3);
			
			if( analysisOnly ) {
				continue;
			}
			
			//////////////////////////////////////
			// Find the original rowptr symbol. //
			//////////////////////////////////////
			row_arrayID = (Identifier)row_array.getArrayName();
			rowptrSym = null;
			parentInMain = null;
			parentProc = null;
			Traversable t = (Traversable)parallelRegion;
			Traversable child_t = null;
			while( (t != null) && !(t.getParent() instanceof Procedure) ) {
				child_t = t;
				t = t.getParent();
			}
			if( (t == null) || (t instanceof TranslationUnit) ) {
				Tools.exit("[ERROR in handleSMVP()] can't find a parent " +
						"procedure of rowptr array, "+row_array);
			}
			t = t.getParent();
			if( t.equals(main_proc) ) {
				parentProc = (Procedure)t;
				parentInMain = (Statement)child_t;
				rowptrSym = SymbolTools.getSymbolOf(row_array);
			} else {
				parentProc = (Procedure)t;
				List paramList = parentProc.getParameters();
				List<FunctionCall> funcCallList = IRTools.getFunctionCalls(main_proc);
				for( FunctionCall funcCall : funcCallList ) {
					if(parentProc.equals(funcCall.getProcedure())) {
						t = funcCall.getStatement();
						while( (t != null) && !(t.getParent() instanceof Procedure) ) {
							child_t = t;
							t = t.getParent();
						}
						parentInMain = (Statement)child_t;
						List argList = funcCall.getArguments();
						for( int i=0; i<paramList.size(); i++ ) {
							////////////////////////////////////////////////////////////////////
							// DEBUG: IRTools.containsSymbol() can not check whether a symbol //
							// is contained in a declaration or not.                          //
							////////////////////////////////////////////////////////////////////
							//if( IRTools.containsSymbol((Declaration)paramList.get(i), 
							//		((Identifier)row_array.getArrayName()).getSymbol())) {
							List declaredSyms = ((Declaration)paramList.get(i)).getDeclaredIDs();
							if( declaredSyms.contains(row_array.getArrayName()) ) {
								rowptrSym = SymbolTools.getSymbolOf((Expression)argList.get(i));
								break;
							}
						}
						break;
					}
				}
			}
			if( rowptrSym == null ) {
				Tools.exit("[ERROR in handleSMVP()] can not find symbol of rowptr array"
						+ row_array);
			}
			
			///////////////////////////////////////////////////////////
			// Find the main statement where the rowptr was defined. //
			// (under construction)                                  //
			///////////////////////////////////////////////////////////
			List<Traversable> mchildren = main_proc.getBody().getChildren();
			Statement lastDefStmt = null;
			for( Traversable child : mchildren ) {
				if( SymbolTools.getAccessedSymbols(child).contains(rowptrSym) ) {
					Set<Symbol> defSet = DataFlowTools.getDefSymbol(child);
					if( defSet.contains(rowptrSym)) {
						lastDefStmt = (Statement)child;
					} else {
						List<FunctionCall> funcCalls = IRTools.getFunctionCalls(child);
						for( FunctionCall funcCall : funcCalls ) {
							if( ipaIsDefined(rowptrSym, funcCall) ) {
								lastDefStmt = (Statement)child;
							}
						}
					}
				}
			}
			
			/////////////////////////////////////////////////////////////////////////////////////
			// Create rowid array, which contains sparse matrix row to kernel block mapping. //
			/////////////////////////////////////////////////////////////////////////////////////
			String rowptrName = rowptrSym.getSymbolName();
			String NBlockName = "NBLOCKS_"+rowptrName;
			if( !rowIDMap.containsKey(rowptrSym) ) {
				StringBuilder str = new StringBuilder(80);
				str.append("rowid__");
				str.append(rowptrName);
				String rowid = str.toString();
				gpu_rowid = new NameID("gpu__"+rowid);
				// The type of the device symbol should be a pointer type 
				gpu_rowid_declarator = new VariableDeclarator(PointerSpecifier.UNQUALIFIED, 
						gpu_rowid);
				VariableDeclaration gpu_rowid_decl = new VariableDeclaration(Specifier.INT, 
						gpu_rowid_declarator);
				List<Traversable> children = main_TU.getChildren();
				Declaration lastCUDADecl = null;
				for( Traversable tdecl : children ) {
					if( tdecl instanceof AnnotationDeclaration ) {
						CommentAnnotation cAnnot = 
							((AnnotationDeclaration)tdecl).getAnnotation(CommentAnnotation.class, "comment");
						if( (cAnnot != null) && ((String)cAnnot.get("comment")).equals("endOfCUDADecls") ) {
							lastCUDADecl = (Declaration)tdecl;
							break;
						}
					}
				}
				if( lastCUDADecl != null ) {
					main_TU.addDeclarationAfter(lastCUDADecl, gpu_rowid_decl);
				} else {
					Tools.exit("[ERROR in handleSMVP] can't find last CUDA-related declaration");
				}
				rowIDMap.put(rowptrSym, gpu_rowid_declarator);
				
				//////////////////////////////////////
				// Sparse matrix rows checking code //
				//////////////////////////////////////
				str = new StringBuilder(2048);
				str.append("#ifndef SPARSE_CHECKED \n");
				str.append("int __ind = 0; \n");
				str.append("int __trowid;\n");
				str.append("int __nzeros = 0; \n");
				str.append("int __nzerosMax = 0; \n");
				str.append("int __ntemps = 0; \n");
				str.append("int __blkcnt = 0;   \n");
				str.append("int __isBlkGen = 0;   \n");
				str.append("__trowid = "+LB.toString()+"; \n");
				str.append("for( __ind="+LB.toString()+"; __ind<="+UB.toString()+"; __ind++ ) {  \n");
				str.append("    __nzeros = ("+rowptrName+"[__ind+1]-"+rowptrName+"[__ind]);\n");
				str.append("    if( __nzeros > __nzerosMax ) {\n");
				str.append("        __nzerosMax = __nzeros;\n");
				str.append("    }\n");
				str.append("    if( __nzeros > BLOCK_SIZE ) { \n");
				str.append("        printf(\"[ERROR] Number of non-zeros (%d) contained in a row is too big \\n \", __nzeros); \n");
				str.append("        printf(\"to be handled by current LoopCollapse implementation;     \\n \"); \n");
				str.append("        printf(\"either increase thread block size, or turn off useLoopCollapse option. \\n \"); \n");
				str.append("        printf(\"(current thread block size = %d) \\n \", BLOCK_SIZE); \n");
				str.append("        exit(-1); \n");
				str.append("    }\n");
				str.append("    __ntemps += __nzeros;   \n");
				str.append("    __isBlkGen = 0;   \n");
				str.append("    if( __ntemps < BLOCK_SIZE ) {\n");
				str.append("        if( (__ind-__trowid+1) == BLOCK_SIZE ) {\n");
				str.append("            __ntemps = 0; \n");
				str.append("            __isBlkGen = 1;   \n");
				str.append("            __blkcnt++;    \n");
				str.append("            __trowid = __ind+1; \n");
				str.append("        }     \n");
				str.append("    }     \n");
				str.append("    else if( __ntemps == BLOCK_SIZE ) {\n");
				str.append("        __ntemps = 0; \n");
				str.append("        __isBlkGen = 1;   \n");
				str.append("        __blkcnt++;    \n");
				str.append("        __trowid = __ind+1; \n");
				str.append("    }     \n");
				str.append("    else { //roll back \n");
				str.append("        __ntemps = __nzeros;\n");
				str.append("        __isBlkGen = 1;   \n");
				str.append("        __blkcnt++;    \n");
				str.append("        __trowid = __ind; \n");
				str.append("    }     \n");
				str.append("}     \n");
				str.append("if( __isBlkGen == 0 ) {// last block is not generated\n");
				str.append("    __blkcnt++;    \n");
				str.append("    if( ("+UB.toString()+"+2-__trowid) > BLOCK_SIZE ) { \n");
				str.append("        printf(\"[ERROR] Number of non-zeros (%d) contained in a row is too big \\n \", "+UB.toString()+"+2-__trowid ); \n");
				str.append("        printf(\"to be handled by current LoopCollapse implementation;     \\n \"); \n");
				str.append("        printf(\"either increase thread block size, or turn off useLoopCollapse option. \\n \"); \n");
				str.append("        printf(\"(current thread block size = %d) \\n \", BLOCK_SIZE); \n");
				str.append("        exit(-1); \n");
				str.append("    }\n");
				str.append("    __trowid = "+UB.toString()+"+1; \n");
				str.append("} else if( __ntemps > 0 ) {    \n");
				str.append("    __blkcnt++;    \n");
				str.append("    __trowid = "+UB.toString()+"+1; \n");
				str.append("}     \n");
				str.append("printf(\"///////////////////////////////////////////////// \\n\"); \n");
				str.append("printf(\"// This is a profile run to check sparse rows. // \\n\"); \n");
				str.append("printf(\"// Max number of nonzeros per row: %3d         // \\n\", __nzerosMax); \n");
				str.append("printf(\"//     - BLOCK_SIZE should be bigger than the  // \\n\"); \n");
				str.append("printf(\"//       above max number.                     // \\n\"); \n");
				str.append("printf(\"// To run the original program, 1) remove the  // \\n\"); \n");
				str.append("printf(\"// following define macro, which is at the top // \\n\"); \n");
				str.append("printf(\"// of the main source file:                    // \\n\"); \n");
				str.append("printf(\"//     #define "+NBlockName+"  -1              // \\n\"); \n");
				str.append("printf(\"// 2) copy the following two macros, paste     // \\n\"); \n");
				str.append("printf(\"// them at the top of the main source file,    // \\n\"); \n");
				str.append("printf(\"// and recompile the source codes.             // \\n\"); \n");
				str.append("printf(\"///////////////////////////////////////////////// \\n\"); \n");
				str.append("printf(\"#define "+NBlockName+" %d\\n\", __blkcnt);\n");
				str.append("printf(\"#define SPARSE_CHECKED\\n\");\n");
				str.append("exit(0); \n");
				str.append("#endif \n");
				CodeAnnotation sparse_check = new CodeAnnotation(str.toString());
				AnnotationStatement sparse_check_stmt = new AnnotationStatement(sparse_check);
				
				str = new StringBuilder(80);
				str.append("#define "+NBlockName+" -1\n");
				CodeAnnotation nBlockCode = new CodeAnnotation(str.toString());
				main_TU.addDeclarationBefore(main_TU.getFirstDeclaration(), new AnnotationDeclaration(nBlockCode));
				
				
				if( lastDefStmt == null ) {
					////////////////////////////////////////////////////////////////////
					//Insert the above annotation statement right after the statement //
					//containing this SPMV computation code.                          //
					////////////////////////////////////////////////////////////////////
					PrintTools.println("[WARNING] Could not find Last Def Stmt \n", 2);
					main_proc.getBody().addStatementAfter(parentInMain, sparse_check_stmt);
				} else {
					////////////////////////////////////////////////////////////////////
					//Insert the above annotation statement right after the statement //
					//where the rowptr array was defined.                             //
					////////////////////////////////////////////////////////////////////
					PrintTools.println("Last Def Stmt => " + lastDefStmt + "\n", 2);
					main_proc.getBody().addStatementAfter(lastDefStmt, sparse_check_stmt);
				}
				
				/////////////////////////////////
				// rowid array generation code //
				/////////////////////////////////
				str = new StringBuilder(2048);
				str.append("#ifdef SPARSE_CHECKED \n");
				// Below local memory allocation can make trouble if the size is too big.
				// Instead, use malloc().
				//str.append("int "+rowid+"["+NBlockName+"+1];\n");
				str.append("int * "+rowid+"; \n");
				str.append("int __ind = 0; \n");
				str.append("int __nzeros = 0; \n");
				str.append("int __ntemps = 0; \n");
				str.append("int __blkcnt = 0;   \n");
				str.append("int __isBlkGen = 0;   \n");
				str.append(rowid + " = (int *)malloc(sizeof(int) * ("+NBlockName+"+1));\n");
				str.append(rowid+"[0] = "+LB.toString()+"; \n");
				str.append("for( __ind="+LB.toString()+"; __ind<="+UB.toString()+"; __ind++ ) {  \n");
				str.append("    __nzeros = ("+rowptrName+"[__ind+1]-"+rowptrName+"[__ind]);\n");
				str.append("    if( __nzeros > BLOCK_SIZE ) { \n");
				str.append("        printf(\"[ERROR] Number of non-zeros (%d) contained in a row is too big \\n \", __nzeros); \n");
				str.append("        printf(\"to be handled by current LoopCollapse implementation;     \\n \"); \n");
				str.append("        printf(\"either increase thread block size, or turn off useLoopCollapse option.     \\n \"); \n");
				str.append("        printf(\"(current thread block size = %d) \\n \", BLOCK_SIZE); \n");
				str.append("        exit(-1); \n");
				str.append("    }\n");
				str.append("    __ntemps += __nzeros;   \n");
				str.append("    __isBlkGen = 0;   \n");
				str.append("    if( __ntemps < BLOCK_SIZE ) {\n");
				str.append("        if( (__ind-"+rowid+"[__blkcnt]+1) == BLOCK_SIZE ) {\n");
				str.append("            __ntemps = 0; \n");
				str.append("            __isBlkGen = 1;   \n");
				str.append("            __blkcnt++;    \n");
				str.append("            "+rowid+"[__blkcnt] = __ind+1; \n");
				str.append("        }     \n");
				str.append("    }     \n");
				str.append("    else if( __ntemps == BLOCK_SIZE ) {\n");
				str.append("        __ntemps = 0; \n");
				str.append("        __isBlkGen = 1;   \n");
				str.append("        __blkcnt++;    \n");
				str.append("        "+rowid+"[__blkcnt] = __ind+1; \n");
				str.append("    }     \n");
				str.append("    else { //roll back \n");
				str.append("        __ntemps = __nzeros;\n");
				str.append("        __isBlkGen = 1;   \n");
				str.append("        __blkcnt++;    \n");
				str.append("        "+rowid+"[__blkcnt] = __ind; \n");
				str.append("    }     \n");
				str.append("}     \n");
				str.append("if( __isBlkGen == 0 ) {// last block is not generated\n");
				str.append("    if( ("+UB.toString()+"+2-"+rowid+"[__blkcnt]) > BLOCK_SIZE ) { \n");
				str.append("        printf(\"[ERROR] Number of non-zeros contained in a row is too big \\n \"); \n");
				str.append("        printf(\"[ERROR] Number of non-zeros (%d) contained in a row is too big \\n \", "+UB.toString()+"+2-"+rowid+"[__blkcnt] ); \n");
				str.append("        printf(\"to be handled by current LoopCollapse implementation;     \\n \"); \n");
				str.append("        printf(\"either increase thread block size, or turn off useLoopCollapse option. \\n \"); \n");
				str.append("        printf(\"(current thread block size = %d) \\n \", BLOCK_SIZE); \n");
				str.append("        exit(-1); \n");
				str.append("    }\n");
				str.append("    __blkcnt++;    \n");
				str.append("    "+rowid+"[__blkcnt] = "+UB.toString()+"+1; \n");
				str.append("} else if( __ntemps > 0 ) {    \n");
				str.append("    __blkcnt++;    \n");
				str.append("    "+rowid+"[__blkcnt] = "+UB.toString()+"+1; \n");
				str.append("}    \n");
				str.append("gpuBytes = (__blkcnt+1) * sizeof(int);\n");
				str.append("gpuGmemSize += gpuBytes;\n");
				str.append("CUDA_SAFE_CALL( cudaMalloc((void**) &"+gpu_rowid+", gpuBytes) );\n");
				str.append("CUDA_SAFE_CALL( cudaMemcpy( "+gpu_rowid+", "+rowid+", gpuBytes,\n");
				str.append("cudaMemcpyHostToDevice) );\n");
				str.append("#endif \n");
				CodeAnnotation rowid_init = new CodeAnnotation(str.toString());
				AnnotationStatement rowid_init_stmt = new AnnotationStatement(rowid_init);
				
				main_proc.getBody().addStatementAfter(sparse_check_stmt, rowid_init_stmt);
			} else {
				gpu_rowid_declarator = rowIDMap.get(rowptrSym);
			}
			
			///////////////////////////////////////////////////////////////////
			// Transform the parallel region to include optimized SPMV code. //
			///////////////////////////////////////////////////////////////////
			NameID tid = new NameID("threadIdx.x");
			Identifier bid = SymbolTools.getOrphanID("_bid");
			Statement currLastStmt = null;
			FunctionCall syncCall = new FunctionCall(new NameID("__syncthreads"));
			Statement syncCallStmt =  new ExpressionStatement(syncCall);
			BinaryExpression biexp1 = null;
			IfStatement ifStmt = null;
			Statement tStmt = null;
			CompoundStatement tBody = null;
			CompoundStatement bBody = null;
			AssignmentExpression assignExp1 = null;
			
			loop_body.getChildren().clear();
			if( !isInitialized ) {
				///////////////////////////////////////////////////////////////////////
				// Create GPU-block-shared data used for loop collapse optimization. //
				//     __shared__ float sh__e_w[BLOCK_SIZE];                         //
				//     __shared__ int sh__rowstr[BLOCK_SIZE+1];                      //
				//     __shared__ int sh__srow, sh__erow, sh__rowsize;               //
				///////////////////////////////////////////////////////////////////////
				List<Specifier> clonedSpecs = new LinkedList<Specifier>();
				clonedSpecs.addAll(((Identifier)out_array.getArrayName()).getSymbol().getTypeSpecifiers());
				clonedSpecs.remove(Specifier.STATIC);
				StringBuilder str = new StringBuilder(80);
				str.append("sh__e_");
				str.append(out_array.getArrayName());
				ArraySpecifier aspec = new ArraySpecifier(new NameID("BLOCK_SIZE"));
				VariableDeclarator arrayV_declarator = new VariableDeclarator(new NameID(str.toString()), aspec);
				List<Specifier> specList = new LinkedList<Specifier>();
				specList.add(CUDASpecifier.CUDA_SHARED);
				specList.addAll(clonedSpecs);
				VariableDeclaration arrayV_decl = 
					new VariableDeclaration(specList, arrayV_declarator); 
				eout_arrayID = new Identifier(arrayV_declarator);
				targetRegion.addDeclaration(arrayV_decl);
				
				str = new StringBuilder(80);
				str.append("sh__");
				str.append(rowptrSym.getSymbolName());
				aspec = new  ArraySpecifier(new BinaryExpression(new NameID("BLOCK_SIZE"), 
						BinaryOperator.ADD, new IntegerLiteral(1)));
				VariableDeclarator rowArray_declarator = new VariableDeclarator(new NameID(str.toString()), aspec);
				specList = new LinkedList<Specifier>();
				specList.add(CUDASpecifier.CUDA_SHARED);
				specList.add(Specifier.INT);
				VariableDeclaration rowArray_decl = 
					new VariableDeclaration(specList, rowArray_declarator); 
				rowArrayID = new Identifier(rowArray_declarator);
				targetRegion.addDeclaration(rowArray_decl);
				
				VariableDeclarator srow_declarator = new VariableDeclarator(new NameID("sh__srow"));
				specList = new LinkedList<Specifier>();
				specList.add(CUDASpecifier.CUDA_SHARED);
				specList.add(Specifier.INT);
				VariableDeclaration srow_decl = new VariableDeclaration(specList, srow_declarator);
				srowID = new Identifier(srow_declarator);
				targetRegion.addDeclaration(srow_decl);
				
				VariableDeclarator erow_declarator = new VariableDeclarator(new NameID("sh__erow"));
				specList = new LinkedList<Specifier>();
				specList.add(CUDASpecifier.CUDA_SHARED);
				specList.add(Specifier.INT);
				VariableDeclaration erow_decl = new VariableDeclaration(specList, erow_declarator);
				erowID = new Identifier(erow_declarator);
				targetRegion.addDeclaration(erow_decl);
				
				VariableDeclarator rowsize_declarator = new VariableDeclarator(new NameID("sh__rowsize"));
				specList = new LinkedList<Specifier>();
				specList.add(CUDASpecifier.CUDA_SHARED);
				specList.add(Specifier.INT);
				VariableDeclaration rowsize_decl = new VariableDeclaration(specList, rowsize_declarator);
				rowsizeID = new Identifier(rowsize_declarator);
				targetRegion.addDeclaration(rowsize_decl);
				
			    ///////////////////////////////////////////////////////
			    // Upload information about rows accessed in a block //
			    ///////////////////////////////////////////////////////
			    // if( tid == 0 ) {                                  //
			    //    sh_srow = rowid[bid];                          //
			    //    sh_erow = rowid[bid+1];                        //
			    //    sh_rowsize = sh_erow - sh_srow;                //
				// }                                                 //
			    // __syncthreads();                                  //
			    ///////////////////////////////////////////////////////
				bBody = new CompoundStatement();
				VariableDeclarator rowid_declarator = 
					new VariableDeclarator(PointerSpecifier.UNQUALIFIED, new NameID("rowid__"+rowptrSym.getSymbolName()));
				Identifier rowidID = new Identifier(rowid_declarator);
				rowid_decl = new VariableDeclaration(Specifier.INT, rowid_declarator);
				tBody = new CompoundStatement();
				assignExp1 = new AssignmentExpression((Identifier)srowID.clone(),
						AssignmentOperator.NORMAL, new ArrayAccess((Identifier)rowidID.clone(),
								(Identifier)bid.clone()));
				tBody.addStatement(new ExpressionStatement(assignExp1));
				assignExp1 = new AssignmentExpression((Identifier)erowID.clone(),
						AssignmentOperator.NORMAL, new ArrayAccess((Identifier)rowidID.clone(),
								new BinaryExpression((Identifier)bid.clone(), BinaryOperator.ADD,
										new IntegerLiteral(1))));
				tBody.addStatement(new ExpressionStatement(assignExp1));
				assignExp1 = new AssignmentExpression((Identifier)rowsizeID.clone(),
						AssignmentOperator.NORMAL, new BinaryExpression( (Identifier)erowID.clone(), 
								BinaryOperator.SUBTRACT, (Identifier)srowID.clone()));
				tBody.addStatement(new ExpressionStatement(assignExp1));
				biexp1 = new BinaryExpression((NameID)tid.clone(),
						BinaryOperator.COMPARE_EQ, new IntegerLiteral(0));
				ifStmt = new IfStatement(biexp1, tBody);
				bBody.addStatement(ifStmt);
				
				bBody.addStatement((Statement)syncCallStmt.clone());

			    //////////////////////////////////////////////////////////////////
			    // Upload used rowptr elements on shared memory                 //
			    //////////////////////////////////////////////////////////////////
			    // if( tid <= sh__rowsize ) {                                   //
			    //     sh__rowptr[tid] = rowptr[sh__srow+tid];                  //
			    // }                                                            //
			    // if( sh__rowsize == BLOCK_SIZE ) {                            //
				//     if( tid == 0 ) {                                         //
			    //         sh__rowptr[BLOCK_SIZE] = rowptr[sh__srow+BLOCK_SIZE];//
				//     }                                                        //
			    // }                                                            //
			    // __syncthreads();                                             //
			    //////////////////////////////////////////////////////////////////
				tBody = new CompoundStatement();
				assignExp1 = new AssignmentExpression(new ArrayAccess((Identifier)rowArrayID.clone(),
						(NameID)tid.clone()), AssignmentOperator.NORMAL, 
						new ArrayAccess((Identifier)row_arrayID.clone(),
								new BinaryExpression((Identifier)srowID.clone(), BinaryOperator.ADD, 
										(NameID)tid.clone())));
				tBody.addStatement(new ExpressionStatement(assignExp1));
				biexp1 = new BinaryExpression((NameID)tid.clone(), BinaryOperator.COMPARE_LE,
						(Identifier)rowsizeID.clone());
				ifStmt = new IfStatement(biexp1, tBody);
				bBody.addStatement(ifStmt);
				tBody = new CompoundStatement();
				assignExp1 = new AssignmentExpression(new ArrayAccess((Identifier)rowArrayID.clone(),
						new NameID("BLOCK_SIZE")), AssignmentOperator.NORMAL, 
						new ArrayAccess((Identifier)row_arrayID.clone(),
								new BinaryExpression((Identifier)srowID.clone(), BinaryOperator.ADD, 
										new NameID("BLOCK_SIZE"))));
				tBody.addStatement(new ExpressionStatement(assignExp1));
				biexp1 = new BinaryExpression((NameID)tid.clone(), BinaryOperator.COMPARE_EQ,
						new IntegerLiteral(0));
				ifStmt = new IfStatement(biexp1, tBody);
				tBody = new CompoundStatement();
				tBody.addStatement(ifStmt);
				biexp1 = new BinaryExpression((Identifier)rowsizeID.clone(), BinaryOperator.COMPARE_EQ,
						new NameID("BLOCK_SIZE"));
				ifStmt = new IfStatement(biexp1, tBody);
				bBody.addStatement(ifStmt);
				bBody.addStatement((Statement)syncCallStmt.clone());
				
				biexp1 = new BinaryExpression((Identifier)bid.clone(), BinaryOperator.COMPARE_LT,
						new NameID(NBlockName));
				ifStmt = new IfStatement(biexp1, bBody);
				
				Statement last_decl_stmt;
				last_decl_stmt = IRTools.getLastDeclarationStatement(targetRegion);
				if( last_decl_stmt != null ) {
					targetRegion.addStatementAfter(last_decl_stmt,ifStmt);
				} else {
					last_decl_stmt = (Statement)targetRegion.getChildren().get(0);
					targetRegion.addStatementBefore(last_decl_stmt,ifStmt);
				}
				isInitialized = true;
			}
			
			/////////////////////////////////////
			// Generate SPMV computation part. //
			/////////////////////////////////////
			bBody = new CompoundStatement();
			////////////////////////////////////////////////////////////////
			//    k = tid + sh_rowptr[0];                                 //
			//                                                            //
		    //    if( tid < (sh__rowptr[sh__rowsize]-sh_rowsptr[0]) ) {   //
		    //        sh_e_w[tid] = a[k]*p[colidx[k]];                    //
		    //    }                                                       //
		    //    __syncthreads();                                        //
			//                                                            //
		    //    if( tid < sh_rowsize ) {                                //
			//        stemp = sh__rowptr[tid];                            //
		    //        sum = 0.0f;                                         //
			//        displace = stemp - sh_rowptr[0];                    //
			//        rsize = sh__rowptr[tid+1] - stemp;                  //
		    //        for( j=0; j<rsize; j++ ) {                          //
		    //            sum = sum + sh__e_w[j+displace];                //
		    //        }                                                   //
		    //        w[tid+sh__srow] = sum;                              //
		    //    }                                                       //
			////////////////////////////////////////////////////////////////
			biexp1 = new BinaryExpression((NameID)tid.clone(), BinaryOperator.ADD,
					new ArrayAccess((Identifier)rowArrayID.clone(),new IntegerLiteral(0)));
			assignExp1 = new AssignmentExpression((Identifier)ivar2.clone(), AssignmentOperator.NORMAL,
					biexp1);
			tStmt = new ExpressionStatement(assignExp1);
			bBody.addStatement(tStmt);
			
			assignExp1 = new AssignmentExpression( new ArrayAccess((Identifier)eout_arrayID.clone(), 
					(NameID)tid.clone()), AssignmentOperator.NORMAL, productExp );
			tBody = new CompoundStatement();
			tBody.addStatement(new ExpressionStatement(assignExp1));
			biexp1 = new BinaryExpression((NameID)tid.clone(), BinaryOperator.COMPARE_LT,
					new BinaryExpression(new ArrayAccess((Identifier)rowArrayID.clone(),
							(Identifier)rowsizeID.clone()), BinaryOperator.SUBTRACT,
							new ArrayAccess((Identifier)rowArrayID.clone(), new IntegerLiteral(0))));
			ifStmt = new IfStatement(biexp1, tBody);
			bBody.addStatement(ifStmt);
			bBody.addStatement((Statement)syncCallStmt.clone());
			
			// stemp = sh__rowptr[tid];
			Identifier stemp = TransformTools.getTempIndex(targetRegion, 0);
			assignExp1 = new AssignmentExpression(stemp, AssignmentOperator.NORMAL,
					new ArrayAccess((Identifier)rowArrayID.clone(), (NameID)tid.clone()));
			tBody = new CompoundStatement();
			tBody.addStatement(new ExpressionStatement(assignExp1));
			if( lsumIsUsed ) {
				tBody.addStatement((Statement)sumInitStmt.clone());
			} else {
				sum = SymbolTools.getTemp(targetRegion, Specifier.FLOAT, "sum");
				assignExp1 = new AssignmentExpression(sum, AssignmentOperator.NORMAL,
						new FloatLiteral(0.0F));
				tBody.addStatement(new ExpressionStatement(assignExp1));
			}
			// displace = stemp - sh_rowptr[0]; 
			biexp1 = new BinaryExpression((Identifier)stemp.clone(), BinaryOperator.SUBTRACT, 
					new ArrayAccess((Identifier)rowArrayID.clone(), new IntegerLiteral(0)));
			Identifier displace = TransformTools.getTempIndex(targetRegion, 1);
			assignExp1 = new AssignmentExpression(displace, AssignmentOperator.NORMAL,
					biexp1);
			tBody.addStatement(new ExpressionStatement(assignExp1));
			// rsize = sh__rowptr[tid+1] - stemp;
			Identifier rsize = TransformTools.getTempIndex(targetRegion, 2);
			biexp1 = new BinaryExpression(new ArrayAccess((Identifier)rowArrayID.clone(),
					new BinaryExpression((NameID)tid.clone(), BinaryOperator.ADD,new IntegerLiteral(1))),
					BinaryOperator.SUBTRACT, (Identifier)stemp.clone());
			assignExp1 = new AssignmentExpression(rsize, AssignmentOperator.NORMAL,
					biexp1);
			tBody.addStatement(new ExpressionStatement(assignExp1));
			
			// j=0;
			assignExp1 = new AssignmentExpression((Identifier)ivar1.clone(),
					AssignmentOperator.NORMAL, new IntegerLiteral(0));
			Statement initStmt = new ExpressionStatement(assignExp1); 
			// j<rsize
			Expression ifcond = new BinaryExpression((Identifier)ivar1.clone(),BinaryOperator.COMPARE_LT,
					(Identifier)rsize.clone());
			// j++ 
			Expression step = new UnaryExpression(UnaryOperator.POST_INCREMENT, (Identifier)ivar1.clone());
			// sum = sum + sh__e_w[j+displace];
			biexp1 = new BinaryExpression((Identifier)sum.clone(), BinaryOperator.ADD,
					new ArrayAccess((Identifier)eout_arrayID.clone(), 
							new BinaryExpression((Identifier)ivar1.clone(), BinaryOperator.ADD,
									(Identifier)displace.clone())));
			assignExp1 = new AssignmentExpression((Identifier)sum.clone(), AssignmentOperator.NORMAL,
					biexp1);
			ForLoop fLoop = new ForLoop(initStmt, ifcond, step, new ExpressionStatement(assignExp1));
			tBody.addStatement(fLoop);
			
			// w[tid+sh__srow] = sum;
			biexp1 = new BinaryExpression((NameID)tid.clone(), BinaryOperator.ADD, 
					(Identifier)srowID.clone());
			assignExp1 = new AssignmentExpression(new ArrayAccess((Identifier)out_arrayID.clone(),
					
					biexp1), AssignmentOperator.NORMAL, (Identifier)sum.clone());
			tBody.addStatement(new ExpressionStatement(assignExp1));
			
			// if( tid < sh_rowsize )
			biexp1 = new BinaryExpression((NameID)tid.clone(), BinaryOperator.COMPARE_LT,
					(Identifier)rowsizeID.clone());
			ifStmt = new IfStatement(biexp1, tBody);
			
			bBody.addStatement(ifStmt);
			
			biexp1 = new BinaryExpression((Identifier)bid.clone(), BinaryOperator.COMPARE_LT,
					new NameID(NBlockName));
			ifStmt = new IfStatement(biexp1, bBody);
			loop_body.addStatement(ifStmt);
			
			////////////////////////////////////////////////////
			//Swap the enclosing for loop with its body       //
			//and move all annotations to the body statement. //
			////////////////////////////////////////////////////
			par_loop.swapWith(loop_body);
			List<Annotation> annot_list = par_loop.getAnnotations();
			for( Annotation tAnnot : annot_list ) {
				loop_body.annotate(tAnnot);
			}
			Expression iterspace = new BinaryExpression(new NameID(NBlockName), 
					BinaryOperator.MULTIPLY, new NameID("BLOCK_SIZE"));
			annot.put("iterspace", iterspace);
		}

		PrintTools.println("[handleSMVP] done", 2);
		return isSMVP;
	}
	
	/** 
	 * Returns GPU rowid Symbol if handleSMVP() uses the symbol for 
	 * Loop Collapse transformation.
	 * Otherwise, return null.
	 * CAUTION: this method assumes that only one rowid is returned 
	 * by a parallel region.
	 * 
	 * @return gpu_rowid_declarator
	 */
	protected VariableDeclarator getGpuRowidSymbol() {
		return gpu_rowid_declarator; 
	}
	
	/**
	 * Returns rowid Declaration if it is used in the handleSMVP().
	 * 
	 * @return rowid_decl rowid declaration, if it is used in the kernel region.
	 *         Otherwise, return null.
	 */
	protected VariableDeclaration getRowidDecl() {
		return rowid_decl;
	}
	
	/**
	 * Interprocedural analysis to check whether the input symbol, Sym, was defined
	 * in the called function, funcCall.
	 * 
	 * @param Sym input symbol to be searched
	 * @param funcCall function call where the input symbol was passed as an argument
	 * @return true if the input symbol was defined in the called function.
	 */
	private boolean ipaIsDefined(Symbol Sym, FunctionCall funcCall ) {
		boolean isDefined = false;
		Symbol paramSym = null;
		Procedure proc = funcCall.getProcedure();
		if (SymbolTools.getAccessedSymbols(funcCall).contains(Sym) ) {
			if( StandardLibrary.isSideEffectFreeExceptIO(funcCall)) {
				isDefined = true;
			} else if( AnalysisTools.isCudaCall(funcCall) ) {
				isDefined = false;
			} else if( proc == null ) {
				isDefined = true;
			}else {
				List paramList = proc.getParameters();
				List<Expression> argList = (List<Expression>)funcCall.getArguments();
				for( int i=0; i<argList.size(); i++ ) {
					if( IRTools.containsSymbol(argList.get(i), Sym)) {
						VariableDeclaration decl = (VariableDeclaration)paramList.get(i);
						paramSym = (VariableDeclarator)decl.getDeclarator(0);
						break;
					}
				}
				List<Traversable> children = proc.getBody().getChildren();
				for( Traversable child : children ) {
					if( SymbolTools.getAccessedSymbols(child).contains(paramSym) ) {
						Set<Symbol> defSet = DataFlowTools.getDefSymbol(child);
						if( defSet.contains(paramSym)) {
							isDefined = true;
							break;
						} else {
							List<FunctionCall> funcCalls = IRTools.getFunctionCalls(child);
							for( FunctionCall fCall : funcCalls ) {
								if( ipaIsDefined(paramSym, fCall) ) {
									isDefined = true;
									break;
								}
							}
						}
					}
				}
			}
		}
		return isDefined;
	}
	
}
