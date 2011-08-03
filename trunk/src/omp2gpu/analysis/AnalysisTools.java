package omp2gpu.analysis;

import java.util.Arrays;
import java.util.Collections;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.LinkedList;

import omp2gpu.hir.CudaAnnotation;
import omp2gpu.hir.CUDASpecifier;
import omp2gpu.hir.CudaStdLibrary;
import omp2gpu.transforms.SplitOmpPRegion;

import cetus.analysis.CFGraph;
import cetus.analysis.DFAGraph;
import cetus.analysis.DFANode;
import cetus.analysis.RangeDomain;
import cetus.analysis.Section;
import cetus.analysis.Section.ELEMENT;
import cetus.analysis.Section.MAP;
import cetus.exec.Driver;
import cetus.hir.Annotatable;
import cetus.hir.AnnotationStatement;
import cetus.hir.ArrayAccess;
import cetus.hir.CompoundStatement;
import cetus.hir.ClassDeclaration;
import cetus.hir.Declaration;
import cetus.hir.DepthFirstIterator;
import cetus.hir.Expression;
import cetus.hir.ForLoop;
import cetus.hir.FunctionCall;
import cetus.hir.IDExpression;
import cetus.hir.Literal;
import cetus.hir.OmpAnnotation;
import cetus.hir.PragmaAnnotation;
import cetus.hir.Procedure;
import cetus.hir.PointerSpecifier;
import cetus.hir.StandardLibrary;
import cetus.hir.Statement;
import cetus.hir.Symbol;
import cetus.hir.Tools;
import cetus.hir.DataFlowTools;
import cetus.hir.SymbolTools;
import cetus.hir.PrintTools;
import cetus.hir.IRTools;
import cetus.hir.TranslationUnit;
import cetus.hir.Traversable;
import cetus.hir.Program;
import cetus.hir.ExpressionStatement;
import cetus.hir.SymbolTable;
import cetus.hir.Specifier;
import cetus.hir.Identifier;
import cetus.hir.UnaryExpression;
import cetus.hir.UnaryOperator;
import cetus.hir.VariableDeclaration;
import cetus.hir.VariableDeclarator;

/**
 * @author Seyong Lee <lee222@purdue.edu>
 *         ParaMount Group 
 *         School of ECE, Purdue University
 */
public abstract class AnalysisTools {
	/**
	 * Java doesn't allow a class to be both abstract and final,
	 * so this private constructor prevents any derivations.
	 */
	private AnalysisTools()
	{
	}

	/* 
	 * List of OpenMP constructs that can not be handled by GPU kernels.
	 * - omp single with nowait or omp master may be executed by GPU kernels only if the attached block
	 *   does not contain any system calls.
	 * - omp critical may be executed by GPU kernels if the attached block contains reduction 
	 *   operations only or an atomic instruction supported by the underlying GPU.
	 * - omp atomic may be executed by GPU kernels only if the underlying GPU supports corresponding
	 *   atomic instruction.
	 */
	static final String[] not_allowed_omp_constructs1 = {
		"flush", "ordered", "barrier", "single", "master"
	};
	static final String[] not_allowed_omp_constructs2 = {
		"critical", "atomic" 
	};
	static final String[] not_allowed_omp_constructs3 = {
		"single", "master"
	};

	public static int checkKernelEligibility(Statement stmt) {
		int eligibility = 0;
		if( (stmt instanceof CompoundStatement) || (stmt instanceof ForLoop) ) {
			if( stmt.containsAnnotation(CudaAnnotation.class, "nogpurun") ) {
				eligibility = 6; // User prevents this statement from executing on the GPU. 
			} else {
				List<OmpAnnotation> omp_annots = IRTools.collectPragmas(stmt, OmpAnnotation.class, "for");
				List not_allowed_consts1 = Arrays.asList(not_allowed_omp_constructs1);
				List not_allowed_consts2 = Arrays.asList(not_allowed_omp_constructs2);
				List not_allowed_consts3 = Arrays.asList(not_allowed_omp_constructs3);
				DepthFirstIterator iter = new DepthFirstIterator(stmt);
				while(iter.hasNext())
				{
					Object obj = iter.next();
					if (obj instanceof Annotatable)
					{
						Annotatable at = (Annotatable)obj;
						List<OmpAnnotation> annotList = at.getAnnotations(OmpAnnotation.class);
						if( (annotList != null) && (annotList.size() > 0) ) {
							if( annotList.size() > 1 ) {
								Tools.exit("[ERROR in checkKernelEligibility()] more than one OmpAnnotations" +
								"were found!");
							}
							OmpAnnotation annot = annotList.get(0);
							if( annot != null ) {
								if( !Collections.disjoint(not_allowed_consts1, annot.keySet()) ) {
									eligibility = 2;
									break;
								} else if( !Collections.disjoint(not_allowed_consts2, annot.keySet()) ) {
									eligibility = 1;
									break;
								} else if( !Collections.disjoint(not_allowed_consts3, annot.keySet()) ) {
									eligibility = 5;
									break;
								} else if( annot.keySet().contains("parallel") ) {
									if( !stmt.equals(obj) ) {
										eligibility = 4; //nested parallel regions
										break;
									}
								}
							}
						}
					}
				}
				if( eligibility == 0 && (omp_annots.size() == 0) ) {
					eligibility = 3;
				}
			}
		} else { //wrong attached statement, stmt
			eligibility = 7;
		}
		return eligibility;
	}
	
	/////////////////////////////////////////////////////////////////////////////
	// Below two methods (addRangeDomainToCFG() and displayCFG()) are directly //
	// copied from cetus.analysis.CommAnalysis; they may need to be changed.   //
	/////////////////////////////////////////////////////////////////////////////
	/**
	 * For now, just union them once in reverse post order.
	 * This method is inefficient because we need to rebuild RangeDomain for
	 * intermediate nodes which do not represent a statement in the IR.
	 * It seems like this is the only way to provide range information correctly.
	 */
	public static void addRangeDomainToCFG(DFAGraph cfg, Map<Statement, RangeDomain> range_map)
	{
		PrintTools.println("[addRangeDomainToCFG] strt", 6);

		TreeMap<Integer,DFANode> work_list = new TreeMap<Integer,DFANode>();

		Iterator<DFANode> iter = cfg.iterator();
		while ( iter.hasNext() )
		{
			DFANode node = iter.next();
			work_list.put((Integer)node.getData("top-order"), node);
		}

		for ( Integer order : work_list.keySet() )
		{
			DFANode node = work_list.get(order);

			Object ir = node.getData(Arrays.asList("super-entry","stmt"));

			RangeDomain rd = range_map.get(ir);

			if ( rd != null )
			{
				node.putData("range", rd.clone());
			}
			else if ( order == 0 )
			{
				RangeDomain range = range_map.get(node.getData("stmt"));

				if ( range == null )
					node.putData("range", new RangeDomain());
				else
					node.putData("range", range.clone());
			}
			else
			{
				RangeDomain range = null;
				int count = 0;
				for ( DFANode pred : node.getPreds() )
				{
					RangeDomain pred_range = (RangeDomain)pred.getData("range");


					if ( pred_range == null )
					{
						pred_range = new RangeDomain();
					}

					if ( range == null )
					{
						range = (RangeDomain)pred_range.clone();
					}
					else
					{
						range.unionRanges(pred_range);
					}
				}
				node.putData("range", range);
			}
		}

		PrintTools.println("[addRangeDomainToCFG] done", 6);

	}

	public static void displayCFG(CFGraph cfg, int debug_level)
	{
		if (debug_level >= 5)
		{
			System.out.println("[displayCFG] strt ----------------------");
			for ( int i=0; i<cfg.size(); i++)
			{
				DFANode node = cfg.getNode(i);
				PrintTools.println("\n" + node.toDot("tag,ir", 1), 5);
	
				Section.MAP may_def_in = (Section.MAP)node.getData("may_def_in");
				if (may_def_in != null) PrintTools.println("    may_def_in" + may_def_in, 9);
	
				Section.MAP must_def_in = (Section.MAP)node.getData("must_def_in");
				if (must_def_in != null) PrintTools.println("    must_def_in" + must_def_in, 9);
	
				Section.MAP may_def_out = (Section.MAP)node.getData("may_def_out");
				if (may_def_out != null) PrintTools.println("    may_def_out" + may_def_out, 5);
	
				Section.MAP must_def_out = (Section.MAP)node.getData("must_def_out");
				if (must_def_out != null) PrintTools.println("    must_def_out" + must_def_out, 5);
	
				Section.MAP ueuse = (Section.MAP)node.getData("ueuse");
				if (ueuse != null) PrintTools.println("    ueuse" + ueuse, 5);
	
				Section.MAP live_out = (Section.MAP)node.getData("live_out");
				if (live_out != null) PrintTools.println("    live_out" + live_out, 5);
			}
			System.out.println("[displayCFG] done ----------------------");
		}
		PrintTools.println(cfg.toDot("tag,ir,ueuse,must_def_out", 3), 5);
	}
	
	/**
	 * Return a statement before the ref_stmt in the parent CompoundStatement
	 * 
	 * @param parent parent CompoundStatement containing the ref_stmt as a child
	 * @param ref_stmt
	 * @return statement before the ref_stmt in the parent CompoundStatement. If ref_stmt 
	 * is not a child of parent or if there is no previous statement, return null.
	 */
	public static Statement getStatementBefore(CompoundStatement parent, Statement ref_stmt) {
		List<Traversable> children = parent.getChildren();
		int index = Tools.indexByReference(children, ref_stmt);
		if( index <= 0 ) {
			return null;
		}
		return (Statement)children.get(index-1);
	}
	
	/**
	 * Return a statement after the ref_stmt in the parent CompoundStatement
	 * 
	 * @param parent parent CompoundStatement containing the ref_stmt as a child
	 * @param ref_stmt
	 * @return statement after the ref_stmt in the parent CompoundStatement. If ref_stmt 
	 * is not a child of parent or if there is no previous statement, return null.
	 */
	public static Statement getStatementAfter(CompoundStatement parent, Statement ref_stmt) {
		List<Traversable> children = parent.getChildren();
		int index = Tools.indexByReference(children, ref_stmt);
		if( (index == -1) || (index == children.size()-1) ) {
			return null;
		}
		return (Statement)children.get(index+1);
	}
	
	/**
	 * Return a set of OpenMP shared variables existing the the input code, tr.
	 * (Shared variables are searched intraprocedurally.)
	 * 
	 * @param tr input code region
	 * @return set of OpenMP shared variable symbols in the input code tr
	 */
	public static Set<Symbol> getOmpSharedVariables(Traversable tr)
	{
		Set<Symbol> ret = new HashSet<Symbol>();

		List<OmpAnnotation>
		omp_annots = IRTools.collectPragmas(tr, OmpAnnotation.class, "shared");

		for (OmpAnnotation annot : omp_annots)
		{ 
			Set<Symbol> shared_set = (Set<Symbol>)annot.get("shared");
			if (shared_set == null)
				Tools.exit("[ERROR] omp shared construct has null shared set");
			else    
			{
				ret.addAll(shared_set);
			}
		}
		return ret;
	}
	
	/**
	 * Return a set of OpenMP shared variables existing the the input code, tr.
	 * This is an interprocedural analysis; if the input code has function calls,
	 * the called functions are searched recursively, and the return set contains 
	 * original symbols. (If shared variable is a function parameter, corresponding
	 * argument symbol is returned.)
	 * 
	 * @param tr input code region
	 * @return set of OpenMP shared variable symbols in the input code tr
	 */
	public static Set<Symbol> getIpOmpSharedVariables(Traversable tr)
	{
		Set<Symbol> ret = new HashSet<Symbol>();
		Set<Symbol> tSet = new HashSet<Symbol>();

		// Search shared variables intraprocedurally.
		List<OmpAnnotation>
		omp_annots = IRTools.collectPragmas(tr, OmpAnnotation.class, "shared");
		for (OmpAnnotation annot : omp_annots)
		{ 
			Set<Symbol> shared_set = (Set<Symbol>)annot.get("shared");
			if (shared_set == null)
				Tools.exit("[ERROR] omp shared construct has null shared set");
			else    
			{
				tSet.addAll(shared_set);
			}
		}
		// Find the original shared variables if they are function parameters.
		for( Symbol iSym : tSet ) {
			List symInfo = AnalysisTools.findOrgSymbol(iSym, tr);
			if( symInfo.size() == 0 ) {
				ret.add(iSym);
			} else {
				ret.add((Symbol)symInfo.get(0));
			}
		}
		
		// Search shared variables interprocedurally.
		List<FunctionCall> calledFuncs = IRTools.getFunctionCalls(tr);
		for( FunctionCall fCall : calledFuncs ) {
			Procedure proc = fCall.getProcedure();
			if( proc != null ) {
				ret.addAll(getIpOmpSharedVariables(proc));
			}
		}
		return ret;
	}
	
	/**
	 * Returns the set of symbols accessed in the traversable object, t.
	 * The returned set includes local symbols in t
	 * and global variables and static variables accessed in t 
	 * and functions called within t.
	 * 
	 *
	 * @param t the traversable object.
	 * @return the set of accessed symbols.
	 */
	public static Set<Symbol> getIpAccessedSymbols(Traversable t) {
		Set<Symbol> accessedSymbols = SymbolTools.getAccessedSymbols(t);
		List<FunctionCall> calledFuncs = IRTools.getFunctionCalls(t);
		for( FunctionCall call : calledFuncs ) {
			Procedure called_procedure = call.getProcedure();
			if( called_procedure != null ) {
				CompoundStatement body = called_procedure.getBody();
				Set<Symbol> procAccessedSymbols = getIpAccessedGlobalorStaticSymbols(body);
				accessedSymbols.addAll(procAccessedSymbols);
			}
		}
		return accessedSymbols;
	}
	
	/**
	 * Returns the set of global or static symbols accessed in the input SymbolTable object, st.
	 * If st contains function calls, each function is recursively checked.
	 *
	 * @param st input SymbolTable object.
	 * @return the set of accessed global or static symbols.
	 */
	public static Set<Symbol> getIpAccessedGlobalorStaticSymbols(SymbolTable st) {
		/////////////////////////////////////////////////////
		// rSet = accessed global symbols + static symbols //
		/////////////////////////////////////////////////////
		Set<Symbol> rSet = new HashSet<Symbol>();
		Set<Symbol> aSet = SymbolTools.getAccessedSymbols(st);
		Set<Symbol> staticSet = extractStaticVariables(aSet);
		rSet.addAll(staticSet);
		aSet.removeAll(staticSet);
		Set<Symbol> localSet = SymbolTools.getLocalSymbols(st);
		aSet.removeAll(localSet);
		/////////////////////////////////////////////////
		// Add accessed global symbols to rSet.        //
		////////////////////////////////////////////////////////////////
		// Accessed global symbols = accessed symbols - local symbols //
		// - static symbols - function parameter symbols              //
		////////////////////////////////////////////////////////////////
		for( Symbol sym : aSet ) {
			if( !SymbolTools.isFormal(sym) ) {
				rSet.add(sym);
			}
		}
		List<FunctionCall> calledFuncs = IRTools.getFunctionCalls(st);
		for( FunctionCall call : calledFuncs ) {
			Procedure called_procedure = call.getProcedure();
			if( called_procedure != null ) {
				CompoundStatement body = called_procedure.getBody();
				Set<Symbol> procAccessedSymbols = getIpAccessedGlobalorStaticSymbols(body);
				rSet.addAll(procAccessedSymbols);
			}
		}
		return rSet;
	}
	
	/**
	 * Find the original symbol of the input symbol, inSym.
	 * If the input symbol is a function parameter, and the corresponding
	 * argument is complex, 
	 *     - return a list of the original symbol only.
	 * Else if the input symbol is a global variable,
	 *     - return a list of the symbol and the parent TranslationUnit
	 * Else if the input symbol is a static variable in a procedure
	 *     - return a list of the original symbol and the parent Procedure
	 * Else if the input symbol is a local variable in a main procedure
	 *     - return a list of the original symbol and the parent Procedure
	 * Else if the input symbol is a local variable or value-passed parameter
	 * of a procedure
	 *     - return a list of the original symbol only.
	 * Else 
	 *     - return an empty list.
	 * 
	 * @param inSym input symbol
	 * @param t procedure where inSym exists
	 * @return list of the original symbol and the parent TranslationUnit/Procedure.
	 */
	public static List findOrgSymbol(Symbol inSym, Traversable t) {
		List symbolInfo = new LinkedList();
		Set<Symbol> symSet = null;
		Procedure p_proc = null;
		TranslationUnit t_unit = null;
		Program program = null;
		// Find a parent Procedure.
		while (true) {
			if (t instanceof Procedure) break;
			t = t.getParent(); 
		}
		p_proc = (Procedure)t;
		// Find a parent TranslationUnit.
		while (true) {
			if (t instanceof TranslationUnit) break;
			t = t.getParent(); 
		}
		t_unit = (TranslationUnit)t;
		program = (Program)t_unit.getParent();
		//////////////////////////////////////////////////
		// Check whether inSym is a function parameter. //
		// If so, find the actual argument.             //
		//////////////////////////////////////////////////
		List<FunctionCall> funcCallList = IRTools.getFunctionCalls(program);
		boolean complexExp = false;
		while (true) {
			symSet = SymbolTools.getVariableSymbols((SymbolTable)p_proc);
			if( symSet.contains(inSym) ) { 
				if( (SymbolTools.isArray(inSym) || SymbolTools.isPointer(inSym)) ) {
					// Find the caller procedure that called this procedure.
					List paramList = p_proc.getParameters();
					Procedure t_proc = p_proc;
					Symbol argSym = null;
					boolean foundArg = false;
					for( FunctionCall funcCall : funcCallList ) {
						///////////////////////////////////////////////////////////////////////
						//DEBUG: below code does not work with the cloning mechanism         //
						//implemented in IpResidentGVariableAnalysis and IpG2CMemTrAnalysis. //
						///////////////////////////////////////////////////////////////////////
						//if(t_proc.equals(funcCall.getProcedure())) {
						if(t_proc.getName().equals(funcCall.getName())) {
							t = funcCall.getStatement();
							while( (t != null) && !(t instanceof Procedure) ) {
								t = t.getParent();
							}
							p_proc = (Procedure)t;
							List argList = funcCall.getArguments();
							for( int i=0; i<paramList.size(); i++ ) {
								////////////////////////////////////////////////////////////////////////
								// DEBUG: IRTools.containsSymbol() works only for searching           //
								// symbols in expression tree; it internally compare Identifier only, //
								// but VariableDeclarator contains NameID instead of Identifier.      //
								////////////////////////////////////////////////////////////////////////
								//if( IRTools.containsSymbol((Declaration)paramList.get(i), 
								//		inSym)) {
								List declaredSyms = ((Declaration)paramList.get(i)).getDeclaredIDs();
								if( declaredSyms.contains(new Identifier(inSym)) ) {
									// Found an actual argument for the inSym. 
									foundArg = true;
									Expression exp = (Expression)argList.get(i);
									///////////////////////////////////////////////////////////////////
									// FIXME: if the passed argument is complex, current translator  //
									// can not calculate the region accessed in the called function. //
									// Therefore, only the following expressions are handled for now.//
									//     - Simple pointer identifier (ex: a)                       //
									//     - Simple unary expression (ex: &b)                        //
									///////////////////////////////////////////////////////////////////
									boolean acceptableArg = false;
									if( exp instanceof Literal ) {
										PrintTools.println("[INFO in findOrgSymbol()] argument (" + exp + 
												") passed for " + "the parameter symbol (" + inSym + 
												") of a procedure, " + t_proc.getName() + ", is a literal", 2);
										symbolInfo.add(inSym);
										return symbolInfo;
									} else if( exp instanceof Identifier ) {
										acceptableArg = true;
									} else if ( exp instanceof UnaryExpression ) {
										UnaryExpression uexp = (UnaryExpression)exp;
										if( uexp.getOperator().equals(UnaryOperator.ADDRESS_OF) &&
											(uexp.getExpression() instanceof Identifier) ) {
												acceptableArg = true;
											}
									}
									if( !acceptableArg ) {
										PrintTools.println("[WARNING in findOrgSymbol()] argument (" + exp + 
												") passed for " + "the parameter symbol (" + inSym + 
												") of a procedure, " + t_proc.getName() + ", has complex expression; " +
												"this symbol may be aliased to other symbols.",1);
										//symbolInfo.add(inSym);
										//return symbolInfo;
										complexExp = true;
									}
									inSym = SymbolTools.getSymbolOf(exp);
									if( argSym == null ) {
										argSym = inSym;
									} else if( !argSym.equals(inSym) ) {
										// Multiple argument symbols are found.
										PrintTools.println("[WARNING in findOrgSymbol()] multiple argments exist " +
												"for the parameter symbol (" + inSym + ") of procedure (" 
												+ t_proc.getSymbolName() + "); can't find the original symbol", 1);
										return symbolInfo;
									}
									break;
								}
							}
						}
					}
					if( !foundArg ) {
						PrintTools.println("[WARNING in findOrgSymbol()] can not find the argument passed for " +
								"the parameter symbol (" + inSym + ") of a procedure, " + t_proc.getName() + ".", 0);
						return symbolInfo;
					}
				} else if( SymbolTools.isScalar(inSym) ) {
					symbolInfo.add(inSym);
					//symbolInfo.add(p_proc);
					return symbolInfo;
				} else {
					// Unknown type.
					PrintTools.println("[WARNING in findOrgSymbol()] Unknown type found", 0);
					return symbolInfo;
				}
			} else {
				break;
			}
		}
		// If actual argument is complex, it is not globally allocated.
		if( complexExp ) {
			symbolInfo.add(inSym);
			return symbolInfo;
		}
		// Check whether inSym is a global symbol of the File scope.
		symSet = SymbolTools.getVariableSymbols((SymbolTable)t_unit);
		if( symSet.contains(inSym) ) {
			symbolInfo.add(inSym);
			symbolInfo.add(t_unit);
			return symbolInfo;
		}
		// Check whether inSym is a local symbol of the parent procedure.
		symSet = SymbolTools.getLocalSymbols((SymbolTable)p_proc.getBody());
		if( symSet.contains(inSym) ) {
			String name = p_proc.getSymbolName();
			List<Specifier> specs = inSym.getTypeSpecifiers();
			symbolInfo.add(inSym);
			////////////////////////////////////////////////////////////////////
			// If the original variable is a static variable in a procedure   //
			// or a local variable in a main procedure, allocate it globally. //
			////////////////////////////////////////////////////////////////////
			if( specs.contains(Specifier.STATIC) 
					|| name.equals("main") || name.equals("MAIN__") ) {
				symbolInfo.add(p_proc);
			}
			return symbolInfo;
		}
		// Can't find the original symbol; return an empty list.
		return symbolInfo;
	}
	
	/**
	 * Find the enclosing parallel region interprocedurally.
	 * If multiple parallel regions exist, the first one will be returned.
	 * 
	 * @param t input code from where search starts.
	 * @return enclosing parallel region
	 */
	public static Traversable findEnclosingParallelRegion(Traversable t) {
		OmpAnnotation pAnnot = null;
		while( !(t instanceof Annotatable) ) {
			t = t.getParent();
		}
		pAnnot = ((Annotatable)t).getAnnotation(OmpAnnotation.class, "parallel");
		while( (pAnnot == null) && !(t instanceof Procedure) ) {
			t = t.getParent();
			pAnnot = ((Annotatable)t).getAnnotation(OmpAnnotation.class, "parallel");
		}
		if( pAnnot != null ) {
			//Found the enclosing parallel region.
			return t;
		}
		//Search the enclosing parallel region interprocedurally.
		if( t instanceof Procedure ) {
			Procedure t_proc = (Procedure)t;
			while( !(t instanceof Program) ) {
				t = t.getParent();
			}
			List<FunctionCall> funcCallList = IRTools.getFunctionCalls(t);
			for( FunctionCall funcCall : funcCallList ) {
				if(t_proc.getName().equals(funcCall.getName())) {
					t = findEnclosingParallelRegion(funcCall.getStatement());
					if( t != null ) {
						return t;
					}
				}
			}
		}
		return null;
	}
	
	/**
	 * Check whether a shared symbol, sharedVar, is accessed in the code, tr.
	 * If tr is a function call, this searches the called function too.
	 * 
	 * @param sharedVar shared variable to be searched
	 * @param tr Traversable tr
	 * @return true if the shared variable, sharedVar, is accessed in the code tr.
	 */
	public static boolean checkSharedVariableAccess( Symbol sharedVar, Traversable tr ) {
		boolean isAccessed = false;
		isAccessed = IRTools.containsSymbol(tr, sharedVar);
		if( !isAccessed ) {
			Expression expr = null;
			if( tr instanceof ExpressionStatement ) {
				expr = ((ExpressionStatement)tr).getExpression();
			} else if( tr instanceof Expression ) {
				expr = (Expression)tr;
			}
			if( (expr != null) && (expr instanceof FunctionCall) ) {
				isAccessed = IRTools.containsSymbol(((FunctionCall)expr).getProcedure(), sharedVar);
			}
		}
		return isAccessed;
	}
	
	/**
	 * check whether the input variable is a member of a class.
	 * 
	 * @param varSym input variable symbol
	 * @return true if input variable is a member of a class
	 */
	public static boolean isClassMember(VariableDeclarator varSym) {
		Traversable t = varSym.getParent();
		boolean foundParentClass = false;
		while( t != null ) {
			if( t instanceof ClassDeclaration ) {
				foundParentClass = true;
				break;
			} else {
				t = t.getParent();
			}
		}
		return foundParentClass;
	}
	
	  /**
	  * Returns a list of pragma annotations that contain the specified string keys
	  * and are attached to annotatable objects within the traversable object
	  * {@code t}. For example, it can collect list of OpenMP pragmas having
	  * a work-sharing directive {@code for} within a specific procedure.
	  * If functions are called within the traversable object (@code t), 
	  * the called functions are recursively searched.
	  *
	  * @param t the traversable object to be searched.
	  * @param pragma_cls the type of pragmas to be searched for.
	  * @param key the keyword to be searched for.
	  * @return the list of matching pragma annotations.
	  */
	  public static <T extends PragmaAnnotation> List<T>
	      ipCollectPragmas(Traversable t, Class<T> pragma_cls, String key)
	  {
	    List<T> ret = new LinkedList<T>();

	    DepthFirstIterator iter = new DepthFirstIterator(t);
	    while ( iter.hasNext() )
	    {
	      Object o = iter.next();
	      if ( o instanceof Annotatable )
	      {
	        Annotatable at = (Annotatable)o;
	        List<T> pragmas = at.getAnnotations(pragma_cls);
	        if( pragmas != null ) {
	          for ( T pragma : pragmas )
	            if ( pragma.containsKey(key) )
	              ret.add(pragma);
	        }
	      } else if( o instanceof FunctionCall ) {
	    	  FunctionCall funCall = (FunctionCall)o;
	    	  if( !StandardLibrary.contains(funCall) ) {
	    		  Procedure calledProc = funCall.getProcedure();
	    		  if( calledProc != null ) { 
	    			  ret.addAll(ipCollectPragmas(calledProc, pragma_cls, key));
	    		  }
	    	  }
	      }
	    }
	    return ret;
	  }
	
	/**
	 * Insert barriers before and after each kernel region, so that other analysis can
	 * easily distinguish kernel regions from CPU regions. 
	 */
	public static void markIntervalForKernelRegions(Program program) {
		/* iterate to search for all Procedures */
		DepthFirstIterator proc_iter = new DepthFirstIterator(program);
		Set<Procedure> proc_list = (Set<Procedure>)(proc_iter.getSet(Procedure.class));
		CompoundStatement target_parent;
		for (Procedure proc : proc_list)
		{
			/* Search for all OpenMP parallel regions in a given Procedure */
			List<OmpAnnotation>
			omp_annots = IRTools.collectPragmas(proc, OmpAnnotation.class, "parallel");
			for ( OmpAnnotation annot : omp_annots )
			{
				Statement target_stmt = (Statement)annot.getAnnotatable();
				int eligibility = AnalysisTools.checkKernelEligibility(target_stmt);
				if( eligibility == 0 ) {
					target_parent = (CompoundStatement)target_stmt.getParent();
					target_parent.addStatementBefore(target_stmt, SplitOmpPRegion.insertBarrier("S2P"));
					target_parent.addStatementAfter(target_stmt, SplitOmpPRegion.insertBarrier("P2S"));
				} else if (eligibility == 3) {
					// Check whether this parallel region is an omp-for loop.
					if( annot.containsKey("for") ) {
						// In the new annotation scheme, the above check is redundant.
						eligibility = 0;
					} else {
						// Check whether called functions have any omp-for loop.
						/////////////////////////////////////////////////////////////////////////////////
						// FIXME: if a function in the function called in the parallel region contains //
						// omp for loop, below checking can not detect it.                             //
						/////////////////////////////////////////////////////////////////////////////////
						List<FunctionCall> funcCalls = IRTools.getFunctionCalls(target_stmt); 
						for( FunctionCall calledProc : funcCalls ) {
							Procedure tProc = calledProc.getProcedure();
							if( tProc != null ) {
								eligibility = AnalysisTools.checkKernelEligibility(tProc.getBody());
								if(  eligibility == 0 ) {
									break;
								}
							}
						}
					}
					if( eligibility == 0 ) {
						target_parent = (CompoundStatement)target_stmt.getParent();
						target_parent.addStatementBefore(target_stmt, SplitOmpPRegion.insertBarrier("S2P"));
						target_parent.addStatementAfter(target_stmt, SplitOmpPRegion.insertBarrier("P2S"));
					}
				} 
			}
		}
	}
	
	/**
	 * Class REGIONMAP represents map from variables to their region. 
	 */
	public static class REGIONMAP extends HashMap<Symbol,String> implements Cloneable
	{
		private static final long serialVersionUID = 14L;	
		/**
		 * Constructs an empty map.
		 */
		public REGIONMAP()
		{
			super();
		}

		/**
		 * Constructs a map with a pair of variable and section.
		 *
		 * @param var the key variable.
		 * @param section the section associated with the variable.
		 */
		public REGIONMAP(Symbol var, String region)
		{
			super();
			put(var, region);
		}

		/**
		 * Returns a clone object.
		 */
		public Object clone()
		{
			REGIONMAP o = new REGIONMAP();

			for ( Symbol var : keySet() )
				o.put(var, get(var));

			return o;
		}
		
		/**
		 * Performs intersection operation between the two region maps.
		 *
		 * @param other the other region map to be intersected with.
		 * @return true if anything is changed.
		 */
		public boolean retainAll(REGIONMAP other)
		{
			boolean changed = false;
			if ( other == null && !isEmpty() ) {
				changed = true;
				clear();
			}
			for ( Symbol var : keySet() )
			{
				String s1 = get(var);
				String s2 = other.get(var);

				if ( s2 == null ) {
					changed = true;
					remove(var);
				} else if( !s1.equals(s2) ) {
					changed = true;
					put(var, new String("Unknown"));
				}
			}
			return changed;
		}

		/**
		 * Performs intersection operation between the two region maps.
		 *
		 * @param other the other region map to be intersected with.
		 * @return the resulting region map after intersection.
		 */
		public REGIONMAP intersectWith(REGIONMAP other)
		{
			REGIONMAP ret = new REGIONMAP();

			if ( other == null )
				return ret;

			for ( Symbol var : keySet() )
			{
				String s1 = get(var);
				String s2 = other.get(var);

				if ( s1 == null || s2 == null )
					continue;

				if( s1.equals(s2) ) {
					ret.put(var, s1);
				} else {
					ret.put(var, new String("Unknown"));
				}
			}
			return ret;
		}
		
		/**
		 * add other region map to this region map.
		 *
		 * @param other the other region map to be added.
		 * @return true if anything is changed or added.
		 */
		public boolean addAll(REGIONMAP other)
		{
			boolean changed = false;
			if ( other == null )
				return changed;

			for ( Symbol var : other.keySet() )
			{
				String s1 = get(var);
				String s2 = other.get(var);

				if ( s1 == null && s2 == null )
					continue;

				if ( s1 == null )
				{
					changed = true;
					put(var, s2);
				}
				else if ( s2 != null )
				{
					if( !s1.equals(s2) ) {
						changed = true;
						put(var, new String("Unknown"));
					}
				}
			}
			return changed;
		}

		/**
		 * Performs union operation between the two region maps.
		 *
		 * @param other the other region map to be united with.
		 * @return the resulting region map after union.
		 */
		public REGIONMAP unionWith(REGIONMAP other)
		{
			if ( other == null )
				return (REGIONMAP)clone();

			REGIONMAP ret = new REGIONMAP();

			Set<Symbol> vars = new HashSet<Symbol>(keySet());
			vars.addAll(other.keySet());

			for ( Symbol var : vars )
			{
				String s1 = get(var);
				String s2 = other.get(var);

				if ( s1 == null && s2 == null )
					continue;

				if ( s1 == null )
				{
					ret.put(var, s2);
				}
				else if ( s2 == null )
					ret.put(var, s1);
				else
				{
					if( s1.equals(s2) ) {
						ret.put(var, s1);
					} else {
						ret.put(var, new String("Unknown"));
					}
				}
			}
			return ret;
		}
		
		/**
		 * Performs overwriting union operation between the two region maps.
		 *
		 * @param other the other region map to be united with.
		 * @return the resulting region map after union.
		 */
		public REGIONMAP overwritingUnionWith(REGIONMAP other)
		{
			if ( other == null )
				return (REGIONMAP)clone();

			REGIONMAP ret = new REGIONMAP();

			Set<Symbol> vars = new HashSet<Symbol>(keySet());
			vars.addAll(other.keySet());

			for ( Symbol var : vars )
			{
				String s1 = get(var);
				String s2 = other.get(var);

				if ( s1 == null && s2 == null )
					continue;

				if ( s1 == null ) {
					ret.put(var, s2);
				} else {
					ret.put(var, s1);
				}
			}
			return ret;
		}
		
		/**
		 * Removes sections that are unsafe in the given traversable object due to
		 * function calls.
		 */
		public void removeSideAffected(Traversable tr)
		{
			DepthFirstIterator iter = new DepthFirstIterator(tr);

			iter.pruneOn(FunctionCall.class);

			while ( iter.hasNext() )
			{
				Object o = iter.next();
				if ( o instanceof FunctionCall )
				{
					Set<Symbol> vars = new HashSet<Symbol>(keySet());
					FunctionCall fc = (FunctionCall)o;
					for ( Symbol var : vars ) {
						Set<Symbol> params = SymbolTools.getAccessedSymbols(fc);
						// Case 1: variables are used as parameters.
						if ( params.contains(var) ) {
							put(var, new String("Unknown"));
						}
						// Case 2: variables are global.
						if ( SymbolTools.isGlobal(var, fc) ) {
							put(var, new String("Unknown"));
						}
					}
				}
			}
		}
		
		/**
		 * Update symbols in this region map.
		 * 
		 * @param t region, from which symbol search starts.
		 */
		public void updateSymbols(Traversable t) {
			HashSet<Symbol> old_set = new HashSet<Symbol>(keySet());
			HashSet<Symbol> new_set = new HashSet<Symbol>();
			AnalysisTools.updateSymbols(t, old_set, new_set, false);
			for( Symbol oSym : old_set ) {
				String old_sym = oSym.getSymbolName();
				String region = get(oSym);
				remove(oSym);
				for( Symbol nSym : new_set ) {
					if( nSym.getSymbolName().equals(old_sym) ) {
						put(nSym, region);
						break;
					}
				}
			}
		}
	}

	/**
	 * For each barrier node, add the following mapping:
	 *     - If a barrier node's type = "S2P", put ("kernelRegion", pStmt) mapping
	 *     - If a barrier node's type = "P2S", put ("pKernelRegion", pStmt) mapping
	 * where pStmt is a parallel region related to the barrier.
	 * This mapping is needed for CFG-based analysis, such as liveGVariableAnalysis() or
	 * reachingGMallocAnalysis(), to identify kernel regions.
	 * 
	 * @param proc
	 * @param cfg
	 */
	public static void annotateBarriers(Procedure proc, CFGraph cfg) {
		HashSet<Statement> s2pBarriers = new HashSet<Statement>();
		HashSet<Statement> p2sBarriers = new HashSet<Statement>();
		HashMap<Statement, Statement> s2pRegions = new HashMap<Statement, Statement>();
		HashMap<Statement, Statement> p2sRegions = new HashMap<Statement, Statement>();
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
				s2pBarriers.add(bstmt);
				s2pRegions.put(bstmt, pstmt);
			} else if( type.equals("P2S") ) {
				bstmt = (Statement)omp_annot.getAnnotatable();
				pstmt = AnalysisTools.getStatementBefore((CompoundStatement)bstmt.getParent(), 
						bstmt);
				p2sBarriers.add(bstmt);
				p2sRegions.put(bstmt, pstmt);
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
			String type = node.getData("type");
			if( type == null ) {
				Tools.exit("[ERROR in AnalysisTools.annotBarriers()] " +
						"DFANode for a barrier does not have type information!");
			} else if( type.equals("S2P") ) {
				for( Statement stmt : s2pBarriers ) {
					if( stmt.equals(IRStmt) ) {
						found_bBarrier = true;
						foundStmt = stmt;
						break;
					}
				}
				if( found_bBarrier ) {
					Statement pStmt = s2pRegions.get(foundStmt);
					node.putData("kernelRegion", pStmt);
				}
			} else if( type.equals("P2S") ) {
				for( Statement stmt : p2sBarriers ) {
					if( stmt.equals(IRStmt) ) {
						found_bBarrier = true;
						foundStmt = stmt;
						break;
					}
				}
				if( found_bBarrier ) {
					Statement pStmt = p2sRegions.get(foundStmt);
					node.putData("pKernelRegion", pStmt);
				}
			}
		}	
	}
	
	/**
	 * Intra-procedural, backward data-flow analysis to compute liveG_out, 
	 * a set of live GPU variables, which may be accessed in later nodes. 
	 * Input  : CFGraph cfg of a procedure called by CPU.
	 * Output : liveG set for each node in cfg
	 *
	 * liveG_out(exit-node) = {}	: only intra-procedural analysis
	 *
	 * for ( node m : successor nodes of node n )
	 * 	liveG_out(n)  += liveG_in(m) // + : union
	 * 
	 * liveG_in(n) = liveG_out(n) + GEN(n) - KILL(n) // + : union
	 *  where,
	 *   GEN(n) = set of shared variables
	 *                - if n is a barrier node before a kernel region
	 *            ()  - otherwise 
	 *   KILL(n) = set of shared variables accessed in a function call    
	 *                 - if n represents a function call
	 *             set of R/O shared scalar variables in a kernel region
	 *              	- if the variables do not exist in liveG_out set
	 *                    and if shrdSclrCachingOnSM option is on
	 *                    and if n is a barrier node before a kernel region.
	 *             ()  - otherwise
	 * 
	 * Since this analysis is MAY analysis, inaccurate analysis will cause that 
	 * some necessary cudaFree() calls are omitted, incurring garbages in 
	 * GPU global memory.
	 * For each kernel region, liveG_out set is stored in a barrier just before 
	 * the kernel region.
	 * To run this analysis, markIntervalForKernelREgins() and annotateBarriers() 
	 * should be called before this method.
	 * 
	 * @param cfg control flow graph of a procedure of interest.
	 * @param includeBackEdge back edge in a For loop is added if true.
	 */
	public static void liveGVariableAnalysis(CFGraph cfg, boolean includeBackEdge) {
		PrintTools.println("Run liveGVariableAnalysis", 1);
		//Check whether shrdSclrCachingOnSM is on or not.
		boolean	shrdSclrCachingOnSM = false;
		String value = Driver.getOptionValue("shrdSclrCachingOnSM");
		if( value != null ) {
			shrdSclrCachingOnSM = true;
		}
		TreeMap work_list = new TreeMap();

		// Enter the exit node in the work_list
		List<DFANode> exit_nodes = cfg.getExitNodes();
		if (exit_nodes.size() > 1)
		{
			PrintTools.println("[liveGVariableAnalysis] Warning: multiple exits in the program", 2);
		}

		for ( DFANode exit_node : exit_nodes )
			work_list.put((Integer)exit_node.getData("top-order"), exit_node);

		// Do iterative steps
		while ( !work_list.isEmpty() )
		{
			DFANode node = (DFANode)work_list.remove(work_list.lastKey());
			Set<Symbol> curr_set = new HashSet<Symbol>();

			// calculate the current live_out to check if there is any change
			for ( DFANode succ : node.getSuccs() )
			{
				Set<Symbol> succ_set = (Set<Symbol>)succ.getData("liveG_in");
				if( succ_set != null ) {
					curr_set.addAll(succ_set);
				}
			}

			// retrieve previous live_out
			Set<Symbol> prev_set = (Set<Symbol>)node.getData("liveG_out");

			if ( prev_set == null || !prev_set.equals(curr_set) )
			{
				// since live_out has been changed, we update it.
				node.putData("liveG_out", curr_set);

				///////////////////////////////////////////
				// Calculate liveG_in = liveG_out + GEN. //
				///////////////////////////////////////////
				// compute liveG_in, a set of live GPU variables, which are accessed 
				// in the current or later nodes.
				Statement stmt = node.getData("kernelRegion");
				Set<Symbol> liveG_in = new HashSet<Symbol>();
				liveG_in.addAll(curr_set);
				if( stmt != null ) {
					OmpAnnotation annot = stmt.getAnnotation(OmpAnnotation.class, "parallel");
					if( (annot != null) && (AnalysisTools.checkKernelEligibility(stmt) == 0) ) {
						Set<Symbol> sharedVars = (Set<Symbol>)annot.get("shared");
						if( sharedVars != null ) {
							Set<Symbol> defSyms = DataFlowTools.getDefSymbol(stmt);
							for(Symbol sSym: sharedVars) {
								////////////////////////////////////////////////////////////
								// If shrdSclrCachingOnSM option is on, R/O shared scalar //
								// variables are not accessed as GPU variables in this    //
								// kernel region.                                         //
								////////////////////////////////////////////////////////////
								if( SymbolTools.isScalar(sSym) && !defSyms.contains(sSym) &&
										shrdSclrCachingOnSM ) {
									continue;
								} else {
									liveG_in.add(sSym);
								}
							}
						}
					}
				}
				//////////////////////
				// Handle KILL Set. //
				//////////////////////
				// If this node is a function call, and some shared variables in liveG_in set are 
				// accessed in the called function, those should be removed from the liveG_in set
				// since we don't know whether those variables are accessed by CPU or GPU; 
				// interprocedural analysis can fix this uncertainty.
				if( liveG_in.size() > 0 ) {
					Traversable ir = node.getData("ir");
					if( (ir != null) && (ir instanceof ExpressionStatement) ) {
						Expression expr = ((ExpressionStatement)ir).getExpression();
						if( expr instanceof FunctionCall ) {
							Set<Symbol> removeSet = new HashSet<Symbol>();
							for( Symbol sym: liveG_in ) {
								if( checkSharedVariableAccess(sym, expr) ) {
									removeSet.add(sym);
								}
							}
							liveG_in.removeAll(removeSet);
						}
					}
				}
				node.putData("liveG_in", liveG_in);

				DFANode temp = (DFANode)node.getData("back-edge-from");
				if( temp == null || includeBackEdge) {
					for ( DFANode pred : node.getPreds() ) {
						work_list.put(pred.getData("top-order"), pred);
					}
				} else {
					for ( DFANode pred : node.getPreds() ) {
						if( temp != pred ) {
							work_list.put(pred.getData("top-order"), pred);
						}
					}
				}
			}
		}
	}
	
	/**
	 * Intra-procedural, backward data-flow analysis to compute advLiveG_out, 
	 * a set of live GPU variables, which may be accessed in later nodes and 
	 * refer to the same cudaMalloced data. 
	 * Input  : CFGraph cfg of a procedure called by CPU.
	 * Output : advLiveG set for each node in cfg
	 *
	 * advLiveG_out(exit-node) = {}	: only intra-procedural analysis
	 *
	 * for ( node m : successor nodes of node n )
	 * 	advLiveG_out(n)  += advLiveG_in(m) // + : union
	 * 
	 * advLiveG_in(n) = advLiveG_out(n) + GEN(n) - KILL(n) // + : union
	 *  where,
	 *   GEN(n) = set of shared variables
	 *                - if n is a barrier node before a kernel region
	 *            ()  - otherwise 
	 *   KILL(n) = set of re-malloced shared variables 
	 *                 - if n is a barrier node before a kernel region
	 *             set of shared variables accessed in a function call    
	 *                 - if n represents a function call
	 *             set of R/O shared scalar variables in a kernel region
	 *              	- if the variables do not exist in advLiveG_out set
	 *                    and if shrdSclrCachingOnSM option is on
	 *                    and if n is a barrier node before a kernel region. 
	 *             ()  - otherwise
	 * 
	 * This analysis differs from liveGVariableAnalysis, since this checks
	 * whether the GPU variables accessed in later nodes refer to the same
	 * cudaMalloced variables. Therefore, in this analysis, live GPU variables 
	 * mean GPU variables that are accessed in later node and NOT reMalloced. 
	 * Since this analysis is MAY analysis, inaccurate analysis will cause that 
	 * some necessary cudaFree() calls are omitted, incurring garbages in 
	 * GPU global memory.
	 * For each kernel region, advLiveG_out set is stored in a barrier just before 
	 * the kernel region.
	 * To run this analysis, markIntervalForKernelREgins(), annotateBarriers(),
	 * liveGVariableAnalysis(), and reachingGMallocAnalysis() should be called 
	 * before this method.
	 * 
	 * @param cfg control flow graph of a procedure of interest.
	 * @param includeBackEdge back edge in a For loop is added if true.
	 */
	public static void advLiveGVariableAnalysis(CFGraph cfg, boolean includeBackEdge) {
		PrintTools.println("Run advLiveGVariableAnalysis", 1);
		TreeMap work_list = new TreeMap();

		// Enter the exit node in the work_list
		List<DFANode> exit_nodes = cfg.getExitNodes();
		if (exit_nodes.size() > 1)
		{
			PrintTools.println("[advLiveGVariableAnalysis] Warning: multiple exits in the program", 2);
		}

		for ( DFANode exit_node : exit_nodes )
			work_list.put((Integer)exit_node.getData("top-order"), exit_node);

		// Do iterative steps
		while ( !work_list.isEmpty() )
		{
			DFANode node = (DFANode)work_list.remove(work_list.lastKey());
			Set<Symbol> curr_set = new HashSet<Symbol>();

			// calculate the current live_out to check if there is any change
			for ( DFANode succ : node.getSuccs() )
			{
				Set<Symbol> succ_set = (Set<Symbol>)succ.getData("advLiveG_in");
				if( succ_set != null ) {
					curr_set.addAll(succ_set);
				}
			}

			// retrieve previous live_out
			Set<Symbol> prev_set = (Set<Symbol>)node.getData("advLiveG_out");

			if ( prev_set == null || !prev_set.equals(curr_set) )
			{
				// since live_out has been changed, we update it.
				node.putData("advLiveG_out", curr_set);

				/////////////////////////////////////////////////
				// Calculate advLiveG_in = advLiveG_out + GEN. //
				/////////////////////////////////////////////////
				Set<Symbol> advLiveG_in = new HashSet<Symbol>();
				advLiveG_in.addAll(curr_set);
				/////////////////////////////
				// Handle GEN - KILL1 set. //
				/////////////////////////////
				String tag = node.getData("tag");
				String type = node.getData("type");
				if( (tag != null) && (type != null) && (tag.equals("barrier")
						&& (type.equals("S2P"))) ) {
					Set<Symbol> liveG_in = node.getData("liveG_in");
					if( liveG_in == null ) {
						Tools.exit("[Error in advLiveGVariableAnalsys()] liveG_in set does not exist; " +
						"Run liveGVariableAnalysis() before this analysis.");
					}	
					advLiveG_in.addAll(liveG_in);
					Set<Symbol> reachingGMalloc_in = node.getData("reachingGMalloc_in");
					Set<Symbol> removeSet = new HashSet<Symbol>();
					if( reachingGMalloc_in == null ) {
						Tools.exit("[Error in advLiveGVariableAnalsys()] reachingGMalloc_in set does not exist; " +
						"Run reachingGMallocAnalysis() before this analysis.");
					}
					if( advLiveG_in.size() > 0 ) {
						for( Symbol sym: advLiveG_in ) {
							if( !reachingGMalloc_in.contains(sym) ) {
								removeSet.add(sym);
							}
						}
						advLiveG_in.removeAll(removeSet);
					}
				}
				
				///////////////////////
				// Handle KILL2 Set. //
				///////////////////////
				// If this node is a function call, and some shared variables in advLiveG_in set are 
				// accessed in the called function, those should be removed from the advLiveG_in set
				// since we don't know whether those variables are accessed by CPU or GPU; 
				// interprocedural analysis can fix this uncertainty.
				if( advLiveG_in.size() > 0 ) {
					Traversable ir = node.getData("ir");
					if( (ir != null) && (ir instanceof ExpressionStatement) ) {
						Expression expr = ((ExpressionStatement)ir).getExpression();
						if( expr instanceof FunctionCall ) {
							Set<Symbol> removeSet = new HashSet<Symbol>();
							for( Symbol sym: advLiveG_in ) {
								if( checkSharedVariableAccess(sym, expr) ) {
									removeSet.add(sym);
								}
							}
							advLiveG_in.removeAll(removeSet);
						}
					}
				}
				node.putData("advLiveG_in", advLiveG_in);

				DFANode temp = (DFANode)node.getData("back-edge-from");
				if( temp == null || includeBackEdge) {
					for ( DFANode pred : node.getPreds() ) {
						work_list.put(pred.getData("top-order"), pred);
					}
				} else {
					for ( DFANode pred : node.getPreds() ) {
						if( temp != pred ) {
							work_list.put(pred.getData("top-order"), pred);
						}
					}
				}
			}
		}
	}

	/**
	 * Intra-procedural, forward data-flow analysis to compute reachingGMalloc_in, 
	 * a set of GPU variables mallocated in the previous nodes.
	 * 
	 * Input  : CFGraph cfg of a procedure called by CPU.
	 * Output : reachingGMalloc set for each node in cfg
	 *
	 * reachingGMalloc_in(entry-node) = {}	: only intra-procedural analysis
	 *
	 * for ( node m : predecessor nodes of node n )
	 * 	reachingGMalloc_in(n)  ^= reachingGMalloc_out(m) // ^ : intersection
	 * 
	 * reachingGMalloc_out(n) = reachingGMalloc_in(n) + GEN(n) - KILL(n) // + : union
	 *  where,
	 *   GEN(n) = set of shared variables
	 *                - if n is a barrier node before a kernel region
	 *            ()  - otherwise 
	 *   KILL(n) = set of cudaFreed shared variables 
	 *                 - if n is a barrier node before a kernel region 
	 *             set of shared variables accessed in a function call    
	 *                 - if n represents a function call
	 *             set of R/O shared scalar variables in a kernel region
	 *             		- if the variables do not exist in reachingGMalloc_in set
	 *                    and if shrdSclrCachingOnSM option is on
	 *                    and if n is a barrier node before a kernel region.
	 *             ()  - otherwise
	 *   
	 * For each kernel region, reachingGMalloc_in set is stored in a barrier just before 
	 * the kernel region.
	 * To run this analysis, markIntervalForKernelRegins(), annotateBarriers(), and 
	 * liveGVariableAnalysis() should be called before this method.
	 * [CAUTION] This analysis assumes that the procedure of interest (cfg) is
	 * called by CPU, even though it can contain kernel function calls.
	 * 
	 * @param cfg control flow graph of a procedure of interest.
	 */
	public static void reachingGMallocAnalysis(CFGraph cfg) {
		//Check whether shrdSclrCachingOnSM is on or not.
		boolean	shrdSclrCachingOnSM = false;
		String value = Driver.getOptionValue("shrdSclrCachingOnSM");
		if( value != null ) {
			shrdSclrCachingOnSM = true;
		}
		
		TreeMap work_list = new TreeMap();
	
		// Enter the entry node in the work_list
		DFANode entry = cfg.getNodeWith("stmt", "ENTRY");
		entry.putData("reachingGMalloc_in", new HashSet<Symbol>());
		work_list.put(entry.getData("top-order"), entry);
		
		////////////////////////////////////////////////////////////////////////////
		// [CAUTION] This analysis assumes that the procedure of interest is      //
		// called by CPU, even though it can contain kernel function calls.       //
		////////////////////////////////////////////////////////////////////////////
		String currentRegion = new String("CPU");
	
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
	
			//Set<Symbol> GMalloc_in = new HashSet<Symbol>();
			HashSet<Symbol> GMalloc_in = null;

			for ( DFANode pred : node.getPreds() )
			{
				// Calculate intersection of previous nodes
				Set<Symbol> pred_GMalloc_out = (Set<Symbol>)pred.getData("reachingGMalloc_out");

				if ( GMalloc_in == null ) {
					if ( pred_GMalloc_out != null ) {
						GMalloc_in = new HashSet<Symbol>();
						GMalloc_in.addAll(pred_GMalloc_out);
					}
				} else {
					// Intersect it with the current data
					if ( pred_GMalloc_out != null ) {
						GMalloc_in.retainAll(pred_GMalloc_out);
					} /* else {
						//This is the first visit to this node; ignore it.
						//GMalloc_in.clear();
					} */
				}
			}
	
			// previous reachingGMalloc_in
			Set<Symbol> p_GMalloc_in = (Set<Symbol>)node.getData("reachingGMalloc_in");
	
			if ( (GMalloc_in == null) || (p_GMalloc_in == null) || !GMalloc_in.equals(p_GMalloc_in) ) {
				node.putData("reachingGMalloc_in", GMalloc_in);
	
				/////////////////////////////////////////////////////////
				// Compute reachingGMalloc_out, a set of GPU variables //
				// mallocated in the current or previous nodes.        //
				/////////////////////////////////////////////////////////
				Set<Symbol> GMalloc_out = new HashSet<Symbol>();
				if( GMalloc_in != null ) {
					GMalloc_out.addAll(GMalloc_in);
				}
				////////////////////////////////////////////////////
				// Calculate reachingGMalloc_out += (GEN - KILL). //
				////////////////////////////////////////////////////
				// Check whether the node contains "kernelRegion" key, which 
				// is stored in a barrier just before a kernel region.
				Statement stmt = node.getData("kernelRegion");
				if( stmt != null ) {
					Set<Symbol> liveG_out = (Set<Symbol>)node.getData("liveG_out");
					if( liveG_out == null ) {
						Tools.exit("[Error in reachingGMallocAnalysis()] liveG_out is null; " +
								"run liveGVariableAnalysis before this analysis");
					}
					OmpAnnotation annot = stmt.getAnnotation(OmpAnnotation.class, "parallel");
					if( annot != null ) {
						Set<Symbol> sharedVars = (Set<Symbol>)annot.get("shared");
						Set<Symbol> defSyms = DataFlowTools.getDefSymbol(stmt);
						if( sharedVars != null ) {
							for(Symbol sSym: sharedVars) {
								////////////////////////////////////////////////////////////
								// If shrdSclrCachingOnSM option is on, R/O shared scalar //
								// variables are not malloced in this kernel region.      //
								////////////////////////////////////////////////////////////
								if( SymbolTools.isScalar(sSym) && !defSyms.contains(sSym) &&
										shrdSclrCachingOnSM ) {
									continue;
								}
								if( liveG_out.contains(sSym) ) {
									GMalloc_out.add(sSym);
								}
							}
						}
					} else {
						Tools.exit("[Error1 in reachingGMallocAnalysis] Incorrect tag in a node: " + node);
					}
				}
				//////////////////////
				// Handle KILL set. //
				//////////////////////
				// If this node is a function call, and some shared variables in GMalloc_out set are 
				// accessed in the called function, those should be removed from the GMalloc_out set
				// since new Malloc will be called for those in the called function.
				if( GMalloc_out.size() > 0 && currentRegion.equals("CPU") ) {
					Traversable ir = node.getData("ir");
					if( (ir != null) && (ir instanceof ExpressionStatement) ) {
						Expression expr = ((ExpressionStatement)ir).getExpression();
						if( expr instanceof FunctionCall ) {
							Set<Symbol> removeSet = new HashSet<Symbol>();
							for( Symbol sym: GMalloc_out ) {
								if( checkSharedVariableAccess(sym, expr) ) {
									removeSet.add(sym);
								}
							}
							GMalloc_out.removeAll(removeSet);
						}
					}
				}
					
				node.putData("reachingGMalloc_out", GMalloc_out);
	
				for ( DFANode succ : node.getSuccs() ) {
					work_list.put(succ.getData("top-order"), succ);
				}
			}
		}
	}
	
	/**
	 * Forward data-flow analysis to compute cudaMallocSet, a set of GPU variables 
	 * that are newly cudaMalloced before a kernel region, and cudaFreeSet, a set of GPU
	 * variables that are cudaFreed after the kernel region. These two sets are stored
	 * in barrier nodes before/after each kernel region.
	 * To run this analysis, markIntervalForKernelRegins(), annotateBarriers(),
	 * liveGVariableAnalysis(), and reachingGMallocAnalysis() should be called before 
	 * this method.
	 * [CAUTION] This analysis assumes that the procedure of interest (cfg) is
	 * called by CPU, even though it can contain kernel function calls.
	 * 
	 * @param cfg control flow graph of a procedure of interest.
	 */
	public static void cudaMallocFreeAnalsys(CFGraph cfg) {
		//Check whether shrdSclrCachingOnSM is on or not.
		boolean	shrdSclrCachingOnSM = false;
		String value = Driver.getOptionValue("shrdSclrCachingOnSM");
		if( value != null ) {
			shrdSclrCachingOnSM = true;
		}
		TreeMap work_list = new TreeMap();
		// Visited Node Set to prevent infinite loops.
		HashSet<DFANode> visitedNodes = new HashSet<DFANode>();
		
		// Enter the entry node in the work_list
		DFANode entry = cfg.getNodeWith("stmt", "ENTRY");
		work_list.put(entry.getData("top-order"), entry);
		
		// Do iterative steps
		while ( !work_list.isEmpty() )
		{
			DFANode node = (DFANode)work_list.remove(work_list.firstKey());

			for ( DFANode succ : node.getSuccs() ) {
				if( !visitedNodes.contains(succ) ) {
					visitedNodes.add(succ);
					work_list.put(succ.getData("top-order"), succ);
				}
			}

			Statement stmt = node.getData("kernelRegion");
			Statement pStmt = null;
			if( stmt != null ) { // barrier node with type == "S2P"
				pStmt = stmt;
			} else {
				stmt = node.getData("pKernelRegion");
				if( stmt != null ) { // barrier node with type == "P2S"
					pStmt = stmt;
				}
			}
			if( pStmt != null ) {
				OmpAnnotation annot = pStmt.getAnnotation(OmpAnnotation.class, "parallel");
				if( annot != null ) {
					Set<Symbol> cudaMallocSet = new HashSet<Symbol>();
					Set<Symbol> cudaFreeSet = new HashSet<Symbol>();
					Set<Symbol> GMalloc_in = (Set<Symbol>)node.getData("reachingGMalloc_in");
					Set<Symbol> liveG_out = (Set<Symbol>)node.getData("liveG_out");
					if( GMalloc_in == null ) {
						PrintTools.println("==> Parallel region: " + pStmt, 0);
						Tools.exit("[Error in cudaMallocFreeAnalysis()] reachingGMalloc_in is null; " +
								"run reachingGMallocAnalysis before this analysis.");
					}
					if( liveG_out == null ) {
						PrintTools.println("==> Parallel region: " + pStmt, 0);
						Tools.exit("[Error in cudaMallocFreeAnalysis()] liveG_out is null; " +
								"run liveGVariableAnalysis before this analysis");
					}
					Set<Symbol> sharedVars = (Set<Symbol>)annot.get("shared");
					if( sharedVars != null ) {
						HashSet<String> noCudaFreeSet = new HashSet<String>();
						HashSet<String> noCudaMallocSet = new HashSet<String>();
						List<CudaAnnotation> cudaAnnots = pStmt.getAnnotations(CudaAnnotation.class);
						if( cudaAnnots != null ) {
							for( CudaAnnotation cannot : cudaAnnots ) {
								HashSet<String> dataSet = (HashSet<String>)cannot.get("nocudamalloc");
								if( dataSet != null ) {
									noCudaMallocSet.addAll(dataSet);
								}
								dataSet = (HashSet<String>)cannot.get("nocudafree");
								if( dataSet != null ) {
									noCudaFreeSet.addAll(dataSet);
								}
							}
						}
						Set<Symbol> defSyms = DataFlowTools.getDefSymbol(pStmt);
						for( Symbol sVar: sharedVars ) {
							////////////////////////////////////////////////////////////
							// If shrdSclrCachingOnSM option is on, R/O shared scalar //
							// variables are not malloced in this kernel region.      //
							////////////////////////////////////////////////////////////
							if( SymbolTools.isScalar(sVar) && !defSyms.contains(sVar) &&
									shrdSclrCachingOnSM ) {
								continue;
							}
							if( !GMalloc_in.contains(sVar) ) {
								if( !noCudaMallocSet.contains(sVar.getSymbolName()) ) {
									cudaMallocSet.add(sVar);
								}
							}
							if( !liveG_out.contains(sVar) ) {
								if( !noCudaFreeSet.contains(sVar.getSymbolName()) ) {
									cudaFreeSet.add(sVar);
								}
							}
						}
					}
					////////////////////////////////////////////////////////////////////////////////
					// Barrier nodes (type == "S2P" and type == "P2S") related to the same kernel //
					// region have different cudaMallocSets; the cudaMallocSet of the node with   //
					// type == "S2P" contains variables that are newly cudaMalloced in the kernel //
					// region, but the node with type == "P2S" has empty cudaMallocSet.           //
					////////////////////////////////////////////////////////////////////////////////
					node.putData("cudaMallocSet", cudaMallocSet);
					//////////////////////////////////////////////////////////////////////////////
					// Both barrier nodes (type == "S2P" and type == "P2S") related to the same //
					// kernel region have identical cudaFreeSets.                               //
					//////////////////////////////////////////////////////////////////////////////
					node.putData("cudaFreeSet", cudaFreeSet);
				} else {
					Tools.exit("[Error1 in cudaMallocFreeAnalysis()] Incorrect tag (kernelRegion) in a node: " + node);
				}
			}

		}
	}
	
	/**
	 * Intra-procedural, forward data-flow analysis to compute residentGVariables, 
	 * a set of GPU variables residing in the GPU global memory.
	 * 
	 * Input  : CFGraph cfg of a procedure called by CPU.
	 * Output : residentGVariable set for each node in cfg
	 *
	 * residentGVars_in(entry-node) = {}	: only intra-procedural analysis
	 *
	 * for ( node m : predecessor nodes of node n )
	 * 	residentGVars_in(n)  ^= residentGVars_out(m) // ^ : intersection
	 * 
	 * residentGVars_out(n) = residentGVars_in(n) + GEN(n) - KILL(n) // + : union
	 *  where,
	 *   GEN(n) = set of shared variables
	 *                - if n is a barrier node after a kernel region
	 *            ()  - otherwise 
	 *   KILL(n) = set of cudaFreed shared variables 
	 *                 - if n is a barrier node after a kernel region 
	 *             set of reduction variables used in the kernel region
	 *                 - if n is a barrier node after a kernel region    
	 *             set of shared variables accessed in a function call    
	 *                 - if n represents a function call
	 *             set of shared variables modified in a CPU region
	 *                 - if n represents a node in a CPU region
	 *             set of R/O shared scalar variables in a kernel region
	 *             		- if the variables do not exist in residentGVars_in set
	 *                    and if shrdSclrCachingOnSM option is on
	 *                    and if n is a barrier node after a kernel region.                 
	 *             ()  - otherwise
	 * 
	 * For each kernel region, residentGVars_in set is stored in a barrier just before 
	 * the kernel region.
	 * To run this analysis, markIntervalForKernelRegins(), annotateBarriers() 
	 * liveGVariableAnalysis(), reachingGMalloc(), and cudaMallocFreeAnalysis() should be 
	 * called before this method.
	 * [CAUTION] This analysis assumes that the procedure of interest (cfg) is
	 * called by CPU, even though it can contain kernel function calls.
	 * 
	 * @param cfg control flow graph of a procedure of interest.
	 */
	public static void residentGVariableAnalysis(CFGraph cfg) {
		//Check whether shrdSclrCachingOnSM is on or not.
		boolean	shrdSclrCachingOnSM = false;
		String value = Driver.getOptionValue("shrdSclrCachingOnSM");
		if( value != null ) {
			shrdSclrCachingOnSM = true;
		}
		
		TreeMap work_list = new TreeMap();
	
		// Enter the entry node in the work_list
		DFANode entry = cfg.getNodeWith("stmt", "ENTRY");
		entry.putData("residentGVars_in", new HashSet<Symbol>());
		work_list.put(entry.getData("top-order"), entry);
		
		////////////////////////////////////////////////////////////////////////////
		// [CAUTION] This analysis assumes that the procedure of interest is      //
		// called by CPU, even though it can contain kernel function calls.       //
		////////////////////////////////////////////////////////////////////////////
		String currentRegion = new String("CPU");
	
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
	
			HashSet<Symbol> residentGVars_in = null;
			
	
			for ( DFANode pred : node.getPreds() )
			{
				Set<Symbol> pred_residentGVars_out = (Set<Symbol>)pred.getData("residentGVars_out");
				if ( residentGVars_in == null ) {
					if ( pred_residentGVars_out != null ) {
						residentGVars_in = new HashSet<Symbol>();
						residentGVars_in.addAll(pred_residentGVars_out);
					}
				} else {
					// Calculate intersection of previous nodes.
					if ( pred_residentGVars_out != null ) {
						residentGVars_in.retainAll(pred_residentGVars_out);
					} /* else {
						//This is the first visit to this node; ignore it
						//residentGVars_in.clear();
					} */
				}
			}
	
			// previous residentGVars_in
			Set<Symbol> p_residentGVars_in = (Set<Symbol>)node.getData("residentGVars_in");
	
			if ( (residentGVars_in == null) || (p_residentGVars_in == null) || !residentGVars_in.equals(p_residentGVars_in) ) {
				node.putData("residentGVars_in", residentGVars_in);
	
				// compute residentGVars_out, a set of GPU variables residing  
				// in the GPU global memory.
				Set<Symbol> residentGVars_out = new HashSet<Symbol>();
				if( residentGVars_in != null ) {
					residentGVars_out.addAll(residentGVars_in);
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
							Set<Symbol> defSyms = DataFlowTools.getDefSymbol(stmt);
							for(Symbol sSym: sharedVars) {
								////////////////////////////////////////////////////////////
								// If shrdSclrCachingOnSM option is on, R/O shared scalar //
								// variables are not malloced in this kernel region.      //
								////////////////////////////////////////////////////////////
								if( SymbolTools.isScalar(sSym) && !defSyms.contains(sSym) &&
										shrdSclrCachingOnSM ) {
									continue;
								} else {
									residentGVars_out.add(sSym);
								}
							}
							Set<Symbol> cudaFreeSet = node.getData("cudaFreeSet");
							if ( cudaFreeSet == null ) {
								Tools.exit("[ERRROR in residentGVariableAnalysis()] cudaFreeSet does not exist; " +
										"run cudaMallocFreeAnalysis() before this analysis.");
							} else {
								for(Symbol sSym: sharedVars) {
									if( cudaFreeSet.contains(sSym) ) {
										residentGVars_out.remove(sSym);
									}
								}
							}
							Set<Symbol> redSyms = findReductionSymbols(stmt);
							if( redSyms.size() > 0 ) {
								for(Symbol rSym : redSyms ) {
									residentGVars_out.remove(rSym);
								}
							}
						}
					} else {
						Tools.exit("[ERROR in residentGVariableAnalysis] Incorrect tag in a node: " + node);
					}
				}
				//////////////////////
				// Handle KILL set. //
				//////////////////////
				if( residentGVars_out.size() > 0 && currentRegion.equals("CPU") ) {
					// If this node is a function call, and some shared variables in residentGVars_out set are 
					// accessed in the called function, those should be removed from the residentGVars_out set
					// since some GPU variables may be modified by CPU in the called function.
					Traversable ir = node.getData("ir");
					if( (ir != null) && (ir instanceof ExpressionStatement) ) {
						Expression expr = ((ExpressionStatement)ir).getExpression();
						if( expr instanceof FunctionCall ) {
							Set<Symbol> removeSet = new HashSet<Symbol>();
							for( Symbol sym: residentGVars_out ) {
								if( checkSharedVariableAccess(sym, expr) ) {
									removeSet.add(sym);
								}
							}
							residentGVars_out.removeAll(removeSet);
						}
					}
					///////////////////////////////////////////////////////////////
					// If shared variables are modified by CPU, remove them from //
					// residentGVars_out set.                                    //
					///////////////////////////////////////////////////////////////
					if( ir != null ) {
						Set<Symbol> defSet = DataFlowTools.getDefSymbol(ir);
						Set<Symbol> removeSet = new HashSet<Symbol>();
						for( Symbol sym: residentGVars_out ) {
							if( defSet.contains(sym) ) {
								removeSet.add(sym);
							}
						}
						residentGVars_out.removeAll(removeSet);
					}
				}
					
				node.putData("residentGVars_out", residentGVars_out);
	
				for ( DFANode succ : node.getSuccs() ) {
					work_list.put(succ.getData("top-order"), succ);
				}
			}
		}
	}

	/**
	 * For each symbol in the old_set, 
	 *   If it is accessed in the region t,
	 *     - find a symbol with the same name in the SymbolTable, 
	 *       and put the new symbol into the new_set.
	 *     - If no symbol is found in the table, put the old symbol into the new_set
	 *     
	 * @param t region, from which symbol search starts.
	 * @param old_set Old Symbol data set from OmpAnnotation.
	 * @param new_set New Symbol data set to be replaced for the old_set.
	 * @param isShared True if this update is for the shared data set.
	 */
	static public void updateSymbols(Traversable t, HashSet<Symbol> old_set, HashSet<Symbol> new_set,
			boolean isShared)
	{
		VariableDeclaration sm_decl = null;
		VariableDeclarator v_declarator = null;
		Traversable tt = t;
		while( !(tt instanceof SymbolTable) ) {
			tt = tt.getParent();
		}
		Set<Symbol> accessedSymbols = null;
		if ( isShared ) {
			accessedSymbols = getIpAccessedSymbols(t);
		} else {
			accessedSymbols = SymbolTools.getAccessedSymbols(t);
		}
		for( Symbol sm : old_set) {
			// Remove symbols that are not accessed in the region t.
			// Because symbols in the region may not have been updated, 
			// use string comparison.
			boolean accessed = false;
			for( Symbol accSym : accessedSymbols ) {
				if( sm.getSymbolName().compareTo(accSym.getSymbolName()) == 0 ) {
					accessed = true;
					break;
				}
			}
			if( accessed ) {
				sm_decl = (VariableDeclaration)SymbolTools.findSymbol((SymbolTable)tt, 
						((VariableDeclarator)sm).getID());
				if( sm_decl == null ) {
					new_set.add(sm);
				} else {
					boolean found_sm = false;
					for( int i=0; i<sm_decl.getNumDeclarators(); i++ ) {
						v_declarator = ((VariableDeclarator)sm_decl.getDeclarator(i));
						if( v_declarator.getSymbolName().compareTo(sm.getSymbolName()) == 0 ) {
							new_set.add(v_declarator);
							found_sm = true;
							break;
						}
					}
					if( !found_sm ) {
						new_set.add(sm);
					}
				}
			}
		}
	}
	
	/**
	 * Converts a collection of symbols to a string of symbol names with the given separator.
	 *
	 * @param symbols the collection of Symbols to be converted.
	 * @param separator the separating string.
	 * @return the converted string, which is a list of symbol names
	 */
	public static String symbolsToString(Collection<Symbol> symbols, String separator)
	{
		if ( symbols == null || symbols.size() == 0 )
			return "";

		StringBuilder str = new StringBuilder(80);

		Iterator<Symbol> iter = symbols.iterator();
		if ( iter.hasNext() )
		{
			str.append(iter.next().getSymbolName());
			while ( iter.hasNext() ) {
				str.append(separator+iter.next().getSymbolName());
			}
		}

		return str.toString();
	}
	
	/**
	 * Converts a collection of symbols to a set of strings of symbol names.
	 *
	 * @param symbols the collection of Symbols to be converted.
	 * @return a set of strings, which contains symbol names
	 */
	public static Set<String> symbolsToStringSet(Collection<Symbol> symbols)
	{
		HashSet<String> strSet = new HashSet<String>();
		if ( symbols == null || symbols.size() == 0 )
			return strSet;


		Iterator<Symbol> iter = symbols.iterator();
		if ( iter.hasNext() )
		{
			strSet.add(iter.next().getSymbolName());
			while ( iter.hasNext() ) {
				strSet.add(iter.next().getSymbolName());
			}
		}

		return strSet;
	}
	
	/**
	 * Find reduction symbols used in the input region.
	 * FIXME: currently, O2GTranslator.reductionTransformation() can not handle the cases 
	 * where reduction is used in a function called in a parallel region. Therefore, this
	 * method does not check reduction variables used in a function called in the input
	 * region.
	 * 
	 * @param region input region
	 * @return HashSet of reduction symbols used in the region
	 */
	public static Set<Symbol> findReductionSymbols(Traversable region) {
		HashMap redMap = null;
		HashSet<Symbol> redSet = new HashSet<Symbol>();
		List<OmpAnnotation> omp_annots = IRTools.collectPragmas(region, OmpAnnotation.class, "reduction");
		for (OmpAnnotation annot : omp_annots)
		{
			redMap = (HashMap)annot.get("reduction");
			for (String ikey : (Set<String>)(redMap.keySet())) {
				redSet.addAll( (HashSet<Symbol>)redMap.get(ikey) );
			}
		}
		return redSet;
	}
	
	/**
	 * Find static symbols contained in the input symbol table.
	 * 
	 * @param st input symbol table
	 * @return set of static symbols
	 */
	public static Set<Symbol> findStaticSymbols(SymbolTable st) {
		HashSet<Symbol> staticSet = new HashSet<Symbol>();
		Set<Symbol> symbols = SymbolTools.getSymbols(st);
		for( Symbol sym : symbols ) {
			List types = sym.getTypeSpecifiers();
			if( types.contains(Specifier.STATIC) ) {
				staticSet.add(sym);
			}
		}
		return staticSet;
	}

	/**
	 * Check whether static data exist in the input region; if function calls
	 * exist, check the called functions too.
	 * 
	 * @param region input region
	 * @return true if static data exist
	 */
	public static boolean ipaStaticDataCheck( CompoundStatement region ) {
		Boolean foundStaticData = false;
		List<FunctionCall> funcCalls = IRTools.getFunctionCalls(region); 
		for( FunctionCall calledProc : funcCalls ) {
			Procedure tProc = calledProc.getProcedure();
			if( tProc != null ) {
				foundStaticData = ipaStaticDataCheck(tProc.getBody());
				if( foundStaticData ) {
					break;
				}
			}
		}
	
		Set<Symbol> localSet = SymbolTools.getLocalSymbols((SymbolTable)region);
		for (Symbol sym : localSet)
		{
			if( foundStaticData ) {
				break;
			}
			List<Specifier> type_specs = sym.getTypeSpecifiers();
			for (Specifier spec : type_specs)
			{
				if ( spec.toString().compareTo("static")==0 )
				{
					foundStaticData = true;
					break;
				}
			}
		}
		return foundStaticData;
	}
	
	/**
	 * For each kernel region, which will be transformed into a CUDA kernel,
	 * 1) add information of the enclosing procedure name and kernel ID.
	 *   The annotation has the following form:
	 * 	 #pragma cuda ainfo procname(procedure-name) kernelid(kernel-id)
	 * 2) apply user directives if existing.
	 * 3) if maxNumOfCudaThreadBlocks option exists, add maxnumofblocks(N)
	 * clause to each kernel region.
	 * 
	 * @param program input program
	 */
	public static void annotateUserDirectives(Program program, 
			HashMap<String, HashMap<String, Object>> userDirectives) {
		boolean userDirectiveExists = false;
		String value = Driver.getOptionValue("maxNumOfCudaThreadBlocks");
		/* iterate to search for all Procedures */
		DepthFirstIterator proc_iter = new DepthFirstIterator(program);
		Set<Procedure> proc_list = (Set<Procedure>)(proc_iter.getSet(Procedure.class));
		for (Procedure proc : proc_list)
		{
			String procName = proc.getSymbolName();
			String kernelName = "";
			int kernelID = 0;
			/* Search for all OpenMP parallel regions in a given Procedure */
			List<OmpAnnotation>
			omp_annots = IRTools.collectPragmas(proc, OmpAnnotation.class, "parallel");
			for ( OmpAnnotation annot : omp_annots )
			{
				kernelName = "";
				Statement target_stmt = (Statement)annot.getAnnotatable();
				int eligibility = AnalysisTools.checkKernelEligibility(target_stmt);
				if (eligibility == 3) {
					// Check whether this parallel region is an omp-for loop.
					if( annot.containsKey("for") ) {
						// In the new annotation scheme, the above check is redundant.
						eligibility = 0;
					} else {
						// Check whether called functions have any omp-for loop.
						List<FunctionCall> funcCalls = IRTools.getFunctionCalls(target_stmt); 
						for( FunctionCall calledProc : funcCalls ) {
							Procedure tProc = calledProc.getProcedure();
							if( tProc != null ) {
								eligibility = AnalysisTools.checkKernelEligibility(tProc.getBody());
								if(  eligibility == 0 ) {
									break;
								}
							}
						}
					}
				} 
				if( eligibility == 0 ) {
					CudaAnnotation cAnnot = target_stmt.getAnnotation(CudaAnnotation.class, "ainfo");
					if( cAnnot == null ) {
						cAnnot = new CudaAnnotation("ainfo", "true");
						target_stmt.annotate(cAnnot);
					}
					cAnnot.put("procname", procName);
					cAnnot.put("kernelid", Integer.toString(kernelID));
					kernelName = kernelName.concat(procName).concat(Integer.toString(kernelID++));
					if( !userDirectives.isEmpty() ) {
						userDirectiveExists = true;
						Set<String> kernelSet = userDirectives.keySet();
						if( kernelSet.contains(kernelName) ) {
							HashMap<String, Object> directives = userDirectives.remove(kernelName);
							for( String clause : directives.keySet() ) {
								if( clause.equals("nogpurun") ) {
									cAnnot =  target_stmt.getAnnotation(CudaAnnotation.class, "nogpurun");
									if( cAnnot == null ) {
										cAnnot = new CudaAnnotation("nogpurun", "true");
										target_stmt.annotate(cAnnot);
									}
									break; //Ignore all remaining clauese for this kernel region.
								}
								Object uObj = directives.get(clause);
								cAnnot =  target_stmt.getAnnotation(CudaAnnotation.class, clause);
								if( cAnnot == null ) {
									cAnnot = new CudaAnnotation("gpurun", "true");
									cAnnot.put(clause, uObj);
									target_stmt.annotate(cAnnot);
								} else {
									Object fObj = cAnnot.get(clause);
									if( fObj instanceof Set ) {
										((Set<String>)fObj).addAll((Set<String>)uObj);
									} else {
										cAnnot.put(clause, uObj);
									}
								}
							}
						}
					}
					if( value != null ) {
						cAnnot = target_stmt.getAnnotation(CudaAnnotation.class, "maxnumofblocks");
						if( cAnnot == null ) {
							cAnnot = new CudaAnnotation("gpurun", "true");
							cAnnot.put("maxnumofblocks", value);
							target_stmt.annotate(cAnnot);
						}
					}
				}
			}
		}
		if( userDirectiveExists ) {
			if( !userDirectives.isEmpty() ) {
				Set<String> kernelSet = userDirectives.keySet();
				PrintTools.println("[WARNING in annotateUserDirectives()] user directives for the following" +
						" set of kernels can not be applicable: " + PrintTools.collectionToString(kernelSet, ","), 0);
			}
		}
	}
	
	public static boolean isCudaCall(FunctionCall fCall) {
		if ( fCall == null )
			return false;

		Set<String> cudaCalls = new HashSet<String>(Arrays.asList(
				"CUDA_SAFE_CALL","cudaFree","cudaMalloc","cudaMemcpy",
				"cudaMallocPitch","tex1Dfetch","cudaBindTexture", "cudaMemcpy2D"
		));

		if ( cudaCalls.contains((fCall.getName()).toString()) ) {
			return true;
		}
		return false;
	}
	
	public static boolean isKernelFunction(Procedure proc) {
		if( proc == null ) {
			return false;
		}
		List return_type = proc.getReturnType();
		if( return_type.contains(CUDASpecifier.CUDA_DEVICE) 
				|| return_type.contains(CUDASpecifier.CUDA_GLOBAL) ) {
			return true;
		} else {
			return false;
		}
	}
	
	/**
	 * Check whether a CUDA kernel function calls C standard library functions 
	 * that are not supported by CUDA runtime systems.
	 * If so, CUDA compiler will fail if they are not inlinable.
	 * 
	 * @param prog
	 */
	public static void checkKernelFunctions(Program prog) {
		List<Procedure> procList = IRTools.getProcedureList(prog);
		List<Procedure> kernelProcs = new LinkedList<Procedure>();
		for( Procedure proc : procList ) {
			List return_type = proc.getReturnType();
			if( return_type.contains(CUDASpecifier.CUDA_DEVICE) 
					|| return_type.contains(CUDASpecifier.CUDA_GLOBAL) ) {
				kernelProcs.add(proc);
			}
		}
		for( Procedure kProc : kernelProcs ) {
			List<FunctionCall> fCalls = IRTools.getFunctionCalls(kProc);
			for( FunctionCall fCall : fCalls ) {
				if( StandardLibrary.contains(fCall) ) {
					if( !CudaStdLibrary.contains(fCall) ) {
						PrintTools.println("[WARNING] C standard library function ("+fCall.getName()+
							") is called in a kernel function,"+kProc.getName()+
							", but not supported by CUDA runtime system V1.1; " +
							"it may cause compilation error if not inlinable.", 0);
					}
				}
			}
		}
	}

	/**
	 * Search input expression and return an ArrayAccess expression if existing;
	 * if there are multiple ArrayAccess expressions, the first one will be returned.
	 * If there is no ArrayAccess, return null.
	 * 
	 * @param iExp input expression to be searched.
	 * @return ArrayAccess expression 
	 */
	public static ArrayAccess getArrayAccess(Expression iExp) {
		ArrayAccess aAccess = null;
		DepthFirstIterator iter = new DepthFirstIterator(iExp);
		while(iter.hasNext()) {
			Object o = iter.next();
			if(o instanceof ArrayAccess)
			{
				aAccess = (ArrayAccess)o;
				break;
			}
		}
		return aAccess;
	}
	
	/**
	 * Search input expression and return a list of ArrayAccess expressions if existing;
	 * If there is no ArrayAccess, return empty list.
	 * 
	 * @param iExp input expression to be searched.
	 * @return list of ArrayAccess expressions 
	 */
	public static List<ArrayAccess> getArrayAccesses(Expression iExp) {
		List<ArrayAccess> aList = new LinkedList<ArrayAccess>();
		DepthFirstIterator iter = new DepthFirstIterator(iExp);
		while(iter.hasNext()) {
			Object o = iter.next();
			if(o instanceof ArrayAccess)
			{
				aList.add((ArrayAccess)o);
			}
		}
		return aList;
	}

	/**
	 * Run interprocedural analysis to find whether the region contains a barrier or not.
	 * @param region : a region to be searched.
	 * @return : returns true if the region contains a barrier.
	 */
	public static boolean ipaContainsBarrierInRegion( Traversable region ) {
		boolean foundBarrier = false;
		if( region instanceof AnnotationStatement) {
			AnnotationStatement annot_stmt = (AnnotationStatement)region;
			OmpAnnotation annot = annot_stmt.getAnnotation(OmpAnnotation.class, "barrier");
			if (annot != null)
			{
				return true;
			} else {
				return false;
			}
		}
		DepthFirstIterator iter = new DepthFirstIterator(region);
		while(iter.hasNext())
		{
			Object obj = iter.next();
			if (obj instanceof AnnotationStatement)
			{
				AnnotationStatement annot_stmt = (AnnotationStatement)obj;
				OmpAnnotation annot = annot_stmt.getAnnotation(OmpAnnotation.class, "barrier");
				if (annot != null)
				{
					foundBarrier = true;
					break;
				}
			}
			else if (obj instanceof FunctionCall)
			{
				FunctionCall call = (FunctionCall)obj;
				// called_procedure is null for system calls 
				Procedure called_procedure = call.getProcedure();
				if (called_procedure != null)
				{	
					foundBarrier = ipaContainsBarrierInRegion(called_procedure.getBody()); // recursive call 
					if( foundBarrier ) {
						break;
					}
				}
			}
		}
		return foundBarrier;
	}

	/**
	 * Return a set of static symbols from the input symbol set, iset
	 * 
	 * @param iset input symbol set
	 * @return a set of static symbols
	 */
	static public HashSet<Symbol> extractStaticVariables(Set<Symbol> iset)
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

	/**
	 * Searches the symbol set and returns the first symbol whose name is the specified string
	 * @param sset		Symbol set being searched
	 * @param symName	symbol name being searched for
	 * @return the first symbol amaong the symbol set whose name is the same as the specified string
	 */
	public static Symbol findsSymbol(Set<Symbol> sset, String symName)
	{
		if ( sset == null )
			return null;
	
		for( Symbol sym : sset ) {
			if( sym.getSymbolName().equals(symName) ) {
				return sym;
			}
		}
		
		return null;
	}

	/**
	 * Returns true if the symbol set contains a symbol whose name is the specified string.
	 * @param sset		Symbol set being searched
	 * @param symName	symbol name being searched for
	 */
	public static boolean containsSymbol(Set<Symbol> sset, String symName)
	{
		if ( sset == null )
			return false;
	
		for( Symbol sym : sset ) {
			if( sym.getSymbolName().equals(symName) ) {
				return true;
			}
		}
		
		return false;
	}

}
