package omp2gpu.analysis;

import java.io.*;
import java.lang.reflect.Method;
import java.util.*;

import cetus.hir.*;
import cetus.exec.*;
import cetus.analysis.*;

/**
 * @author Seyong Lee <lee222@purdue.edu>
 *         ParaMount Group 
 *         School of ECE, Purdue University
 *
 * This pass analyzes openmp pragmas and converts them into Cetus Annotations, 
 * the same form as what parallelization passes generate. The original OpenMP 
 * pragmas are removed after the analysis.
 */
public class OmpAnalysis extends AnalysisPass
{
	private int debug_level;
	private int debug_tab;
	private HashSet<String> visitedProcs;

	static
	{
		Class[] params = new Class[2];
		params[0] = OmpAnnotation.class;
		params[1] = OutputStream.class;
	}

	public OmpAnalysis(Program program)
	{
		super(program);
		debug_level = Integer.valueOf(Driver.getOptionValue("verbosity")).intValue();
		debug_tab = 0;
	}

	public String getPassName()
	{
		return new String("[OmpAnalysis]");
	}

	public void start()
	{
		//////////////////////////////////////////////////////////////////////////////////////////////
		// shared_analysis() assumes that each annotatable contains only one OpenMP annotation.     //
		// To satisfy this condition, multiple OpenMP annotations belonging to the same annotatable //
		// are merged into one OmpAnnotation.                                                       //
		//////////////////////////////////////////////////////////////////////////////////////////////
		DepthFirstIterator iter = new DepthFirstIterator(program);
		while(iter.hasNext())
		{
			Object obj = iter.next();
			if (obj instanceof Annotatable)
			{
				Annotatable at = (Annotatable)obj;
				List<OmpAnnotation> ompAnnotList = at.getAnnotations(OmpAnnotation.class);
				if ( (ompAnnotList != null) && (ompAnnotList.size() > 1) ) 
				{
					OmpAnnotation ompAnnot = new OmpAnnotation();
					for( OmpAnnotation annot : ompAnnotList ) {
						for( String key: annot.keySet() ) {
							if( ompAnnot.containsKey(key) ) {
								HashSet hset = ompAnnot.get(key);
								hset.addAll((HashSet)annot.get(key));
							} else {
								ompAnnot.put(key, annot.get(key));
							}
						}
					}
					at.removeAnnotations(OmpAnnotation.class);
					at.annotate(ompAnnot);
				}
			}
		}
		
		/*
		 * Omp threadprivate directive is not used as a clause for omp parallel or 
		 * work-sharing constructs. To assist other analysis, however, we insert
		 * threadprivate data list into omp parallel and for-loop constructs. 
		 * CAUTION: For correct OpenMP program, these threadprivate clauses added in 
		 * the omp parallel and for-loop constructs should be removed later.
		 */
		/////////////////////////////////////////////
		// Collect OpenMP threadprivate variables. //
		/////////////////////////////////////////////
		List<OmpAnnotation> ThreadPrivateAnnots = 
			IRTools.collectPragmas(program, OmpAnnotation.class, "threadprivate");
		HashSet<String> threadprivSet = new HashSet<String>();
		for( OmpAnnotation tp_annot : ThreadPrivateAnnots ) {
			threadprivSet.addAll((HashSet<String>)tp_annot.get("threadprivate"));
		}
		///////////////////////////////////////////////////////////////////////
		// Insert threadprivate variables into each omp parallel or for-loop //
		// construct.                                                        //
		///////////////////////////////////////////////////////////////////////
		HashSet<OmpAnnotation> OmpPAnnotSet = new HashSet<OmpAnnotation>();
		OmpPAnnotSet.addAll( IRTools.collectPragmas(program, OmpAnnotation.class, "parallel") );
		OmpPAnnotSet.addAll( IRTools.collectPragmas(program, OmpAnnotation.class, "for") );
		for( OmpAnnotation annot : OmpPAnnotSet ) {
			if( !annot.containsKey("threadprivate") ) {
				HashSet<String> tpSet = new HashSet<String>();
				tpSet.addAll(threadprivSet);
				annot.put("threadprivate", tpSet);
			}
		}


		/*
		 *	OpenMP can express shared variables within a parallel region using shared
		 * clause, but it is often the case that not all shared variables are listed 
		 * in shared clause when default is shared
		 * The shared_analysis method finds all shared variables used within
		 * a parallel region inter-procedurally and provide that information in the
		 * form of Annotation attached to the parallel region (structured block).
		 */
		shared_analysis();
		
		/*
		 * Convert a set of Strings in each OpenMP reduction clause into a set of Symbols.
		 * - This method should be called right after OmpAnalysis.start() if your pass wants
		 * to handle OpenMP reduction clauses.
		 * CAUTION: this method may modify shared set information if reduction clauses 
		 * include shared variables; if a shared variable is used as a reduction variable, 
		 * it should be removed from shared set.
		 */
		convertReductionClauses();
		
		/*
		 *	Debugging
		 */
		display();
	}

	/**
		* shared_analysis 
		* Limitations: the following two cases are not tested 
		* - Heap-allocated variables are shared
		* - Formal arguments of called routines in the region that are passed by 
		*   reference inherit the data-sharing attributes of the associated actual argument.
		*/
	public void shared_analysis()
	{
		PrintTools.println("shared_anlaysis strt", 5);
		
		visitedProcs = new HashSet<String>();

		List<OmpAnnotation> OmpParallelAnnots = IRTools.collectPragmas(program, OmpAnnotation.class, "parallel");

		/////////////////////////////////////////////////////////////////////////////////
		// CAUTION: this analysis assumes that each OpenMP parallel or worksharing     //
		// construct has only one OpenMP pragma, which consists of a single statement; //
		/////////////////////////////////////////////////////////////////////////////////

		// shared variable analysis for every parallel region
		for (OmpAnnotation annot : OmpParallelAnnots)
		{
			// ---------------------------------------------------------------------------
			// PART I : gather phase
			// finds all shared variables within a parallel region inter-procedurally
			// include downward inter-procedural analysis
			// ---------------------------------------------------------------------------
			HashSet<Symbol> SharedSet = findSharedInRegion(annot);

			if( annot.keySet().contains("shared") ) {
				annot.remove("shared");
			}
			annot.put("shared", SharedSet);

			// ---------------------------------------------------------------------------
			// PART II : distribution phase
			// the following while loop searches all parallel for-loops annotated 
			// with "omp for" and find all shared variable for each for-loop
			// ---------------------------------------------------------------------------
			runIpaSharedInForLoop(annot, (Statement)annot.getAnnotatable());
			
			// Handle copyin clause
			// FIXME: If threadprivate variable is accessed only in a function
			//        called in the parallel region, below conversion will omit it.
			if( annot.keySet().contains("copyin") ) {
				HashSet<String>tmp_set = (HashSet<String>)annot.remove("copyin");
				HashSet<Symbol> OmpCopyinSet = 
					convertString2Symbol(tmp_set, (Statement)annot.getAnnotatable());
				annot.put("copyin", OmpCopyinSet);
			}
		}

		PrintTools.println("shared_anlaysis done", 5);
	}

	/**
		* Finds all parallel for-loops (we also need to include critical sections, too) and
		* find shared variables within a for-loop and attach shared variables to the for-loop.
		*
		* @param omp_annot is an OpenMP annotation for the parallel region.
		* @param stmt is a target statement to be searched.
		*/
	private void runIpaSharedInForLoop(OmpAnnotation omp_annot, Statement stmt)
	{
		PrintTools.println("[runIpaSharedInForLoop] strt", 5);
		DepthFirstIterator iter = new DepthFirstIterator(stmt);
		while(iter.hasNext())
		{
			Object obj = iter.next();
			if (obj instanceof Annotatable)
			{
				Annotatable at = (Annotatable)obj;
				//////////////////////////////////////////////////////////////////////////
				// CAUTION: below assumes that Omp-for loop has only one OmpAnnotation. //
				//////////////////////////////////////////////////////////////////////////
				OmpAnnotation for_annot = at.getAnnotation(OmpAnnotation.class, "for");
				if ( for_annot != null ) 
				{
					if (at instanceof ForLoop)
					{
						ForLoop loop = (ForLoop)at;
						findSharedInForLoop(omp_annot, for_annot, loop);
					}
					else
					{
						Tools.exit("[shared_analysis] Wrong statement contains omp-for annotation");
					}
				}
			}
			else if (obj instanceof FunctionCall)
			{
				FunctionCall call = (FunctionCall)obj;
				// called_procedure is null for system calls 
				Procedure called_procedure = call.getProcedure();
	/**
		* Seungjai: FIXME! 
		* If a shared variable in the par_map is passed to a function through parameter list,
		* then we need to find a way to replace this shared variable in the par_map with the
		* actual parameter in the called function.
		*/
				if (called_procedure != null)
				{	
					/////////////////////////////////////////////////////////////////////////////////
					// FIXME: Below use assumption that even if a function is called several times //
					//in different sites, the function is always called in the same context.       //
					// ==> For context-sensitive analysis, below should be modified.               //
					/////////////////////////////////////////////////////////////////////////////////
					if( !visitedProcs.contains(called_procedure.getSymbolName()) ) {
						PrintTools.println("[runIpaSharedInForLoop] going down to proc: " + called_procedure.getName(), 5);
						runIpaSharedInForLoop(omp_annot, called_procedure.getBody()); // recursive call 
						visitedProcs.add(called_procedure.getSymbolName());
					}
				}
			}
		}
		PrintTools.println("[runIpaSharedInForLoop] done", 5);
	}

	/**
		* attach AnnotationStatement for shared variable to the loop.
		* [CAUTION] this method should be called only once for each function.
		* For context-aware analysis, versioned cloning for multiply-called function
		* should be used.
		*/
	private void findSharedInForLoop(HashMap par_map, HashMap for_map, ForLoop loop)
	{ 
		PrintTools.println("[findSharedInForLoop] start", 5);
		HashSet<String> tmp_set;

		// private variables annotated by "omp private" to the current for-loop
		tmp_set = (HashSet<String>)for_map.get("private");
		HashSet<Symbol> OmpPrivSet = convertString2Symbol(tmp_set, loop);
		displaySet("OmpPrivSet in a loop", OmpPrivSet);
		
		// firstprivate variables annotated by "omp firstprivate" to the current for-loop
		tmp_set = (HashSet<String>)for_map.get("firstprivate");
		HashSet<Symbol> OmpFirstPrivSet = convertString2Symbol(tmp_set, loop);
		displaySet("OmpFirstPrivSet in a loop", OmpFirstPrivSet);
		
		// threadprivate variables annotated by "omp threadprivate" to the current for-loop
		tmp_set = (HashSet<String>)for_map.get("threadprivate");
		HashSet<Symbol> OmpThreadPrivSet = convertString2Symbol(tmp_set, loop);
		displaySet("OmpThreadPrivSet in a loop", OmpThreadPrivSet);

		/*
		 * "omp for" does not have shared clause, but for other analysis, we 
		 * keep shared set for each "omp for" loop
		 */
		// shared variables annotated by "omp shared" to the current for-loop
		tmp_set = (HashSet<String>)for_map.get("shared");
		HashSet<Symbol> OmpSharedSet = convertString2Symbol(tmp_set, loop);
		displaySet("OmpSharedSet in a loop", OmpSharedSet);

		// shared variables found by shared_analysis to the enclosing parallel region
		Set<Symbol> ParSharedSet = (Set<Symbol>)par_map.get("shared");
		// private variables in the enclosing parallel region
		Set<Symbol> ParPrivateSet = (Set<Symbol>)par_map.get("private");
		
		// Put loop index variable into private set
		Expression ivar_expr = LoopTools.getIndexVariable(loop);
		Symbol ivar = SymbolTools.getSymbolOf(ivar_expr);
		OmpPrivSet.add(ivar);
		
		// all variables accessed within the current for-loop
		//Set<Symbol> ForAccessedSet = SymbolTools.getAccessedSymbols(loop);
		Set<Symbol> ForAccessedSet = getAccessedVariables(loop);
		
		// find private variables for the current for-loop
		for (Symbol symbol : ParPrivateSet)
		{
			if (ForAccessedSet.contains(symbol))
			{
				OmpPrivSet.add(symbol);
			}
		}
		
		// find shared variables for the current for-loop
		// FIXME: If shared variable is passed as a function parameter,
		// and if this for-loop uses the parameter as shared variable,
		// below update can not catch the parameter.
		Set<Symbol> ForSharedSet = new HashSet<Symbol> ();
		for (Symbol symbol : ParSharedSet)
		{
			if (ForAccessedSet.contains(symbol))
			{
				ForSharedSet.add(symbol);
			}
		}

		// Shared = Shared + OmpShared - OmpPriv
		ForSharedSet.addAll(OmpSharedSet);
		ForSharedSet.removeAll(OmpPrivSet);
		ForSharedSet.removeAll(OmpFirstPrivSet);

		if (for_map.keySet().contains("shared"))
		{
			for_map.remove("shared");
		}
		for_map.put("shared", ForSharedSet);
		if (for_map.keySet().contains("private"))
		{
			for_map.remove("private");
		}
		for_map.put("private", OmpPrivSet);
		if (for_map.keySet().contains("firstprivate"))
		{
			for_map.remove("firstprivate");
		}
		for_map.put("firstprivate", OmpFirstPrivSet);
		if (for_map.keySet().contains("threadprivate"))
		{
			for_map.remove("threadprivate");
		}
		for_map.put("threadprivate", OmpThreadPrivSet);

		displaySet("shared variables in a loop", ForSharedSet);
		displaySet("private variables in a loop", OmpPrivSet);
		PrintTools.println("[findSharedInForLoop] done", 5);
	}

	/** 
		* @param omp_annot is an OmpAnnotation containing data attributes 
		* of an OpenMP parallel region.
		*/
	private HashSet<Symbol> findSharedInRegion(OmpAnnotation omp_annot)
	{
		HashSet<String> tmp_set;
		debug_tab++;
		PrintTools.println("[findSharedInRegion] strt: " + 
				omp_annot.getAnnotatable().getClass().getName(), 8);
		
		///////////////////////////////////////////////////////////////////////
		// OpenMP parallel pragma is attached to either CompoundStatement or //
		// ForLoop.                                                          //
		///////////////////////////////////////////////////////////////////////
		Statement annot_container = (Statement)omp_annot.getAnnotatable();

		// shared variables explicitly defined by the OpenMP directive
		tmp_set = (HashSet<String>)omp_annot.get("shared");
		HashSet<Symbol> OmpSharedSet = convertString2Symbol(tmp_set, annot_container);
		displaySet("OmpSharedSet in a region", OmpSharedSet);

		// private variables explicitly defined by the OpenMP directive
		tmp_set = (HashSet<String>)omp_annot.get("private");
		HashSet<Symbol> OmpPrivSet = convertString2Symbol(tmp_set, annot_container);
		displaySet("OmpPrivSet in a region", OmpPrivSet);
		
		// firstprivate variables explicitly defined by the OpenMP directive
		tmp_set = (HashSet<String>)omp_annot.get("firstprivate");
		HashSet<Symbol> OmpFirstPrivSet = convertString2Symbol(tmp_set, annot_container);
		displaySet("OmpFirstPrivSet in a region", OmpFirstPrivSet);

		// In "C", the syntax is default(shared|none)
		// In a parallel or task construct, the data-sharing attributes of variables
		// are determined by the default clause, if present.
		// In a parallel construct, if no default clause if present, variables are shared
		boolean default_shared = true;
		if ( omp_annot.keySet().contains("default") )
		{
			String default_value = (String)(omp_annot.get("default"));
			if ( default_value.equals("none") ) default_shared = false;
		}

		// add all accessed variable symbols in the procedure
		Set<Symbol> AccessedSet = getAccessedVariables(annot_container);
		displaySet("AccessedSet in a region", AccessedSet);

		// -------------------------------------------------------------------
		// find the local variables declared locally within this parallel region
		// (both CompoundStatement and ForLoop have SymbolTable interface)
		// LocalPrivSet = local variables - static local - threadprivate 
		// -------------------------------------------------------------------

		Set<Symbol> LocalPrivSet = new HashSet<Symbol>();

		// if annot_container is CompoundStatement or ForLoop, it has SymbolTable interface
		if ( annot_container instanceof SymbolTable ) {
			LocalPrivSet.addAll( SymbolTools.getLocalSymbols((SymbolTable)annot_container) );	
			Set<Symbol> StaticLocalSet = AnalysisTools.extractStaticVariables(LocalPrivSet);
			displaySet("static local variables in a region", StaticLocalSet);
			LocalPrivSet.removeAll(StaticLocalSet);
		}

		/*
		 * "omp parallel" does not have threadprivate clause, but for other analysis, we 
		 * keep threadprivate set for each "omp parallel" region
		 * FIXME: convertString2Symbol() find threadprivate symbols accessed in the target
		 * statement, annot_container in this case; if the target statement has a function call
		 * whose body contains a threadprivate variable, it can not be included by this 
		 * conversion.
		 */
		tmp_set = (HashSet<String>)omp_annot.get("threadprivate");
		HashSet<Symbol> OmpThreadPrivSet = convertString2Symbol(tmp_set, annot_container);
		displaySet("threadprivate variables in a region", OmpThreadPrivSet);
		LocalPrivSet.removeAll(OmpThreadPrivSet);

		// add loop index variables of parallel for loops to the LocalPrivSet
		HashSet<Symbol> LoopIndexVariables = new HashSet<Symbol>();
		if( annot_container instanceof CompoundStatement ) {
			LoopIndexVariables = getLoopIndexVarSet(annot_container);
		} 
		else if( annot_container instanceof ForLoop ) {
			Expression ivar_expr = LoopTools.getIndexVariable((ForLoop)annot_container);
			Symbol ivar = SymbolTools.getSymbolOf(ivar_expr);
			LoopIndexVariables.add(ivar);
		}
		displaySet("parallel loop index variables in a region", LoopIndexVariables);
		LocalPrivSet.addAll(LoopIndexVariables);
		displaySet("LocalPrivSet = Local - Static - ThreadPrivate + ParallelLoopIndex", LocalPrivSet);
		
		//Omp private set contains local variables in current parallel region, annot_container,
		//but does not include local variables in the called procedures.
		if( omp_annot.keySet().contains("private") ) {
			omp_annot.remove("private");
		}
		OmpPrivSet.addAll(LocalPrivSet);
		omp_annot.put("private", OmpPrivSet);
		
		if( omp_annot.keySet().contains("firstprivate") ) {
			omp_annot.remove("firstprivate");
		}
		omp_annot.put("firstprivate", OmpFirstPrivSet);
		
		if( omp_annot.keySet().contains("threadprivate") ) {
			omp_annot.remove("threadprivate");
		}
		omp_annot.put("threadprivate", OmpThreadPrivSet);
		
		// ------------------------------------------------------------------------------------------
		// Downward inter-procedural analysis
		// ipaSharedSet is a set of shared variables in the functions called within the current scope
		// ------------------------------------------------------------------------------------------
		Set<Symbol> IpaSharedSet = getIpaSharedSet(omp_annot);
		displaySet("IpaSharedSet", IpaSharedSet);

		// if default is shared
		//   SharedSet = AccessedSet + IpaSharedSet + OmpShared - OmpPriv - Local - ThreadPrivate
		// if default is none
		//   SharedSet = IpaSharedSet + OmpShared - OmpPriv - Local - ThreadPrivate
		HashSet<Symbol> SharedSet = new HashSet<Symbol> ();
		if (default_shared)
		{
			SharedSet.addAll(AccessedSet);
		}
		///////////////////////////////////////////////////////////////////////////////////////////////
		// FIXME: If a global-scope shared variable is used in current parallel region via parameter //
		// passing and also used in the functions called in the current region but without parameter //
		// passing, the final SheredSet will contain two symbols: the global shared variable and the //
		// function parameter corresponding to the shared variable.                                  //
		///////////////////////////////////////////////////////////////////////////////////////////////
		SharedSet.addAll(IpaSharedSet);
		SharedSet.addAll(OmpSharedSet);
		SharedSet.removeAll(OmpPrivSet);
		SharedSet.removeAll(OmpFirstPrivSet);
		SharedSet.removeAll(OmpThreadPrivSet);
//  SharedSet.removeAll(LocalPrivSet);

		displaySet("Final SharedSet in a region", SharedSet);
		
		PrintTools.println("[findSharedInRegion] done: " + annot_container.getClass().getName(), 8);
		debug_tab--;
		return SharedSet;
	}

	/**
		* collect loop index variables for parallel for loops
		*/
	private	HashSet<Symbol> getLoopIndexVarSet(Statement stmt)
	{
		HashSet<Symbol> ret = new HashSet<Symbol> ();
		DepthFirstIterator iter = new DepthFirstIterator(stmt);
		while(iter.hasNext())
		{
			Object obj = iter.next();
			if (obj instanceof Annotatable)
			{
				Annotatable at = (Annotatable)obj;
				OmpAnnotation for_annot = at.getAnnotation(OmpAnnotation.class, "for");
				if ( for_annot != null )
				{
					ForLoop loop = (ForLoop)at;
					Expression ivar_expr = LoopTools.getIndexVariable(loop);
					Symbol ivar = SymbolTools.getSymbolOf(ivar_expr);
					if (ivar==null)
						Tools.exit("[getLoopIndexVariables] Cannot find symbol:" + ivar.toString());
					else
						ret.add(ivar);
				}
			}
		}
		return ret;
	}

	/**
		* Interprocedural shared variable analysis driver
		*/
	private HashSet<Symbol> getIpaSharedSet(OmpAnnotation omp_annot)
	{
		///////////////////////////////////////////////////////////////////////
		// OpenMP parallel pragma is attached to either CompoundStatement or //
		// ForLoop.                                                          //
		///////////////////////////////////////////////////////////////////////
		Statement annot_container = (Statement)omp_annot.getAnnotatable();
		HashSet<Symbol> SharedSet = new HashSet<Symbol> ();
		DepthFirstIterator iter = new DepthFirstIterator(annot_container);
		while(iter.hasNext())
		{
			Object obj = iter.next();
			if (obj instanceof FunctionCall)
			{
				FunctionCall call = (FunctionCall)obj;
				// called_procedure is null for system calls 
				Procedure called_procedure = call.getProcedure();
				Set<Symbol> procSharedSet = null;
				if (called_procedure != null)
				{	
					//System.out.println("[getIpaSharedSet] proc="+call.toString());
					// recursive call to findSharedInRegion routine 
					procSharedSet = findSharedInProcedure(omp_annot, called_procedure);
					if (procSharedSet != null) {
						displaySet("procSharedSet in " + called_procedure.getName(),  procSharedSet);
						SharedSet.addAll(procSharedSet);
					}
				}
			}
		}
		return SharedSet;
	}

	/**
		* returns a set of all accessed symbols except Procedure symbols and 
		* member symbols of a class
		*/
	private Set<Symbol> getAccessedVariables(Statement stmt)
	{
		Set<Symbol> set = SymbolTools.getAccessedSymbols(stmt);
		HashSet<Symbol> ret = new HashSet<Symbol> ();
		for (Symbol symbol : set)
		{
			if( symbol instanceof VariableDeclarator ) {
				if( !AnalysisTools.isClassMember((VariableDeclarator)symbol) ) {
					ret.add(symbol);
				}
			} else if( symbol instanceof AccessSymbol ) {
				Symbol base = ((AccessSymbol)symbol).getIRSymbol();
				if( base != null ) {
					ret.add(base);
				}
			} else if( symbol instanceof NestedDeclarator ){
				//FIXME: How to handle NestedDeclarator?
				ret.add(symbol);
			}
		}
		return ret;
	}

	/**
		* Data-sharing Attribute Rules for Variables Referenced in a Region but not 
		* in a Construct.
		* (1) Static variables declared in called routines in the region are shared.
		* (2) Variables with const-qualified type having no mutable member, and that 
		*     are declared in called routines, are shared.
		* (3) File-scope or namespace-scope variables referenced in called routines in 
		*     the region are shared unless they appear in a threadprivate directive.
		* (4) Variables with heap-allocated storage are shared.
		* (5) Static data members are shared unless they appear in a threadprivate directive.
		* (6) Formal arguments of called routines in the region that are passed by 
		*     reference inherit the data-sharing attributes of the associated actual argument.
		* (7) Other variables declared in called routines in the region are private.
		*/

	/**
		* recursive function call
		*/
	private Set<Symbol> findSharedInProcedure(OmpAnnotation omp_annot, Procedure proc)
	{
		debug_tab++;
/*
		IDExpression expr = proc.getName();
		Symbol sss = expr.getSymbol();
		String str = sss.getSymbolName();	// NullPointerException here for conj_grad in CG
*/
		PrintTools.println("[findSharedInProcedure] strt: "+proc.getName().toString(), 5);
		CompoundStatement proc_body = proc.getBody();

		// find all local variables declared in the procedure body
		Set<Symbol> LocalPrivSet = SymbolTools.getLocalSymbols((SymbolTable)proc_body);
		displaySet("All local variables in a procedure, " + proc.getName().toString(), LocalPrivSet);
		
		Set<Symbol> SharedSet = new HashSet<Symbol>();
		// add all accessed variable symbols in the procedure
		Set<Symbol> accessedSymbols = getAccessedVariables(proc_body);
		displaySet("All accessed variables in a procedure, " + proc.getName().toString(), accessedSymbols);
		for( Symbol sm : accessedSymbols ) {
			///////////////////////
			// DEBUG: deprecated //
			///////////////////////
			//Traversable decl = ((SymbolTable)proc).getTable().get(((VariableDeclarator)sm).getSymbol()); 
			if( !proc.containsSymbol(sm) ) { //symbol is not passed as a parameter.
				SharedSet.add(sm); //it can be either a global or local variable.
			} else {
				if(SymbolTools.isPointerParameter(sm)) { 
					// Formal arguments of called routines that are passed by reference 
					// inherit the data-sharing attributes of the associated argument.
					PrintTools.println("[OmpAnalsys.findSharedInProcedure()] Call-by-reference parameter, " +
							sm.getSymbolName()+ ", in "+ proc.getName().toString() + 
							" should be handled by a caller", 2);
				} else if( SymbolTools.isScalar(sm) ) { //Call-by-value parameter is a local variable
					LocalPrivSet.add(sm);
				}
			}
		}

		// find static variables in a procedure to check rule (1)
		Set<Symbol> StaticLocalSet = AnalysisTools.extractStaticVariables(LocalPrivSet);
		displaySet("static local variables in a procedure", StaticLocalSet);

		// LocalPrivSet = LocalPrivSet - StaticLocalSet - ThreadPrivSet
		LocalPrivSet.removeAll(StaticLocalSet);

		//////////////////////////////////////////////////////////////////////////////////////
		// FIXME: if the called function has a threadprivate variable that was not accessed //
		// by the enclosing parallel region, below threadprivate set does not contain the   //
		// threadprivate variable.                                                          //
		//////////////////////////////////////////////////////////////////////////////////////
		HashSet<Symbol> OmpThreadPrivSet = (HashSet<Symbol>)omp_annot.get("threadprivate");
		displaySet("threadprivate variables in a procedure", OmpThreadPrivSet);
		LocalPrivSet.removeAll(OmpThreadPrivSet);

		// SharedSet = SharedSet - LocalSet - ThreadPrivSet
		SharedSet.removeAll(LocalPrivSet);
		SharedSet.removeAll(OmpThreadPrivSet);

		// index variables of omp for-loops are predetermined private variables, 
		// which may or may not be listed in the data-sharing attribute clauses
		// In case if they are not listed, we add them to the private variables
		// SharedSet = SharedSet - LoopIndexVariables
		Set LoopIndexVariables = getLoopIndexVarSet(proc_body);
		SharedSet.removeAll(LoopIndexVariables);
		
		//Below is disabled so that omp private set does not contain private 
		//variables in called procedures. 
		//Update private set in the map
		//HashSet<Symbol> OmpPrivSet = (HashSet<Symbol>)map.get("private");
		//OmpPrivSet.addAll(LocalPrivSet);
		//OmpPrivSet.addAll(LoopIndexVariables);

		DepthFirstIterator iter = new DepthFirstIterator(proc_body);
		while(iter.hasNext())
		{
			Object obj = iter.next();
			if (obj instanceof FunctionCall)
			{
				FunctionCall call = (FunctionCall)obj;

				// called_proc is null for system calls 
				Procedure called_proc = call.getProcedure();

				if (called_proc != null)
				{	
					// recursive call to findSharedInProcedure routine
					PrintTools.println("Performing IPA into the procedure: " + called_proc.getName(), 5);
					Set<Symbol> procSharedSet = findSharedInProcedure(omp_annot, called_proc);
					if (procSharedSet != null) {
						displaySet("procSharedSet in " + called_proc.getName(),  procSharedSet);
						SharedSet.addAll(procSharedSet);
					}
				}
			}
		}

		PrintTools.println("[findSharedInProcedure] done: "+proc.getName().toString(), 5);
		PrintTools.println("--------------------------------------", 5);

		debug_tab--;
		return SharedSet;
	}

	/**
		* convert a set of String into a set of Symbols
		* @param stmt is either a CompoundStatement or ForLoop, where a matching symbol 
		* is searched for a given String. We assume that there should be only one matching 
		* symbol within a stmt
		*/
	private HashSet<Symbol> convertString2Symbol(Set<String> iset, Statement stmt)
	{
		HashSet<Symbol> ret = new HashSet<Symbol> ();
		if (iset == null) return ret;

		Set<Symbol> accessed_set = getAccessedVariables(stmt);
		for (Symbol sym : accessed_set)
		{
			String str = sym.getSymbolName();
			if ( iset.contains(str) )
			{
				ret.add(sym);
			}
		}
		return ret;
	}


	/**
		*	This method is for debugging purpose; it shows the statement that
		*	has an OpenMP pragma.
		*/
	public void display()
	{
		if (debug_level < 8) return;

		DepthFirstIterator iter = new DepthFirstIterator(program);
		
		System.out.println("====> This program contains the following OpenMP pragmas.");
		while( iter.hasNext() )
		{
			Object o = iter.next();
			if( o instanceof Annotatable ) {
				Annotatable annot_container = (Annotatable)o;
				List<OmpAnnotation> omp_annots = 
					annot_container.getAnnotations(OmpAnnotation.class);
				if( omp_annots != null ) {
					for( OmpAnnotation annot : omp_annots ) {
						System.out.println(annot.toString());
					}
				}
			}
		}
	}

	public void displayList(LinkedList<String> list)
	{
		int cnt = 0;
		if (debug_level > 1)
		{
			for (int i=0; i<debug_tab; i++) System.out.print("  ");
			for (String ilist : list)
			{
				if ( (cnt++)!=0 ) System.out.print(", ");
				System.out.print(ilist);
				}
		}
	}

	public void displaySet(Set iset)
	{
		int cnt = 0;
		if (iset == null) return;
		if (debug_level > 1)
		{
			for (int i=0; i<debug_tab; i++) System.out.print("  ");
			for ( Object obj : iset )
			{
				if ( (cnt++)!=0 ) System.out.print(", ");
				if (obj instanceof String)
					System.out.print((String)obj);
				else if (obj instanceof Symbol)
					System.out.print(((Symbol)obj).getSymbolName());
			}
		}
	}

	public void displaySet(String name, Set iset)
	{
		int cnt = 0;
		if (iset == null) return;
		if (debug_level > 1)
		{
			for (int i=0; i<debug_tab; i++) System.out.print("  ");
			System.out.print(name + ": ");
			for ( Object obj : iset )
			{
				if ( (cnt++)!=0 ) System.out.print(", ");
				if (obj instanceof String)
					System.out.print((String)obj);
				else if (obj instanceof Symbol)
					System.out.print(((Symbol)obj).getSymbolName());
			}
			System.out.println("\n");
		}
	}
	
	static private String[] omp_pragma_w_attach = {
		"parallel",  "for",  "sections", "section", "single", "task", 
		"master", "critical", "atomic", "ordered"
	};
/*	static private String[] omp_pragma_wo_attach = {
		"barrier", "taskwait", "flush", "threadprivate"
	};*/
	
	/**
	  * Convert a set of Strings in each OpenMP reduction clause into a set of Symbols.
	  * - This method should be called right after OmpAnalysis.start() if your pass wants
	  * to handle OpenMP reduction clauses.
	  * CAUTION: this method may modify shared set information if reduction clauses include 
	  * shared variables; if a shared variable is used as a reduction variable, it should 
	  * be removed from shared set (but if the reduction clause belongs to a work-sharing 
	  * construct, the list item that appears in the reduction clause must be shared in 
	  * the parallel region to which any of work-sharing regions arising from the 
	  * work-sharing construct bind).
	  */
	public void convertReductionClauses() {
		List<OmpAnnotation> redAnnotList = IRTools.collectPragmas(program, OmpAnnotation.class, "reduction");

		for (OmpAnnotation omp_annot : redAnnotList)
		{
			Statement atstmt = (Statement)omp_annot.getAnnotatable();
			HashMap reduction_map = (HashMap)omp_annot.remove("reduction");
			HashMap newreduction_map = new HashMap(4);
			for (String ikey : (Set<String>)(reduction_map.keySet())) {
				HashSet<String> tmp_set = (HashSet<String>)reduction_map.get(ikey);
				HashSet<Symbol> itemSet = convertString2Symbol(tmp_set, atstmt);
				newreduction_map.put(ikey, itemSet);
				//newreduction_map.put(BinaryOperator.fromString(ikey), itemSet);
			}
			omp_annot.put("reduction", newreduction_map);
			//Update reduction clause; remove unused reduction variables from 
			//the reduction itemlist, and remove shared variables from the shared 
			//set if they are included in the reduction itemlist.
			updateReductionClause(atstmt, omp_annot);
		}
	}

	/*
	 * Restriction to the reduction clause (OpenMP API V3)
	 *     - A list item that appears in a reduction clause of a worksharing construct
	 *     must be shared in the parallel regions to which any of the worksharing regions
	 *     arising from the worksharing construct bind.
	 */
	
	/**
	 * - For each symbol in the reduction itemlist,
	 *   If it is accessed in the region t,
	 *     - find a symbol with the same name in the SymbolTable, 
	 *       and put the new symbol into the reduction itemlist.
	 *     - If no symbol is found in the table, put the old symbol into the itemlist.
	 * - If no symbol in the reduction itemlist is used in region, t,
	 *   remove the reduction clause.
	 *   FIXME: this method does not check whether any reduction variable is used 
	 *   in a function called within the region, t.
	 * - If any shared variable is included in the reduction itemlist,
	 *   it should be removed from the shared set.
	 *     
	 * @param t region, from which symbol search starts.
	 * @param omp_map OmpAnnotation, which is a HashMap containing OpenMP clauses. 
	 */
	static public void updateReductionClause(Traversable t, OmpAnnotation omp_map)
	{
		VariableDeclaration sm_decl = null;
		VariableDeclarator v_declarator = null;
		Traversable tt = t;
		while( !(tt instanceof SymbolTable) ) {
			tt = tt.getParent();
		}
		Set<Symbol> accessedSymbols = SymbolTools.getAccessedSymbols(t);
		HashMap reduction_map = (HashMap)omp_map.remove("reduction");
		HashMap newreduction_map = new HashMap(4);
		HashSet<Symbol> allItemsSet = new HashSet<Symbol>();
		Collection tCollect = null;
		for (String ikey : (Set<String>)(reduction_map.keySet())) {
			tCollect = (Collection)reduction_map.get(ikey);
			HashSet<Symbol> old_set = new HashSet<Symbol>();
			old_set.addAll(tCollect);
			HashSet<Symbol> new_set = new HashSet<Symbol>();
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
			if( new_set.size() > 0 ) {
				//newreduction_map.put(BinaryOperator.fromString(ikey), itemSet);
				newreduction_map.put(ikey, new_set);
				allItemsSet.addAll(new_set);
			}
		}
		if( allItemsSet.size() > 0 ) {
			omp_map.put("reduction", newreduction_map);
			//If any shared variable is included in the reduction itemlist,
			//it should be removed from the shared set.
			tCollect = (Collection)omp_map.get("shared");
			HashSet<Symbol> sharedSet = new HashSet<Symbol>();
			if( tCollect != null ) {
				sharedSet.addAll(tCollect);
				omp_map.put("shared", sharedSet);
			}
			HashSet<Symbol> deleteSet = new HashSet<Symbol>();
			for( Symbol svar : sharedSet ) {
				for( Symbol redVar : allItemsSet ) {
					if( svar.getSymbolName().compareTo(redVar.getSymbolName()) == 0 ) {
						deleteSet.add(svar);
						break;
					}
				}
			}
			sharedSet.removeAll(deleteSet);
		}
	}



}


