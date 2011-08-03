package cetus.analysis;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Vector;

import cetus.hir.Annotatable;
import cetus.hir.Annotation;
import cetus.hir.AnnotationStatement;
import cetus.hir.ArrayAccess;
import cetus.hir.ArraySpecifier;
import cetus.hir.AssignmentExpression;
import cetus.hir.AssignmentOperator;
import cetus.hir.ClassDeclaration;
import cetus.hir.CommentAnnotation;
import cetus.hir.CompoundStatement;
import cetus.hir.Declaration;
import cetus.hir.DeclarationStatement;
import cetus.hir.Declarator;
import cetus.hir.DepthFirstIterator;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.FlatIterator;
import cetus.hir.FunctionCall;
import cetus.hir.GotoStatement;
import cetus.hir.IDExpression;
import cetus.hir.Identifier;
import cetus.hir.Label;
import cetus.hir.NameID;
import cetus.hir.NestedDeclarator;
import cetus.hir.NullStatement;
import cetus.hir.PointerSpecifier;
import cetus.hir.Procedure;
import cetus.hir.ProcedureDeclarator;
import cetus.hir.Program;
import cetus.hir.ReturnStatement;
import cetus.hir.Specifier;
import cetus.hir.Statement;
import cetus.hir.Symbol;
import cetus.hir.SymbolTable;
import cetus.hir.SymbolTools;
import cetus.hir.Tools;
import cetus.hir.TranslationUnit;
import cetus.hir.Traversable;
import cetus.hir.UserSpecifier;
import cetus.hir.VariableDeclaration;
import cetus.hir.VariableDeclarator;
import cetus.transforms.SingleReturn;

/**
 * Performs simple function inlining. It assumes that the code to be inlined is already compiled correctly and has no errors.
 * Following are taken care of: 
 * 		* Functions that result in recursion are not inlined
 * 		* Functions that use static external variables are not inlined
 * 		* Global variables used in the function to be inlined are handled by declaring them as extern
 * 		* Variable reshaping (array-dimensions) is handled
 * 		* Long function call chains (a->b->c->d->...) are handled as long as they do not result in a recursion
 * 		* Switches are provided for controlling variable names and their length
 * 		* Support for making log
 * 		* Comments with the inlined code
 */
public class InlineExpansion {
	
	/** Default option for making log */
	private static final boolean MAKE_LOG = true;
	/** Default prefix for the new variable name for a parameter */
	private static final String PARAM_PREFIX = "param";
	/** Default prefix for the new variable name for a local variable */
	private static final String LOCAL_PREFIX = "local";
	/** Default prefix for the new name for a label */
	private static final String LABEL_PREFIX = "label";
	/** Default prefix for the name of the variable used to hold the return value of the function that has been inlined */
	private static final String RESULT_PREFIX = "result";
	/** Default maximum length of variables' names that inlining introduces */
	private static final int MAX_VAR_LENGTH = 256;
	/** Default option for variable naming, in case of fully qualified names name of function is also appended, short form does not include that */
	private static final boolean FULLY_QUALIFIED_NAMES = true;
	
	/** Current option for making log */
	private boolean makeLog = MAKE_LOG;
	/** Currently used prefix for the new variable name for a parameter */
	private String paramPrefix = PARAM_PREFIX;
	/** Currently used prefix for the new variable name for a local variable */
	private String localVarPrefix = LOCAL_PREFIX;
	/** Currently used prefix for the new name for a label */
	private String labelPrefix = LABEL_PREFIX;
	/** Currently used prefix for the name of the variable used to hold the return value of the function that has been inlined */
	private String resultPrefix = RESULT_PREFIX;
	/** Currently used maximum length of variables' names that inlining introduces */
	private int maxVarLength = MAX_VAR_LENGTH;
	/** Current option for variable naming, in case of fully qualified names name of function is also appended, short form does not include that */
	private boolean fullyQualifiedName = FULLY_QUALIFIED_NAMES;
	
	/**
	 * sets the make-log flag
	 * @param makeLog - new value for the currently used make-log option
	 */
	public void setMakeLog(boolean makeLog) {
		this.makeLog = makeLog;
	}
	/**
	 * sets the currently used prefix for new variables replacing old function parameters
	 * @param paramPrefix - the new prefix
	 */
	public void setParamPrefix(String paramPrefix) {
		this.paramPrefix = paramPrefix;
	}
	/**
	 * sets the currently used prefix for new variables used to hold return value of the function that has been inlined
	 * @param resultPrefix - the new prefix
	 */
	public void setResultPrefix(String resultPrefix) {
		this.resultPrefix = resultPrefix;
	}
	/**
	 * sets the currently used prefix for new variables replacing old local variables
	 * @param localVarPrefix - the new prefix for local variables in the inlined code
	 */
	public void setLocalVarPrefix(String localVarPrefix) {
		this.localVarPrefix = localVarPrefix;
	}
	/**
	 * sets the currently used prefix for new labels replacing old labels
	 * @param labelPrefix - the new prefix for labels in the inlined code
	 */
	public void setLabelPrefix(String labelPrefix) {
		this.labelPrefix = labelPrefix;
	}
	/**
	 * sets the maximum length for the variables introduced by inlining
	 * @param length - new length for variables
	 */
	public void setMaxVarLength(int length) {
		this.maxVarLength = length;
	}
	/**
	 * sets the option to have verbose variable names
	 * @param verboseName - the option to have verbose names
	 */
	public void setFullyQualifiedName(boolean verboseName) {
		this.fullyQualifiedName = verboseName;
	}
	
	/**
	 * tells if the log is being made
	 */
	public boolean isMakeLog() {
		return makeLog;
	}
	/**
	 * returns the prefix used in naming variables that replace parameters in the function to be inlined
	 */
	public String getParamPrefix() {
		return paramPrefix;
	}
	/**
	 * returns the prefix used in naming variables that replace local variables in the inlined code
	 */
	public String getLocalVarPrefix() {
		return localVarPrefix;
	}
	/**
	 * returns the prefix used in naming labels that replace labels in the inlined code
	 */
	public String getLabelPrefix() {
		return labelPrefix;
	}
	/**
	 * returns the prefix used in naming the variable that holds the return value of the function that has been inlined
	 */
	public String getResultPrefix() {
		return resultPrefix;
	}
	/**
	 * returns the maximum allowed length for the variable names that inlining introduces
	 */
	public int getMaxVarLength() {
		return maxVarLength;
	}
	/**
	 * tells if the inlining code uses verbose names for variables that it introuduces
	 */
	public boolean isFullyQualifiedName() {
		return fullyQualifiedName;
	}
	/**
	 * performs inlining in the body of the function passed, after performing inlining in the functions
	 * in the call graph of this function. does so in a bottom up fashion. Clients that are only interested
	 * in one-level inlining should call the <code>inlineFunction<code> instead.
	 * Following are taken care of:
	 * 		* Functions that result in recursion are not inlined
	 * 		* Functions that use static external variables are not inlined
	 * 		* Global variables used in the function to be inlined are handled by declaring them as extern
	 * 		* Variable reshaping (array-dimensions) is handled
	 * 		* Long function call chains (a->b->c->d->...) are handled as long as they do not result in a recursion
	 * 		* Switches are provided for controlling variable names and their lengths
	 * 		* Support for making log
	 * 		* Comments with the inlined code
	 *   
	 * @param proc - the function in whose code, inlining is to be performed, usually the main function
	 */
	public void inline(Procedure proc) {
		
		// get the program
		Program program = getProgram(proc);
		if(program == null)
			return;
		
	    // make the call graph
	    CallGraph callGraph = new CallGraph(program);
	    
	    // first inline the callee functions deep in the call graph, do it bottom up
	    List<Procedure> procs = callGraph.getTopologicalCallList();
	    for (Procedure procedure : procs) {
	    	if(!procedure.equals(proc))
	    		inlineFunction(procedure, callGraph);
		}
	    
	    // now inline the main procedure
	    inlineFunction(proc, callGraph);
	}
	
	/**
	 * inlines the function calls in the body of the given procedure, doesn't go deeper than that
	 * i.e. does not inline functions called by the procedure recursively.
	 * @param proc - the procedure 
	 * @param callGraph - the call graph, using call graph this method makes sure that the function calls that result in recursion
	 *                  - are not inlined. If that is not a concern, clients can pass null for this parameter
	 *                  
	 * Following are taken care of:
	 * 		* Function calls that result in recursion are not inlined (it can be bypassed by passing null for the call graph)
	 * 		* Functions that use static external variables are not inlined
	 * 		* Global variables used in the function to be inlined are handled by declaring them as extern
	 * 		* Variable reshaping (array-dimensions) is handled
	 * 		* Switches are provided for controlling variable names and their lengths
	 * 		* Support for making log
	 * 		* Comments with the inlined code
	 */
	public void inlineFunction(Procedure proc, CallGraph callGraph) {
		
	    CompoundStatement functionBody = proc.getBody();
	    DepthFirstIterator dfi = new DepthFirstIterator(proc);
	    Vector<Statement> statementsWithFunctionCalls = new Vector<Statement>();
	    
	    // gather all the statements that have function calls
	    Object obj;
	    while(dfi.hasNext()) {
	    	if((obj = dfi.next()) instanceof FunctionCall){
		    	FunctionCall call = (FunctionCall)obj;
	    		Statement s = getStatement(call);
	    		if(!statementsWithFunctionCalls.contains(s))
	    			statementsWithFunctionCalls.addElement(s);
	    	}	
	    }
	    // find the first non-declaration statement, new variables will be declared before it
	    Statement firstNonDeclarationStmtInFunc = getFirstNonDeclarationStatement(functionBody);
	    if(statementsWithFunctionCalls.size() > 0){
	    	if(firstNonDeclarationStmtInFunc == null || firstNonDeclarationStmtInFunc == statementsWithFunctionCalls.elementAt(0)){
	    		firstNonDeclarationStmtInFunc = statementsWithFunctionCalls.elementAt(0);
	    	}	
	    }
	    	
	    // for each statement, inline every function that is called from it 
	    for(int i = 0; i < statementsWithFunctionCalls.size(); i++){
	    	
	    	Statement statementWithFunctionCall = statementsWithFunctionCalls.elementAt(i);
	    	CompoundStatement enclosingCompoundStmt = getEnclosingCompoundStmt(statementWithFunctionCall);
	    	if(enclosingCompoundStmt == null)
	    		enclosingCompoundStmt = functionBody;
	    	
	    	Statement firstNonDeclarationStmt = firstNonDeclarationStmtInFunc;
	    	if(enclosingCompoundStmt != functionBody)
	    		firstNonDeclarationStmt = getFirstNonDeclarationStatement(enclosingCompoundStmt);
	    	
	    	if(Tools.indexByReference(enclosingCompoundStmt.getChildren(), statementWithFunctionCall) < 
	    	      Tools.indexByReference(enclosingCompoundStmt.getChildren(), firstNonDeclarationStmt)){

	    		if(makeLog){
					System.out.println("function calls in following statement in fucntion \'" + proc.getName().toString() + "\' can't be inlined because this statement appears before declaration statement(s) or is one itself");
					System.out.println(statementWithFunctionCall.toString());
	    		}	
				continue;
	    	}
	    	
	    	// There may be many function calls inside one statement
	    	dfi = new DepthFirstIterator(statementWithFunctionCall);
	    	Vector<FunctionCall> callsInStatement = new Vector<FunctionCall>();
	    	while(dfi.hasNext()){
    			if( (obj = dfi.next()) instanceof FunctionCall){
	    			callsInStatement.addElement((FunctionCall) obj);
    			}
	    	}
	    	
	    	// for each function call, inline the called function. Make sure to do it in reverse order.
	    	for (int j = callsInStatement.size()-1; j >= 0; j--) {
				
    			FunctionCall fc = callsInStatement.elementAt(j);
//    			Procedure function = (Procedure)(fc.getProcedure().clone()); // What? Procedure can't be cloned.. why? Okay, clone the body then. (note: in latest code it can be cloned)
    			Procedure function = fc.getProcedure();
    			boolean inline = true;

    			// function can be null, e.g. in case if the function call involved function pointer(s)
    			if(function == null){
    				if(makeLog){
    					System.out.println("function call in following statement in function \'" + proc.getName().toString() + "\' calls unrecognized or library function, or possibly involves function pointer(s) so we are not inlining it");
    					System.out.println(getStatement(fc).toString());
    				}	
    				continue;
    			}

    			String functionName = function.getName().toString();
    			
    			// don't expand if it is a recursive call (self or otherwise).
    			if(callGraph != null && callGraph.isRecursive(function)){
    				if(makeLog){
    					System.out.println("calling " + functionName + " function in the following statement in function \'" + proc.getName().toString() + "\' results in a recursion around the function so we are not inlining it");
    					System.out.println(getStatement(fc).toString());
    				}
    				continue;
    			}
    			// self-recursion
    			if(function.equals(proc)){
    				if(makeLog){
    					System.out.println("calling " + functionName + " function in the following statement in function \'" + proc.getName().toString() + "\' results in self-recursion so we are not inlining it");
    					System.out.println(getStatement(fc).toString());
    				}	
    				continue;
    			}
    			// check other conditions (like use of function pointers or static variables)
    			if(!canInline(function, fc, proc)){
    				continue;
    			}
    			
				// get the arguments and return type of the function that is to be inlined
    			List<Expression> args = fc.getArguments();
    			List<Specifier> returnTypes = function.getReturnType();
    			List parameters = function.getParameters();

				List<String> newParamNames = new LinkedList<String>();
    			List<IDExpression> oldParams = new LinkedList<IDExpression>();
    			
    			Vector<Declaration> declarations = new Vector<Declaration>();
    			Vector<Statement> statements = new Vector<Statement>();
    			Vector<Declaration> addedGlobalDeclarations = new Vector<Declaration>();
    			Vector<Declaration> removedGlobalDeclarations = new Vector<Declaration>();
    			
    			
				// get the code to be inlined (don't forget to clone)
				CompoundStatement codeToBeInlined = (CompoundStatement)(function.getBody().clone());

				// if there are multiple return statements in the function, use the single return transformation to have only one
				if(hasMultipleReturnStmts(codeToBeInlined)){
					Procedure p = (Procedure)function.clone();
					new SingleReturn(getProgram(function)).transformProcedure(p);
					codeToBeInlined = p.getBody();
					codeToBeInlined.setParent(null);
				}
				
				// rename local variables in the inlined code, except for extern declarations
		        dfi = new DepthFirstIterator(codeToBeInlined);
		        List<IDExpression> locals = new LinkedList<IDExpression>();
		        List <IDExpression> labels = new LinkedList<IDExpression>();
		        List<String> newLocals = new LinkedList<String>();
		        // get all the local variables
		        obj = null;
		        while (dfi.hasNext()){
		        	if( (obj = dfi.next()) instanceof DeclarationStatement){
		        		Declaration d = ((DeclarationStatement)obj).getDeclaration();
		        		// skip the extern declarations
		        		if(d instanceof VariableDeclaration){
		        			List specs = ((VariableDeclaration)d).getSpecifiers();
		        			if(specs.contains(Specifier.EXTERN))
		        				continue;

		        			List<IDExpression> ids = ((VariableDeclaration)d).getDeclaredIDs();
		    				for (int k = 0; k < ids.size(); k++) {
								locals.add((IDExpression)(ids.get(k).clone())); // don't forget to clone
							}
		        		}
		        	}
		        	// also handle labels
		        	else if(obj instanceof Label){
		        		labels.add(((Label)obj).getName());
		        	}
				}
		        // come up with unique names for new variables and labels.
		        for (int k = 0; k < locals.size(); k++) {
					newLocals.add(getUniqueIdentifier(enclosingCompoundStmt, localVarPrefix + "_" + functionName, locals.get(k).toString()).getName());
				}
		        for (int k = 0; k < labels.size(); k++) {
		        	locals.add(labels.get(k));
					newLocals.add(getUniqueLabel(proc, labelPrefix + "_" + functionName, labels.get(k).toString()).getName());
				}
		        HashMap<String, ArrayAccess> actualArgs = new HashMap<String, ArrayAccess>();
		        // replace the old variables with the new variables, remove them from the symbol table of the inlined code
		        // and add new variables in it
		        replaceVariableNames(codeToBeInlined, locals, newLocals, true, actualArgs);
				
    			if(parameters.size() == args.size()){
			        
    				// for each parameter, come up with a new variable, declare it and assign the actual parameter value to it
					// but for array parameters use the original variable if possible
    				ArrayParameterAnalysis arrayParameterAnalysis = new ArrayParameterAnalysis(getProgram(proc));
    				arrayParameterAnalysis.start();
    				
    				for(int k = 0; k < parameters.size(); k++){
    					
						Expression actualArg = args.get(k);
						if(actualArg instanceof IDExpression){

							VariableDeclaration pdn = (VariableDeclaration)parameters.get(k);
							Declarator pdd = pdn.getDeclarator(0);
							if(pdd instanceof VariableDeclarator && pdd.getArraySpecifiers() != null && pdd.getArraySpecifiers().size() > 0){
								Expression exp = arrayParameterAnalysis.getCompatibleArgument(fc, (VariableDeclarator)pdd);
								if(exp != null){
									oldParams.add((IDExpression)(((IDExpression)pdn.getDeclaredIDs().get(0)).clone())); // don't forget to clone
				    				newParamNames.add(exp.toString());
				    				continue;
								}
							}
							// following is my version, but right now i'm using hansang's method which is more conservative
//							Declaration dn = getEnclosingCompoundStmt(getStatement(fc)).findSymbol((Identifier)actualArg);
//							if(dn instanceof VariableDeclaration){
//								Declarator dd = ((VariableDeclaration)dn).getDeclarator(0);
//								if(dd.getArraySpecifiers() != null && dd.getArraySpecifiers().size() > 0){
//									dn = (Declaration) parameters.get(k);
//									dd = ((VariableDeclaration)dn).getDeclarator(0);
//									if(dd.getArraySpecifiers() != null && dd.getArraySpecifiers().size() > 0){
//										oldParams.add((IDExpression)(((IDExpression)dn.getDeclaredSymbols().get(0)).clone())); // don't forget to clone
//					    				newParamNames.add(actualArg.toString());
//					    				continue;
//									}
//								}	
//							}
						}
						else if(actualArg instanceof ArrayAccess){
							Declaration dn = (Declaration) parameters.get(k);
							Declarator dd = ((VariableDeclaration)dn).getDeclarator(0);
							if(dd instanceof VariableDeclarator && dd.getArraySpecifiers() != null && dd.getArraySpecifiers().size() > 0){
								Expression exp = arrayParameterAnalysis.getCompatibleArgument(fc, (VariableDeclarator)dd);
								if(exp != null){
									oldParams.add((IDExpression)(((IDExpression)dn.getDeclaredIDs().get(0)).clone())); // don't forget to clone
				    				newParamNames.add(exp.toString());
				    				actualArgs.put(exp.toString(), (ArrayAccess)exp);
				    				continue;
								}
							}	
							// following is my version, but right now i'm using hansang's method which is more conservative
//							if(dd.getArraySpecifiers() != null && dd.getArraySpecifiers().size() > 0){
//								oldParams.add((IDExpression)(((IDExpression)dn.getDeclaredSymbols().get(0)).clone())); // don't forget to clone
//			    				newParamNames.add(actualArg.toString());
//			    				actualArgs.put(actualArg.toString(), (ArrayAccess)actualArg);
//			    				continue;
//							}
							
						}
						
	    				Declaration d = (Declaration) parameters.get(k);
	    				List<IDExpression> params = d.getDeclaredIDs();
	    				if(d instanceof VariableDeclaration && params.size() == 1){
	    					VariableDeclaration p = (VariableDeclaration)d;
		    				Declarator decl = p.getDeclarator(0); // in case of parameters it seems safe to get 0th declarator, there should be only one
							oldParams.add((IDExpression)(params.get(0).clone())); // don't forget to clone
		    				String paramName = ((NameID)params.get(0)).getName();
		    				// find out a new variable name for this parameter, must be unique
		    				NameID nameId = getUniqueIdentifier(enclosingCompoundStmt, paramPrefix + "_" + functionName, paramName);
		    				Identifier id = null;
		    				newParamNames.add(nameId.getName());
		    				// declare the new variable
		    				VariableDeclaration newDeclaration = null;
		    				// it may be using a nested declarator, e.g. int (*param_name)[4]
		    				if(decl instanceof NestedDeclarator){
		    					// clone the original declarator, find the identifier and change its name to new name
		    					NestedDeclarator nd = (NestedDeclarator)decl;
		    					VariableDeclarator vd = new VariableDeclarator(nameId);
		    					id = new Identifier(vd);
		    					NestedDeclarator newND = new NestedDeclarator(vd, nd.getParameters());
//		    					List<Traversable> children = nd.getChildren();
//		    					for (int l = 0; l < children.size(); l++) {
//									if(children.get(l) instanceof VariableDeclarator){
//										children = ((VariableDeclarator)children.get(l)).getChildren();
//										for (int l2 = 0; l2 < children.size(); l2++) {
//											//TODO: can't use setName do something else
//											if(children.get(l2) instanceof Identifier){
//												((Identifier)children.get(l2)).setName(id.getName());
//											}	
//										}
//									}
//								}
		    					newDeclaration = new VariableDeclaration(p.getSpecifiers(), newND);
		    				}
		    				else if(decl instanceof VariableDeclarator){
		    					VariableDeclarator declarator = (VariableDeclarator)decl;
                  VariableDeclaration declaration = (VariableDeclaration)declarator.getDeclaration();
			    				// if the parameter is an array declare the new variable as pointer, handle multi-dimensional arrays with care
			    				if(declarator.getArraySpecifiers().size() > 0){
			    					List arraySpecs = declarator.getArraySpecifiers();
			    					if(arraySpecs.size() == 1){
			    						int n = ((ArraySpecifier)arraySpecs.get(0)).getNumDimensions();
			    						// we assume that all the dimensions except the first will have the size mentioned in them
			    						// we are assuming that we are inlining code which is correctly compiled before
										if(n == 1) {
					    					//List<Specifier> newSpecs = declarator.getTypeSpecifiers();
                      List<Specifier> newSpecs = new ArrayList<Specifier>(declarator.getSpecifiers());
											newSpecs.add(PointerSpecifier.UNQUALIFIED);
											VariableDeclarator vd = new VariableDeclarator(newSpecs, nameId);
											id = new Identifier(vd);
					    					//newDeclaration = new VariableDeclaration(vd);
                      newDeclaration = new VariableDeclaration(declaration.getSpecifiers(), vd);
										}
										else if(n > 1) {
					    					List<Specifier> newSpecs = new ArrayList<Specifier>();
					    					newSpecs.add(PointerSpecifier.UNQUALIFIED);
											VariableDeclarator vd = new VariableDeclarator(newSpecs, nameId);
											id = new Identifier(vd);
					    					List<Specifier> newTrailingSpecs = new ArrayList<Specifier>();
					    					List<Expression> dimensions = new ArrayList<Expression>();
				    						for (int m = 1; m < n; m++) {
				    							dimensions.add((Expression)((ArraySpecifier)arraySpecs.get(0)).getDimension(m).clone());
											}
					    					ArraySpecifier arraySpecifier = new ArraySpecifier(dimensions);
			    							newTrailingSpecs.add(arraySpecifier);
											//NestedDeclarator nd = new NestedDeclarator(declarator.getTypeSpecifiers(), vd, null, newTrailingSpecs);
											//newDeclaration = new VariableDeclaration(nd);
                      NestedDeclarator nd = new NestedDeclarator(declarator.getSpecifiers(), vd, null, newTrailingSpecs);
                      newDeclaration = new VariableDeclaration(declaration.getSpecifiers(), nd);
										}
									}
			    					else{
			    						System.err.println("this case not handled in InlineExpansion.java ... 1");
			    					}
			    				}
			    				else{
			    					//VariableDeclarator vd = new VariableDeclarator(declarator.getTypeSpecifiers(), nameId);
                    VariableDeclarator vd = new VariableDeclarator(declarator.getSpecifiers(), nameId);
			    					id = new Identifier(vd);
			    					//newDeclaration = new VariableDeclaration(vd);	    					
                    newDeclaration = new VariableDeclaration(declaration.getSpecifiers(), vd);
			    				}
		    				}
		    				else{
		    					System.err.println("unexpected Declarator type, this case not handled in InlineExpansion.java ... 2");
		    				}
		    				if(newDeclaration != null && id != null){
			    				// add the new variable declaration
			    				declarations.addElement(newDeclaration);
			    		        Expression arg = (Expression)fc.getArgument(k).clone(); // don't forget to clone
			    		        // and assign to it the expression that was being passed as a function call, this should be done before the statement with function call 
			    		        ExpressionStatement exprStmt = new ExpressionStatement(new AssignmentExpression(id, AssignmentOperator.NORMAL, arg));
			    		        statements.addElement(exprStmt);
		    				}    
	    				}
	    				else{
	    					// shouldn't get here
	    					System.err.println("FIXME in InlineExpansion.java, wasn't expecting it ... 3");
	    				}
	    			}
					
			        // in the code to be inlined replace the parameters with new variables, but don't declare them in the symbol table
					// of the inlined code as we have already declared them in the surrounding function
					replaceVariableNames(codeToBeInlined, oldParams, newParamNames, false, actualArgs);
    			}	
				// deal with the global variables that the original function was using, now we need to extern them
				Map<IDExpression, Declaration> usedGlobalVars = getUsedGlobalVariables(function, codeToBeInlined, newLocals, newParamNames);
				Iterator<IDExpression> iterator = usedGlobalVars.keySet().iterator();
				while(iterator.hasNext()){
					IDExpression varId = (IDExpression)iterator.next(); //should be safe to cast to Identifier since we checked the equality in getUsedGlobalVariables function
					if(usedGlobalVars.get(varId) instanceof VariableDeclaration){
						VariableDeclaration varDec = (VariableDeclaration)usedGlobalVars.get(varId).clone();
						VariableDeclarator decl = getDeclarator(varDec, varId);
						if(decl != null){
							List<Specifier> spec = new ArrayList<Specifier>();
							spec.addAll(varDec.getSpecifiers());
							if(spec.contains(Specifier.STATIC)){
								// we have a static variable declared in the same translation unit, make sure its declaration appears before
								// the function, otherwise move it up.
								TranslationUnit tUnit = getTranslationUnit(function);
								FlatIterator fi = new FlatIterator(tUnit);
								boolean functionTraversed = false;
								while(fi.hasNext()){
									obj = fi.next();
									if(obj instanceof Procedure && proc.equals(obj)){
										functionTraversed = true;
									}
									if(obj instanceof VariableDeclaration && usedGlobalVars.get(varId).equals(obj)){
										if(functionTraversed){
											// move the declaration up
											removedGlobalDeclarations.addElement((VariableDeclaration)obj);
											addedGlobalDeclarations.addElement((VariableDeclaration)obj);
										}
										break;
									}
								}
								// don't extern the static variable
								continue;
							}
							if(!spec.contains(Specifier.EXTERN))
								spec.add(0, Specifier.EXTERN);

							NameID name = new NameID(varId.getName());
							VariableDeclarator newDecl = new VariableDeclarator(decl.getSpecifiers(), name, decl.getTrailingSpecifiers());
							varId = new Identifier(newDecl);
							VariableDeclaration newExternDeclaration = new VariableDeclaration(spec, newDecl);
							declarations.addElement(newExternDeclaration);
							//TODO: enclosing compound statement may already has such extern declaration, would be nice to avoid
							// multiple externs of the same variable.
						}	
					}
				}
				// if some function calls in the code to be inlined are not inlined, we need to make sure that we have the declarations
				// for those function calls, if not we got to get them from the callee's header files
				// if they are using user specifier types in the parameters, we need to get their declarations as well
				// and declare them in caller's translation. while doing so, we should follow the chains of typedefs as well the struct
				// elements. However, in case of a name conflict, we would give up and not inline.
				dfi = new DepthFirstIterator(codeToBeInlined);
				TranslationUnit callerUnit = getTranslationUnit(proc);
				TranslationUnit calleeUnit = getTranslationUnit(function);
				
				while(dfi.hasNext()){
					if((obj = dfi.next()) instanceof FunctionCall){
						FunctionCall functionCall = (FunctionCall)obj;
						// if we don't have the declaration, get it from the callee's translation unit and declare it
						// but if the arguments are typedefed we don't want to create problems, so we won't inline
						if(!isDeclarationAvailable(functionCall, callerUnit)){
							Object o = getFunctionDeclaration(functionCall, calleeUnit);
							if(o != null){
								VariableDeclaration d = null;
								if(o instanceof VariableDeclaration){
									d = (VariableDeclaration)((VariableDeclaration)o).clone();
								}
								if(o instanceof Procedure){
									Procedure p = (Procedure)o;
									d = new VariableDeclaration(p.getReturnType(), (Declarator)p.getDeclarator().clone());
								}

								if(d.getNumDeclarators() != 1 && !(d.getDeclarator(0) instanceof ProcedureDeclarator))
									System.err.println("unexpected..needs to be fixed in InlineExpansion.java ... 4");
								
								ProcedureDeclarator pd = (ProcedureDeclarator)d.getDeclarator(0);
								List<Declaration> p = pd.getParameters();
								
					CHECK_PARAMS: for (int l = 0; l < p.size(); l++) {
									if(!(p.get(l) instanceof VariableDeclaration))
										System.err.println("unexpected .. needs to be fixed in InlineExpansion.java ... 5");
										
									List<Specifier> specs = ((VariableDeclaration)p.get(l)).getSpecifiers();
									Vector<UserSpecifier> toBeResolved = new Vector<UserSpecifier>();
									for (int m = 0; m < specs.size(); m++) {
										if(specs.get(m) instanceof UserSpecifier){
											toBeResolved.addElement((UserSpecifier)specs.get(m));
										}
									}
										
									while(!toBeResolved.isEmpty()){
										UserSpecifier userSpec = toBeResolved.remove(0);
										IDExpression userSpecName = userSpec.getIDExpression();
										Declaration original = null;
										// check if it is in the callee's symbol table
										if((original = calleeUnit.findSymbol(userSpecName)) != null){
											
											// we might have already decided to add this declaration to the caller's symbol table
											if(addedGlobalDeclarations.contains(original))
												continue;
											
											// it might be a typedef
											if(original instanceof VariableDeclaration){
												VariableDeclaration vd = (VariableDeclaration)original;
												Vector<UserSpecifier> us = new Vector<UserSpecifier>();
												boolean isTypedef = false;
												for (Specifier spec : vd.getSpecifiers()) {
													if(spec.equals(Specifier.TYPEDEF)){
														isTypedef = true;
													}
													else if(spec instanceof UserSpecifier){
														us.addElement((UserSpecifier)spec);
													}
												}
												if(isTypedef){
													// check the declaration in the caller's symbol table
													Declaration callerDecl = null;
													boolean addDeclaration = true;
													if((callerDecl = callerUnit.findSymbol(userSpecName)) != null){
														// if we have a declaration with this name but it doesn't match we got a name conflict, we should give up and not online
														if(callerDecl instanceof VariableDeclaration && callerDecl.toString().equals(original.toString())){
//															// if it matches we got to remove the original one if it is in the .c file and add the new one in the beginning
//															if(callerUnit.getTable().containsKey(userSpecName))
//																removedGlobalDeclarations.addElement(callerDecl);
//															else
															addDeclaration = false;
														}
														else{
															if(makeLog){
																System.out.println("function \'" + function.getName().toString() +  "\' in the following function call inside function \'" + proc.getName().toString() + 
																		"\' can't be inlined because the declaration of one of the non-inlined functions in it contains parameter(s) whose type " +
																		"clashes with type(s) in the caller translation unit, so we can't inline");
																System.out.println(statementWithFunctionCall.toString());
												    		}	
															inline = false;
															break CHECK_PARAMS;
														}
													}
													
													if(addDeclaration){
														// the declaration should be added
														addedGlobalDeclarations.addElement(vd);
														// if it involves other user specified types, we need to deal with those too
														toBeResolved.addAll(us);
													}	
												}
												else{
													System.err.println("handle this case..in InlineExpansion.java ... 6");
												}
												
											}
											// it might be a struct
											else if(original instanceof ClassDeclaration){
												ClassDeclaration cd = (ClassDeclaration)original;
												// check if the caller already has such declaration
												Declaration callerDecl = null;
												boolean addDeclaration = true;
												if((callerDecl = callerUnit.findSymbol(userSpecName)) != null){
													// if they don't match, we got a name conflict, we should give up and not inline
													if(callerDecl instanceof ClassDeclaration && callerDecl.toString().equals(original.toString())){
//														// if it matches and is in the .c file we got to remove the original one and add the new one in the beginning
//														if(callerUnit.getTable().containsKey(userSpecName))
//															removedGlobalDeclarations.addElement(callerDecl);
//														else
														addDeclaration = false;
													}
													else{
														if(makeLog){
															System.out.println("function \'" + function.getName().toString() +  "\' in the following function call inside function \'" + proc.getName().toString() + 
																	"\' can't be inlined because the declaration of one of the non-inlined functions in it contains parameter(s) whose type " +
																	"clashes with type(s) in the caller translation unit, so we can't inline");
															System.out.println(statementWithFunctionCall.toString());
											    		}	
														inline = false;
														break CHECK_PARAMS;
													}
												}
												
												if(addDeclaration){
													// the declaration should be added
													addedGlobalDeclarations.addElement(cd);
	
													// if declarations inside the struct use other structs or typedefs, handle them here
													for (Traversable traversable : cd.getChildren()) {
														if(traversable instanceof DeclarationStatement){
															DeclarationStatement ds = (DeclarationStatement)traversable;
															if(!(ds.getDeclaration() instanceof VariableDeclaration))
																System.err.println("not handled ... handle in InlineExpression.java ... 7");
															
															VariableDeclaration vd = (VariableDeclaration)ds.getDeclaration();
															for (Specifier spec : vd.getSpecifiers()) {
																if(spec instanceof UserSpecifier){
																	toBeResolved.addElement((UserSpecifier)spec);
																}
															}
														}
														else{
															System.err.println("not handled ... handle in InlineExpansion.java ... 8");
														}	
													}
												}	
											}
										}
										else{
											if(makeLog){
												System.out.println("function \'" + function.getName().toString() +  "\' in the following function call inside function \'" + proc.getName().toString() + 
														"\' can't be inlined because the declaration of one of the non-inlined functions in it contains parameter(s) whose type is unknown, so we can't extern it");
												System.out.println(statementWithFunctionCall.toString());
								    		}	
											inline = false;
											break CHECK_PARAMS;
										}
									}
								}
								if(inline){
									// add the declaration itself, in the beginning of the caller's translation unit
									callerUnit.addDeclarationFirst(d);
								}	
							}
							else{
								if(makeLog){
									System.out.println("function \'" + function.getName().toString() +  "\' in the following function call inside function \'" + proc.getName().toString() + 
											"\' can't be inlined because we couldn't find declaration of one of the non-inlined functions in it");
									System.out.println(statementWithFunctionCall.toString());
					    		}	
								inline = false;
							}
						}
					}
				}
				
				if(inline){
					
					// modify the body of the function and the translation unit with the inlined code and necessary declarations and assignment statements
					for (Declaration d : declarations) {
						if(statementWithFunctionCall instanceof DeclarationStatement) {
							enclosingCompoundStmt.addDeclarationBefore(((DeclarationStatement)statementWithFunctionCall).getDeclaration(), d);
						}
						else {
							enclosingCompoundStmt.addDeclaration(d);							
						}						
					}
					for (Statement s : statements) {
						enclosingCompoundStmt.addStatementBefore(statementWithFunctionCall, s);
					}
					for (Declaration d : removedGlobalDeclarations) {
						callerUnit.removeChild(d);
					}
					for (Declaration d : addedGlobalDeclarations) {
						// these should go to the beginning of the file, but clone them first as we haven't done that earlier
						callerUnit.addDeclarationFirst((Declaration)d.clone());
					}
					
					// deal with the return statement
					ReturnStatement returnStmt = getReturnStatement(codeToBeInlined);
					Identifier returnVarId = null;
					if(returnStmt != null){
						if(returnStmt.getExpression() != null){
							// store the return expression in new unique variable defined in the scope of the surrounding function and remove the return statement
							Expression returnExpr = (Expression)returnStmt.getExpression().clone();
							NameID returnVarNameId = getUniqueIdentifier(enclosingCompoundStmt, resultPrefix, function.getSymbolName());
							//VariableDeclarator vd = new VariableDeclarator(returnVarNameId);
              VariableDeclarator vd = new VariableDeclarator(function.getDeclarator().getSpecifiers(), returnVarNameId);
							returnVarId = new Identifier(vd);
							List<Specifier> returnType = function.getReturnType();
              returnType.removeAll(function.getDeclarator().getSpecifiers()); // removes declarator specs already included in vd
							// for a static function return type also has static, we don't want that otherwise our result variable would be
							// declared as static
							returnType.remove(Specifier.STATIC);							
							enclosingCompoundStmt.addDeclaration(new VariableDeclaration(returnType, vd));
							ExpressionStatement exprStmt = new ExpressionStatement(new AssignmentExpression(returnVarId, AssignmentOperator.NORMAL, returnExpr));
							returnStmt.swapWith(exprStmt);
						}
						else{
							NullStatement nullStmt = new NullStatement();
							returnStmt.swapWith(nullStmt);
						}
					}
				
					// add the comment
					enclosingCompoundStmt.addStatementBefore(statementWithFunctionCall, new AnnotationStatement(new CommentAnnotation("inlining function " + functionName + " in the body of function " + proc.getName().toString())));
					// include the inlined code before the statement with function call
					enclosingCompoundStmt.addStatementBefore(statementWithFunctionCall, codeToBeInlined);
					
					// deal with the function call in the statement 
					if(returnStmt != null && returnVarId != null){
						// replace the function call with the variable that holds the return value  
						fc.swapWith(returnVarId.clone());
					}
					else{ 
						// it is a void function
						// just remove the function call
						//TODO: removeChild can't be used ... do something else
						//((Statement)fc.getParent()).removeChild(fc);
						Statement original = (Statement)fc.getParent();
						// if it is just one function call (and for void function calls it should be the case)
						if(original.getChildren().size() == 1 && original.getParent() instanceof CompoundStatement){
							((CompoundStatement)original.getParent()).removeStatement(original);
						}
						else{
						    int index = Tools.indexByReference(original.getChildren(), fc);
						    if (index != -1){
						    	original.setChild(index, new NullStatement());
						    }    
						    else{
						    	System.err.println("Fix me in InlineExpansion.java, couldn't remove function call");
						    }
						}    
					}
				}	
    		}
	    }	
	}
	/**
	 * returns a unique identifier for the given compound statement based on the provided hints
	 * depending upon the fullyQualifiedName flag, this method returns names in two different formats
	 * when the flag is true, it returns "_[prefix]_[suffix]", in case of conflict it adds number to it, i.e. "_[prefix]_x[suffix]"
	 * where x is 2,3,4,...
	 * when the flag is false, it does not add local variable prefix with the suffix if it already has local variable prefix in front of it
	 * all it appends in this case is an underscore
	 * Here are the two examples:
	 * a) _local_func1__local_Func2__param_Func3_array // when fullyQualifiedName flag is set to true
	 * b) __local_Func2__param_Func3_array // when the flag is false
	 * The length of the variable name is also kept into account, to make sure it does not exceed the maximum allowed limit
	 * @param compoundStmt - the compound statement (may be function body) for which unique identifier is to be sought
	 * @param prefix - prefix hint
	 * @param suffix - suffix hint
	 * @return - the unique identifier
	 */
	private NameID getUniqueIdentifier(CompoundStatement compoundStmt, String prefix, String suffix){
		int i = 1;
		String newName = "_" + prefix + "_" + suffix;
		if(!fullyQualifiedName && prefix.startsWith(localVarPrefix) && suffix.indexOf(localVarPrefix) != -1)
			newName = "_" + suffix;
		newName = adjustLength(newName);
		NameID id = new NameID(newName);
		while(compoundStmt.findSymbol(id) != null){
			newName = "_" + prefix + "_" + (++i) + "_" + suffix;
			if(!fullyQualifiedName && prefix.startsWith(localVarPrefix) && suffix.indexOf(localVarPrefix) != -1)
				newName = "_" + i + suffix;
			newName = adjustLength(newName);
			id = new NameID(newName);
		}
		return id;
	}
	/**
	 * returns a unique label for the given function based on the provided hints
	 * depending upon the fullyQualifiedName flag, this method returns names in two different formats
	 * when the flag is true, it returns "_[prefix]_[suffix]", in case of conflict it adds number to it, i.e. "_[prefix]_x[suffix]"
	 * where x is 2,3,4,...
	 * when the flag is false, it does not add label prefix with the suffix if it already has label prefix in front of it
	 * all it appends in this case is an underscore
	 * Here are the two examples:
	 * The length of the variable name is also kept into account, to make sure it does not exceed the maximum allowed limit
	 * @param proc - the function
	 * @param prefix - prefix hint
	 * @param suffix - suffix hint
	 * @return - the unique identifier
	 */
	private NameID getUniqueLabel(Procedure proc, String prefix, String suffix){
		int i = 1;
		String newName = "_" + prefix + "_" + suffix;
		if(!fullyQualifiedName && suffix.startsWith(labelPrefix))
			newName = "_" + suffix;
		newName = adjustLength(newName);
		NameID newLabel = new NameID(newName);
		CompoundStatement body = proc.getBody();
		DepthFirstIterator dfi = new DepthFirstIterator(body);
		Vector<NameID> labels = new Vector<NameID>();
		Object next = null;
		while(dfi.hasNext()){
			if((next = dfi.next()) instanceof Label)
				labels.addElement((NameID)((Label)next).getName());
		}
		while(labels.contains(newLabel)){
			newName = "_" + prefix + "_" + (++i) + "_" + suffix;
			if(!fullyQualifiedName && prefix.startsWith(localVarPrefix) && suffix.indexOf(localVarPrefix) != -1)
				newName = "_" + i + suffix;
			newName = adjustLength(newName);
			newLabel = new NameID(newName);
		}
		return newLabel;
	}
	/**
	 * makes sure that the given name is not longer than maximum allowed length. If it is, it is renamed. If there are underscores
	 * in the beginning of the name (e.g. when the fullyQualifiedName flag is false), they are removed and a string "_t_" is appended
	 * to mean that the name has been truncated. Otherwise the last half is chopped off.  
	 * @param name - name of a variable
	 * @return - variable name after making sure it is not longer than the maximum allowed length, returned name may be different from the 
	 *           one passed
	 */
	private String adjustLength(String name) {
		if(name.length() < maxVarLength)
			return name;
		for (int i = 0; i < name.length(); i++) {
			if(Character.isLetter(name.charAt(i))){
				String temp = name.substring(i);
				if(temp.length() + 3 < maxVarLength)
					return "_t_" + temp;
				else
					break;
			}	
		}
		return name.substring(0, maxVarLength/2);
	}
	/**
	 * returns the return statement in the body of a function specified by the compound statement
	 * Note: shouldn't it be moved to Procedure class?
	 * @param functionBody - body of the function 
	 * @return - the return statement or null if there is none
	 */
	private ReturnStatement getReturnStatement(CompoundStatement functionBody){
		DepthFirstIterator dfi = new DepthFirstIterator(functionBody);
		Object next = null;
		while(dfi.hasNext()){
			if( (next = dfi.next()) instanceof ReturnStatement){
				return (ReturnStatement)next;
			}
		}
		return null;
	}
	/**
	 * replaces variable names, also removes the old symbols from the symbol table
	 * @param functionBody - the function body
	 * @param oldVars - list of old variables (IDEXpression instances)
	 * @param newVarNames - list of new variable names
	 * @param addVarsInTable - adds the new variables in the symbol table if this argument is true
	 * @param args - map containing actual ArrayAccess instances
	 */
	private void replaceVariableNames(CompoundStatement functionBody, List<IDExpression> oldVars, List<String> newVarNames, boolean addVarsInTable, HashMap<String, ArrayAccess> args){
		if(oldVars.size() == newVarNames.size()){
			for (int i = 0; i < oldVars.size(); i++) {
//				Tools.renameSymbol(functionBody, oldVars.get(i).toString(), newVarNames.get(i));
//				if(true)
//					continue;
//				Declaration d = functionBody.getTable().remove(oldVars.get(i));
				String oldVarName = oldVars.get(i).toString();//getSymbol().getSymbolName(); in case of label symbol is null, so use toString instead
				DepthFirstIterator dfi = new DepthFirstIterator(functionBody);
				Object next = null;
				while(dfi.hasNext()){
					if( (next = dfi.next()) instanceof IDExpression){
						if(((IDExpression)next).getName().equals(oldVarName)){
							Traversable t = ((IDExpression)next).getParent();
							try{
								if(t instanceof Symbol){
									//renameVarInDeclarator((Declarator)t, (Identifier)next, newVarNames.get(i), args);
									((Symbol)t).setName(newVarNames.get(i));
								}
								else if(t != null){
									int index = Tools.indexByReference(t.getChildren(), next);
									if(index != -1){
										if(newVarNames.get(i).indexOf('[') >= 0 && t instanceof ArrayAccess){
											List<Expression> indices = ((ArrayAccess)t).getIndices();
											ArrayAccess original = args.get(newVarNames.get(i));
											if(original != null){
												List<Expression> callerIndices = original.getIndices();
												List<Expression> newIndices = new ArrayList<Expression>();
												for(Expression ind : callerIndices){
													newIndices.add(ind.clone());
												}
												for(Expression ind : indices){
													ind.setParent(null);
													newIndices.add(ind);
												}
												((ArrayAccess)t).setIndices(newIndices);
												t.setChild(index, original.getArrayName().clone());
											}
											else{
												//TODO: use of setChild is dangerous and NameID won't make into symbol table
												t.setChild(index, new NameID(newVarNames.get(i)));
											}	
										}
										else{
											//TODO: use of setChild is dangerous and NameID won't make into symbol table
											t.setChild(index, new NameID(newVarNames.get(i)));
										}
									}
									else{
										System.out.println("name could not be changed ... check it");
									}
								}	
							}
							catch(UnsupportedOperationException ex){
								System.out.println("handle me in Inliner...." + t.getClass().getName() + "does not support setChild()");
							}
						}	
					}
					// also change the variable names inside annotations
					else if(next instanceof Annotatable){
						// first deal with the labels
						if(next instanceof Label){
							if(((Label)next).getName().getName().equals(oldVarName)){
								((Label)next).setName(new NameID(newVarNames.get(i)));
							}
						}
						// change the labels in the goto statements
						else if(next instanceof GotoStatement){
							Expression exp = ((GotoStatement)next).getExpression();
							if(exp != null && exp.toString() != null && exp.toString().equals(oldVarName)){
								((GotoStatement)next).setExpression(new NameID(newVarNames.get(i)));
							}
						}
						// modify the annotations as well
						List<Annotation> annotations = ((Annotatable)next).getAnnotations();
						if(annotations != null){
							for(Annotation a : annotations){
								Iterator<String> iter = a.keySet().iterator();
								while(iter.hasNext()){
									String key = iter.next();
									Object val = a.get(key);
									
									if(val instanceof String && ((String)val).equals(oldVarName)){
										// replace the value
										a.put(key, newVarNames.get(i));
									}
									else if(val instanceof Collection){
										replaceNameInCollection((Collection)val, oldVarName, newVarNames.get(i));
									}
									else if(val instanceof Map){
										replaceNameInMap((Map)val, oldVarName, newVarNames.get(i));
									}
								}
							}	
						}
					}
				}
			}
			if(addVarsInTable){
				DepthFirstIterator dfi = new DepthFirstIterator(functionBody);
				Object next = null;
				while(dfi.hasNext()){
					if( (next = dfi.next()) instanceof Declaration){
						SymbolTools.addSymbols(functionBody, (Declaration)next);
					}
				}
			}
		}
	}
	// Following is not needed, now we can call setName() on symbols
//	/**
//	 * renames variable in the given declarator
//	 * @param d - the declarator
//	 * @param id - identifier for the variable that needs to be renamed
//	 * @param newVarName - new name for the variable
//	 * @param args - 
//	 */
//	private void renameVarInDeclarator(Declarator d, Identifier id, String newVarName, Map<String, ArrayAccess>args){
//		if(d instanceof VariableDeclarator){
//			Declarator oldD = (VariableDeclarator)d;
//			Declarator newD = new VariableDeclarator(oldD.getSpecifiers(), new Identifier(newVarName), oldD.getArraySpecifiers());
//			newD.setInitializer(oldD.getInitializer());
//			Traversable t;
//			while( (t = oldD.getParent()) instanceof NestedDeclarator){
//				oldD = (NestedDeclarator)t;
//				newD = new NestedDeclarator(oldD.getSpecifiers(), newD, oldD.getParameters(), oldD.getArraySpecifiers());
//			}
//			int index = Tools.indexByReference(t.getChildren(), oldD);
//			SymbolTable table = getSymbolTable(t);
//			if(table != null){
//				if(index != -1 && t instanceof Declaration){
//					SymbolTools.removeSymbols(table, (Declaration)t);
//					t.setChild(index, newD);
//					SymbolTools.addSymbols(table, (Declaration)t);
//				}
//				
//			}
//			else{
//				System.out.println("fix me in inliner");
//			}
//		}
//	}
	
	private SymbolTable getSymbolTable(Traversable t){
		while(t != null && !(t instanceof SymbolTable))
			t = t.getParent();
		
		if(t instanceof SymbolTable)
			return (SymbolTable)t;
		
		return null;
	}
	
	/**
	 * replaces the given old name in the given collection with the new name, if the values are themselves collections or maps 
	 * recursively handles them also
	 */
	private void replaceNameInCollection(Collection c, String name, String newName){
		Iterator iter = ((Collection)c).iterator();
		while(iter.hasNext()){
			Object val = iter.next();
			if(val instanceof String && ((String)val).equals(name)){
				c.remove(val);
				c.add(newName);
				break;
			}
			else if(val instanceof Collection){
				replaceNameInCollection((Collection)val, name, newName);
			}
			else if(val instanceof Map){
				replaceNameInMap((Map)val, name, newName);
			}
		}
	}
	/**
	 * replaces the given old name in the values of given map with the new name, if the values are themselves collections or maps 
	 * recursively handles them also
	 */
	private void replaceNameInMap(Map m, String name, String newName){
		Iterator<String> iter = m.keySet().iterator();
		while(iter.hasNext()){
			String key = iter.next();
			Object val = m.get(key);
			if(val instanceof String && ((String)val).equals(name)){
				// replace the value
				m.put(key, newName);
			}
			else if(val instanceof Collection){
				replaceNameInCollection((Collection)val, name, newName);
			}
			else if(val instanceof Map){
				replaceNameInMap((Map)val, name, newName);
			}
		}
	}
	/**
	 * For a given function call, this method returns the overall statement that encloses this call 
	 * @param call - the function call
	 * @return - the maximally complete statement which encloses the given function call
	 */
	private Statement getStatement(FunctionCall call){
		Traversable t = call;
		Traversable parent;
		while( (parent = t.getParent()) != null && !(parent instanceof CompoundStatement)){
			t = parent;
		}
		return (Statement)t;
	}
	/**
	 * For a given statement, this method returns the enclosing compound statement 
	 * @param stmt - the given statement
	 * @return - the enclosing compound statement (can be same as the compound statement in the function or may be another compound statement in it)
	 */
	private CompoundStatement getEnclosingCompoundStmt(Statement stmt){
		Traversable t = stmt;
		while( t != null && !(t instanceof CompoundStatement)){
			t = t.getParent();
		}
		return (CompoundStatement)t;
	}
	/**
	 * returns the first non-declaration statement in the given compound statement (which may be function body)
	 * @param compoundStmt - the compound statement (which may be function body) 
	 * @return - first non-declaration statement, or null if it can't find one
	 */
	private Statement getFirstNonDeclarationStatement(CompoundStatement compoundStmt) {
		FlatIterator fi = new FlatIterator(compoundStmt);
		while(fi.hasNext()){
			Object next = fi.next();
			if(next instanceof Statement && !(next instanceof DeclarationStatement || next instanceof AnnotationStatement) )
				return (Statement)next;
		}
		return null;
	}
	/**
	 * Tells if the called function can be inlined.
	 * Currently it makes sure that the number of actual and formal parameters match and that function pointers and static variables are not used.  
	 * This method does not check for any recursion resulted from the statements inside the passed procedure
	 * That should be checked by the call graph in the calling method.
	 * @param functionToBeInlined - the function to be inlined
	 * @param fc - the function call
	 * @param enclosingFunction - the function in whose code the called function would be inlined in
	 * @return - true if the function can be inlined, false otherwise (it will later be changed to return the cost)
	 */
	private boolean canInline(Procedure functionToBeInlined, FunctionCall fc, Procedure enclosingFunction){
		// check the number of parameters
		if(fc.getNumArguments() != functionToBeInlined.getNumParameters()){
			// handle the foo(void){} case
			if(fc.getNumArguments() == 0 && functionToBeInlined.getNumParameters() == 1){
				// List params would be null, we should get the children from the declarator, second would be the specifier 
				Declarator d = functionToBeInlined.getDeclarator();
				List c = d.getChildren();
				if(c.size() > 1){
					Object o = c.get(1);
					if(o instanceof VariableDeclaration){
						List s = ((VariableDeclaration)o).getSpecifiers();
						if(s.size() == 1 && s.get(0).equals(Specifier.VOID))
							return true;
					}
				}	
			}
			if(makeLog){
				System.out.println("number of actual and formal parameters in calling " + functionToBeInlined.getName().toString() + " function in the following statement does not match so we are not inlining it");
				System.out.println(fc.toString());
			}	
			return false;
		}
		// check every parameter independently and return false in case of function pointers
		//TODO: Tools.getExpressionType() can be used to evaluate the type of expressions passed as actual parameters
		// however I'm not sure how complete this function is, so I'm not using it right now.
		// Moreover this is just an extra check as we are already assuming that the code is already compiled correctly before inlining.

		// check if the called procedure accepts function pointers, we won't inline such functions
		List params = functionToBeInlined.getParameters();
		for (int i = 0; i < params.size(); i++) {
			Declaration d = (Declaration)params.get(i);
			List<Traversable> children = d.getChildren();
			for (int j = 0; j < children.size(); j++) {
				if(children.get(j) instanceof NestedDeclarator){
					if(((NestedDeclarator)children.get(j)).isProcedure()){
						if(makeLog){
							System.out.println(functionToBeInlined.getName().toString() + " function in the following statement accepts function pointer(s) so we are not inlining it");
							System.out.println(fc.toString());
						}	
						return false;
					}	
				}
			}
		}
		// check if the called procedure has any static local variables, we won't inline such functions
		Collection<Declaration> localVars = functionToBeInlined.getBody().getDeclarations();
		for(Declaration local : localVars){
			if(local instanceof VariableDeclaration){
				List<Specifier> specs = ((VariableDeclaration)local).getSpecifiers();
				for(Specifier spec : specs){
					if(spec.equals(Specifier.STATIC)){
						if(makeLog){
							System.out.println(functionToBeInlined.getName().toString() + " function in the following statement in function \'" + enclosingFunction.getName() + "\' has static local variable(s) so we are not inlining it");
							System.out.println(fc.toString());
						}	
						return false;
					}
				}
			}	
		}
		// Make sure that the function to be inlined does not use any static external variables
		// but if the function to be inlined is in the same file as the function in whose code it is inlined then allow it
		// would be a problem if they are declared after they are used, can't extern static variables declared later.
		if(!getTranslationUnit(functionToBeInlined).equals(getTranslationUnit(enclosingFunction))){
			List<IDExpression> staticVars = getExternalStaticVars(functionToBeInlined);
			Set<Symbol> symbols_locals = functionToBeInlined.getBody().getSymbols();
			Set<Symbol> symbols_params = functionToBeInlined.getSymbols();
			Set<String> locals = new HashSet<String>(symbols_locals.size());
			Set<String> functionParams = new HashSet<String>(symbols_params.size());
			for(Symbol s : symbols_locals) {
				locals.add(s.getSymbolName());
			}
			for(Symbol s : symbols_params) {
				functionParams.add(s.getSymbolName());
			}
			
			if(staticVars.size() > 0){
				DepthFirstIterator dfi = new DepthFirstIterator(functionToBeInlined.getBody());
				Object next = null;
				HashMap<String, String> checked = new HashMap<String, String>();
				while(dfi.hasNext()){
					if((next = dfi.next()) instanceof IDExpression){
						String staticVar = ((IDExpression)next).getName();
						if(!checked.containsKey(staticVar) && !locals.contains(((IDExpression)next).toString()) && !functionParams.contains(((IDExpression)next).toString())){
							checked.put(staticVar, null);
							for (int i = 0; i < staticVars.size(); i++) {
								try{
									if(staticVars.get(i).toString().equals(staticVar)){
										if(makeLog){
											System.out.println(functionToBeInlined.getName().toString() + " function in the following statement in function \'" + enclosingFunction.getName() + "\' uses external static variable(s) so we are not inlining it");
											System.out.println(fc.toString());
										}	
										return false;
									}	
								}
								catch(NullPointerException ex){
									System.err.println("Fix me in inline code..");
								}
							}
						}
					}
				}
			}
		}	
		return true;
	}
	/**
	 * returns the program the given procedure belongs to
	 * @param proc - the procedure
	 */
	private Program getProgram(Procedure proc) {
		Traversable t = proc;
		while(t != null){
			if(t instanceof Program)
				return (Program)t;
			t = t.getParent();
		}
		return null;
	}
	/**
	 * returns the translation unit the given procedure belongs to
	 * @param proc - the procedure
	 */
	private TranslationUnit getTranslationUnit(Procedure proc) {
		Traversable t = proc;
		while(t != null){
			if(t instanceof TranslationUnit)
				return (TranslationUnit)t;
			t = t.getParent();
		}
		return null;
	}
	/**
	 * Given a function, this method returns a list of static external variables found in the file of the function
	 * @param proc - the procedure of whose file we are looking into to find static external variables
	 * @return - list of static external variables
	 */
	private List<IDExpression> getExternalStaticVars(Procedure proc) {
		TranslationUnit tUnit = getTranslationUnit(proc);
		List<IDExpression> staticVars = new ArrayList<IDExpression>();
		if(tUnit != null){
			Set<Declaration> declarations = tUnit.getDeclarations();
			for(Declaration d : declarations){
				if(d instanceof VariableDeclaration){
					VariableDeclaration vd = (VariableDeclaration)d;
					List<Specifier> specs = vd.getSpecifiers();
					for (int l = 0; l < specs.size(); l++) {
						if(specs.get(l).equals(Specifier.STATIC)){
							staticVars.addAll(d.getDeclaredIDs());
						}	
					}
				}
				
			}
		}
		return staticVars;
	}
	/**
	 * For a given function this method returns a map of those global variables and their declarations that are used in 
	 * the body of the function
	 * 
	 * @param function - the function
	 * @param codeToBeInlined - body of the function in which we are trying to find the use of global variables
	 *                          note: this may be modified inlined code
	 * @param newLocals - list of new variables introduced as local variables by inlining code
	 * @param newParamNames - list of new variables introduced to replace parameters by the inlining code
	 * @return - map of global variables and their declarations
	 */
	private Map<IDExpression, Declaration> getUsedGlobalVariables(Procedure function, CompoundStatement codeToBeInlined, List<String> newLocals, List<String> newParamNames) {
		Map<IDExpression, Declaration> usedGlobalVars = new HashMap<IDExpression, Declaration> ();
		TranslationUnit tUnit = getTranslationUnit(function);
		List<SymbolTable> tables = getAllSymbolTables(tUnit);
		DepthFirstIterator dfi = new DepthFirstIterator(codeToBeInlined);
		Vector<String> functions = new Vector<String>();
		// go through the body of the function and gather all identifiers that may be global variables
		Vector<IDExpression> globalVars = new Vector<IDExpression>(); 
		Object next = null;
		while(dfi.hasNext()){
			next = dfi.next();
			if(next instanceof FunctionCall)
				functions.addElement(((FunctionCall)next).getName().toString());
			
			if(next instanceof IDExpression){
				String id = ((IDExpression)next).getName();
				if(!newLocals.contains(id) && !newParamNames.contains(id) && !functions.contains(id)){
					globalVars.addElement((IDExpression)next);
				}
			}
		}
		// for each guessed global variable, find out if it is really a global variable, if it is put it in the map to be returned 
		for (int i = 0; i < globalVars.size(); i++) {
			IDExpression id = globalVars.elementAt(i);
			for (int j = 0; j < tables.size(); j++) {
				Set<Symbol> symbols = tables.get(j).getSymbols();
				Set<Declaration> declarations = tables.get(j).getDeclarations();
				for (Symbol s : symbols) {
					if(s.getSymbolName().equals(id.getName())) {
						for (Declaration d : declarations) {
							if(d.equals(s.getDeclaration()))
								usedGlobalVars.put(id, d);
						}	
					}	
				}
			}
		}
		return usedGlobalVars;
	}
	/**
	 * For the given translation unit, this method returns all symbol tables (including the given, as well as the parent)
	 */
	private List<SymbolTable> getAllSymbolTables(TranslationUnit tUnit) {
		List<SymbolTable> tables = new ArrayList<SymbolTable>();
		Traversable t = tUnit;
		while(t != null){
			if(t instanceof SymbolTable)
				tables.add((SymbolTable)t);
			
			t = t.getParent();
		}
		return tables;
	}
	/**
	 * A variable declaration can have many declarators, this method returns the one which involves the given identifier
	 * @param declaration - the variable declaration
	 * @param id - the identifier we are interested in
	 * @return - the declarator involving the given identifier, null if there is none
	 */
	private VariableDeclarator getDeclarator(VariableDeclaration declaration, IDExpression id){
		int n = declaration.getNumDeclarators();
		for (int i = 0; i < n; i++) {
			VariableDeclarator d = (VariableDeclarator)declaration.getDeclarator(i);
			if(id.getName().equals(d.getSymbolName())) // but usually direct declarator is empty so we need to find the id in children
				return d;
			List<Traversable> children = d.getChildren();
			for (int j = 0; j < children.size(); j++) {
				if(children.get(j) instanceof IDExpression && id.getName().equals(((IDExpression)children.get(j)).getName())){
					return d;
				}
			}
		}
		return null;
	}
	/**
	 * tells if the body of the function supplied, has multiple return statements or not
	 * @param functionBody - compound statement representing function body
	 * @return - true if the function body has multiple return statements in it, false if it has one or zero
	 */
	private boolean hasMultipleReturnStmts(CompoundStatement functionBody){
        DepthFirstIterator dfi = new DepthFirstIterator(functionBody);
        boolean returnStatement = false;
        while(dfi.hasNext()){
        	Object obj = dfi.next();
        	if(obj instanceof ReturnStatement){
        		if(returnStatement)
        			return true;
        		returnStatement = true;
        	}
        }
        return false;
	}
	/**
	 * returns a list containing function declarations and definitions visible in the specified translation unit
	 * @param tUnit - the translation unit
	 * @return - list of visible function declarations and definitions
	 */
	private List getAvailableFunctions(TranslationUnit tUnit){
		List functions = new LinkedList();
		FlatIterator fi = new FlatIterator(tUnit);
		Object obj = null;
		while(fi.hasNext()){
			obj = fi.next();
			
			if(obj instanceof VariableDeclaration){
				VariableDeclaration d = (VariableDeclaration)obj;
				for(int i = 0; i < d.getNumDeclarators(); i++){
					if(d.getDeclarator(i) instanceof ProcedureDeclarator){
						functions.add(d);
						break;
					}
				}
			}
			if(obj instanceof Procedure){
				functions.add(obj);
			}
			
		}
		return functions;
	}
	/**
	 * tells if the function declaration/definition of the function called in the specified function call is available in 
	 * the provided translation unit
	 * @param fc - the function call
	 * @param tUnit - the translation unit
	 * @return - true if the declaration/definition of the function is available/visible, false otherwise
	 */
	private boolean isDeclarationAvailable(FunctionCall fc, TranslationUnit tUnit){
		return getFunctionDeclaration(fc, tUnit) != null;
	}
	
	/**
	 * returns the declaration/definition of the function, called in the passed function call, if it is available in the 
	 * provided translation unit, null otherwise
	 * @param fc - the function call
	 * @param tUnit - the translation unit
	 * @return - declaration/definition of the function if it is available in the passed translation unit, null otherwise
	 */
	private Object getFunctionDeclaration(FunctionCall fc, TranslationUnit tUnit){
		String name = fc.getName().toString();
		// check if we have the function or its declaration
		List availableFuncs = getAvailableFunctions(tUnit);
		boolean declared = false;
		for (int k = 0; k < availableFuncs.size(); k++) {
			Object o = availableFuncs.get(k);
			if(o instanceof Procedure){
				// NOTE: we are only comparing function names (good enough for C, as no function overriding)
				if(((Procedure)o).getName().equals(name)){
					return o;
				}
			}
			else if(o instanceof VariableDeclaration){
				VariableDeclaration v = (VariableDeclaration)o;
				for (int l = 0; l < v.getNumDeclarators(); l++) {
					if(v.getDeclarator(l) instanceof ProcedureDeclarator){
						if(((ProcedureDeclarator)v.getDeclarator(l)).getSymbolName().equals(name)){
							return v;
						}
					}
				}
			}
		}
		return null;
	}
//	/**
//	 * returns a list containing all typedefs for the given translation unit
//	 * @param tUnit - the translation unit
//	 * @return - list containing all typedefs
//	 */
//	private List<VariableDeclaration> getTypedefs(TranslationUnit tUnit){
//		List<VariableDeclaration> typedefs = new LinkedList<VariableDeclaration>();
//		FlatIterator fi = new FlatIterator(tUnit);
//		Object obj = null;
//		while(fi.hasNext()){
//			obj = fi.next();
//			
//			if(obj instanceof VariableDeclaration){
//				VariableDeclaration d = (VariableDeclaration)obj;
//				List<Specifier> specs = d.getSpecifiers();
//				for(int i = 0; i < specs.size(); i++){
//					if(Specifier.TYPEDEF.equals(specs.get(i))){
//						typedefs.add(d);
//						break;
//					}
//				}
//			}
//		}
//		return typedefs;
//	}
}
