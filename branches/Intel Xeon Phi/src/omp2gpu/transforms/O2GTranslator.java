package omp2gpu.transforms;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Set;

import omp2gpu.analysis.AnalysisTools;
import omp2gpu.hir.CUDASpecifier;
import omp2gpu.hir.CudaAnnotation;
import omp2gpu.hir.CudaStdLibrary;
import omp2gpu.hir.Dim3Specifier;
import omp2gpu.hir.KernelFunctionCall;
import omp2gpu.hir.TextureSpecifier;
import cetus.analysis.LoopTools;
import cetus.exec.Driver;
import cetus.hir.Annotatable;
import cetus.hir.Annotation;
import cetus.hir.AnnotationDeclaration;
import cetus.hir.AnnotationStatement;
import cetus.hir.ArrayAccess;
import cetus.hir.ArraySpecifier;
import cetus.hir.AssignmentExpression;
import cetus.hir.AssignmentOperator;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.BreadthFirstIterator;
import cetus.hir.ChainedList;
import cetus.hir.CodeAnnotation;
import cetus.hir.CommentAnnotation;
import cetus.hir.CompoundStatement;
import cetus.hir.DataFlowTools;
import cetus.hir.Declaration;
import cetus.hir.DeclarationStatement;
import cetus.hir.Declarator;
import cetus.hir.DepthFirstIterator;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.FlatIterator;
import cetus.hir.FloatLiteral;
import cetus.hir.ForLoop;
import cetus.hir.FunctionCall;
import cetus.hir.IDExpression;
import cetus.hir.IRTools;
import cetus.hir.Identifier;
import cetus.hir.IfStatement;
import cetus.hir.Initializer;
import cetus.hir.IntegerLiteral;
import cetus.hir.Literal;
import cetus.hir.Loop;
import cetus.hir.MinMaxExpression;
import cetus.hir.NameID;
import cetus.hir.OmpAnnotation;
import cetus.hir.PointerSpecifier;
import cetus.hir.PrintTools;
import cetus.hir.Procedure;
import cetus.hir.ProcedureDeclarator;
import cetus.hir.Program;
import cetus.hir.ReturnStatement;
import cetus.hir.SizeofExpression;
import cetus.hir.Specifier;
import cetus.hir.StandardLibrary;
import cetus.hir.Statement;
import cetus.hir.StringLiteral;
import cetus.hir.Symbol;
import cetus.hir.SymbolTable;
import cetus.hir.SymbolTools;
import cetus.hir.Symbolic;
import cetus.hir.Tools;
import cetus.hir.TranslationUnit;
import cetus.hir.Traversable;
import cetus.hir.Typecast;
import cetus.hir.UnaryExpression;
import cetus.hir.UnaryOperator;
import cetus.hir.VariableDeclaration;
import cetus.hir.VariableDeclarator;

/**
 * @author Seyong Lee <lee222@purdue.edu>
 *         ParaMount Group
 *         School of ECE, Purdue University
 *
 * OpenMP-to-GPGPU Translator, which performs OpenMP to CUDA-GPGPU translation.
 * Extracts a parallel region (a CompoundStatement or a ForLoop) and converts to a kernel
 * function. Assumes the SingleDeclarator pass and OmpAnalysis pass have been
 * run on the program.
 */
public class O2GTranslator {
	private static String pass_name = "[O2GTranslator]";
	private static Program program;
	private static TranslationUnit main_TrUnt;
	private static Procedure main;
	private static Declaration lastMainCudaDecl; // last cuda-related declaration in
	                                         // the main translation unit.
	private static Statement firstMainStmt;  // The first statement in main procedure;
	private static List<Statement> FlushStmtList; //List of the last flush statement
	private static int defaultBlockSize = 256;
	private static int maxGridDimSize = 65535; //Max. dim. size of Grid = 65535
	private static int defaultGridDimSize = 10000;
	private static int defaultGMemSize = 1600000000;
	private static int defaultSMemSize = 16384;
	private static String nvccVersion = "1.1";
	private static int MemTrOptLevel = 2;
	private static int cudaMallocOptLevel = 0;
	private static String tuningParamFile = null;
	/*
	 * Counter of cuda-related statements, which are inserted by this translator,
	 * in the parent CompoundStatement of each target kernel region statement.
	 */
	private static int num_cudastmts = 0;

	private static boolean opt_MallocPitch = false;
	private static boolean opt_MatrixTranspose = false;
	private static boolean opt_LoopCollapse = false;
	private static boolean opt_UnrollingOnReduction = false;
	private static boolean opt_addSafetyCheckingCode = false;
	private static boolean opt_addCudaErrorCheckingCode = false;
	private static boolean opt_forceSyncKernelCall = false;
	private static boolean opt_shrdSclrCachingOnSM = false;
	private static boolean opt_globalGMalloc = false;
	///////////////////////////////////////////
	// Below variables are not used for now. //
	///////////////////////////////////////////
	//private static boolean opt_ParallelLoopSwap = false;
	//private static boolean assumeNonZeroTripLoops = false;
	//private static boolean opt_shrdSclrCachingOnReg = false;
	//private static boolean opt_shrdArryElmtCachigOnReg = false;
	//private static boolean opt_prvtArryCachingOnSM = false;
	//private static boolean opt_shrdArryCachingOnTM = false;

	/*
	 * HashMaps used to keep information about data transfered between CPU and GPU,
	 * which will be used to identify duplicate memory transfers.
	 */
	private static HashMap c2gMemTr = null;
	private static HashMap g2cMemTr = null;
	private static HashMap c2gMap = null;

	private static HashMap<Symbol, VariableDeclaration> pitchMap = null;
	/////////////////////////////////////////////////////////////////////////////
	// HashSet containing kernel function call statements which are results of //
	// kernel region transformation.                                           //
	/////////////////////////////////////////////////////////////////////////////
	private static HashSet<Annotatable> kernelCallStmtSet = null;
	/*
	 * HashMaps containing share/threadprivate variable to its local variable mapping.
	 * This information is needed for correct transformation of functions called within
	 * a kernel function.
	 */
	private static HashMap<Identifier, Expression> tempMap = null;

	/*
	 * This HashMap keeps information of reduction variables used in each parallel region.
	 */
	private static HashMap<Statement, HashSet> redVarMap = null;

	////////////////////////////////////////////////////////////////
	// This HashMap contains mapping of (CPU symbol, GPU symbol). //
	// (This map is used if opt_globalGMalloc is true.)           //
	////////////////////////////////////////////////////////////////
	private static HashMap<VariableDeclarator, VariableDeclarator> gC2GMap = null;


	///////////////////////////////////////////////////////////////////////////////////
	// If caching optimizations are applied, shared variables with locality will be  //
	// loaded into caches (registers or shared memory) at the beginning of kernel    //
	// region. However, if the region is a for-loop and the loaded variable is used  //
	// in the condition expression of the loop, the initial loading statement should //
	// be inserted before the converted kernel region. For this, the statement has   //
	// to be inserted after the for-loop is converted into a kernel function.        //
	///////////////////////////////////////////////////////////////////////////////////
	private static HashSet<Statement> cacheLoadingStmts = null;

	private static LoopCollapse loopCollapseHandler = null;

	/**
	 * Insert CUDA-related initialization code into the input OpenMP program
	 *
	 * @param program : Input OpenMP program
	 */
	public static void CUDAInitializer(Program program) {
		Statement stmt1, stmt2, stmt3;
		Statement optPrintStmt;
		Statement confPrintStmt;
		List<Statement> optPrintStmts = new LinkedList<Statement>();
		List<Statement> confPrintStmts = new LinkedList<Statement>();
		boolean found_main = false;

		/////////////////////////////////////////////////////////////////
		// Read command-line options and set corresponding parameters. //
		/////////////////////////////////////////////////////////////////
		String value = Driver.getOptionValue("useMallocPitch");
		if( value != null ) {
			//opt_MallocPitch = Boolean.valueOf(value).booleanValue();
			opt_MallocPitch = true;
			FunctionCall optMPPrintCall = new FunctionCall(new NameID("printf"));
			optMPPrintCall.addArgument(new StringLiteral("====> MallocPitch Opt is used.\\n"));
			optPrintStmts.add( new ExpressionStatement(optMPPrintCall) );
		}

		value = Driver.getOptionValue("useMatrixTranspose");
		if( value != null ) {
			opt_MatrixTranspose = true;
			FunctionCall optMTPrintCall = new FunctionCall(new NameID("printf"));
			optMTPrintCall.addArgument(new StringLiteral("====> MatrixTranspose Opt is used.\\n"));
			optPrintStmts.add( new ExpressionStatement(optMTPrintCall) );
		}

		value = Driver.getOptionValue("useParallelLoopSwap");
		if( value != null ) {
			FunctionCall optPLPrintCall = new FunctionCall(new NameID("printf"));
			optPLPrintCall.addArgument(new StringLiteral("====> ParallelLoopSwap Opt is used.\\n"));
			optPrintStmts.add( new ExpressionStatement(optPLPrintCall) );
		}

		value = Driver.getOptionValue("useLoopCollapse");
		if( value != null ) {
			opt_LoopCollapse = true;
			FunctionCall optLCPrintCall = new FunctionCall(new NameID("printf"));
			optLCPrintCall.addArgument(new StringLiteral("====> LoopCollapse Opt is used.\\n"));
			optPrintStmts.add( new ExpressionStatement(optLCPrintCall) );
		}

		value = Driver.getOptionValue("addSafetyCheckingCode");
		if( value != null ) {
			opt_addSafetyCheckingCode = true;
			FunctionCall optSCPrintCall = new FunctionCall(new NameID("printf"));
			optSCPrintCall.addArgument(new StringLiteral("====> Safety-checking code is added.\\n"));
			optPrintStmts.add( new ExpressionStatement(optSCPrintCall) );
		}

		value = Driver.getOptionValue("forceSyncKernelCall");
		if( value != null ) {
			opt_forceSyncKernelCall = true;
			FunctionCall optSKCPrintCall = new FunctionCall(new NameID("printf"));
			optSKCPrintCall.addArgument(new StringLiteral("====> Explicit synchronization is forced.\\n"));
			optPrintStmts.add( new ExpressionStatement(optSKCPrintCall) );
		}

		value = Driver.getOptionValue("addCudaErrorCheckingCode");
		if( value != null ) {
			opt_addCudaErrorCheckingCode = true;
			///////////////////////////////////////////////////////////////////////////
			// If this option is on, forceSyncKernelCall option is suppressed, since //
			// the error checking code contains a built-in synchronization call.     //
			///////////////////////////////////////////////////////////////////////////
			if( opt_forceSyncKernelCall ) {
				opt_forceSyncKernelCall = false;
			} else {
				FunctionCall optSKCPrintCall = new FunctionCall(new NameID("printf"));
				optSKCPrintCall.addArgument(new StringLiteral("====> Explicit synchronization is forced.\\n"));
				optPrintStmts.add( new ExpressionStatement(optSKCPrintCall) );
			}
			FunctionCall optCECPrintCall = new FunctionCall(new NameID("printf"));
			optCECPrintCall.addArgument(new StringLiteral("====> CUDA-error-checking code is added.\\n"));
			optPrintStmts.add( new ExpressionStatement(optCECPrintCall) );
		}

		value = Driver.getOptionValue("cudaThreadBlockSize");
		if( value != null ) {
			defaultBlockSize = Integer.valueOf(value).intValue();
		} else {
			if( opt_LoopCollapse ) {
				defaultBlockSize = 512;
			}
		}
		FunctionCall BLKSizePrintCall = new FunctionCall(new NameID("printf"));
		BLKSizePrintCall.addArgument(new StringLiteral("====> GPU Block Size: "+defaultBlockSize+" \\n"));
		confPrintStmts.add( new ExpressionStatement(BLKSizePrintCall) );

		value = Driver.getOptionValue("useUnrollingOnReduction");
		if( value != null ) {
/*			if( defaultBlockSize < 64 ) {
				opt_UnrollingOnReduction = false;
				PrintTools.println("To use Unrolling optimization on reduction, " +
						"thread block size (BLOCK_SIZE) should be no less than 64;" +
						"this option will be ignored.", 0);
			} else {*/
				opt_UnrollingOnReduction = true;
				FunctionCall optRUPrintCall = new FunctionCall(new NameID("printf"));
				optRUPrintCall.addArgument(new StringLiteral("====> Unrolling-on-reduction Opt is used.\\n"));
				optPrintStmts.add( new ExpressionStatement(optRUPrintCall) );
			//}
		}

		value = Driver.getOptionValue("cudaGlobalMemSize");
		if( value != null ) {
			defaultGMemSize = Integer.valueOf(value).intValue();
		}

		value = Driver.getOptionValue("cudaMaxGridDimSize");
		if( value != null ) {
			maxGridDimSize = Integer.valueOf(value).intValue();
		}

		value = Driver.getOptionValue("cudaGridDimSize");
		if( value != null ) {
			defaultGridDimSize = Integer.valueOf(value).intValue();
		}

		value = Driver.getOptionValue("nvccVersion");
		if( value != null ) {
			nvccVersion = value;
		}

		value = Driver.getOptionValue("useGlobalGMalloc");
		if( value != null ) {
			opt_globalGMalloc = true;
			gC2GMap = new HashMap<VariableDeclarator, VariableDeclarator>();
			FunctionCall optGMallocCall = new FunctionCall(new NameID("printf"));
			optGMallocCall.addArgument(new StringLiteral("====> Allocate GPU variables as global ones.\\n"));
			optPrintStmts.add( new ExpressionStatement(optGMallocCall) );
		}

		value = Driver.getOptionValue("globalGMallocOpt");
		if( value != null ) {
			FunctionCall optGMallocOptCall = new FunctionCall(new NameID("printf"));
			optGMallocOptCall.addArgument(new StringLiteral("====> Optimize globally allocated GPU variables .\\n"));
			optPrintStmts.add( new ExpressionStatement(optGMallocOptCall) );
		}

		value = Driver.getOptionValue("cudaMemTrOptLevel");
		if( value != null ) {
			MemTrOptLevel = Integer.valueOf(value).intValue();
		}
		FunctionCall optMemTrPrintCall = new FunctionCall(new NameID("printf"));
		optMemTrPrintCall.addArgument(new StringLiteral("====> CPU-GPU Mem Transfer Opt Level: "
				+ MemTrOptLevel + "\\n"));
		optPrintStmts.add( new ExpressionStatement(optMemTrPrintCall) );

		value = Driver.getOptionValue("cudaMallocOptLevel");
		if( value != null ) {
			cudaMallocOptLevel = Integer.valueOf(value).intValue();
		}
		FunctionCall optCudaMallocPrintCall = new FunctionCall(new NameID("printf"));
		optCudaMallocPrintCall.addArgument(new StringLiteral("====> Cuda Malloc Opt Level: "
				+ cudaMallocOptLevel + "\\n"));
		optPrintStmts.add( new ExpressionStatement(optCudaMallocPrintCall) );

		value = Driver.getOptionValue("assumeNonZeroTripLoops");
		if( value != null ) {
			FunctionCall assmNnZrTrLpsPrintCall = new FunctionCall(new NameID("printf"));
			assmNnZrTrLpsPrintCall.addArgument(new StringLiteral("====> Assume that all loops have non-zero iterations.\\n"));
			optPrintStmts.add( new ExpressionStatement(assmNnZrTrLpsPrintCall) );
		}

		value = Driver.getOptionValue("shrdSclrCachingOnSM");
		if( value != null ) {
			opt_shrdSclrCachingOnSM = true;
			FunctionCall shrdSclrCachingOnSMPrintCall = new FunctionCall(new NameID("printf"));
			shrdSclrCachingOnSMPrintCall.addArgument(new StringLiteral("====> Cache shared scalar variables onto GPU shared memory.\\n"));
			optPrintStmts.add( new ExpressionStatement(shrdSclrCachingOnSMPrintCall) );
		}

		value = Driver.getOptionValue("shrdSclrCachingOnReg");
		if( value != null ) {
			FunctionCall shrdSclrCachingOnRegPrintCall = new FunctionCall(new NameID("printf"));
			if( opt_shrdSclrCachingOnSM ) {
				shrdSclrCachingOnRegPrintCall.addArgument(
						new StringLiteral("====> Cache shared scalar variables onto GPU registers.\\n"
								        + "      (Because shrdSclrCachingOnSM is on, R/O shared scalar variables\\n"
								        + "       are cached on shared memory, instead of registers.)\\n"));
			} else {
				shrdSclrCachingOnRegPrintCall.addArgument(
						new StringLiteral("====> Cache shared scalar variables onto GPU registers.\\n"));
			}
			optPrintStmts.add( new ExpressionStatement(shrdSclrCachingOnRegPrintCall) );
		}

		value = Driver.getOptionValue("shrdArryElmtCachingOnReg");
		if( value != null ) {
			FunctionCall shrdArryElmtCachingOnRegPrintCall = new FunctionCall(new NameID("printf"));
			shrdArryElmtCachingOnRegPrintCall.addArgument(new StringLiteral("====> Cache shared array elements onto GPU registers.\\n"));
			optPrintStmts.add( new ExpressionStatement(shrdArryElmtCachingOnRegPrintCall) );
		}

		value = Driver.getOptionValue("prvtArryCachingOnSM");
		if( value != null ) {
			FunctionCall prvtArryCachingOnSMPrintCall = new FunctionCall(new NameID("printf"));
			prvtArryCachingOnSMPrintCall.addArgument(new StringLiteral("====> Cache private array variables onto GPU shared memory.\\n"));
			optPrintStmts.add( new ExpressionStatement(prvtArryCachingOnSMPrintCall) );
		}

		value = Driver.getOptionValue("shrdArryCachingOnTM");
		if( value != null ) {
			FunctionCall shrdArryCachingOnTMPrintCall = new FunctionCall(new NameID("printf"));
			shrdArryCachingOnTMPrintCall.addArgument(new StringLiteral("====> Cache 1-dimensional, R/O shared array variables onto GPU texture memory.\\n"));
			optPrintStmts.add( new ExpressionStatement(shrdArryCachingOnTMPrintCall) );
		}

		value = Driver.getOptionValue("extractTuningParameters");
		if( value != null ) {
			if( value.equals("1") ) {
				tuningParamFile="TuningOptions.txt";
			} else {
				tuningParamFile=value;
			}
		}

		redVarMap = new HashMap<Statement, HashSet>();

		if( opt_LoopCollapse ) {
			loopCollapseHandler = new LoopCollapse(program);
		}

		for ( Traversable tt : program.getChildren() )
		{
			TranslationUnit tu = (TranslationUnit)tt;
			PrintTools.println(pass_name + "Input file name = " + tu.getInputFilename(), 5);
			boolean main_TU = false;
			/* find main()procedure */
			if( !found_main ) {
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
						main_TrUnt = tu;
						found_main = true;
						main_TU = true;
						break;
					}
				}

				if( found_main ) {
					/* 1) Insert CUDA initialization call at the beginning of the main() */
					/*     - CUT_DEVICE_INIT(argc, argv);                                */
					/* 2) Insert CUDA Exit call at the end of the main()                 */
					/*     - CUT_EXIT(argc, argv);                                       */
					/*     - ====> Use the following: fflush(stdout); fflush(stderr);    */
					/* In fact, above calls are not procedures, but macro                */
					FlushStmtList = new LinkedList<Statement>();
					FunctionCall cudaInit_call = new FunctionCall(new NameID("CUT_DEVICE_INIT"));
					//FunctionCall cudaExit_call = new FunctionCall(new NameID("CUT_EXIT"));
					FunctionCall fflush1_call = new FunctionCall(new NameID("fflush"));
					FunctionCall fflush2_call = new FunctionCall(new NameID("fflush"));
					FunctionCall optPrintCall = new FunctionCall(new NameID("printf"));
					optPrintCall.addArgument(new StringLiteral("/**********************/ \\n" +
							                                   "/* Used Optimizations */ \\n" +
							                                   "/**********************/ \\n"));
					FunctionCall confPrintCall = new FunctionCall(new NameID("printf"));
					confPrintCall.addArgument(new StringLiteral("/***********************/ \\n" +
							                                    "/* Input Configuration */ \\n" +
							                                    "/***********************/ \\n"));
					confPrintStmt = new ExpressionStatement(confPrintCall);
					List arglist = main.getParameters();
					for(Declaration arg: (List<Declaration>)arglist) {
						//VariableDeclaration cloned_decl = (VariableDeclaration)arg.clone();
						Expression expr = ((VariableDeclaration)arg).getDeclarator(0).getID();
						Expression cloned_expr = expr.clone();
						cudaInit_call.addArgument(cloned_expr);
						//cloned_expr = (Expression)expr.clone();
						//cudaExit_call.addArgument(cloned_expr);
					}
					fflush1_call.addArgument(new NameID("stdout"));
					fflush2_call.addArgument(new NameID("stderr"));
					CompoundStatement mainBody = main.getBody();
					stmt1 = new ExpressionStatement(cudaInit_call);
					//stmt2 = new ExpressionStatement(cudaExit_call);
					// Use fflush() calls instead of CUT_EXIT() call.
					stmt2 = new ExpressionStatement(fflush1_call);
					stmt3 = new ExpressionStatement(fflush2_call);
					Statement flushStmt = null;
					optPrintStmt = new ExpressionStatement(optPrintCall);
					Declaration last_decl = IRTools.getLastDeclaration(mainBody);
					// Insert cuda init statement only if main arguments exist.
					if( arglist.size() > 0 ) {
						mainBody.addStatementAfter((Statement)last_decl.getParent(), stmt1);
						firstMainStmt = AnalysisTools.getStatementAfter(mainBody, stmt1);
					} else {
						firstMainStmt = AnalysisTools.getStatementAfter(mainBody,
								(Statement)last_decl.getParent());
					}
					//PrintTools.println("FirstMainStmt: "+firstMainStmt, 0);
				    /*
				     * Find return statements in the main function, and add CUDA Exit call
				     * just before each return statement.
				     */
				    LinkedList<ReturnStatement> return_list = new LinkedList<ReturnStatement>();
				    BreadthFirstIterator riter = new BreadthFirstIterator(mainBody);
				    riter.pruneOn(Expression.class); /* optimization */
				    for (;;)
				    {
				      ReturnStatement stmt = null;

				      try {
				        stmt = (ReturnStatement)riter.next(ReturnStatement.class);
				      } catch (NoSuchElementException e) {
				        break;
				      }

				      return_list.add(stmt);
				    }
				    for( Statement rstmt : return_list ) {
				    	mainBody.addStatementBefore(rstmt, confPrintStmt.clone());
				    	for(Statement confStmt : confPrintStmts) {
				    		mainBody.addStatementBefore(rstmt, confStmt.clone());
				    	}
				    	mainBody.addStatementBefore(rstmt, optPrintStmt.clone());
				    	for(Statement optStmt : optPrintStmts) {
				    		mainBody.addStatementBefore(rstmt, optStmt.clone());
				    	}
				    	flushStmt =stmt2.clone();
				    	FlushStmtList.add(flushStmt);
				    	//mainBody.addStatementBefore(rstmt, (Statement)stmt2.clone());
				    	mainBody.addStatementBefore(rstmt, flushStmt);
				    	mainBody.addStatementBefore(rstmt, stmt3.clone());
				    }
				    ////////////////////////////////////////////////////////////
				    // If main() does not have any explicit return statement, //
				    // add CUDA exit call at the end of the main().           //
				    ////////////////////////////////////////////////////////////
				    if( return_list.size() == 0 ) {
				    	mainBody.addStatement(confPrintStmt.clone());
				    	for(Statement confStmt : confPrintStmts) {
				    		mainBody.addStatement(confStmt.clone());
				    	}
				    	mainBody.addStatement(optPrintStmt.clone());
				    	for(Statement optStmt : optPrintStmts) {
				    		mainBody.addStatement(optStmt.clone());
				    	}
				    	flushStmt =stmt2.clone();
				    	FlushStmtList.add(flushStmt);
				    	//mainBody.addStatement((Statement)stmt2.clone());
				    	mainBody.addStatement(flushStmt);
				    	mainBody.addStatement(stmt3.clone());
				    }
				}
			}

			/* Insert CUDA-related header files and macros */
			StringBuilder str = new StringBuilder(2048);
			str.append("/******************************************/\n");
			str.append("/* Added codes for OpenMP2GPU translation */\n");
			str.append("/******************************************/\n");
			str.append("#include <cutil.h>\n");
			str.append("#include <math.h>\n");
			str.append("#define MAX(a,b) (((a) > (b)) ? (a) : (b))\n");
			str.append("\n");
			str.append("/**********************************************************/\n");
			str.append("/* Maximum width of linear memory bound to texture memory */\n");
			str.append("/**********************************************************/\n");
			str.append("/* width in bytes */\n");
			str.append("#define LMAX_WIDTH    134217728\n");
			str.append("/* width in words */\n");
			str.append("#define LMAX_WWIDTH  33554432\n");
			str.append("/**********************************/\n");
			str.append("/* Maximum memory pitch (in bytes)*/\n");
			str.append("/**********************************/\n");
			str.append("#define MAX_PITCH   262144\n");
			str.append("/****************************************/\n");
			str.append("/* Maximum allowed GPU global memory    */\n");
			str.append("/* (less than actual size (1609891840)) */\n");
			str.append("/****************************************/\n");
			str.append("#define MAX_GMSIZE  " + defaultGMemSize + "\n");
			str.append("/****************************************/\n");
			str.append("/* Maximum allowed GPU shared memory    */\n");
			str.append("/****************************************/\n");
			str.append("#define MAX_SMSIZE  " + defaultSMemSize + "\n");
			str.append("/********************************************/\n");
			str.append("/* Maximum size of each dimension of a grid */\n");
			str.append("/********************************************/\n");
			str.append("#define MAX_GDIMENSION  " + maxGridDimSize + "\n");
			str.append("#define MAX_NDIMENSION  " + defaultGridDimSize + "\n");
			str.append("\n");
			str.append("#define BLOCK_SIZE  " + defaultBlockSize +"\n");
			str.append("\n");
			//tu.setHeader(str.toString());
			CodeAnnotation headerAnnot = new CodeAnnotation(str.toString());
			AnnotationDeclaration headerDecl = new AnnotationDeclaration(headerAnnot);
			Declaration firstDecl = tu.getFirstDeclaration();
			Declaration lastCudaDecl = null;
			if( (main_TU) || (!nvccVersion.equals("1.1")) ) {
				tu.addDeclarationBefore(firstDecl, headerDecl);
			}

			/*                                                     */
			/* Insert variables used for GPU-Kernel Initialization */
			/*                                                     */
			List<Specifier> specs = null;
			Declaration totalNumThreads_decl = null;
			//////////////////////////////////////////////////////////////////////
			// FIXME: Assume nvcc V2.0 supports inlining of external functions. //
			//        If this assumption is false, below is incorrect.          //
			//////////////////////////////////////////////////////////////////////
			if( (main_TU) || (!nvccVersion.equals("1.1")) ) {
				VariableDeclarator numThreads_declarator = new VariableDeclarator(new NameID("gpuNumThreads"));
				numThreads_declarator.setInitializer(new Initializer(new NameID("BLOCK_SIZE")));
				specs = new LinkedList<Specifier>();
				specs.add(Specifier.STATIC);
				specs.add(Specifier.INT);
				Declaration numThreads_decl = new VariableDeclaration(specs, numThreads_declarator);
				//tu.addDeclarationAfter(annot, numThreads_decl);
				tu.addDeclarationAfter(headerDecl, numThreads_decl);
				Identifier numThreads = new Identifier(numThreads_declarator);

				VariableDeclarator numBlocks_declarator = new VariableDeclarator(new NameID("gpuNumBlocks"));
				specs = new LinkedList<Specifier>();
				specs.add(Specifier.STATIC);
				specs.add(Specifier.INT);
				Declaration numBlocks_decl = new VariableDeclaration(specs, numBlocks_declarator);
				tu.addDeclarationAfter(numThreads_decl, numBlocks_decl);
				Identifier numBlocks = new Identifier(numBlocks_declarator);

				VariableDeclarator numBlocks_declarator1 = new VariableDeclarator(new NameID("gpuNumBlocks1"));
				specs = new LinkedList<Specifier>();
				specs.add(Specifier.STATIC);
				specs.add(Specifier.INT);
				Declaration numBlocks_decl1 = new VariableDeclaration(specs, numBlocks_declarator1);
				tu.addDeclarationAfter(numBlocks_decl, numBlocks_decl1);
				Identifier numBlocks1 = new Identifier(numBlocks_declarator1);

				VariableDeclarator numBlocks_declarator2 = new VariableDeclarator(new NameID("gpuNumBlocks2"));
				specs = new LinkedList<Specifier>();
				specs.add(Specifier.STATIC);
				specs.add(Specifier.INT);
				Declaration numBlocks_decl2 = new VariableDeclaration(specs, numBlocks_declarator2);
				tu.addDeclarationAfter(numBlocks_decl1, numBlocks_decl2);
				Identifier numBlocks2 = new Identifier(numBlocks_declarator2);

				VariableDeclarator totalNumThreads_declarator = new VariableDeclarator(new NameID("totalNumThreads"));
				specs = new LinkedList<Specifier>();
				specs.add(Specifier.STATIC);
				specs.add(Specifier.INT);
				totalNumThreads_decl = new VariableDeclaration(specs,
						totalNumThreads_declarator);
				tu.addDeclarationAfter(numBlocks_decl2, totalNumThreads_decl);
				Identifier totalNumThreads = new Identifier(totalNumThreads_declarator);
				lastCudaDecl = totalNumThreads_decl;
			}

			Identifier gpuMemSize = null;
			VariableDeclarator gpuMemSize_declarator = null;
			Declaration gpuMemSize_decl = null;
			Identifier smemSize = null;
			VariableDeclarator smemSize_declarator = null;
			Declaration smemSize_decl = null;
			if( main_TU ) {
				gpuMemSize_declarator = new VariableDeclarator(new NameID("gpuGmemSize"));
				gpuMemSize_declarator.setInitializer(new Initializer(new IntegerLiteral(0)));
				specs = new LinkedList<Specifier>();
				specs.add(Specifier.UNSIGNED);
				specs.add(Specifier.INT);
				gpuMemSize_decl = new VariableDeclaration(specs, gpuMemSize_declarator);
				tu.addDeclarationAfter(totalNumThreads_decl, gpuMemSize_decl);
				gpuMemSize = new Identifier(gpuMemSize_declarator);
				smemSize_declarator = new VariableDeclarator(new NameID("gpuSmemSize"));
				smemSize_declarator.setInitializer(new Initializer(new IntegerLiteral(0)));
				specs = new LinkedList<Specifier>();
				specs.add(Specifier.UNSIGNED);
				specs.add(Specifier.INT);
				smemSize_decl = new VariableDeclaration(specs, smemSize_declarator);
				tu.addDeclarationAfter(gpuMemSize_decl, smemSize_decl);
				smemSize = new Identifier(smemSize_declarator);
				lastCudaDecl = smemSize_decl;
			} else {
				if( !nvccVersion.equals("1.1") ) {
					gpuMemSize_declarator = new VariableDeclarator(new NameID("gpuGmemSize"));
					specs = new LinkedList<Specifier>();
					specs.add(Specifier.EXTERN);
					specs.add(Specifier.UNSIGNED);
					specs.add(Specifier.INT);
					gpuMemSize_decl = new VariableDeclaration(specs, gpuMemSize_declarator);
					tu.addDeclarationAfter(totalNumThreads_decl, gpuMemSize_decl);
					gpuMemSize = new Identifier(gpuMemSize_declarator);
					smemSize_declarator = new VariableDeclarator(new NameID("gpuSmemSize"));
					specs = new LinkedList<Specifier>();
					specs.add(Specifier.EXTERN);
					specs.add(Specifier.UNSIGNED);
					specs.add(Specifier.INT);
					smemSize_decl = new VariableDeclaration(specs, smemSize_declarator);
					tu.addDeclarationAfter(gpuMemSize_decl, smemSize_decl);
					smemSize = new Identifier(smemSize_declarator);
					lastCudaDecl = smemSize_decl;
				}
			}
			if( (main_TU) || (!nvccVersion.equals("1.1")) ) {
				VariableDeclarator bytes_declarator = new VariableDeclarator(new NameID("gpuBytes"));
				bytes_declarator.setInitializer(new Initializer(new IntegerLiteral(0)));
				specs = new LinkedList<Specifier>();
				specs.add(Specifier.STATIC);
				specs.add(Specifier.UNSIGNED);
				specs.add(Specifier.INT);
				Declaration bytes_decl = new VariableDeclaration(specs, bytes_declarator);
				tu.addDeclarationAfter(smemSize_decl, bytes_decl);
				Identifier bytes = new Identifier(bytes_declarator);
				lastCudaDecl = bytes_decl;
			}
			if( main_TU ) {
				CommentAnnotation endComment = new CommentAnnotation("endOfCUDADecls");
				endComment.setSkipPrint(true);
				AnnotationDeclaration endCommentDecl = new AnnotationDeclaration(endComment);
				tu.addDeclarationAfter(lastCudaDecl, endCommentDecl);
				lastCudaDecl = endCommentDecl;
				lastMainCudaDecl = lastCudaDecl;
			}
		}
	}

	/**
	 * Convert each OpenMP parallel region in a program into a CUDA kernel function.
	 * Assumes the SingleDeclarator pass and OmpAnalysis pass have
	 * been run on the program.
	 *
	 * @param program : Input program
	 */
	public static void convParRegionToKernelFunc(Program program)
	{
		int numOfKernelRegions = 0;
		c2gMemTr = new HashMap();
		g2cMemTr = new HashMap();
		c2gMap = new HashMap();
		pitchMap = new HashMap<Symbol, VariableDeclaration>();
		kernelCallStmtSet = new HashSet<Annotatable>();
		cacheLoadingStmts = new HashSet<Statement>();
		/* iterate to search for all Procedures */
		DepthFirstIterator proc_iter = new DepthFirstIterator(program);
		Set<Procedure> proc_list = (proc_iter.getSet(Procedure.class));
		for (Procedure proc : proc_list)
		{
			PrintTools.println(pass_name + "====> Find parallel regions in a procedure "+proc.getName(), 9);
			/* func_num is used to differentiate converted funcs in the same proc */
			int func_num = 0;

			/* Counter of non-OpenMP-parallel statements in the parent
			 * CompoundStatement of each target kernel region statement.
			 * If the parent statement contains more than one target regions,
			 * this counter is adjusted because previous call of setKernelConfParameters()
			 * inserts additional statements into the parent statement for O2G translation.
			 */
			num_cudastmts = 0;

			/* Reset c2gMemTr and g2cMemTr mapping */
			c2gMemTr.clear();
			g2cMemTr.clear();
			// FIXME: c2gMap and redMap may need to be reset at every procedure.
			c2gMap.clear();
			pitchMap.clear();
			kernelCallStmtSet.clear();
			cacheLoadingStmts.clear();

			/* Search for all OpenMP parallel regions in a given Procedure */
			List<OmpAnnotation>
			omp_annots = IRTools.collectPragmas(proc, OmpAnnotation.class, "parallel");

			//////////////////////////////////////////////////////////////////////////////
			// For each CompoundStatement containing at least one parallel region, find //
			// reduction variables used within the CompoundStatement and store the      //
			// information into redVarMap.                                              //
			//////////////////////////////////////////////////////////////////////////////
			findRedVarsPerPRegion(omp_annots);

			Statement prev_parent = null;
			for ( OmpAnnotation annot : omp_annots )
			{
				Statement target_stmt = (Statement)annot.getAnnotatable();
				int eligibility = AnalysisTools.checkKernelEligibility(target_stmt);
				if( eligibility == 0 ) {
					Statement target_parent = (Statement)target_stmt.getParent();
					if( prev_parent != target_parent) {
						num_cudastmts = 0;
					}
					prev_parent = target_parent;
					extractKernelRegion(program, proc, annot, func_num++);
				} else if (eligibility == 1) {
					PrintTools.println("[INFO in convParRegionToKernelFunc()] parallel region annotated by " +
							"the following cetus annotation will not be executed by GPU.", 0);
					PrintTools.println("  OmpAnnotation: " + annot, 0);
					PrintTools.println("  Enclosing procedure: " + proc.getSymbolName(), 0);
					PrintTools.println("  ====> parallel regions containing " +
							"critical or atomic construct" +
							" are not handled by current O2G translater" +
							"(these regions will be executed on the CPU).", 0);
					Statement target_parent = (Statement)target_stmt.getParent();
					if( prev_parent != target_parent) {
						num_cudastmts = 0;
					}
					prev_parent = target_parent;
					handleOtherOmpParallelRegion(proc, annot);
				} else if (eligibility == 2) {
					PrintTools.println("[INFO in convParRegionToKernelFunc()] parallel region annotated by " +
							"the following cetus annotation will not be executed by GPU.", 0);
					PrintTools.println("  OmpAnnotation: " + annot, 0);
					PrintTools.println("  Enclosing procedure: " + proc.getSymbolName(), 0);
					PrintTools.println("  ====> parallel regions containing " +
							"flush, ordered, or barrier construct" +
							" are not handled by current O2G translater" +
							"(these regions will be executed on the CPU).", 0);
					Statement target_parent = (Statement)target_stmt.getParent();
					if( prev_parent != target_parent) {
						num_cudastmts = 0;
					}
					prev_parent = target_parent;
					handleOtherOmpParallelRegion(proc, annot);
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
						Statement target_parent = (Statement)target_stmt.getParent();
						if( prev_parent != target_parent) {
							num_cudastmts = 0;
						}
						prev_parent = target_parent;
						extractKernelRegion(program, proc, annot, func_num++);
					} else {
						PrintTools.println("[INFO in convParRegionToKernelFunc()] parallel region annotated by " +
								"the following cetus annotation will not be executed by GPU.", 0);
						PrintTools.println("  OmpAnnotation: " + annot, 0);
					PrintTools.println("  Enclosing procedure: " + proc.getSymbolName(), 0);
						PrintTools.println("  ====> parallel regions that do not contain " +
								"any omp-for loop construct" +
								" are not eligible for GPU kernel execution" +
								"(these regions will be executed on the CPU).", 0);
						Statement target_parent = (Statement)target_stmt.getParent();
						if( prev_parent != target_parent) {
							num_cudastmts = 0;
						}
						prev_parent = target_parent;
						handleOtherOmpParallelRegion(proc, annot);
					}
				} else if (eligibility == 4) {
					PrintTools.println("[Error in convParRegionToKernelFunc()] parallel region annotated by " +
							"the following cetus annotation has an error.", 0);
					PrintTools.println("  OmpAnnotation: " + annot, 0);
					PrintTools.println("  Enclosing procedure: " + proc.getSymbolName(), 0);
					Tools.exit("  ====> nested parallel regions, which are not allowed by" +
							" O2G translator, are found.");
				} else if (eligibility == 5) {
					PrintTools.println("[INFO in convParRegionToKernelFunc()] parallel region annotated by " +
							"the following cetus annotation will not be executed by GPU.", 0);
					PrintTools.println("  OmpAnnotation: " + annot, 0);
					PrintTools.println("  Enclosing procedure: " + proc.getSymbolName(), 0);
					PrintTools.println("  ====> parallel regions containing " +
							"single or master construct are not handled by current O2G translater" +
							"(these regions will be executed on the CPU).", 0);
					Statement target_parent = (Statement)target_stmt.getParent();
					if( prev_parent != target_parent) {
						num_cudastmts = 0;
					}
					prev_parent = target_parent;
					handleOtherOmpParallelRegion(proc, annot);
				} else if (eligibility == 6) {
					PrintTools.println("[INFO in convParRegionToKernelFunc()] parallel region annotated by " +
							"the following cetus annotation will not be executed by GPU.", 0);
					PrintTools.println("  OmpAnnotation: " + annot, 0);
					PrintTools.println("  Enclosing procedure: " + proc.getSymbolName(), 0);
					PrintTools.println("  ====> User directive (pragma cuda nogpurun) is attached to this region", 0);
					Statement target_parent = (Statement)target_stmt.getParent();
					if( prev_parent != target_parent) {
						num_cudastmts = 0;
					}
					prev_parent = target_parent;
					handleOtherOmpParallelRegion(proc, annot);
				} else {
					Tools.exit("[ERROR in convParRegionToKernelFunc()] omp parallel annotation is attached to wrong statement");
				}
			}
			numOfKernelRegions += func_num;
			PrintTools.println(pass_name + "====> End search", 9);
		}
		PrintTools.println("[INFO] number of transformed kernel regions: "+numOfKernelRegions, 0);
		// Update linkSymbols.
		//SymbolTools.linkSymbol(program);
	}

	/**
	 * Extracts a parallel region (a CompoundStatement or a ForLoop) and converts
	 * to kernel function. Assumes the SingleDeclarator pass and OmpAnalysis pass
	 * have been run on the program.
	 *
	 * @param proc Procedure that contains the kernel region to be transformed
	 * @param annot OmpAnnotation attached to the kernel region
	 * @param func_num trail number used to create a unique new function name
	 */
	private static void extractKernelRegion(Program program, Procedure proc, OmpAnnotation annot,
			int func_num) {
		Statement region = (Statement)annot.getAnnotatable();
		String new_func_name = proc.getName().toString() + "_kernel" + func_num;
		SymbolTable global_table = (SymbolTable) proc.getParent();
		TranslationUnit tu = (TranslationUnit)proc.getParent();

		///////////////////////////////////////////////////////////////
		// Extract Cuda directives attached to this parallel region. //
		///////////////////////////////////////////////////////////////
		HashSet<String> cudaC2GMemTrSet = new HashSet<String>();
		HashSet<String> cudaNoC2GMemTrSet = new HashSet<String>();
		HashSet<String> cudaG2CMemTrSet = new HashSet<String>();
		HashSet<String> cudaNoG2CMemTrSet = new HashSet<String>();
		HashSet<String> cudaRegisterROSet = new HashSet<String>();
		HashSet<String> cudaRegisterSet = new HashSet<String>();
		HashSet<String> cudaSharedROSet = new HashSet<String>();
		HashSet<String> cudaSharedSet = new HashSet<String>();
		HashSet<String> cudaTextureSet = new HashSet<String>();
		HashSet<String> cudaConstantSet = new HashSet<String>();
		HashSet<String> cudaNoRedUnrollSet = new HashSet<String>();
		HashSet<String> cudaNoMallocSet = new HashSet<String>();
		HashSet<String> cudaNoFreeSet = new HashSet<String>();
		HashSet<String> cudaFreeSet = new HashSet<String>();
		String cudaMaxNumOfBlocks = null;
		String cudaThreadBlockSize = null;
		boolean noloopcollapse = false;
		List<CudaAnnotation> cudaAnnots = region.getAnnotations(CudaAnnotation.class);
		if( cudaAnnots != null ) {
			for( CudaAnnotation cannot : cudaAnnots ) {
				HashSet<String> dataSet = (HashSet<String>)cannot.get("c2gmemtr");
				if( dataSet != null ) {
					cudaC2GMemTrSet.addAll(dataSet);
				}
				dataSet = (HashSet<String>)cannot.get("noc2gmemtr");
				if( dataSet != null ) {
					cudaNoC2GMemTrSet.addAll(dataSet);
				}
				dataSet = (HashSet<String>)cannot.get("g2cmemtr");
				if( dataSet != null ) {
					cudaG2CMemTrSet.addAll(dataSet);
				}
				dataSet = (HashSet<String>)cannot.get("nog2cmemtr");
				if( dataSet != null ) {
					cudaNoG2CMemTrSet.addAll(dataSet);
				}
				dataSet = (HashSet<String>)cannot.get("registerRO");
				if( dataSet != null ) {
					cudaRegisterSet.addAll(dataSet);
					cudaRegisterROSet.addAll(dataSet);
				}
				dataSet = (HashSet<String>)cannot.get("registerRW");
				if( dataSet != null ) {
					cudaRegisterSet.addAll(dataSet);
				}
				dataSet = (HashSet<String>)cannot.get("sharedRO");
				if( dataSet != null ) {
					cudaSharedSet.addAll(dataSet);
					cudaSharedROSet.addAll(dataSet);
				}
				dataSet = (HashSet<String>)cannot.get("sharedRW");
				if( dataSet != null ) {
					cudaSharedSet.addAll(dataSet);
				}
				dataSet = (HashSet<String>)cannot.get("texture");
				if( dataSet != null ) {
					cudaTextureSet.addAll(dataSet);
				}
				/////////////////////////////////////////////////////////
				// FIXME: Currently, constant clause is not supported. //
				/////////////////////////////////////////////////////////
				dataSet = (HashSet<String>)cannot.get("constant");
				if( dataSet != null ) {
					cudaConstantSet.addAll(dataSet);
					PrintTools.println("[INFO in extractKernelRegion()] constant clause is not yet suppored", 0);
				}
				String sData = (String)cannot.get("maxnumofblocks");
				if( sData != null ) {
					cudaMaxNumOfBlocks = sData;
				}
				///////////////////////////////////////////////////////////////
				// FIXME: Currently, theadblocksize clause is not supported. //
				///////////////////////////////////////////////////////////////
				sData = (String)cannot.get("threadblocksize");
				if( sData != null ) {
					cudaThreadBlockSize = sData;
					PrintTools.println("[INFO in extractKernelRegion()] threadblocksize clause is not yet suppored", 0);
				}
				dataSet = (HashSet<String>)cannot.get("noreductionunroll");
				if( dataSet != null ) {
					cudaNoRedUnrollSet.addAll(dataSet);
				}
				dataSet = (HashSet<String>)cannot.get("nocudamalloc");
				if( dataSet != null ) {
					cudaNoMallocSet.addAll(dataSet);
				}
				dataSet = (HashSet<String>)cannot.get("nocudafree");
				if( dataSet != null ) {
					cudaNoFreeSet.addAll(dataSet);
				}
				dataSet = (HashSet<String>)cannot.get("cudafree");
				if( dataSet != null ) {
					cudaFreeSet.addAll(dataSet);
				}
				sData = (String)cannot.get("noloopcollapse");
				if( sData != null ) {
					noloopcollapse = true;
				}
			}

		}


		///////////////////////////////////////////////////////////////////////
		// c2gMap is changed to a static member of O2GTranslator, which will //
		// be reset at every procedure of a program.                         //
		///////////////////////////////////////////////////////////////////////
		//Annotation cudaAnnot = (Annotation)tu.getChildren().get(0);
		//HashMap c2gMap = (HashMap)cudaAnnot.getMap();
		HashMap redMap = (HashMap)c2gMap.get("_redMap_");
		if( redMap == null ) {
			///////////////////////////////////////////////////////////////////////////
			// Add empty HashMap, which is used to keep reduction variable to        //
			// CUDA global variable mapping. This additional HashMap is needed       //
			// because the same shared variable may be used as a reduction variable. //
			///////////////////////////////////////////////////////////////////////////
			redMap = new HashMap();
			c2gMap.put("_redMap_", redMap);
		}
		CompoundStatement kernelRegion = new CompoundStatement();
		boolean use_MallocPitch = opt_MallocPitch;
		boolean use_TextureMemory = false;
		boolean use_SharedMemory = false;
		tempMap = new HashMap<Identifier, Expression>();

		/* Generate GPU global data initialization part */
		CompoundStatement procbody = proc.getBody();

		PrintTools.println(pass_name + " Creating a new kernel procedure "
				+ new_func_name, 2);

		PrintTools.println(pass_name + "The following code section will be "
				+ " extracted as a kernel region: \n"
				+ region + "\n", 8);

		/*
		 * need to create 2 things: the new procedure where we are moving the
		 * region to, and a function call to the new procedure which we will use
		 * to replace the region
		 */
		List<Specifier> new_proc_ret_type = new LinkedList<Specifier>();
		new_proc_ret_type.add(CUDASpecifier.CUDA_GLOBAL);
		new_proc_ret_type.add(Specifier.VOID);

		Procedure new_proc = new Procedure(new_proc_ret_type,
				new ProcedureDeclarator(new NameID(new_func_name),
						new LinkedList()), new CompoundStatement());
		List<Expression> kernelConf = new ArrayList<Expression>();
		KernelFunctionCall call_to_new_proc = new KernelFunctionCall(new NameID(
				new_func_name), new LinkedList(), kernelConf);
		Statement kernelCall_stmt = new ExpressionStatement(call_to_new_proc);

		/* Extract data sharing attributes from OpenMP pragma */
		HashSet<Symbol> OmpSharedSet = null;
		HashSet<Symbol> OmpPrivSet = null;
		HashSet<Symbol> OmpFirstPrivSet = null;
		HashSet<Symbol> OmpLastPrivSet = null;
		HashSet<Symbol> OmpThreadPrivSet = null;
		HashSet<Symbol> OmpCopyinSet = null;
		HashSet<Symbol> OmpCopyPrivSet = null;
		if (annot.keySet().contains("shared")) {
			OmpSharedSet = (HashSet<Symbol>) annot.get("shared");
		}
		if (annot.keySet().contains("private")) {
			OmpPrivSet = (HashSet<Symbol>) annot.get("private");
		}
		if (annot.keySet().contains("firstprivate")) {
			OmpFirstPrivSet = (HashSet<Symbol>) annot.get("firstprivate");
		}
		if (annot.keySet().contains("lasstprivate")) {
			OmpLastPrivSet = (HashSet<Symbol>) annot.get("lastprivate");
			if( !OmpLastPrivSet.isEmpty() ) {
				Tools.exit("[ERROR] Current translator does not support lastprivate clause; exit");
			}
		}
		if (annot.keySet().contains("threadprivate")) {
			OmpThreadPrivSet = (HashSet<Symbol>) annot.get("threadprivate");
		}
		if (annot.keySet().contains("copyin")) {
			OmpCopyinSet = (HashSet<Symbol>) annot.get("copyin");
		}
		if (annot.keySet().contains("copyprivate")) {
			OmpCopyPrivSet = (HashSet<Symbol>) annot.get("copyprivate");
			if( !OmpCopyPrivSet.isEmpty() ) {
				Tools.exit("[ERROR] Current translator does not support copyprivate clause; exit");
			}
		}


		/*
		 * Find reference statement so that CPU-GPU-memory-transfer-related statements
		 * can be added before or after the reference statement.
		 * If the parallel region is enclosed by a loop, and if the parallel region
		 * is the only codes enclosed in the loop, the enclosing loop
		 * becomes the reference statement.
		 * Otherwise, the parallel region becomes the reference statement.
		 */
		Statement refstmt = region; //Ref-point used for inserting Shared-related statements.
		Statement refstmt1 = region; //Optimal ref-point that can be used for inserting Shared-related statements.
		                             //If a shared variable is used as reduction variable, this ref is not used.
		Statement refstmt2 = null; //Ref-point used for inserting Threadprivate-related statements.
		Traversable grandparent = region.getParent().getParent();
		Traversable greatgrandparent = grandparent.getParent();
		if( MemTrOptLevel > 0 ) {
			boolean containsParallelROnly =
				containsKernelRegionsOnly((CompoundStatement)region.getParent(), true);
			if( grandparent instanceof Loop ) {
				if( containsParallelROnly ) {
					refstmt1 = (Statement)grandparent;
				}
			} else if( (greatgrandparent instanceof Loop) && containsParallelROnly  ) {
				if( containsKernelRegionsOnly((CompoundStatement)grandparent, false) ) {
					refstmt1 = (Statement)greatgrandparent;
				}
			}
		}
		/////////////////////////////////////////////////////////////////////////
		// Shared data MallocPoint is always the out-most enclosing region     //
		// that is a child of a enclosing procedure body.                      //
		/////////////////////////////////////////////////////////////////////////
		Statement orgMallocPoint = region;
		Statement mallocPoint = region;
		Traversable t = region;
		Traversable p = t.getParent();
		Statement parentRegion = (Statement)p;
		while (p != null)
		{
			if (p.getParent() instanceof Procedure) {
				break;
			}
			else {
				t = p;
				p = p.getParent();
			}
		}
		orgMallocPoint = (Statement)t;
		mallocPoint = orgMallocPoint;

		// Pointer to the first CUDA-related statement
		Statement firstCudaStmt = null; //for shared-data handling statements.
		Statement firstCudaStmt2 = null;//for threadprivate, firstprivate, or reduction handling statements.

		//////////////////////////////////////////////////////////////
		//cacheLoadingStmts set is reset at every kernel conversion //
		//////////////////////////////////////////////////////////////
		cacheLoadingStmts.clear();

		// Auxiliary variables used for GPU kernel conversion
		VariableDeclaration bytes_decl = (VariableDeclaration)SymbolTools.findSymbol(global_table, "gpuBytes");
		Identifier cloned_bytes = new Identifier((VariableDeclarator)bytes_decl.getDeclarator(0));
		VariableDeclaration gmem_decl = (VariableDeclaration)SymbolTools.findSymbol(global_table, "gpuGmemSize");
		Identifier gmemsize = new Identifier((VariableDeclarator)gmem_decl.getDeclarator(0));
		VariableDeclaration smem_decl = (VariableDeclaration)SymbolTools.findSymbol(global_table, "gpuSmemSize");
		Identifier smemsize = new Identifier((VariableDeclarator)smem_decl.getDeclarator(0));
		VariableDeclaration totalThreads_decl = (VariableDeclaration)SymbolTools.findSymbol(global_table, "totalNumThreads");
		Identifier totalNumThreads = new Identifier((VariableDeclarator)totalThreads_decl.getDeclarator(0));
		ExpressionStatement gpuBytes_stmt = null;
		ExpressionStatement orgTPBytes_stmt = null;
		ExpressionStatement gMemAdd_stmt = new ExpressionStatement( new AssignmentExpression(gmemsize,
				AssignmentOperator.ADD, cloned_bytes.clone()) );
		ExpressionStatement gMemSub_stmt = new ExpressionStatement( new AssignmentExpression(gmemsize.clone(),
				AssignmentOperator.SUBTRACT, cloned_bytes.clone()) );
		Identifier textureRefID = null;
		VariableDeclarator rowidSymbol = null;

		/////////////////////////////////////////////////////////////////////////////
		// Apply LoopCollapse optimization; currently LoopCollapse optimization is //
		// applied to Sparse Matrix-Vector Product (SPMV) patterns only.           //
		/////////////////////////////////////////////////////////////////////////////
		if( opt_LoopCollapse && !noloopcollapse ) {
			loopCollapseHandler.handleSMVP(region, false);
			rowidSymbol = loopCollapseHandler.getGpuRowidSymbol();
			if( rowidSymbol != null) {
				if( region instanceof ForLoop ) {
					region = ((ForLoop)region).getBody();
				}
				call_to_new_proc.addArgument(new Identifier(rowidSymbol));
				new_proc.addDeclaration(loopCollapseHandler.getRowidDecl());
			}
		}

		//////////////////////////////////////////////////////////////////////////////
		// If Max # of Thread Blocks is set by cuda maxnumofblocks(nblocks) clause, //
		// apply cyclic unrolling for each OMP-for loop in this parallel region.    //
		// DEBUG: If useLoopCollapse option is on and if LoopCollapse is applicable //
		// to an OMP-for loop with maxnumofblocks clause, this cyclic unrolling will//
		// be skipped, since LoopCollapse optimization swaps the loop with its      //
		// transformed body statement.                                              //
		// DEBUG: below conversion works only if the index variable of the          //
		// original loop increment by 1.                                            //
		//////////////////////////////////////////////////////////////////////////////
		// Original loop:                                                           //
		//     for( k = LB; k <= UB; k++ ) { }                                      //
		// Cyclic-unrolled loop:                                                    //
		//     for( i = 0; i < totalNumOfThreads; i++ ) {                           //
		//          for( k = i+LB; k <= UB; k += totalNumOfThreads ) { }            //
		//     }                                                                    //
		//////////////////////////////////////////////////////////////////////////////
		List<OmpAnnotation> omp_annots = null;
		if( cudaMaxNumOfBlocks != null ) {
			int maxNumOfBlocks = Integer.parseInt(cudaMaxNumOfBlocks);
			int totalNumOfThreads = maxNumOfBlocks * defaultBlockSize;
			CompoundStatement targetRegion = null;
			if( region instanceof ForLoop ) {
				targetRegion = (CompoundStatement)((ForLoop)region).getBody();
			} else {
				targetRegion = (CompoundStatement)region;
			}
			omp_annots = IRTools.collectPragmas(region, OmpAnnotation.class, "for");
			for ( OmpAnnotation fannot : omp_annots ) {
				Statement target_stmt = (Statement)fannot.getAnnotatable();
				if( target_stmt instanceof ForLoop ) {
					ForLoop ploop = (ForLoop)target_stmt;
					//////////////////////////////////////////////////////////
					// Check the increment of the omp-for loop is 1 or not. //
					//////////////////////////////////////////////////////////
					Expression incrExp = LoopTools.getIncrementExpression(ploop);
					if( incrExp instanceof IntegerLiteral ) {
						if( ((IntegerLiteral)incrExp).getValue() != 1 ) {
							PrintTools.println("[WARNING in O2GTranslator()] cyclic unrolling is not " +
									"applicable to an omp-for loop whose increment is not 1",0);
							continue;
						}
					} else {
						PrintTools.println("[WARNING in O2GTranslator()] cyclic unrolling is not " +
								"applicable to an omp-for loop whose increment is not 1",0);
						continue;
					}
					CompoundStatement forBody = new CompoundStatement();
					Identifier index = null;
					if( targetRegion == region ) {
						index = TransformTools.getTempIndex(targetRegion, 100);
					} else {
						index = TransformTools.getTempIndex(forBody, 100);
					}
					Expression expr1 = new AssignmentExpression(index, AssignmentOperator.NORMAL,
							new IntegerLiteral(0));
					Statement initStmt = new ExpressionStatement(expr1);
					expr1 = new BinaryExpression(index.clone(), BinaryOperator.COMPARE_LT,
							new IntegerLiteral(totalNumOfThreads));
					Expression expr2 = new UnaryExpression(
							UnaryOperator.POST_INCREMENT, index.clone());
					ForLoop wLoop = new ForLoop(initStmt, expr1, expr2, forBody);
					// Swap the new loop (wLoop) with the old loop (ploop).
					wLoop.swapWith(ploop);
					expr1 = new BinaryExpression(index.clone(), BinaryOperator.ADD,
							LoopTools.getLowerBoundExpression(ploop));
					Expression oldindex = LoopTools.getIndexVariable(ploop);
					expr2 = new AssignmentExpression(oldindex.clone(), AssignmentOperator.NORMAL,
							expr1);
					initStmt = new ExpressionStatement(expr2);
					ploop.getInitialStatement().swapWith(initStmt);
/*					expr1 = new BinaryExpression((Expression)oldindex.clone(), BinaryOperator.COMPARE_LE,
							LoopTools.getUpperBoundExpression(ploop));
					ploop.getCondition().swapWith(expr1);*/
					expr2 = new AssignmentExpression(oldindex.clone(), AssignmentOperator.ADD,
							new IntegerLiteral(totalNumOfThreads));
					ploop.getStep().swapWith(expr2);
					forBody.addStatement(ploop);
					///////////////////////////////////////////////
					// Move all Annotations of ploop into wLoop. //
					///////////////////////////////////////////////
					List<Annotation> annot_list = ploop.getAnnotations();
					for( Annotation tAnnot : annot_list ) {
						wLoop.annotate(tAnnot);
					}
					ploop.removeAnnotations();
					if( region instanceof ForLoop ) {
						if( refstmt == region ) {
							refstmt = wLoop;
						}
						if( refstmt1 == region ) {
							refstmt1 = wLoop;
						}
						if( orgMallocPoint == region ) {
							orgMallocPoint = wLoop;
						}
						if( mallocPoint == region ) {
							mallocPoint = wLoop;
						}
						region = wLoop;
						annot = fannot;

					}
				}
			}
		}

		////////////////////////////////////////////////////////////////////////////////
		// Calculate iteration space sizes of omp-for loops included in the parallel  //
		// region, region.                                                            //
		////////////////////////////////////////////////////////////////////////////////
		Set<Symbol> ispaceSymbols = TransformTools.calcLoopItrSize(region, annot);

		////////////////////////////////////////////////////////////////////////////////
		// Check whether grid-size (= max of iteration space sizes) is not changed in //
		// a region referenced by refstmt1. If so, reduction-related statements and   //
		// Threadprivate-related statements can be inserted using refstmt1.           //
		////////////////////////////////////////////////////////////////////////////////
		Set<Symbol> defSet = DataFlowTools.getDefSymbol(refstmt1);
		boolean gridSizeNotChanged = false;
		ispaceSymbols.retainAll(defSet);
		if( ispaceSymbols.size() == 0 ) {
			gridSizeNotChanged = true;
			//////////////////////////////////////////////////////////////////////////
			//If a function called in the refstmt1 has omp-for loop, conservatively //
			//assume that the grid-size may be changed in the called function.      //
			//////////////////////////////////////////////////////////////////////////
			List<FunctionCall> callList = IRTools.getFunctionCalls(refstmt1);
			for( FunctionCall fcall : callList ) {
				if( (fcall instanceof KernelFunctionCall) || AnalysisTools.isCudaCall(fcall)
						|| StandardLibrary.contains(fcall) ) {
					continue;
				}
				Procedure cProc = fcall.getProcedure();
				if( cProc == null ) {
					continue;
				} else {
					List<OmpAnnotation> fList = IRTools.collectPragmas(cProc, OmpAnnotation.class, "for");
					if( fList.size() > 0 ) {
						gridSizeNotChanged = false;
						break;
					}
				}
			}
		}

/*		if( gridSizeNotChanged ) {
			PrintTools.println("grid-size is not changed", 0);
		} else {
			PrintTools.println("grid-size is changed", 0);
		}*/

		/*
		 * Fill in the parameter list of the kernel function and the argument list
		 * of the new call.
		 */

		/////////////////////////////////////////////////////////////////////////////////
		// If a shared variable is used as a reduction variable in any enclosed        //
		// omp-for loops but not used as a shared one, code generation for the shared  //
		// variable should be skipped.                                                 //
		/////////////////////////////////////////////////////////////////////////////////
		// Privatize shared variables included in a private/firstprivate clause of any //
		// omp-for loop in this parallel region.                                       //
		/////////////////////////////////////////////////////////////////////////////////
		if( region instanceof CompoundStatement ) {
			TransformTools.privatizeSharedData(annot, (CompoundStatement)region);
		}

		////////////////////////////////
        // Handle OMP reduction data. //
		////////////////////////////////
		List redData = reductionTransformation(proc, region, redMap, call_to_new_proc,
				new_proc, cudaNoRedUnrollSet, refstmt1, gridSizeNotChanged);
		firstCudaStmt2 = (Statement)redData.get(0);
		// FIXME: below is redundant assignment.
		HashSet<Symbol> redItemSet = (HashSet<Symbol>)redData.get(1);

		if( redVarMap.containsKey(parentRegion) ) {
			redItemSet.addAll( redVarMap.get(parentRegion));
		}


		/////////////////////////////
		// Handle OMP Shared data. //
		/////////////////////////////
		if (OmpSharedSet != null) {
			for( Symbol shared_var : OmpSharedSet ) {
				////////////////////////////////////////////////////////////////////////////////////
				// If a shared variable is used as a reduction variable, the shared variable      //
				// should be transferred back to CPU after corresponding kernel function returns. //
				////////////////////////////////////////////////////////////////////////////////////
				if( redItemSet.contains(shared_var) ) {
					refstmt = region;
				} else {
					refstmt = refstmt1;
				}
				if( shared_var instanceof VariableDeclarator ) {
					VariableDeclaration decl = (VariableDeclaration)((VariableDeclarator)shared_var).getParent();
					/*
					 * Create a cloned Declaration of the shared variable.
					 */
					VariableDeclarator cloned_declarator =
						((VariableDeclarator)shared_var).clone();
					cloned_declarator.setInitializer(null);
					/////////////////////////////////////////////////////////////////////////////////
					// __device__ and __global__ functions can not declare static variables inside //
					// their body.                                                                 //
					/////////////////////////////////////////////////////////////////////////////////
					List<Specifier> clonedspecs = new ChainedList<Specifier>();
					clonedspecs.addAll(decl.getSpecifiers());
					clonedspecs.remove(Specifier.STATIC);
					VariableDeclaration cloned_decl = new VariableDeclaration(clonedspecs, cloned_declarator);
					Identifier cloned_ID = new Identifier(cloned_declarator);

					Identifier gpu_var = null;
					Identifier pitch_var = null;
					Identifier pointer_var = null;

					boolean c2gMap_added = false;

					////////////////////////////////////////////////////////////////////////////////
					// Pass R/O shared scalar variable as a kernel parameter instead of using GPU //
					// global memory, which has the effect of caching it on the GPU Shared Memory.//                                                     //
					////////////////////////////////////////////////////////////////////////////////
					if( cudaSharedROSet.contains(cloned_ID.getName()) ) {
						// Create a GPU kernel parameter corresponding to shared_var
						VariableDeclarator gpu_declarator = new VariableDeclarator(new NameID(cloned_ID.getName()));
						VariableDeclaration gpu_decl = new VariableDeclaration(cloned_decl.getSpecifiers(),
								gpu_declarator);
						gpu_var = new Identifier(gpu_declarator);
						new_proc.addDeclaration(gpu_decl);

						// Insert argument to the kernel function call
						call_to_new_proc.addArgument(new Identifier(shared_var));

						// Replace the instance of shared variable with the new gpu_var.
						IRTools.replaceAll(region, cloned_ID, gpu_var);

						continue;
					}

					/////////////////////////////////////////////////////////////////
					// If insertGMalloc == 0, GPU variables are locally allocated. //
					// Else if insertGMalloc == 1, allocate GPU variable globally. //
					// Else if insertGMalloc == 2, GPU variable has already been   //
					//                    allocated globally; no action is needed. //
					/////////////////////////////////////////////////////////////////
					int insertGMalloc = 0;
					SymbolTable symTable = null;
					CompoundStatement mainBody = null;
					VariableDeclarator orgSym = null;
					if( opt_globalGMalloc ) {
						//////////////////////////////////////////////////////////////
						// Allocate GPU variables as global variables.              //
						// GPU variables are declared in the main translation unit, //
						// and they are allocated in the main procedure.            //
						//////////////////////////////////////////////////////////////
						List symInfo = AnalysisTools.findOrgSymbol(shared_var, proc);
						if( symInfo.size() == 0 ) {
							mallocPoint = orgMallocPoint;
							PrintTools.println("[WARNING in extractKernelRegion()] can't find the original " +
									"symbol of "+ shared_var + " in a procedure ("+proc.getSymbolName()+
									"); this will be allocated locally.", 0);
						} else if( symInfo.size() == 1 ) {
							mallocPoint = orgMallocPoint;
							PrintTools.println("[INFO in extractKernelRegion()] a symbol, " +
									shared_var + ", in a procedure ("+proc.getSymbolName()+
									") will be allocated locally.", 0);
						} else {
							orgSym = (VariableDeclarator)symInfo.get(0);
							symTable = (SymbolTable)symInfo.get(1);
							mainBody = main.getBody();
							if( gC2GMap.containsKey(orgSym) ) {
								insertGMalloc = 2;
								mallocPoint = firstMainStmt;
								VariableDeclarator gpu_declarator = gC2GMap.get(orgSym);
								if( !c2gMap.containsKey(shared_var) ) {
									c2gMap.put(shared_var, gpu_declarator);
								}
							} else {
								insertGMalloc = 1;
								c2gMap_added = true;
								mallocPoint = firstMainStmt;
								// Create a GPU device variable corresponding to shared_var
								// Ex: float * gpu__b;
								// Give a new name for the device variable
								StringBuilder str = new StringBuilder(80);
								str.append("gpu__");
								str.append(orgSym.getSymbolName());
								if( symTable instanceof Procedure ) {
									str.append("__"+((Procedure)symTable).getSymbolName());
								}
								// The type of the device symbol should be a pointer type
								VariableDeclarator gpu_declarator = new VariableDeclarator(PointerSpecifier.UNQUALIFIED,
										new NameID(str.toString()));
								VariableDeclaration gpu_decl = new VariableDeclaration(cloned_decl.getSpecifiers(),
										gpu_declarator);
								gpu_var = new Identifier(gpu_declarator);
								main_TrUnt.addDeclarationAfter(lastMainCudaDecl, gpu_decl);
								lastMainCudaDecl = gpu_decl;
								// Add mapping from orgSym to the gpu_declarator.
								gC2GMap.put(orgSym, gpu_declarator);
								// Add mapping from shared_var to gpu_declarator.
								c2gMap.put(shared_var, gpu_declarator);
							}

						}
					}

					/*
					 * c2gMap contains a mapping from a shared/threadprivate variable to corresponding GPU variable.
					 */
					if( c2gMap.containsKey(shared_var)) {
						// clone GPU device symbol corresponding to shared_var
						gpu_var = new Identifier((VariableDeclarator)c2gMap.get(shared_var));
					} else {
						// Create a GPU device variable corresponding to shared_var
						// Ex: float * gpu__b;
						// Give a new name for the device variable
						StringBuilder str = new StringBuilder(80);
						str.append("gpu__");
						str.append(cloned_ID.toString());
						// The type of the device symbol should be a pointer type
						VariableDeclarator gpu_declarator = new VariableDeclarator(PointerSpecifier.UNQUALIFIED,
								new NameID(str.toString()));
						VariableDeclaration gpu_decl = new VariableDeclaration(cloned_decl.getSpecifiers(),
								gpu_declarator);
						gpu_var = new Identifier(gpu_declarator);
						procbody.addDeclaration(gpu_decl);
						// Add mapping from shared_var to gpu_declarator
						c2gMap.put(shared_var, gpu_declarator);
						c2gMap_added = true;
					}

					///////////////////////////////////////////////////////
					// Check duplicate Malloc() for the shared variable. //
					///////////////////////////////////////////////////////
					boolean insertMalloc = false;
					boolean insertMalloc2 = false;
					boolean insertFree2 = false;
					boolean insertGFree = false; //control cudaFree of globally allocated GPU variable
					HashSet<String> memTrSet = null;
					StringBuilder str = new StringBuilder(80);
					str.append("malloc_");
					str.append(gpu_var.getName());
					if( c2gMemTr.containsKey(mallocPoint) ) {
						memTrSet = (HashSet<String>)c2gMemTr.get(mallocPoint);
						if( !memTrSet.contains(str.toString()) ) {
							memTrSet.add(str.toString());
							insertMalloc = true;
						}
					} else {
						memTrSet = new HashSet<String>();
						memTrSet.add(str.toString());
						c2gMemTr.put(mallocPoint, memTrSet);
						insertMalloc = true;
					}
					if( insertMalloc ) {
						insertMalloc2 = true;
						insertFree2 = true;
					}
					///////////////////////////////////////////////////////////////////////
					// If nocudamalloc clause contains this symbol, insertMalloc2 should //
					// set to false.                                                     //
					///////////////////////////////////////////////////////////////////////
					if( cudaNoMallocSet.contains(cloned_ID.getName()) ) {
						insertMalloc2 = false;
					}
					/////////////////////////////////////////////////////////////////
					// If cudafree clause contains this symbol, insertFree2 should //
					// set to true.                                                //
					/////////////////////////////////////////////////////////////////
					if( cudaFreeSet.contains(cloned_ID.getName()) ) {
						insertFree2 = true;
					}
					///////////////////////////////////////////////////////////////////
					// If nocudafree clause contains this symbol, insertFree2 should //
					// set to false.                                                 //
					///////////////////////////////////////////////////////////////////
					if( cudaNoFreeSet.contains(cloned_ID.getName()) ) {
						insertFree2 = false;
					}
					/////////////////////////////////////////////////////////
					// Check duplicate cudaFree() for the shared variable. //
					/////////////////////////////////////////////////////////
					str = new StringBuilder(80);
					str.append("cudaFree_");
					str.append(gpu_var.getName());
					if( insertFree2 ) {
						if( g2cMemTr.containsKey(mallocPoint) ) {
							memTrSet = (HashSet<String>)g2cMemTr.get(mallocPoint);
							if( !memTrSet.contains(str.toString()) ) {
								memTrSet.add(str.toString());
								//insertFree2 = true;
							} else {
								insertFree2 = false;
							}
						} else {
							memTrSet = new HashSet<String>();
							memTrSet.add(str.toString());
							g2cMemTr.put(mallocPoint, memTrSet);
							//insertFree2 = true;
						}
					}

					/////////////////////////////////////////////////////////////////////
					// If opt_globalGMalloc is on, cuda Malloc and Free are controlled //
					// by the following statements.                                    //
					/////////////////////////////////////////////////////////////////////
					if( opt_globalGMalloc ) {
						if( insertGMalloc == 1 ) {
							insertMalloc2 = true;
							insertFree2 = false;
							insertGFree = true;
						} else if( insertGMalloc == 2){
							insertMalloc2 = false;
							insertFree2 = false;
						}
					}
					/*
					 * Check duplicate CPU to GPU memory transfers.
					 * Currently, simple name-only analysis is conducted; if the same array
					 * is transferred multiply at the same program point, insert only one memory transfer.
					 */
					boolean insertC2GMemTr = false;
					if( c2gMemTr.containsKey(refstmt) ) {
						memTrSet = (HashSet<String>)c2gMemTr.get(refstmt);
						if( !memTrSet.contains(gpu_var.getName()) ) {
							//memTrSet.add(gpu_var.getName());
							insertC2GMemTr = true;
						}
					} else {
						memTrSet = new HashSet<String>();
						//memTrSet.add(gpu_var.getName());
						c2gMemTr.put(refstmt, memTrSet);
						insertC2GMemTr = true;
					}
					/////////////////////////////////////////////////////////////////////////
					// If refstmt != region, and if this shared variable is modified by    //
					// CPU before this kernel region, memory transfers before and after    //
					// this kernel call may be needed; insertC2GMemTr2 and insertG2CMemTr2 //
					// are used for this purpose.                                          //
					/////////////////////////////////////////////////////////////////////////
					boolean insertC2GMemTr2 = false;
					boolean insertG2CMemTr2 = false;
					if( (refstmt != region) && c2gMap_added ) {
						List<Traversable> children = parentRegion.getChildren();
						int currIndex = Tools.indexByReference(children, region);
						currIndex -= 1;
						while( currIndex >= 0 ) {
							Statement currStmt = (Statement)children.get(currIndex);
							if( !kernelCallStmtSet.contains(currStmt)) {
								Set<Symbol> DefSet = DataFlowTools.getDefSymbol(currStmt);
								if( DefSet.contains(shared_var)) {
									insertC2GMemTr2 = true;
									insertG2CMemTr2 = true;
									break;
								}
							}
							currIndex--;
						}
					}
					/////////////////////////////////////////////////////////////////////////
					// If Cuda c2gmemtr clause contains this symbol, insertC2GMemTr should //
					// set to true.                                                        //
					/////////////////////////////////////////////////////////////////////////
					if( cudaC2GMemTrSet.contains(cloned_ID.getName()) ) {
						insertC2GMemTr = false;
						insertC2GMemTr2 = true;
					}
					///////////////////////////////////////////////////////////////////////////
					// If Cuda noc2gmemtr clause contains this symbol, insertC2GMemTr should //
					// set to false.                                                         //
					///////////////////////////////////////////////////////////////////////////
					if( cudaNoC2GMemTrSet.contains(cloned_ID.getName()) ) {
						insertC2GMemTr = false;
						insertC2GMemTr2 = false;
					}
					if( insertC2GMemTr ) {
						memTrSet.add(gpu_var.getName());
					}
					//////////////////////////////////////////////////////////////
					// Below part is needed for handleOtherOmpParallelRegion(). //
					//////////////////////////////////////////////////////////////
					if( ((refstmt == region) && insertC2GMemTr) ||
							(insertC2GMemTr2)) {
						if( c2gMemTr.containsKey(kernelCall_stmt) ) {
							memTrSet = (HashSet<String>)c2gMemTr.get(kernelCall_stmt);
							if( !memTrSet.contains(gpu_var.getName()) ) {
								memTrSet.add(gpu_var.getName());
							}
						} else {
							memTrSet = new HashSet<String>();
							memTrSet.add(gpu_var.getName());
							c2gMemTr.put(kernelCall_stmt, memTrSet);
						}
					}

					/*
					 * Check duplicate GPU to CPU memory transfers
					 * Currently, simple name-only analysis is conducted; if the same array
					 * is transferred multiply at the same program point, insert only one memory transfer.
					 */
					boolean insertG2CMemTr = false;
					if( g2cMemTr.containsKey(refstmt) ) {
						memTrSet = (HashSet<String>)g2cMemTr.get(refstmt);
						if( !memTrSet.contains(gpu_var.getName()) ) {
							//memTrSet.add(gpu_var.getName());
							insertG2CMemTr = true;
						}
					} else {
						memTrSet = new HashSet<String>();
						//memTrSet.add(gpu_var.getName());
						g2cMemTr.put(refstmt, memTrSet);
						insertG2CMemTr = true;
					}
					/////////////////////////////////////////////////////////////////////////
					// If Cuda g2cmemtr clause contains this symbol, insertG2CMemTr should //
					// set to true.                                                        //
					/////////////////////////////////////////////////////////////////////////
					if( cudaG2CMemTrSet.contains(cloned_ID.getName()) ) {
						insertG2CMemTr = false;
						insertG2CMemTr2 = true;
					}
					///////////////////////////////////////////////////////////////////////////
					// If Cuda nog2cmemtr clause contains this symbol, insertG2CMemTr should //
					// set to false.                                                         //
					///////////////////////////////////////////////////////////////////////////
					if( cudaNoG2CMemTrSet.contains(cloned_ID.getName()) ) {
						insertG2CMemTr = false;
						insertG2CMemTr2 = false;
					}
					if( insertG2CMemTr ) {
						memTrSet.add(gpu_var.getName());
					}
					//////////////////////////////////////////////////////////////
					// Below part is needed for handleOtherOmpParallelRegion(). //
					//////////////////////////////////////////////////////////////
					if( ((refstmt == region) && insertG2CMemTr) ||
							(insertG2CMemTr2)) {
						if( g2cMemTr.containsKey(kernelCall_stmt) ) {
							memTrSet = (HashSet<String>)g2cMemTr.get(kernelCall_stmt);
							if( !memTrSet.contains(gpu_var.getName()) ) {
								memTrSet.add(gpu_var.getName());
							}
						} else {
							memTrSet = new HashSet<String>();
							memTrSet.add(gpu_var.getName());
							g2cMemTr.put(kernelCall_stmt, memTrSet);
						}
					}

					/////////////////////////////////////////////////////////////////////////
					// Memory allocation for the device variable                           //
					/////////////////////////////////////////////////////////////////////////
					// - Insert cudaMalloc() function before the region.                   //
					// Ex: CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu_b)), gpuBytes)); //
					// Ex2: CUDA_SAFE_CALL( cudaMallocPitch((void**) &d_x, &pitch_x,       //
					//      width*sizeof(float), height) );                                //
					/////////////////////////////////////////////////////////////////////////
					FunctionCall malloc_call = null;
					List<Specifier> specs = new ArrayList<Specifier>(4);
					specs.add(Specifier.VOID);
					specs.add(PointerSpecifier.UNQUALIFIED);
					specs.add(PointerSpecifier.UNQUALIFIED);
					List<Expression> arg_list = new ArrayList<Expression>();
					arg_list.add(new Typecast(specs, new UnaryExpression(UnaryOperator.ADDRESS_OF,
							gpu_var.clone())));
					SizeofExpression sizeof_expr = new SizeofExpression(cloned_decl.getSpecifiers());
					if( SymbolTools.isPointer(shared_var) ) {
						Tools.exit(pass_name + "[ERROR] extractKernelRegion() needs to support Pointer type shared variable: "
								+ shared_var.toString());
					} else if( SymbolTools.isScalar(shared_var) ) {
						use_TextureMemory = false;
						use_MallocPitch = false;
						malloc_call = new FunctionCall(new NameID("cudaMalloc"));
						// Insert "gpuBytes = sizeof(varType);" statement
						AssignmentExpression assignex = new AssignmentExpression(cloned_bytes.clone(),AssignmentOperator.NORMAL,
								sizeof_expr);
						ExpressionStatement estmt = new ExpressionStatement(assignex);
						gpuBytes_stmt = estmt.clone();
						//if( insertMalloc ) {
						if( insertMalloc2 ) {
							((CompoundStatement)mallocPoint.getParent()).addStatementBefore(mallocPoint, estmt);
//							if( firstCudaStmt == null ) {
//								firstCudaStmt = estmt;
//							}
							if( mallocPoint == region ) {
								num_cudastmts-=1;
							}
							if( opt_addSafetyCheckingCode ) {
								// Insert "gpuGmemSize += gpuBytes;" statement
								((CompoundStatement)mallocPoint.getParent()).addStatementBefore(mallocPoint,
										gMemAdd_stmt.clone());
								if( mallocPoint == region ) {
									num_cudastmts-=1;
								}
							}
						}

						// Create a parameter Declaration for the kernel function
						// - Change the scalar variable to a pointer type
						boolean useRegister = false;
						boolean ROData = false;
						if( cudaRegisterSet.contains(cloned_ID.getName()) ) {
							useRegister = true;
						}
						if( cudaRegisterROSet.contains(cloned_ID.getName()) ) {
							ROData = true;
						}
						VariableDeclarator pointerV_declarator = scalarVariableConv(cloned_declarator, new_proc, region,
								useRegister, ROData);
						pointer_var = new Identifier(pointerV_declarator);

						// Insert argument to the kernel function call
						call_to_new_proc.addArgument(gpu_var.clone());
					} else if( SymbolTools.isArray(shared_var) ) {
						// Insert "gpuBytes = (dimension1 * dimension2 * ..) * sizeof(varType);" statement
						List aspecs = shared_var.getArraySpecifiers();
						ArraySpecifier aspec = (ArraySpecifier)aspecs.get(0);
						int dimsize = aspec.getNumDimensions();
						VariableDeclaration pitch_decl = null;
						VariableDeclarator pointerV_declarator =  null;

						if( aspec.getDimension(0) == null ) {
							Tools.exit(pass_name + " [Error in transforming a parallel region in a function, " +
									proc.getSymbolName() + "()] the first dimension of a shared array, "
									+ shared_var + ", is missing; for the O2G translator " +
									"to allocate GPU memory for this array, the exact dimension size of accessed" +
									" array section should be specified." );
						}

						// cudaMallocPitch() is used only for 2 dimensional arrays.
						if( dimsize == 2 ) {
							use_MallocPitch = opt_MallocPitch;
						}
						else {
							use_MallocPitch = false;
						}
						if( cudaTextureSet.contains(cloned_ID.getName())) {
							if( dimsize == 1 ) {
								use_TextureMemory = true;
								use_MallocPitch = false;
							} else {
								use_TextureMemory = false;
								PrintTools.println("[WARNING] Texture memory can be allocated for 1 dimension array only" +
										", but shared variable, " + cloned_ID + ", is a " + dimsize +
										" dimension array; O2G translator will not apply texture memory!", 0);
							}
						} else {
							use_TextureMemory = false;
						}
/*						if( cudaSharedSet.contains(cloned_ID.getName())) {
							use_SharedMemory = true;
							use_TextureMemory = false;
							use_MallocPitch = false;
						}*/

						if( use_TextureMemory ) {
							/////////////////////////////////////////////////////////////////////////
							// Texture reference is declared in a TranslationUnit and can be used  //
							// by multiple procedures existing in the same TranslationUnit.        //
							// (cf. c2gMap is reset for each procedure.)                           //
							// CAUTION: Current translator calls cudaMalloc() at each procedure    //
							// for a shared variable; if a shared variable is accessed in multiple //
							// procedures, multiple texture references should be created.          //
							// If the translator is optimized to allocate GPU memory only once for //
							// each shared variable, only one texture reference will be needed.    //
							/////////////////////////////////////////////////////////////////////////
							// Check whether the enclosing TranslationUnit contains the texture //
							// reference of this shared variable.                               //
							//////////////////////////////////////////////////////////////////////
							str = new StringBuilder(80);
							str.append("texture__");
							if( opt_globalGMalloc && (orgSym != null) ) {
								str.append(orgSym.getSymbolName());
							} else {
								str.append(cloned_ID.toString());
							}
							str.append("_at_"+proc.getSymbolName());
							Set<Symbol> symbolset = SymbolTools.getSymbols(global_table);
							VariableDeclaration texture_decl = null;
							for (Symbol sym : symbolset)
							{
								if( sym.getSymbolName().compareTo(str.toString()) == 0 ) {
									texture_decl = (VariableDeclaration)((Declarator)sym).getParent();
									if (texture_decl != null) {
										break;
									}
								}
							}
							if( texture_decl == null ) {
								TextureSpecifier texturespec = new TextureSpecifier(clonedspecs);
								VariableDeclarator textureRef_declarator = new VariableDeclarator(new NameID(str.toString()));
								Declaration textureRef_decl = new VariableDeclaration(texturespec,
										textureRef_declarator);
								textureRefID = new Identifier(textureRef_declarator);
								tu.addDeclarationAfter(tu.getFirstDeclaration(), textureRef_decl);
							} else {
								textureRefID = new Identifier((VariableDeclarator)texture_decl.getDeclarator(0));
							}
						}

						if( use_MallocPitch ) {
							malloc_call = new FunctionCall(new NameID("cudaMallocPitch"));
							// Give a new name for a new pitch variable
							str = new StringBuilder(80);
							str.append("pitch_");
							if( opt_globalGMalloc && (orgSym != null) ) {
								str.append(orgSym.getSymbolName());
							} else {
								str.append(cloned_ID.toString());
							}
							if( symTable instanceof Procedure ) {
								str.append("__"+((Procedure)symTable).getSymbolName());
							}
							/////////////////////////////////////////////////////////////////////////
							// FIXME: is it correct to use insertMalloc instead of insertC2GMemTr? //
							/////////////////////////////////////////////////////////////////////////
							//if( insertC2GMemTr ) {
							//if( insertMalloc ) {
							if( insertMalloc2 ) {
								// Create a device local variable to keep pitch value
								// Ex: size_t pitch_b;
								// The type of the device symbol should be a pointer type
								VariableDeclarator pitch_declarator = new VariableDeclarator(new NameID(str.toString()));
								pitch_decl = new VariableDeclaration(CUDASpecifier.SIZE_T,
										pitch_declarator);
								pitch_var = new Identifier(pitch_declarator);
								if( insertGMalloc == 1 ) {
									main_TrUnt.addDeclarationAfter(lastMainCudaDecl, pitch_decl);
									lastMainCudaDecl = pitch_decl;
								} else {
									procbody.addDeclaration(pitch_decl);
								}
							} else {
								if( opt_globalGMalloc ) {
									pitch_decl = (VariableDeclaration)SymbolTools.findSymbol(main_TrUnt, str.toString());
									if( pitch_decl == null ) {
										pitch_decl = (VariableDeclaration)SymbolTools.findSymbol(procbody, str.toString());
									}
								} else {
									pitch_decl = (VariableDeclaration)SymbolTools.findSymbol(procbody, str.toString());
								}
								pitch_var = new Identifier((VariableDeclarator)pitch_decl.getDeclarator(0));
							}
							BinaryExpression biexp = new BinaryExpression(pitch_var.clone(),
									BinaryOperator.MULTIPLY, aspec.getDimension(0).clone());
							AssignmentExpression assignex = new AssignmentExpression(cloned_bytes.clone(),AssignmentOperator.NORMAL,
									biexp);
							gpuBytes_stmt = new ExpressionStatement(assignex);
							/*
							 * gpuBytes_stmt should be inserted after cudaMallocPitch() is called.
							 * Therefore, this statement is not inserted at this time
							 */
						} else {
							malloc_call = new FunctionCall(new NameID("cudaMalloc"));
							// Add malloc size (gpuBytes) statement
							// Ex: gpuBytes=(((2048+2)*(2048+2))*sizeof (float));
							Object o1 = aspec.getDimension(0).clone();
							Expression biexp = (Expression)o1;
							for( int i=1; i<dimsize; i++ )
							{
								Object o2 = aspec.getDimension(i).clone();
								if (o2 instanceof Expression)
									biexp = new BinaryExpression((Expression)o1, BinaryOperator.MULTIPLY, (Expression)o2);
								else
									throw new IllegalArgumentException("all list items must be Expressions; found a "
											+ o2.getClass().getName() + " instead");
								o1 = biexp;
							}
							BinaryExpression biexp2 = new BinaryExpression(biexp, BinaryOperator.MULTIPLY, sizeof_expr);
							AssignmentExpression assignex = new AssignmentExpression(cloned_bytes.clone(),AssignmentOperator.NORMAL,
									biexp2);
							ExpressionStatement estmt = new ExpressionStatement(assignex);
							gpuBytes_stmt = estmt.clone();
							//if( insertMalloc ) {
							if( insertMalloc2 ) {
								((CompoundStatement)mallocPoint.getParent()).addStatementBefore(mallocPoint, estmt);
/*								if( firstCudaStmt == null ) {
									firstCudaStmt = estmt;
								}*/
								if( mallocPoint == region ) {
									num_cudastmts-=1;
								}
								// Insert "gpuGmemSize += gpuBytes;" statement
								if( opt_addSafetyCheckingCode ) {
									((CompoundStatement)mallocPoint.getParent()).addStatementBefore(mallocPoint,
											gMemAdd_stmt.clone());
									if( mallocPoint == region ) {
										num_cudastmts-=1;
									}
								}
							}
						}

						arrayCachingOnRegister(region, cudaRegisterSet, cudaRegisterROSet, cloned_declarator);

						if( use_TextureMemory ) {
							DepthFirstIterator iter = new DepthFirstIterator(region);
							for (;;)
							{
								ArrayAccess aAccess = null;

								try {
									aAccess = (ArrayAccess)iter.next(ArrayAccess.class);
								} catch (NoSuchElementException e) {
									break;
								}
								IDExpression arrayID = (IDExpression)aAccess.getArrayName();
								if( arrayID.equals(cloned_ID) ) {
									FunctionCall texAccessCall = new FunctionCall(new NameID("tex1Dfetch"));
									texAccessCall.addArgument(textureRefID.clone());
									texAccessCall.addArgument(aAccess.getIndex(0).clone());
									aAccess.swapWith(texAccessCall);
								}
							}
						}

						// Create a parameter Declaration for the kernel function
						if( dimsize == 1 ) {
							// Change to a pointer type
							// Ex:  "float* b"
							pointerV_declarator = new VariableDeclarator(PointerSpecifier.UNQUALIFIED,
									new NameID(shared_var.getSymbolName()));
							VariableDeclaration pointerV_decl = new VariableDeclaration(cloned_decl.getSpecifiers(),
									pointerV_declarator);
							pointer_var = new Identifier(pointerV_declarator);
							new_proc.addDeclaration(pointerV_decl);
						} else {
							if( use_MallocPitch ) {
								VariableDeclaration cloned_pitch_decl = pitch_decl.clone();
								///////////////////////////////////////////////////////////////////
								// DEBUG: if declaration is added to a TranslationUnit, internal //
								// needs_semi_colon field is set to true and thus clone of that  //
								// declaration will add semicolon when printed. To remove the    //
								// semicolon, the field should be reset.                         //
								// cf: If declaration is regenenated the field will be reset by  //
								// default; no need to explicit reset.                           //
								///////////////////////////////////////////////////////////////////
								cloned_pitch_decl.setSemiColon(false);
								// Change to a pointer type
								// Ex:  "float* b"
								pointerV_declarator = pitchedAccessConv((VariableDeclarator)shared_var,
										new_proc, cloned_pitch_decl, region);
								pointer_var = new Identifier(pointerV_declarator);
								pitchMap.put(pointerV_declarator, cloned_pitch_decl);
							} else {
								// Keep the original array type
								// Ex: "float b[(2048+2)][(2048+2)]"
								new_proc.addDeclaration(cloned_decl);
							}
						}

						// Insert argument to the kernel function call
						if( dimsize == 1 ) {
							// Simply pass address of the pointer
							// Ex:  "gpu_b"
							call_to_new_proc.addArgument(gpu_var.clone());
						} else {
							if( use_MallocPitch ) {
								// Simply pass address of the pointer
								// Ex:  "gpu_b"
								call_to_new_proc.addArgument(gpu_var.clone());
								// Insert corresponding pitch argument to the kernel function call
								call_to_new_proc.addArgument(pitch_var.clone());
							} else {
								//Cast the gpu variable to pointer-to-array type
								// Ex: (float (*)[dimesion2]) gpu_b
								List castspecs = new LinkedList();
								castspecs.addAll(cloned_decl.getSpecifiers());
								/*
								 * FIXME: ArrayAccess was used for (*)[SIZE2], but this may not be
								 * semantically correct way to represent (*)[SIZE2] in IR.
								 */
								List tindices = new LinkedList();
								for( int i=1; i<dimsize; i++) {
									tindices.add(aspec.getDimension(i).clone());
								}
								ArrayAccess castArray = new ArrayAccess(new NameID("(*)"), tindices);
								castspecs.add(castArray);
								call_to_new_proc.addArgument(new Typecast(castspecs, gpu_var.clone()));
							}
						}

						// Replace all instances of the shared variable to the parameter variable
						// Below replacement seems to be OK since Identifier.equals() method compares string name only
						if( dimsize == 1 ) {
							IRTools.replaceAll(region, cloned_ID, pointer_var);
						} else if( !use_MallocPitch ){
							//pitchedAccessConv() handles necessary replacements for use_MallocPitch case.
							//Thus, there is no need to be handled here.
							IRTools.replaceAll(region, cloned_ID, cloned_ID);
						}
					} else {
						Tools.exit(pass_name + "[ERROR] extractKernelRegion() found unsupported shared symbols."
								+ shared_var.toString());
					}
					BinaryExpression hostWidthBytes = null;
					ArraySpecifier aspec = null;
					if( use_MallocPitch ) {
						arg_list.add(new UnaryExpression(UnaryOperator.ADDRESS_OF,pitch_var.clone()));
						List aspecs = shared_var.getArraySpecifiers();
						aspec = (ArraySpecifier)aspecs.get(0);
						hostWidthBytes = new BinaryExpression(aspec.getDimension(1).clone(),
								BinaryOperator.MULTIPLY, sizeof_expr.clone());
						arg_list.add(hostWidthBytes);
						arg_list.add(aspec.getDimension(0).clone());
						malloc_call.setArguments(arg_list);
					} else {
						// Add gpuBytes argument to cudaMalloc() call
						//cloned_bytes = (Identifier)bytes_decl.getDeclarator(0).getSymbol().clone();
						arg_list.add(cloned_bytes.clone());
						malloc_call.setArguments(arg_list);
					}
					FunctionCall safe_call = new FunctionCall( new NameID("CUDA_SAFE_CALL") );
					safe_call.addArgument(malloc_call);
					ExpressionStatement malloc_stmt = new ExpressionStatement(safe_call);
					//if( insertMalloc ) {
					if( insertMalloc2 ) {
						((CompoundStatement)mallocPoint.getParent()).addStatementBefore(mallocPoint, malloc_stmt);
/*						if( firstCudaStmt == null ) {
							firstCudaStmt = malloc_stmt;
						}*/
						if( mallocPoint == region ) {
							num_cudastmts-=1;
						}
						/*
						 * If cudaMallocPitch() is used, gpuBytes_stmt should be inserted after cudaMallocPitch()
						 * is inserted. Current implementation uses cudaMemcpy2D() for mallocpitched data, and
						 * thus gpuBytes_stmt is needed only when safety checking code is added.
						 */
						if( use_MallocPitch ) {
							if( opt_addSafetyCheckingCode ) {
								///////////////////////////
								// Insert gpuBytes_stmt. //
								///////////////////////////
								((CompoundStatement)mallocPoint.getParent()).addStatementBefore(mallocPoint,
										gpuBytes_stmt.clone());
								if( mallocPoint == region ) {
									num_cudastmts-=1;
								}
								/////////////////////////////////////
								// Insert pitch size checkin code. //
								//////////////////////////////////////////////////////////////////////////////////////
								//    if( pitch_x > MAX_PITCH ) {                                                   //
								//        printf("Size of pitch, pitch_x, is bigger than the maximum size;\n");     //
								//        printf("Please turn off usaMallocPitch option \n");                       //
								//        exit(1);                                                                  //
								//    }                                                                             //
								//////////////////////////////////////////////////////////////////////////////////////
								Expression condExp = new BinaryExpression(pitch_var.clone(),
										BinaryOperator.COMPARE_GT, new NameID("MAX_PITCH"));
								CompoundStatement ifBody = new CompoundStatement();
								FunctionCall printfCall = new FunctionCall(new NameID("printf"));
								printfCall.addArgument(new StringLiteral("Size (%d) of pitch, "+pitch_var.toString()+
								", is bigger than the maximum size (%d); \\n"));
								printfCall.addArgument(pitch_var.clone());
								printfCall.addArgument(new NameID("MAX_PITCH"));
								ifBody.addStatement(new ExpressionStatement(printfCall));
								printfCall = new FunctionCall(new NameID("printf"));
								printfCall.addArgument(new StringLiteral("Please turn off useMallocPitch option.\\n"));
								ifBody.addStatement(new ExpressionStatement(printfCall));
								FunctionCall exitCall = new FunctionCall(new NameID("exit"));
								exitCall.addArgument(new IntegerLiteral(1));
								ifBody.addStatement(new ExpressionStatement(exitCall));
								((CompoundStatement)mallocPoint.getParent()).addStatementBefore(mallocPoint,
										new IfStatement(condExp, ifBody));
								((CompoundStatement)mallocPoint.getParent()).addStatementBefore(mallocPoint,
										gMemAdd_stmt.clone());
								if( mallocPoint == region ) {
									num_cudastmts-=2;
								}
							}
						}
					}
					if( use_TextureMemory ) {
						str = new StringBuilder(80);
						str.append("texture__");
						str.append(cloned_ID.toString());
						str.append("_at_"+proc.getSymbolName());
						if( insertMalloc2 || (!c2gMap.containsKey(str.toString())) ) {
							c2gMap.put(str.toString(), textureRefID);
							PrintTools.println("Texture Mapping (GPU variable: "
									+gpu_var+" binded to "+textureRefID+")", 3);
							///////////////////////////////////////////////////////////////////
							// If cudaMalloc for this variable is called during the previous //
							// kernel transformation, correct gpyBytes statement should be   //
							// inserted before adding current texture-binding statements.    //
							///////////////////////////////////////////////////////////////////
							if( !insertMalloc2 ) {
								((CompoundStatement)mallocPoint.getParent()).addStatementBefore(mallocPoint,
									gpuBytes_stmt.clone());
							}
							//////////////////////////////////////////////////////
							// Add texture-bound GPU memory usage check code.   //
							//////////////////////////////////////////////////////
							//     if( gpuBytes > LMAX_WIDTH ) {                //
							//         printf("[Error] ....");                  //
							//         exit(1);                                 //
							//     }                                            //
							//////////////////////////////////////////////////////
							Expression MemCheckExp = new BinaryExpression(cloned_bytes.clone(),
									BinaryOperator.COMPARE_GT, new NameID("LMAX_WIDTH"));
							FunctionCall MemWarningCall = new FunctionCall(new NameID("printf"));
							StringLiteral warningMsg = new StringLiteral("[Error] size of linear memory " +
									"bound to a texture reference" +
							" (%d) exceeds the given limit (%d)\\n");
							MemWarningCall.addArgument(warningMsg);
							MemWarningCall.addArgument(cloned_bytes.clone());
							MemWarningCall.addArgument( new NameID("LMAX_WIDTH"));
							FunctionCall ExitCall = new FunctionCall(new NameID("exit"));
							ExitCall.addArgument(new IntegerLiteral(1));
							CompoundStatement ifbody = new CompoundStatement();
							ifbody.addStatement(new ExpressionStatement(MemWarningCall));
							ifbody.addStatement(new ExpressionStatement(ExitCall));
							IfStatement MemCheckStmt = new IfStatement(MemCheckExp,
									ifbody);
							((CompoundStatement)mallocPoint.getParent()).addStatementBefore(
									mallocPoint, MemCheckStmt);
							//////////////////////////////////////////////////////////////
							// Bind a texture reference to a GPU global memory.         //
							//////////////////////////////////////////////////////////////
							//    cudaBindTexture(0, texture__var, gpu__var, gpuBytes); //
							//////////////////////////////////////////////////////////////
							FunctionCall textureBindCall = new FunctionCall(new NameID("cudaBindTexture"));
							textureBindCall.addArgument(new IntegerLiteral(0));
							textureBindCall.addArgument(textureRefID.clone());
							textureBindCall.addArgument(gpu_var.clone());
							textureBindCall.addArgument(cloned_bytes.clone());
							((CompoundStatement)mallocPoint.getParent()).addStatementBefore(
									mallocPoint, new ExpressionStatement(textureBindCall));
							if( mallocPoint == region ) {
								num_cudastmts-=2;
							}
						}
					}

					/*
					 * Insert cudaFree() to deallocate device memory.
					 * Because cuda-related statements are added in reverse order,
					 * this function call is added first.
					 */
					//if( insertMalloc ) {
					if( insertFree2 || insertGFree ) {
						if( opt_addSafetyCheckingCode && !insertGFree ) {
							// Insert "gpuGmemSize -= gpuBytes;" statement
							((CompoundStatement)mallocPoint.getParent()).addStatementAfter(mallocPoint,
									gMemSub_stmt.clone());
							if( mallocPoint == region ) {
								num_cudastmts-=1;
							}
						}
						// Insert "CUDA_SAFE_CALL(cudaFree(gpu_a));"
						FunctionCall cudaFree_call = new FunctionCall(new NameID("cudaFree"));
						cudaFree_call.addArgument(gpu_var.clone());
						safe_call = new FunctionCall( new NameID("CUDA_SAFE_CALL"));
						safe_call.addArgument(cudaFree_call);
						ExpressionStatement cudaFree_stmt = new ExpressionStatement(safe_call);
						if( insertGFree ) {
							for( Statement lastFlushStmt : FlushStmtList ) {
								((CompoundStatement)lastFlushStmt.getParent()).addStatementBefore(lastFlushStmt,
										cudaFree_stmt.clone());
							}
						} else {
							((CompoundStatement)mallocPoint.getParent()).addStatementAfter(mallocPoint, cudaFree_stmt);
							// Remove mapping from shared_var to gpu_declarator
							//c2gMap.remove(shared_var);
							if( mallocPoint == region ) {
								num_cudastmts-=1;
							}
							if( mallocPoint != refstmt ) {
								((CompoundStatement)mallocPoint.getParent()).addStatementAfter(mallocPoint,
										gpuBytes_stmt.clone());
								if( mallocPoint == region ) {
									num_cudastmts-=1;
								}
							}
						}
					}

					if( use_MallocPitch ) {
						if( insertC2GMemTr ) {
							/* Insert memory copy function from CPU to GPU */
							// Ex: CUDA_SAFE_CALL(cudaMemcpy2D(gpu_b, pitch_b, b, width*sizeof(float),
							// width*sizeof(float), height, cudaMemcpyHostToDevice));
							FunctionCall memCopy_call = new FunctionCall(new NameID("cudaMemcpy2D"));
							List<Expression> arg_list2 = new ArrayList<Expression>();
							arg_list2.add(gpu_var.clone());
							arg_list2.add(pitch_var.clone());
							arg_list2.add(new Identifier(shared_var));
							arg_list2.add(hostWidthBytes.clone());
							arg_list2.add(hostWidthBytes.clone());
							arg_list2.add(aspec.getDimension(0).clone());
							arg_list2.add(new NameID("cudaMemcpyHostToDevice"));
							memCopy_call.setArguments(arg_list2);
							safe_call = new FunctionCall( new NameID("CUDA_SAFE_CALL"));
							safe_call.addArgument(memCopy_call);
							ExpressionStatement memCopy_stmt = new ExpressionStatement(safe_call);
							((CompoundStatement)refstmt.getParent()).addStatementBefore(refstmt, memCopy_stmt);
							if( firstCudaStmt == null ) {
								firstCudaStmt = memCopy_stmt;
							}
							if( refstmt == region ) {
								num_cudastmts--;
							}
						}
						if( insertC2GMemTr2 ) {
							/* Insert memory copy function from CPU to GPU */
							// Ex: CUDA_SAFE_CALL(cudaMemcpy2D(gpu_b, pitch_b, b, width*sizeof(float),
							// width*sizeof(float), height, cudaMemcpyHostToDevice));
							FunctionCall memCopy_call = new FunctionCall(new NameID("cudaMemcpy2D"));
							List<Expression> arg_list2 = new ArrayList<Expression>();
							arg_list2.add(gpu_var.clone());
							arg_list2.add(pitch_var.clone());
							arg_list2.add(new Identifier(shared_var));
							arg_list2.add(hostWidthBytes.clone());
							arg_list2.add(hostWidthBytes.clone());
							arg_list2.add(aspec.getDimension(0).clone());
							arg_list2.add(new NameID("cudaMemcpyHostToDevice"));
							memCopy_call.setArguments(arg_list2);
							safe_call = new FunctionCall( new NameID("CUDA_SAFE_CALL"));
							safe_call.addArgument(memCopy_call);
							ExpressionStatement memCopy_stmt = new ExpressionStatement(safe_call);
							((CompoundStatement)region.getParent()).addStatementBefore(region, memCopy_stmt);
							if( firstCudaStmt == null ) {
								firstCudaStmt = memCopy_stmt;
							}
							num_cudastmts--;
						}
						if( insertG2CMemTr ) {
							/* Insert memory copy function from GPU to CPU */
							// Ex: CUDA_SAFE_CALL(cudaMemcpy2D(b, width*sizeof(float), gpu_b, pitch_b,
							// width*sizeof(float), height, cudaMemcpyHostToDevice));
							FunctionCall memCopy_call2 = new FunctionCall(new NameID("cudaMemcpy2D"));
							List<Expression> arg_list3 = new ArrayList<Expression>();
							arg_list3.add(new Identifier(shared_var));
							arg_list3.add(hostWidthBytes.clone());
							arg_list3.add(gpu_var.clone());
							arg_list3.add(pitch_var.clone());
							arg_list3.add(hostWidthBytes.clone());
							arg_list3.add(aspec.getDimension(0).clone());
							arg_list3.add(new NameID("cudaMemcpyDeviceToHost"));
							memCopy_call2.setArguments(arg_list3);
							safe_call = new FunctionCall( new NameID("CUDA_SAFE_CALL"));
							safe_call.addArgument(memCopy_call2);
							ExpressionStatement memCopy_stmt = new ExpressionStatement(safe_call);
							((CompoundStatement)refstmt.getParent()).addStatementAfter(refstmt, memCopy_stmt);
							((CompoundStatement)refstmt.getParent()).addStatementAfter(refstmt, gpuBytes_stmt);
							if( refstmt == region ) {
								num_cudastmts-=2;
							}
						}
						if( insertG2CMemTr2 ) {
							/* Insert memory copy function from GPU to CPU */
							// Ex: CUDA_SAFE_CALL(cudaMemcpy2D(b, width*sizeof(float), gpu_b, pitch_b,
							// width*sizeof(float), height, cudaMemcpyHostToDevice));
							FunctionCall memCopy_call2 = new FunctionCall(new NameID("cudaMemcpy2D"));
							List<Expression> arg_list3 = new ArrayList<Expression>();
							arg_list3.add(new Identifier(shared_var));
							arg_list3.add(hostWidthBytes.clone());
							arg_list3.add(gpu_var.clone());
							arg_list3.add(pitch_var.clone());
							arg_list3.add(hostWidthBytes.clone());
							arg_list3.add(aspec.getDimension(0).clone());
							arg_list3.add(new NameID("cudaMemcpyDeviceToHost"));
							memCopy_call2.setArguments(arg_list3);
							safe_call = new FunctionCall( new NameID("CUDA_SAFE_CALL"));
							safe_call.addArgument(memCopy_call2);
							ExpressionStatement memCopy_stmt = new ExpressionStatement(safe_call);
							((CompoundStatement)region.getParent()).addStatementAfter(region, memCopy_stmt);
							((CompoundStatement)region.getParent()).addStatementAfter(region,
									gpuBytes_stmt.clone());
							num_cudastmts-=2;
						}
					} else {
						if( insertC2GMemTr ) {
							/* Insert memory copy function from CPU to GPU */
							// Ex: CUDA_SAFE_CALL(cudaMemcpy(gpu_b, b, gpuBytes, cudaMemcpyHostToDevice));
							if( !insertMalloc2 || (mallocPoint != refstmt) ) {
								Statement gpuBytesStmt = gpuBytes_stmt.clone();
								((CompoundStatement)refstmt.getParent()).addStatementBefore(refstmt,
										gpuBytesStmt);
								if( firstCudaStmt == null ) {
									firstCudaStmt = gpuBytesStmt;
								}
								if( refstmt == region ) {
									num_cudastmts-=1;
								}
							}
							FunctionCall memCopy_call = new FunctionCall(new NameID("cudaMemcpy"));
							List<Expression> arg_list2 = new ArrayList<Expression>();
							arg_list2.add(gpu_var.clone());
							if( SymbolTools.isScalar(shared_var)) {
								arg_list2.add( new UnaryExpression(UnaryOperator.ADDRESS_OF,
												new Identifier(shared_var)));
							} else {
								arg_list2.add(new Identifier(shared_var));
							}
							arg_list2.add(cloned_bytes.clone());
							arg_list2.add(new NameID("cudaMemcpyHostToDevice"));
							memCopy_call.setArguments(arg_list2);
							safe_call = new FunctionCall( new NameID("CUDA_SAFE_CALL"));
							safe_call.addArgument(memCopy_call);
							ExpressionStatement memCopy_stmt = new ExpressionStatement(safe_call);
							((CompoundStatement)refstmt.getParent()).addStatementBefore(refstmt, memCopy_stmt);
							if( firstCudaStmt == null ) {
								firstCudaStmt = memCopy_stmt;
							}
							if( refstmt == region ) {
								num_cudastmts-=1;
							}
						}
						if( insertC2GMemTr2 ) {
							/* Insert memory copy function from CPU to GPU */
							// Ex: CUDA_SAFE_CALL(cudaMemcpy(gpu_b, b, gpuBytes, cudaMemcpyHostToDevice));
							if( !insertMalloc2 || (mallocPoint != region) ) {
								Statement gpuBytesStmt = gpuBytes_stmt.clone();
								((CompoundStatement)region.getParent()).addStatementBefore(region,
										gpuBytesStmt);
								if( firstCudaStmt == null ) {
									firstCudaStmt = gpuBytesStmt;
								}
								num_cudastmts-=1;
							}
							FunctionCall memCopy_call = new FunctionCall(new NameID("cudaMemcpy"));
							List<Expression> arg_list2 = new ArrayList<Expression>();
							arg_list2.add(gpu_var.clone());
							if( SymbolTools.isScalar(shared_var)) {
								arg_list2.add( new UnaryExpression(UnaryOperator.ADDRESS_OF,
												new Identifier(shared_var)));
							} else {
								arg_list2.add(new Identifier(shared_var));
							}
							arg_list2.add(cloned_bytes.clone());
							arg_list2.add(new NameID("cudaMemcpyHostToDevice"));
							memCopy_call.setArguments(arg_list2);
							safe_call = new FunctionCall( new NameID("CUDA_SAFE_CALL"));
							safe_call.addArgument(memCopy_call);
							ExpressionStatement memCopy_stmt = new ExpressionStatement(safe_call);
							((CompoundStatement)region.getParent()).addStatementBefore(region, memCopy_stmt);
							if( firstCudaStmt == null ) {
								firstCudaStmt = memCopy_stmt;
							}
							num_cudastmts-=1;
						}
						if( insertG2CMemTr ) {
							/* Insert memory copy function from GPU to CPU */
							// Ex: gpuBytes = (4096 * sizeof(float));
							//     CUDA_SAFE_CALL(cudaMemcpy(a, gpu_a, gpuBytes, cudaMemcpyDeviceToHost));
							FunctionCall memCopy_call2 = new FunctionCall(new NameID("cudaMemcpy"));
							List<Expression> arg_list3 = new ArrayList<Expression>();
							if( SymbolTools.isScalar(shared_var)) {
								arg_list3.add( new UnaryExpression(UnaryOperator.ADDRESS_OF,
												new Identifier(shared_var)));
							} else {
								arg_list3.add(new Identifier(shared_var));
							}
							arg_list3.add(gpu_var.clone());
							arg_list3.add((Identifier)cloned_bytes.clone());
							arg_list3.add(new NameID("cudaMemcpyDeviceToHost"));
							memCopy_call2.setArguments(arg_list3);
							safe_call = new FunctionCall( new NameID("CUDA_SAFE_CALL"));
							safe_call.addArgument(memCopy_call2);
							ExpressionStatement memCopy_stmt = new ExpressionStatement(safe_call);
							((CompoundStatement)refstmt.getParent()).addStatementAfter(refstmt, memCopy_stmt);
							((CompoundStatement)refstmt.getParent()).addStatementAfter(refstmt, gpuBytes_stmt);
							if( refstmt == region ) {
								num_cudastmts-=2;
							}
						}
						if( insertG2CMemTr2 ) {
							/* Insert memory copy function from GPU to CPU */
							// Ex: gpuBytes = (4096 * sizeof(float));
							//     CUDA_SAFE_CALL(cudaMemcpy(a, gpu_a, gpuBytes, cudaMemcpyDeviceToHost));
							FunctionCall memCopy_call2 = new FunctionCall(new NameID("cudaMemcpy"));
							List<Expression> arg_list3 = new ArrayList<Expression>();
							if( SymbolTools.isScalar(shared_var)) {
								arg_list3.add( new UnaryExpression(UnaryOperator.ADDRESS_OF,
												new Identifier((VariableDeclarator)shared_var)));
							} else {
								arg_list3.add(new Identifier((VariableDeclarator)shared_var));
							}
							arg_list3.add((Identifier)gpu_var.clone());
							arg_list3.add((Identifier)cloned_bytes.clone());
							arg_list3.add(new NameID("cudaMemcpyDeviceToHost"));
							memCopy_call2.setArguments(arg_list3);
							safe_call = new FunctionCall( new NameID("CUDA_SAFE_CALL"));
							safe_call.addArgument(memCopy_call2);
							ExpressionStatement memCopy_stmt = new ExpressionStatement(safe_call);
							((CompoundStatement)region.getParent()).addStatementAfter(region, memCopy_stmt);
							((CompoundStatement)region.getParent()).addStatementAfter(region,
									(Statement)gpuBytes_stmt.clone());
							num_cudastmts-=2;
						}
					}
				} else {
					Tools.exit(pass_name + "[ERROR] extractKernelRegion() supports VariableDeclarator shared symbols only;" +
							"current procedure = " + proc.getSymbolName() + " current symbol = " + shared_var);
				}
			}
		}

		////////////////////////////////////
		// Handle OMP ThreadPrivate data. //
		////////////////////////////////////
		if (OmpThreadPrivSet != null) {
			if( OmpThreadPrivSet.size() > 0 ) {
				refstmt2 = region;
				if( gridSizeNotChanged ) {
					mallocPoint = refstmt1;
				} else {
					mallocPoint = region;
				}
			}
			for( Symbol threadPriv_var : OmpThreadPrivSet ) {
				if( threadPriv_var instanceof VariableDeclarator ) {
					VariableDeclaration decl = (VariableDeclaration)((VariableDeclarator)threadPriv_var).getParent();
					/*
					 * Create a cloned Declaration of the threadprivate variable.
					 */
					VariableDeclarator cloned_declarator =
						(VariableDeclarator)((VariableDeclarator)threadPriv_var).clone();
					cloned_declarator.setInitializer(null);
					/////////////////////////////////////////////////////////////////////////////////
					// __device__ and __global__ functions can not declare static variables inside //
					// their body.                                                                 //
					/////////////////////////////////////////////////////////////////////////////////
					List<Specifier> clonedspecs = new ChainedList<Specifier>();
					clonedspecs.addAll(decl.getSpecifiers());
					clonedspecs.remove(Specifier.STATIC);
					VariableDeclaration cloned_decl = new VariableDeclaration(clonedspecs, cloned_declarator);
					Identifier cloned_ID = new Identifier(cloned_declarator);
					VariableDeclarator gpu_declarator = null;
					VariableDeclarator extended_declarator = null;
					Identifier gpu_var = null;
					Identifier pitch_var = null;
					Identifier pointer_var = null;
					Identifier extended_var = null;
					Identifier array_var = null;
					ArraySpecifier aspec = null;
					int dimsize = 0;

					/*
					 * c2gMap contains a mapping from a shared/threadprivate variable to corresponding GPU variable.
					 */
					if( c2gMap.containsKey(threadPriv_var)) {
						// clone GPU device symbol corresponding to threadPriv_var
						gpu_declarator = (VariableDeclarator)c2gMap.get(threadPriv_var);
						gpu_var = new Identifier(gpu_declarator);
					} else {
						// Create a GPU device variable corresponding to threadPriv_var
						// Ex: float * gpu_b;
						// Give a new name for the device variable
						StringBuilder str = new StringBuilder(80);
						str.append("gpu__");
						str.append(cloned_ID.toString());
						// The type of the device symbol should be a pointer type
						gpu_declarator = new VariableDeclarator(PointerSpecifier.UNQUALIFIED,
								new NameID(str.toString()));
						VariableDeclaration gpu_decl = new VariableDeclaration(cloned_decl.getSpecifiers(),
								gpu_declarator);
						gpu_var = new Identifier(gpu_declarator);
						procbody.addDeclaration(gpu_decl);
						// Add mapping from threadPriv_var to gpu_declarator
						c2gMap.put(threadPriv_var, gpu_declarator);
					}

					/////////////////////////////
					// Check duplicate Malloc. //
					/////////////////////////////
					boolean insertMalloc = false;
					HashSet<String> memTrSet = null;
					StringBuilder str = new StringBuilder(80);
					str.append("malloc_");
					str.append(gpu_var.getName());
					if( c2gMemTr.containsKey(mallocPoint) ) {
						memTrSet = (HashSet<String>)c2gMemTr.get(mallocPoint);
						if( !memTrSet.contains(str.toString()) ) {
							memTrSet.add(str.toString());
							insertMalloc = true;
						}
					} else {
						memTrSet = new HashSet<String>();
						memTrSet.add(str.toString());
						c2gMemTr.put(mallocPoint, memTrSet);
						insertMalloc = true;
					}

					boolean insertC2GMemTr = false;
					/*
					 * For Omp Threadprivate variable, "CPU to GPU memory transfer" is needed only if
					 * the variable is included in Omp copyin clause.
					 */
					boolean containSymbol = false;
					if( AnalysisTools.containsSymbol(OmpCopyinSet, threadPriv_var.getSymbolName()) ) {
						containSymbol = true;
					}
					insertC2GMemTr = containSymbol;
					//////////////////////////////////////////////////////////////////////////////
					// FIXME: below code block is useless in that it can not identify redundant //
					//        memory allocation or C2G memory transfers; to find redundant      //
					//        memory allocation, involved kernel regions should be invoked with //
					//        the same number of threads.                                       //
					//////////////////////////////////////////////////////////////////////////////
/*					if( c2gMemTr.containsKey(refstmt2) ) {
						memTrSet = (HashSet<String>)c2gMemTr.get(refstmt2);
						if( !memTrSet.contains(gpu_var.getName()) ) {
							memTrSet.add(gpu_var.getName());
							insertMalloc = true;
							insertC2GMemTr = containSymbol;
						}
					} else {
						memTrSet = new HashSet<String>();
						memTrSet.add(gpu_var.getName());
						c2gMemTr.put(refstmt2, memTrSet);
						insertMalloc = true;
						insertC2GMemTr = containSymbol;
					}*/
					/*
					 * Check duplicate GPU to CPU memory transfers
					 * Currently, simple name-only analysis is conducted; if the same array
					 * is transferred multiply at the same program point, insert only one memory transfer.
					 */
					boolean insertG2CMemTr = true;
					//////////////////////////////////////////////////////////////////////////////
					// FIXME: below code block is useless in that it can not identify redundant //
					//        G2C memory transfers; to find redundant memory transfers,         //
					//        involved kernel regions should be invoked with the same number of //
					//        threads.                                                          //
					//////////////////////////////////////////////////////////////////////////////
/*					boolean insertG2CMemTr = false;
					if( g2cMemTr.containsKey(refstmt2) ) {
						memTrSet = (HashSet<String>)g2cMemTr.get(refstmt2);
						if( !memTrSet.contains(gpu_var.getName()) ) {
							memTrSet.add(gpu_var.getName());
							insertG2CMemTr = true;
						}
					} else {
						memTrSet = new HashSet<String>();
						memTrSet.add(gpu_var.getName());
						g2cMemTr.put(refstmt2, memTrSet);
						insertG2CMemTr = true;
					}*/
					///////////////////////////////////////////////////////////////////////////
					// If Cuda nog2cmemtr clause contains this symbol, insertG2CMemTr should //
					// set to false.                                                         //
					///////////////////////////////////////////////////////////////////////////
					if( cudaNoG2CMemTrSet.contains(cloned_ID.getName()) ) {
						insertG2CMemTr = false;
					}

					// Memory allocation for the device variable
					// Insert cudaMalloc() function before the region
					// Ex: CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu_b)), gpuBytes));
					// Ex2: CUDA_SAFE_CALL( cudaMallocPitch((void**) &d_x, &pitch_x, width*sizeof(float), height) );
					FunctionCall malloc_call = null;
					List<Specifier> specs = new ArrayList<Specifier>(4);
					specs.add(Specifier.VOID);
					specs.add(PointerSpecifier.UNQUALIFIED);
					specs.add(PointerSpecifier.UNQUALIFIED);
					List<Expression> arg_list = new ArrayList<Expression>();
					arg_list.add(new Typecast(specs, new UnaryExpression(UnaryOperator.ADDRESS_OF,
							(Identifier)gpu_var.clone())));
					SizeofExpression sizeof_expr = new SizeofExpression(cloned_decl.getSpecifiers());
					if( SymbolTools.isPointer(threadPriv_var) ) {
						Tools.exit(pass_name + "[ERROR] extractKernelRegion() needs to support Pointer type threadprivate variable: "
								+ threadPriv_var.toString());
					} else if( SymbolTools.isScalar(threadPriv_var) ) {
						use_MallocPitch = false;
						malloc_call = new FunctionCall(new NameID("cudaMalloc"));
						// Insert "gpuBytes = totalNumThreads * sizeof(varType);" statement
						AssignmentExpression assignex = new AssignmentExpression((Identifier)cloned_bytes.clone(),
								AssignmentOperator.NORMAL, new BinaryExpression((Expression)totalNumThreads.clone(),
										BinaryOperator.MULTIPLY, sizeof_expr));
						ExpressionStatement estmt = new ExpressionStatement(assignex);
						gpuBytes_stmt = (ExpressionStatement)estmt.clone();
						assignex = new AssignmentExpression((Identifier)cloned_bytes.clone(),
								AssignmentOperator.NORMAL, (Expression)sizeof_expr.clone());
						//Generate "gpuBytes = sizeof(varType)" statement, which will be used for G2C transfer.
						orgTPBytes_stmt = new ExpressionStatement(assignex);
						if( insertMalloc ) {
							((CompoundStatement)mallocPoint.getParent()).addStatementBefore(mallocPoint, estmt);
							if( firstCudaStmt2 == null ) {
								firstCudaStmt2 = estmt;
							}
							if( mallocPoint == region ) {
								num_cudastmts-=1;
							}
							if( opt_addSafetyCheckingCode ) {
								// Insert "gpuGmemSize += gpuBytes;" statement
								((CompoundStatement)mallocPoint.getParent()).addStatementBefore(mallocPoint,
										(Statement)gMemAdd_stmt.clone());
								if( mallocPoint == region ) {
									num_cudastmts-=1;
								}
							}
						}

						// Create a parameter Declaration for the kernel function
						// Change the scalar variable to a pointer type
						boolean useRegister = false;
						boolean ROData = false;
						if( cudaRegisterSet.contains(cloned_ID.getName())) {
							useRegister = true;
						}
						if( cudaRegisterROSet.contains(cloned_ID.getName())) {
							ROData = true;
						}
						VariableDeclarator pointerV_declarator =
							scalarVariableConv2(cloned_declarator, new_proc, region, useRegister, ROData);
						pointer_var = new Identifier(pointerV_declarator);

						// Insert argument to the kernel function call
						call_to_new_proc.addArgument((Identifier)gpu_var.clone());
					} else if( SymbolTools.isArray(threadPriv_var) ) {
						// Insert "gpuBytes = totalNumThreads * (dimension1 * dimension2 * ..)
						// * sizeof(varType);" statement
						List aspecs = threadPriv_var.getArraySpecifiers();
						aspec = (ArraySpecifier)aspecs.get(0);
						dimsize = aspec.getNumDimensions();
						VariableDeclaration pitch_decl = null;
						VariableDeclarator pointerV_declarator =  null;
						VariableDeclarator arrayV_declarator =  null;
						////////////////////////////////////////////////////////////////////////////////
						// For threadprivate data, cudaMallocPitch() is used only for MatrixTranspose //
						// optimization.                                                              //
						// DEBUG: Matrix transpose without using cudaMallocPitch() is not possible if //
						// total number of threads are not known at compile time; When a[SIZE1] is    //
						// expanded to a[SIZE1][# of threads], the right-most dimension size should   //
						// be known at compile time. Otherwise, it will cause compile error.          //
						// Current O2G translator always uses cudaMallocPitch() for MatrixTranspose,  //
						// and thus if the pich size is too big, CUDA compiler may not be able to     //
						// compile it.                                                                //
						////////////////////////////////////////////////////////////////////////////////
						if( dimsize == 1 ) {
							use_MallocPitch = opt_MatrixTranspose;
						}
						else {
							use_MallocPitch = false;
						}
						if( aspec.getDimension(0) == null ) {
							Tools.exit(pass_name + " [Error in transforming a parallel region in a function, " +
									proc.getSymbolName() + "()] the first dimension of a threadprivate array, "
									+ threadPriv_var + ", is missing; for the O2G translator " +
									"to allocate GPU memory for this array, the exact dimension size of accessed" +
									" array section should be specified." );
						}
						/*
						 * Create cudaMalloc() or cudaMallocPitch() statement for GPU variable.
						 */
						if( use_MallocPitch ) {
							malloc_call = new FunctionCall(new NameID("cudaMallocPitch"));
							// Give a new name for a new pitch variable
							str = new StringBuilder(80);
							str.append("pitch_");
							str.append(cloned_ID.toString());
							if( insertMalloc ) {
								// Create a device local variable to keep pitch value
								// Ex: size_t pitch_b;
								// The type of the device symbol should be a pointer type
								VariableDeclarator pitch_declarator = new VariableDeclarator(new NameID(str.toString()));
								pitch_decl = new VariableDeclaration(CUDASpecifier.SIZE_T,
										pitch_declarator);
								pitch_var = new Identifier(pitch_declarator);
								procbody.addDeclaration(pitch_decl);
							} else {
								pitch_decl = (VariableDeclaration)SymbolTools.findSymbol(procbody, str.toString());
								pitch_var = new Identifier((VariableDeclarator)pitch_decl.getDeclarator(0));
							}
							BinaryExpression biexp = new BinaryExpression((Expression)pitch_var.clone(),
									BinaryOperator.MULTIPLY, (Expression)aspec.getDimension(0).clone());
							AssignmentExpression assignex = new AssignmentExpression((Identifier)cloned_bytes.clone(),
									AssignmentOperator.NORMAL, biexp);
							/*
							 * gpuBytes_stmt should be inserted after cudaMallocPitch() is called.
							 * Therefore, this statement is not inserted at this time
							 */
							gpuBytes_stmt = new ExpressionStatement(assignex);
							orgTPBytes_stmt = (ExpressionStatement)gpuBytes_stmt.clone();
							// Insert argument to the kernel function call
							//call_to_new_proc.addArgument((Identifier)pitch_var.clone());
							// Insert a parameter for the kernel procedure
							//new_proc.addDeclaration((Declaration)pitch_decl.clone());
						} else {
							malloc_call = new FunctionCall(new NameID("cudaMalloc"));
							// Add malloc size (gpuBytes) statement
							// Ex: gpuBytes= totalNumThreads * (((2048+2)*(2048+2))*sizeof (float));
							Object o1 = aspec.getDimension(0).clone();
							Expression biexp = (Expression)o1;
							for( int i=1; i<dimsize; i++ )
							{
								Object o2 = aspec.getDimension(i).clone();
								if (o2 instanceof Expression)
									biexp = new BinaryExpression((Expression)o1, BinaryOperator.MULTIPLY, (Expression)o2);
								else
									throw new IllegalArgumentException("all list items must be Expressions; found a "
											+ o2.getClass().getName() + " instead");
								o1 = biexp;
							}
							BinaryExpression biexp2 = new BinaryExpression(biexp, BinaryOperator.MULTIPLY, sizeof_expr);
							AssignmentExpression assignex = new AssignmentExpression((Expression)cloned_bytes.clone(),
									AssignmentOperator.NORMAL, biexp2);
							orgTPBytes_stmt = new ExpressionStatement(assignex);
							biexp = new BinaryExpression((Expression)totalNumThreads.clone(),
									BinaryOperator.MULTIPLY, (Expression)biexp2.clone());
							assignex = new AssignmentExpression((Expression)cloned_bytes.clone(),AssignmentOperator.NORMAL,
									biexp);
							ExpressionStatement estmt = new ExpressionStatement(assignex);
							gpuBytes_stmt = (ExpressionStatement)estmt.clone();
							if( insertMalloc ) {
								((CompoundStatement)mallocPoint.getParent()).addStatementBefore(mallocPoint, estmt);
								if( firstCudaStmt2 == null ) {
									firstCudaStmt2 = estmt;
								}
								if( mallocPoint == region ) {
									num_cudastmts-=1;
								}
								if( opt_addSafetyCheckingCode ) {
									// Insert "gpuGmemSize += gpuBytes;" statement
									((CompoundStatement)mallocPoint.getParent()).addStatementBefore(mallocPoint,
											(Statement)gMemAdd_stmt.clone());
									if( mallocPoint == region ) {
										num_cudastmts-=1;
									}
								}
							}
						}

						// Create a parameter Declaration for the kernel function
						if( use_MallocPitch ) {
							// Change to a pointer type
							// Ex:  "float* b"
							pointerV_declarator = pitchedAccessConv2((VariableDeclarator)threadPriv_var,
									new_proc, (VariableDeclaration)pitch_decl.clone(), region);
							pointer_var = new Identifier(pointerV_declarator);
							pitchMap.put(pointerV_declarator, pitch_decl);
						} else {
							// Create an extended array type
							// Ex1: "float b[][SIZE1]"
							// Ex2: "float b[][SIZE1][SIZE2]"
							arrayV_declarator = CreateExtendedArray(threadPriv_var, new_proc, region);
							array_var = new Identifier(arrayV_declarator);
						}

						// Insert argument to the kernel function call
						if( use_MallocPitch ) {
							// Simply pass address of the pointer
							// Ex:  "gpu_b"
							call_to_new_proc.addArgument((Identifier)gpu_var.clone());
							// Insert pitch argument to the kernel function call
							call_to_new_proc.addArgument((Identifier)pitch_var.clone());
						} else {
							//Cast the gpu variable to pointer-to-array type
							// Ex: (float (*)[dimesion2]) gpu_b
							List castspecs = new LinkedList();
							castspecs.addAll(cloned_decl.getSpecifiers());
							/*
							 * FIXME: ArrayAccess was used for (*)[SIZE1][SIZE2], but this may not be
							 * semantically correct way to represent (*)[SIZE1][SIZE2] in IR.
							 */
							List tindices = new LinkedList();
							for( int i=0; i<dimsize; i++) {
								tindices.add(aspec.getDimension(i).clone());
							}
							ArrayAccess castArray = new ArrayAccess(new NameID("(*)"), tindices);
							castspecs.add(castArray);
							call_to_new_proc.addArgument(new Typecast(castspecs, (Identifier)gpu_var.clone()));
						}
					} else {
						Tools.exit(pass_name + "[ERROR] extractKernelRegion() found unsupported threadprivate symbols."
								+ threadPriv_var.toString());
					}


					List<Specifier> castspecs = null;
					AssignmentExpression assignex = null;
					BinaryExpression hostWidthBytes = null;
					if( use_MallocPitch ) {
						arg_list.add(new UnaryExpression(UnaryOperator.ADDRESS_OF,(Identifier)pitch_var.clone()));
						hostWidthBytes = new BinaryExpression((Identifier)totalNumThreads.clone(),
								BinaryOperator.MULTIPLY, (Expression)sizeof_expr.clone());
						arg_list.add(hostWidthBytes);
						arg_list.add((Expression)aspec.getDimension(0).clone());
						malloc_call.setArguments(arg_list);
					} else {
						// Add gpuBytes argument to cudaMalloc() call
						//cloned_bytes = (Identifier)bytes_decl.getDeclarator(0).getSymbol().clone();
						arg_list.add((Identifier)cloned_bytes.clone());
						malloc_call.setArguments(arg_list);
					}
					FunctionCall safe_call = new FunctionCall( new NameID("CUDA_SAFE_CALL") );
					safe_call.addArgument(malloc_call);
					ExpressionStatement malloc_stmt = new ExpressionStatement(safe_call);
					if( insertMalloc ) {
						((CompoundStatement)mallocPoint.getParent()).addStatementBefore(mallocPoint, malloc_stmt);
						if( mallocPoint == region ) {
							num_cudastmts--;
						}
						if( firstCudaStmt2 == null ) {
							firstCudaStmt2 = malloc_stmt;
						}
						/*
						 * If cudaMallocPitch() is used, gpyBytes_stmt should be inserted after cudaMallocPitch()
						 * is inserted.
						 */
						if( use_MallocPitch ) {
							if( opt_addSafetyCheckingCode ) {
								//////////////////////////////////////////////////////////////////////////////////////
								//    if( pitch_x > MAX_PITCH ) {
								//        printf("Size of pitch, pitch_x, is bigger than the maximum size;\n");
								//        printf("Please turn off usaMallocPitch or useMatrixTranspose option \n");
								//        exit(1);
								//    }
								//////////////////////////////////////////////////////////////////////////////////////
								Expression condExp = new BinaryExpression((Identifier)pitch_var.clone(),
										BinaryOperator.COMPARE_GT, new NameID("MAX_PITCH"));
								CompoundStatement ifBody = new CompoundStatement();
								FunctionCall printfCall = new FunctionCall(new NameID("printf"));
								printfCall.addArgument(new StringLiteral("Size (%d) of pitch, "+pitch_var.toString()+
										", is bigger than the maximum size (%d); \\n"));
								printfCall.addArgument((Identifier)pitch_var.clone());
								printfCall.addArgument(new NameID("MAX_PITCH"));
								ifBody.addStatement(new ExpressionStatement(printfCall));
								printfCall = new FunctionCall(new NameID("printf"));
								printfCall.addArgument(new StringLiteral("Please turn off useMatrixTranspose option.\\n"));
								ifBody.addStatement(new ExpressionStatement(printfCall));
								FunctionCall exitCall = new FunctionCall(new NameID("exit"));
								exitCall.addArgument(new IntegerLiteral(1));
								ifBody.addStatement(new ExpressionStatement(exitCall));
								((CompoundStatement)mallocPoint.getParent()).addStatementBefore(mallocPoint,
									new IfStatement(condExp, ifBody));
								if( mallocPoint == region ) {
									num_cudastmts-=1;
								}
							}
							((CompoundStatement)mallocPoint.getParent()).addStatementBefore(mallocPoint,
									(Statement)gpuBytes_stmt.clone());
							if( mallocPoint == region ) {
								num_cudastmts-=1;
							}
							if( opt_addSafetyCheckingCode ) {
								// Insert "gpuGmemSize += gpuBytes;" statement
								((CompoundStatement)mallocPoint.getParent()).addStatementBefore(mallocPoint,
										(Statement)gMemAdd_stmt.clone());
								if( mallocPoint == region ) {
									num_cudastmts-=1;
								}
							}
						}
					}

					/*
					 * Create a temporary array that is an extended version of the threadprivate variable.
					 * - The extended array is primarily used for CPU-to-GPU memory transfers, but it is also used
					 * for GPU-to-CPU memory transfers when MallocPitch is used.
					 */
					if( insertC2GMemTr || (insertG2CMemTr&&use_MallocPitch) ) {
						/*
						 * c2gMap also contains GPU threadprivate variable to extended variable mapping.
						 */
						if( c2gMap.containsKey(gpu_declarator)) {
							// clone GPU device symbol corresponding to threadPriv_var
							extended_declarator = (VariableDeclarator)c2gMap.get(gpu_declarator);
							extended_var = new Identifier(extended_declarator);
						} else {
							// Create a temporary pointer variable pointing to the temporary array.
							// Ex: float * x__extended;
							str = new StringBuilder(80);
							str.append(cloned_ID.toString());
							str.append("__extended");
							// The type of the device symbol should be a pointer type
							extended_declarator = new VariableDeclarator(PointerSpecifier.UNQUALIFIED,
									new NameID(str.toString()));
							VariableDeclaration extended_decl = new VariableDeclaration(cloned_decl.getSpecifiers(),
									extended_declarator);
							extended_var = new Identifier(extended_declarator);
							procbody.addDeclaration(extended_decl);
							// Add mapping from gpu_declarator to extended_declarator
							c2gMap.put(gpu_declarator, extended_declarator);
						}
						/*
						 * Create malloc() statement, "x__extended = (float *)malloc(gpuBytes);"
						 * FIXME: for now, malloc() and free() are called at every kernel call site.
						 *        Advanced analysis may be able to remove redundant malloc()/free().
						 */
						FunctionCall tempMalloc_call = new FunctionCall(new NameID("malloc"));
						tempMalloc_call.addArgument((Expression)cloned_bytes.clone());
						castspecs = new LinkedList<Specifier>();
						castspecs.addAll(cloned_decl.getSpecifiers());
						castspecs.add(PointerSpecifier.UNQUALIFIED);
						assignex = new AssignmentExpression((Identifier)extended_var.clone(),
								AssignmentOperator.NORMAL, new Typecast(castspecs, tempMalloc_call));
						ExpressionStatement eMallocStmt = new ExpressionStatement(assignex);
						((CompoundStatement)mallocPoint.getParent()).addStatementBefore(mallocPoint, eMallocStmt);
						if( mallocPoint == region ) {
							num_cudastmts--;
						}
						if( firstCudaStmt2 == null ) {
							firstCudaStmt2 = eMallocStmt;
						}
					}
					/*
					 * Initialize the extended variable by replicating the corresponding
					 * threadprivate variable.
					 */
					if( insertC2GMemTr ) {
						Statement loop_init = null;
						Expression condition = null;
						Expression step = null;
						CompoundStatement loop_body = null;
						ForLoop innerLoop = null;
						Identifier index_var = null;
						if(SymbolTools.isScalar(threadPriv_var)) {
							/*
							 * Insert initialization of x__extended;
							 * Example for threadprivate variable float x:
							 *      for(i=0; i<totalNumThreads; i++) {
							 *          x__extended[i] = x;
							 *      }
							 */
							index_var = TransformTools.getTempIndex(procbody, 0);
							assignex = new AssignmentExpression((Identifier)index_var.clone(),
									AssignmentOperator.NORMAL, new IntegerLiteral(0));
							loop_init = new ExpressionStatement(assignex);
							condition = new BinaryExpression((Identifier)index_var.clone(),
									BinaryOperator.COMPARE_LT, (Identifier)totalNumThreads.clone());
							step = new UnaryExpression(UnaryOperator.POST_INCREMENT,
									(Identifier)index_var.clone());
							assignex = new AssignmentExpression(new ArrayAccess((Identifier)extended_var.clone(),
									(Identifier)index_var.clone()), AssignmentOperator.NORMAL,
									(Identifier)cloned_ID.clone());
							loop_body = new CompoundStatement();
							loop_body.addStatement(new ExpressionStatement(assignex));
							innerLoop = new ForLoop(loop_init, condition, step, loop_body);
							((CompoundStatement)refstmt2.getParent()).addStatementBefore(refstmt2, innerLoop);
							if( refstmt2 == region ) {
								num_cudastmts--;
							}
						} else if( SymbolTools.isArray(threadPriv_var)) {
							/*
							 * Create or find temporaray pointers that are used to replicate
							 * a threadprivate variable into corresponding extended variable.
							 * c2gMap also contains extended variable to row_temps list mapping.
							 */
							List<Identifier> row_temps = null;
							if( c2gMap.containsKey(extended_declarator)) {
								// clone GPU device symbol corresponding to threadPriv_var
								row_temps = (List<Identifier>)
										c2gMap.get(extended_declarator);
							} else {
								row_temps = new ArrayList<Identifier>(dimsize+1);
								for( int i=0; i<dimsize; i++ ) {
									row_temps.add(SymbolTools.getPointerTemp(procbody,
											clonedspecs, "row_temp"));
								}
								row_temps.add((Identifier)extended_var.clone());
								// Add mapping from extended_declarator to row_temps list
								c2gMap.put(extended_declarator, row_temps);
							}
							/*
							 * Create or find temporary index variables that are used to replicate
							 * a threadprivate variable into corresponding extended variable.
							 */
							List<Identifier> index_vars = new LinkedList<Identifier>();
							for( int i=0; i<=dimsize; i++ ) {
								index_vars.add(TransformTools.getTempIndex(procbody, i));
							}
							/*
							 * Initialize the extended variable.
							 */
							if( use_MallocPitch ) {
								/*
								 * Insert initialization of x__extended;
								 * Example for threadprivate variable float x[SIZE1]:
								 *      for(int i=0; i<SIZE1; i++) {
								 *      	row_temp0 = (float*)((char*)x__extended + i*pitch_x);
								 *      	for(int k=0; k<totalNumThreads; k++) {
								 *      		row_temp0[k] = x[i];
								 *      	}
								 *      }
								 */
								for( int i=0; i<=1; i++ ) {
									index_var = index_vars.get(i);
									assignex = new AssignmentExpression((Identifier)index_var.clone(),
											AssignmentOperator.NORMAL, new IntegerLiteral(0));
									loop_init = new ExpressionStatement(assignex);
									step = new UnaryExpression(UnaryOperator.POST_INCREMENT,
											(Identifier)index_var.clone());
									loop_body = new CompoundStatement();
									if( i==0  ) {
										condition = new BinaryExpression((Identifier)index_var.clone(),
												BinaryOperator.COMPARE_LT, (Identifier)totalNumThreads.clone());
										assignex = new AssignmentExpression(new ArrayAccess(
												(Identifier)row_temps.get(0).clone(), (Identifier)index_var.clone()),
												AssignmentOperator.NORMAL,  new ArrayAccess((Identifier)cloned_ID.clone(),
														(Identifier)index_vars.get(1).clone()));
									} else {
										condition = new BinaryExpression((Identifier)index_var.clone(),
												BinaryOperator.COMPARE_LT, (Expression)aspec.getDimension(0).clone());
										castspecs = new ArrayList<Specifier>(2);
										castspecs.add(Specifier.CHAR);
										castspecs.add(PointerSpecifier.UNQUALIFIED);
										Typecast tcast1 = new Typecast(castspecs, (Identifier)row_temps.get(1).clone());
										BinaryExpression biexp1 = null;
										BinaryExpression biexp2 = null;
										biexp1 = new BinaryExpression((Identifier)index_var.clone(),
												BinaryOperator.MULTIPLY, (Identifier)pitch_var.clone());
										biexp2 = new BinaryExpression(tcast1, BinaryOperator.ADD, biexp1);
										castspecs = new ArrayList<Specifier>();
										castspecs.addAll(cloned_decl.getSpecifiers());
										castspecs.add(PointerSpecifier.UNQUALIFIED);
										tcast1 = new Typecast(castspecs, biexp2);
										assignex = new AssignmentExpression((Identifier)row_temps.get(0).clone(),
												AssignmentOperator.NORMAL, tcast1);

									}
									loop_body.addStatement(new ExpressionStatement(assignex));
									if( innerLoop != null ) {
										loop_body.addStatement(innerLoop);
									}
									innerLoop = new ForLoop(loop_init, condition, step, loop_body);
								}
								((CompoundStatement)refstmt2.getParent()).addStatementBefore(refstmt2, innerLoop);
								if( refstmt2 == region ) {
									num_cudastmts--;
								}
							} else {
								/*
								 * Insert initialization of x__extended;
								 * Example for threadpriave variable float x[SIZE1][SIZE2]:
								 * 	for(int i=0; i<totalNumThreads; i++) {
								 * 		row_temp1 = (float*)((char*)x__extended + i*SIZE1*SIZE2*sizeof(float));
								 * 		for(int k=0; k<SIZE1; k++) {
								 * 			row_temp0 = (float*)((char*)row_temp1 + k*SIZE2*sizeof(float));
								 * 			for(int m=0; m<SIZE2; m++) {
								 * 				row_temp0[m] = x[k][m];
								 * 			}
								 * 		}
								 * 	}
								 */
								// Create the nested loops.
								for( int i=0; i<=dimsize; i++ ) {
									index_var = index_vars.get(i);
									assignex = new AssignmentExpression((Identifier)index_var.clone(),
											AssignmentOperator.NORMAL, new IntegerLiteral(0));
									loop_init = new ExpressionStatement(assignex);
									if( i<dimsize ) {
										condition = new BinaryExpression((Identifier)index_var.clone(),
												BinaryOperator.COMPARE_LT,
												(Expression)aspec.getDimension(dimsize-1-i).clone());
									} else {
										condition = new BinaryExpression((Identifier)index_var.clone(),
												BinaryOperator.COMPARE_LT, (Identifier)totalNumThreads.clone());
									}
									step = new UnaryExpression(UnaryOperator.POST_INCREMENT,
											(Identifier)index_var.clone());
									loop_body = new CompoundStatement();
									if( i==0  ) {
										List<Expression> indices = new LinkedList<Expression>();
										for( int k=dimsize-1; k>=0; k-- ) {
											indices.add((Expression)index_vars.get(k).clone());
										}
										assignex = new AssignmentExpression(new ArrayAccess(
												(Identifier)row_temps.get(0).clone(), (Identifier)index_var.clone()),
												AssignmentOperator.NORMAL,  new ArrayAccess((Identifier)cloned_ID.clone(),
														indices));
									} else {
										castspecs = new ArrayList<Specifier>(2);
										castspecs.add(Specifier.CHAR);
										castspecs.add(PointerSpecifier.UNQUALIFIED);
										Typecast tcast1 = new Typecast(castspecs, (Identifier)row_temps.get(i).clone());
										BinaryExpression biexp1 = new BinaryExpression((Expression)sizeof_expr.clone(),
												BinaryOperator.MULTIPLY, (Expression)aspec.getDimension(dimsize-1).clone());
										BinaryExpression biexp2 = null;
										for( int k=1; k<i; k++ ) {
											biexp2 = new BinaryExpression(biexp1, BinaryOperator.MULTIPLY,
													(Expression)aspec.getDimension(dimsize-1-k).clone());
											biexp1 = biexp2;
										}
										biexp2 = new BinaryExpression((Expression)index_var.clone(),
												BinaryOperator.MULTIPLY, biexp1);
										biexp1 = new BinaryExpression(tcast1, BinaryOperator.ADD, biexp2);
										castspecs = new ArrayList<Specifier>();
										castspecs.addAll(cloned_decl.getSpecifiers());
										castspecs.add(PointerSpecifier.UNQUALIFIED);
										tcast1 = new Typecast(castspecs, biexp1);
										assignex = new AssignmentExpression((Identifier)row_temps.get(i-1).clone(),
												AssignmentOperator.NORMAL, tcast1);

									}
									loop_body.addStatement(new ExpressionStatement(assignex));
									if( innerLoop != null ) {
										loop_body.addStatement(innerLoop);
									}
									innerLoop = new ForLoop(loop_init, condition, step, loop_body);
								}
								((CompoundStatement)refstmt2.getParent()).addStatementBefore(refstmt2, innerLoop);
								if( refstmt2 == region ) {
									num_cudastmts--;
								}
							}
						}
					}

					/*
					 * Insert cudaFree() to deallocate device memory.
					 * Because cuda-related statements are added in reverse order,
					 * this function call is added first.
					 */
					//if( insertG2CMemTr ) {
					if( insertMalloc ) {
						if( opt_addSafetyCheckingCode ) {
							// Insert "gpuGmemSize -= gpuBytes;" statement.
							((CompoundStatement)mallocPoint.getParent()).addStatementAfter(mallocPoint, (Statement)gMemSub_stmt.clone());
							if( mallocPoint == region ) {
								num_cudastmts-=1;
							}
						}
						// Insert "CUDA_SAFE_CALL(cudaFree(gpu_x));" statement.
						FunctionCall cudaFree_call = new FunctionCall(new NameID("cudaFree"));
						cudaFree_call.addArgument((Identifier)gpu_var.clone());
						safe_call = new FunctionCall( new NameID("CUDA_SAFE_CALL"));
						safe_call.addArgument(cudaFree_call);
						ExpressionStatement free_stmt = new ExpressionStatement(safe_call);
						((CompoundStatement)mallocPoint.getParent()).addStatementAfter(mallocPoint, free_stmt);
						if( mallocPoint == region ) {
							num_cudastmts-=1;
						}
						// Insert free(x__extended);
						if( c2gMap.containsKey(gpu_declarator)) {
							// clone GPU device symbol corresponding to threadPriv_var
							extended_var = new Identifier((VariableDeclarator)
									c2gMap.get(gpu_declarator));
							FunctionCall free_call = new FunctionCall(new NameID("free"));
							free_call.addArgument(extended_var);
							free_stmt = new ExpressionStatement(free_call);
							((CompoundStatement)mallocPoint.getParent()).addStatementAfter(mallocPoint, free_stmt);
							if( mallocPoint == region ) {
								num_cudastmts--;
							}
						}
						// Remove mapping from shared_var to gpu_declarator
						//c2gMap.remove(threadPriv_var);
						// Remove mapping from gpu_declarator to extended_declarator
						//c2gMap.remove(gpu_declarator);
					}

					if( use_MallocPitch ) {
						if( insertC2GMemTr ) {
							/* Insert memory copy function from CPU to GPU */
							// Ex: CUDA_SAFE_CALL(cudaMemcpy2D(gpu_b, pitch_b, b, width*sizeof(float),
							// width*sizeof(float), height, cudaMemcpyHostToDevice));
							FunctionCall memCopy_call = new FunctionCall(new NameID("cudaMemcpy2D"));
							List<Expression> arg_list2 = new ArrayList<Expression>();
							arg_list2.add((Identifier)gpu_var.clone());
							arg_list2.add((Identifier)pitch_var.clone());
							arg_list2.add((Identifier)extended_var.clone());
							arg_list2.add((Expression)hostWidthBytes.clone());
							arg_list2.add((Expression)hostWidthBytes.clone());
							arg_list2.add((Expression)aspec.getDimension(0).clone());
							arg_list2.add(new NameID("cudaMemcpyHostToDevice"));
							memCopy_call.setArguments(arg_list2);
							safe_call = new FunctionCall( new NameID("CUDA_SAFE_CALL"));
							safe_call.addArgument(memCopy_call);
							ExpressionStatement memCopy_stmt = new ExpressionStatement(safe_call);
							((CompoundStatement)refstmt2.getParent()).addStatementBefore(refstmt2, memCopy_stmt);
							if( refstmt2 == region ) {
								num_cudastmts--;
							}
							if( firstCudaStmt2 == null ) {
								firstCudaStmt2 = memCopy_stmt;
							}
						}
						if( insertG2CMemTr ) {
							/* Insert memory copy function from GPU to CPU */
							// Ex: CUDA_SAFE_CALL(cudaMemcpy2D(b__extended, width*sizeof(float), gpu_b, pitch_b,
							// width*sizeof(float), height, cudaMemcpyHostToDevice));
							FunctionCall memCopy_call2 = new FunctionCall(new NameID("cudaMemcpy2D"));
							List<Expression> arg_list3 = new ArrayList<Expression>();
							arg_list3.add((Identifier)extended_var.clone());
							arg_list3.add((Expression)hostWidthBytes.clone());
							arg_list3.add((Identifier)gpu_var.clone());
							arg_list3.add((Identifier)pitch_var.clone());
							arg_list3.add((Expression)hostWidthBytes.clone());
							arg_list3.add((Expression)aspec.getDimension(0).clone());
							arg_list3.add(new NameID("cudaMemcpyDeviceToHost"));
							memCopy_call2.setArguments(arg_list3);
							safe_call = new FunctionCall( new NameID("CUDA_SAFE_CALL"));
							safe_call.addArgument(memCopy_call2);
							ExpressionStatement memCopy_stmt = new ExpressionStatement(safe_call);
							/*
							 * Insert memory copy function from GPU to CPU.
							 * Only threadprivate data on the first GPU thread need to be moved back to CPU.
							 * For simplicity, however, the whole gpu_x datat is copied back to CPU,
							 * and only the portion of the first GPU is overwritten to the original x.
							 * Example for threadprivate variable float x:
							 *      for(i=0; i<SIZE1; i++) {
							 *          x[i] = *((float*)((char*)x__extended + i*pitch_x));
							 *      }
							 */
							Identifier index_var = TransformTools.getTempIndex(procbody, 0);
							assignex = new AssignmentExpression((Identifier)index_var.clone(),
									AssignmentOperator.NORMAL, new IntegerLiteral(0));
							ExpressionStatement loop_init = new ExpressionStatement(assignex);
							BinaryExpression condition = new BinaryExpression((Expression)index_var.clone(),
									BinaryOperator.COMPARE_LT, (Expression)aspec.getDimension(0).clone());
							UnaryExpression step = new UnaryExpression(UnaryOperator.POST_INCREMENT,
									(Identifier)index_var.clone());
							castspecs = new ArrayList<Specifier>(2);
							castspecs.add(Specifier.CHAR);
							castspecs.add(PointerSpecifier.UNQUALIFIED);
							Typecast tcast1 = new Typecast(castspecs, (Identifier)extended_var.clone());
							BinaryExpression biexp1 = null;
							BinaryExpression biexp2 = null;
							biexp1 = new BinaryExpression((Expression)index_var.clone(), BinaryOperator.MULTIPLY,
									(Expression)pitch_var.clone());
							biexp2 = new BinaryExpression(tcast1, BinaryOperator.ADD, biexp1);
							castspecs = new ArrayList<Specifier>();
							castspecs.addAll(cloned_decl.getSpecifiers());
							castspecs.add(PointerSpecifier.UNQUALIFIED);
							tcast1 = new Typecast(castspecs, biexp2);
							assignex = new AssignmentExpression(new ArrayAccess(
									new Identifier((VariableDeclarator)threadPriv_var),
									(Identifier)index_var.clone()), AssignmentOperator.NORMAL,
									new UnaryExpression(UnaryOperator.DEREFERENCE, tcast1));
							CompoundStatement loop_body = new CompoundStatement();
							loop_body.addStatement(new ExpressionStatement(assignex));
							ForLoop copyLoop = new ForLoop(loop_init, condition, step, loop_body);
							((CompoundStatement)refstmt2.getParent()).addStatementAfter(refstmt2, copyLoop);
							((CompoundStatement)refstmt2.getParent()).addStatementAfter(refstmt2, memCopy_stmt);
							((CompoundStatement)refstmt2.getParent()).addStatementAfter(refstmt2, gpuBytes_stmt);
							if( refstmt2 == region ) {
								num_cudastmts-=3;
							}
						}
					} else {
						if( insertC2GMemTr ) {
							/* Insert memory copy function from CPU to GPU */
							// Ex: CUDA_SAFE_CALL(cudaMemcpy(gpu_b, b__extended, gpuBytes, cudaMemcpyHostToDevice));
							FunctionCall memCopy_call = new FunctionCall(new NameID("cudaMemcpy"));
							List<Expression> arg_list2 = new ArrayList<Expression>();
							arg_list2.add((Identifier)gpu_var.clone());
							arg_list2.add((Identifier)extended_var.clone());
							arg_list2.add((Identifier)cloned_bytes.clone());
							arg_list2.add(new NameID("cudaMemcpyHostToDevice"));
							memCopy_call.setArguments(arg_list2);
							safe_call = new FunctionCall( new NameID("CUDA_SAFE_CALL"));
							safe_call.addArgument(memCopy_call);
							ExpressionStatement memCopy_stmt = new ExpressionStatement(safe_call);
							((CompoundStatement)refstmt2.getParent()).addStatementBefore(refstmt2, memCopy_stmt);
							if( refstmt2 == region ) {
								num_cudastmts--;
							}
							if( firstCudaStmt2 == null ) {
								firstCudaStmt2 = memCopy_stmt;
							}
						}
						if( insertG2CMemTr ) {
							/*
							 * Insert memory copy function from GPU to CPU.
							 * Only threadprivate data on the first GPU thread are moved back to CPU.
							 * */
							// Ex: gpuBytes = sizeof(float);
							//     CUDA_SAFE_CALL(cudaMemcpy(a, gpu_a, gpuBytes, cudaMemcpyDeviceToHost));
							//     gpuBytes = (totalNumThreads * sizeof(float));
							FunctionCall memCopy_call2 = new FunctionCall(new NameID("cudaMemcpy"));
							List<Expression> arg_list3 = new ArrayList<Expression>();
							if( SymbolTools.isScalar(threadPriv_var)) {
								arg_list3.add( new UnaryExpression(UnaryOperator.ADDRESS_OF,
												new Identifier((VariableDeclarator)threadPriv_var)));
							} else {
								arg_list3.add(new Identifier((VariableDeclarator)threadPriv_var));
							}
							arg_list3.add((Identifier)gpu_var.clone());
							arg_list3.add((Identifier)cloned_bytes.clone());
							arg_list3.add(new NameID("cudaMemcpyDeviceToHost"));
							memCopy_call2.setArguments(arg_list3);
							safe_call = new FunctionCall( new NameID("CUDA_SAFE_CALL"));
							safe_call.addArgument(memCopy_call2);
							ExpressionStatement memCopy_stmt = new ExpressionStatement(safe_call);
							((CompoundStatement)refstmt2.getParent()).addStatementAfter(refstmt2, gpuBytes_stmt);
							((CompoundStatement)refstmt2.getParent()).addStatementAfter(refstmt2, memCopy_stmt);
							((CompoundStatement)refstmt2.getParent()).addStatementAfter(refstmt2, orgTPBytes_stmt);
							if( refstmt2 == region ) {
								num_cudastmts-=3;
							}
						}
					}
				} else {
					Tools.exit(pass_name + "[ERROR] extractKernelRegion() supports VariableDeclarator threadprivate symbols only; " +
							"current procedure = " + proc.getSymbolName() + " current symbol = " + threadPriv_var);
				}
			}
		}

		///////////////////////////////////
		// Handle OMP firstprivate data. //
		///////////////////////////////////
		if (OmpFirstPrivSet != null) {
			if( OmpFirstPrivSet.size() > 0 ) {
				/////////////////////////////////////////////////////////////////////////////////////////
				// DEBUG: each kernel region is conservatively used for both refstmt and mallocPoint.  //
				// To use optimal ref-point (refstmt1), malloc-related optimizations should consider   //
				// firstprivate variables; current malloc optimizations consider shared variables only //
				/////////////////////////////////////////////////////////////////////////////////////////
				refstmt = region;
				mallocPoint = region;
			}
			CompoundStatement targetStmt = null;
			if( region instanceof CompoundStatement ) {
				targetStmt = (CompoundStatement)region;
			} else if( region instanceof ForLoop ) {
				targetStmt = (CompoundStatement)((ForLoop)region).getBody();
			} else {
				Tools.exit(pass_name + "[ERROR] Unknwon region in extractKernelRegion(): "
						+ region.toString());
			}
			for( Symbol fpriv_var : OmpFirstPrivSet ) {
				if( fpriv_var instanceof VariableDeclarator ) {
					VariableDeclaration decl = (VariableDeclaration)((VariableDeclarator)fpriv_var).getParent();
					/*
					 * Create a cloned Declaration of the original shared variable.
					 */
					VariableDeclarator cloned_declarator =
						(VariableDeclarator)((VariableDeclarator)fpriv_var).clone();
					cloned_declarator.setInitializer(null);
					/////////////////////////////////////////////////////////////////////////////////
					// __device__ and __global__ functions can not declare static variables inside //
					// their body.                                                                 //
					/////////////////////////////////////////////////////////////////////////////////
					List<Specifier> clonedspecs = new ChainedList<Specifier>();
					clonedspecs.addAll(decl.getSpecifiers());
					clonedspecs.remove(Specifier.STATIC);
					VariableDeclaration cloned_decl = new VariableDeclaration(clonedspecs, cloned_declarator);
					Identifier cloned_ID = new Identifier(cloned_declarator);
					Identifier gpu_var = null;

					//////////////////////////////////////////////////////////////////////////////
					// If firstprivate variable is scalar, the corresponding shared variable is //
					// passed as a kernel parameter instead of using GPU global memory, which   //
					// has the effect of caching it on the GPU Shared Memory.                   //
					//////////////////////////////////////////////////////////////////////////////
					if( SymbolTools.isScalar(fpriv_var) && !SymbolTools.isPointer(fpriv_var) ) {
						// Create a GPU kernel parameter corresponding to fpriv_var
						StringBuilder str = new StringBuilder(80);
						str.append("param__");
						str.append(cloned_ID.toString());
						VariableDeclarator gpu_declarator = new VariableDeclarator(new NameID(str.toString()));
						VariableDeclaration gpu_decl = new VariableDeclaration(cloned_decl.getSpecifiers(),
								gpu_declarator);
						gpu_var = new Identifier(gpu_declarator);
						new_proc.addDeclaration(gpu_decl);

						// Insert argument to the kernel function call
						call_to_new_proc.addArgument(new Identifier((VariableDeclarator)fpriv_var));

						// Replace the instance of the firstprivate variable with the cloned device gpu variable
						IRTools.replaceAll(region, cloned_ID, cloned_ID);
						/////////////////////////////////////////////////////////////////////////////////
						// Add the declaration of the cloned device gpu variable into kernel region.   //
						// DEBUG: below statement is commented out since necessary declaration will    //
						// be added in later private data handling part; if below statement is enabled //
						// duplicate declaration statements will be added, since the cloned_declarator //
						// is not the same symbol as fpriv_var even though names are identical.        //
						/////////////////////////////////////////////////////////////////////////////////
						// targetStmt.addDeclaration(cloned_decl);

						///////////////////////////////////////////////////////////////////////////////
						// Load the value of the passed shared variable to the firstprivate variable //
						///////////////////////////////////////////////////////////////////////////////
						Statement estmt = new ExpressionStatement(new AssignmentExpression(cloned_ID.clone(),
								AssignmentOperator.NORMAL, gpu_var.clone()));
						Statement last_decl_stmt;
						last_decl_stmt = IRTools.getLastDeclarationStatement(targetStmt);
						if( last_decl_stmt != null ) {
							targetStmt.addStatementAfter(last_decl_stmt,(Statement)estmt);
						} else {
							last_decl_stmt = (Statement)targetStmt.getChildren().get(0);
							targetStmt.addStatementBefore(last_decl_stmt,(Statement)estmt);
						}
						continue;
					}

					/*
					 * c2gMap contains a mapping from a shared/threadprivate/firstprivate variable
					 * to corresponding GPU variable.
					 */
					if( c2gMap.containsKey(fpriv_var)) {
						// clone GPU device symbol corresponding to fpriv_var
						VariableDeclarator gpu_declarator = (VariableDeclarator)c2gMap.get(fpriv_var);
						gpu_var = new Identifier(gpu_declarator);
					} else {
						// Create a GPU device variable corresponding to fpriv_var
						// Ex: float * gpufp_b;
						// Give a new name for the device variable
						StringBuilder str = new StringBuilder(80);
						str.append("gpufp__");
						str.append(cloned_ID.toString());
						// The type of the device symbol should be a pointer type
						VariableDeclarator gpu_declarator = new VariableDeclarator(PointerSpecifier.UNQUALIFIED,
								new NameID(str.toString()));
						VariableDeclaration gpu_decl = new VariableDeclaration(cloned_decl.getSpecifiers(),
								gpu_declarator);
						gpu_var = new Identifier(gpu_declarator);
						procbody.addDeclaration(gpu_decl);
						// Add mapping from fpriv_var to gpu_declarator
						c2gMap.put(fpriv_var, gpu_declarator);
					}

					/////////////////////////////
					// Check duplicate Malloc. //
					/////////////////////////////
					boolean insertMalloc = false;
					boolean insertFree = false;
					HashSet<String> memTrSet = null;
					StringBuilder str = new StringBuilder(80);
					str.append("malloc_");
					str.append(gpu_var.getName());
					if( c2gMemTr.containsKey(mallocPoint) ) {
						memTrSet = (HashSet<String>)c2gMemTr.get(mallocPoint);
						if( !memTrSet.contains(str.toString()) ) {
							memTrSet.add(str.toString());
							insertMalloc = true;
						}
					} else {
						memTrSet = new HashSet<String>();
						memTrSet.add(str.toString());
						c2gMemTr.put(mallocPoint, memTrSet);
						insertMalloc = true;
					}
					////////////////////////////////////////////////////////////////
					// Currently, cudaMalooc() and cudaFree() are called for each //
					// kernel region.                                             //
					////////////////////////////////////////////////////////////////
					insertFree = insertMalloc;

					//////////////////////////////////////////////////////////////////
					// Currently, CPU-to-GPU memory transfer is done at each kernel //
					// region.                                                      //
					//////////////////////////////////////////////////////////////////
					boolean insertC2GMemTr = insertMalloc;
					boolean insertG2CMemTr = false;

					/////////////////////////////////////////////////////////////////////////
					// Memory allocation for the device variable                           //
					/////////////////////////////////////////////////////////////////////////
					// - Insert cudaMalloc() function before the region.                   //
					// Ex: CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu_b)), gpuBytes)); //
					// Ex2: CUDA_SAFE_CALL( cudaMallocPitch((void**) &d_x, &pitch_x,       //
					//      width*sizeof(float), height) );                                //
					/////////////////////////////////////////////////////////////////////////
					FunctionCall malloc_call = null;
					List<Specifier> specs = new ArrayList<Specifier>(4);
					specs.add(Specifier.VOID);
					specs.add(PointerSpecifier.UNQUALIFIED);
					specs.add(PointerSpecifier.UNQUALIFIED);
					List<Expression> arg_list = new ArrayList<Expression>();
					arg_list.add(new Typecast(specs, new UnaryExpression(UnaryOperator.ADDRESS_OF,
							(Identifier)gpu_var.clone())));
					SizeofExpression sizeof_expr = new SizeofExpression(cloned_decl.getSpecifiers());
					if( SymbolTools.isPointer(fpriv_var) ) {
						Tools.exit(pass_name + "[ERROR] extractKernelRegion() needs to support Pointer type firstprivate variable: "
								+ fpriv_var.toString());
					} else if( SymbolTools.isScalar(fpriv_var) ) {
						//////////////////////////////////////////////////////////////////////////////////////////////////
						// This case is already handled in the previous section by passing the original shared variable //
						// as a kernel function parameter without allocating GPU memory.                                //
						//////////////////////////////////////////////////////////////////////////////////////////////////
						continue;
					} else if( SymbolTools.isArray(fpriv_var) ) {
						// Insert "gpuBytes = (dimension1 * dimension2 * ..) * sizeof(varType);" statement
						List aspecs = fpriv_var.getArraySpecifiers();
						ArraySpecifier aspec = (ArraySpecifier)aspecs.get(0);
						int dimsize = aspec.getNumDimensions();
						VariableDeclarator pointerV_declarator =  null;

						if( aspec.getDimension(0) == null ) {
							Tools.exit(pass_name + " [Error in transforming a parallel region in a function, " +
									proc.getSymbolName() + "()] the first dimension of a firstprivate array, "
									+ fpriv_var + ", is missing; for the O2G translator " +
									"to allocate GPU memory for this array, the exact dimension size of accessed" +
									" array section should be specified." );
						}

						malloc_call = new FunctionCall(new NameID("cudaMalloc"));
						// Add malloc size (gpuBytes) statement
						// Ex: gpuBytes=(((2048+2)*(2048+2))*sizeof (float));
						Object o1 = aspec.getDimension(0).clone();
						Expression biexp = (Expression)o1;
						for( int i=1; i<dimsize; i++ )
						{
							Object o2 = aspec.getDimension(i).clone();
							if (o2 instanceof Expression)
								biexp = new BinaryExpression((Expression)o1, BinaryOperator.MULTIPLY, (Expression)o2);
							else
								throw new IllegalArgumentException("all list items must be Expressions; found a "
										+ o2.getClass().getName() + " instead");
							o1 = biexp;
						}
						BinaryExpression biexp2 = new BinaryExpression(biexp, BinaryOperator.MULTIPLY, sizeof_expr);
						AssignmentExpression assignex = new AssignmentExpression(cloned_bytes.clone(),AssignmentOperator.NORMAL,
								biexp2);
						ExpressionStatement estmt = new ExpressionStatement(assignex);
						gpuBytes_stmt = (ExpressionStatement)estmt.clone();
						if( insertMalloc ) {
							((CompoundStatement)mallocPoint.getParent()).addStatementBefore(mallocPoint, estmt);
							if( firstCudaStmt2 == null ) {
								firstCudaStmt2 = estmt;
							}
							if( mallocPoint == region ) {
								num_cudastmts-=1;
							}
							// Insert "gpuGmemSize += gpuBytes;" statement
							if( opt_addSafetyCheckingCode ) {
								((CompoundStatement)mallocPoint.getParent()).addStatementBefore(mallocPoint,
										(Statement)gMemAdd_stmt.clone());
								if( mallocPoint == region ) {
									num_cudastmts-=1;
								}
							}
						}

						/////////////////////////////////////////////////////////////
						// Create a parameter Declaration for the kernel function. //
						// Keep the original array type, but change name           //
						// Ex: "float param_b[(2048+2)][(2048+2)]"                 //
						/////////////////////////////////////////////////////////////
						str = new StringBuilder(80);
						str.append("param__");
						str.append(cloned_ID.toString());
						VariableDeclarator param_declarator = ((VariableDeclarator)fpriv_var).clone();
						param_declarator.setName(str.toString());
						VariableDeclaration param_decl = new VariableDeclaration(cloned_decl.getSpecifiers(),
								param_declarator);
						Identifier param_var = new Identifier(param_declarator);
						new_proc.addDeclaration(param_decl);

						//////////////////////////////////////////////////
						// Insert argument to the kernel function call. //
						//////////////////////////////////////////////////
						if( dimsize == 1 ) {
							// Simply pass address of the pointer
							// Ex:  "gpu_b"
							call_to_new_proc.addArgument((Identifier)gpu_var.clone());
						} else {
							//Cast the gpu variable to pointer-to-array type
							// Ex: (float (*)[dimesion2]) gpu_b
							List castspecs = new LinkedList();
							castspecs.addAll(cloned_decl.getSpecifiers());
							/*
							 * FIXME: ArrayAccess was used for (*)[SIZE2], but this may not be
							 * semantically correct way to represent (*)[SIZE2] in IR.
							 */
							List tindices = new LinkedList();
							for( int i=1; i<dimsize; i++) {
								tindices.add(aspec.getDimension(i).clone());
							}
							ArrayAccess castArray = new ArrayAccess(new NameID("(*)"), tindices);
							castspecs.add(castArray);
							call_to_new_proc.addArgument(new Typecast(castspecs, (Identifier)gpu_var.clone()));
						}

						///////////////////////////////////////////////////////////////////////////////
						// Load the value of the passed shared variable to the firstprivate variable //
						///////////////////////////////////////////////////////////////////////////////
						///////////////////////////////////////////////////////
						// Ex: for(i=0; i<SIZE1; i++) {                      //
						//         for(k=0; k<SIZE2; k++) {                  //
						//             device_var[i][k] = param_var[i][k];   //
						//         }                                         //
						//      }                                            //
						///////////////////////////////////////////////////////
						//////////////////////////////////////// //////
						// Create or find temporary index variables. //
						//////////////////////////////////////// //////
						List<Identifier> index_vars = new LinkedList<Identifier>();
						for( int i=0; i<dimsize; i++ ) {
							index_vars.add(TransformTools.getTempIndex(targetStmt, i));
						}
						Identifier index_var = null;
						assignex = null;
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
										new Identifier(fpriv_var), indices1),
										AssignmentOperator.NORMAL,
										new ArrayAccess(param_var, indices2));
								loop_body.addStatement(new ExpressionStatement(assignex));
							} else {
								loop_body.addStatement(innerLoop);
							}
							innerLoop = new ForLoop(loop_init, condition, step, loop_body);
						}
						Statement last_decl_stmt;
						last_decl_stmt = IRTools.getLastDeclarationStatement(targetStmt);
						if( last_decl_stmt != null ) {
							targetStmt.addStatementAfter(last_decl_stmt,(Statement)innerLoop);
						} else {
							last_decl_stmt = (Statement)targetStmt.getChildren().get(0);
							targetStmt.addStatementBefore(last_decl_stmt,(Statement)innerLoop);
						}
					} else {
						Tools.exit(pass_name + "[ERROR] extractKernelRegion() found unsupported firstprivate symbols."
								+ fpriv_var.toString());
					}
					// Add gpuBytes argument to cudaMalloc() call
					//cloned_bytes = (Identifier)bytes_decl.getDeclarator(0).getSymbol().clone();
					arg_list.add((Identifier)cloned_bytes.clone());
					malloc_call.setArguments(arg_list);
					FunctionCall safe_call = new FunctionCall( new NameID("CUDA_SAFE_CALL") );
					safe_call.addArgument(malloc_call);
					ExpressionStatement malloc_stmt = new ExpressionStatement(safe_call);
					if( insertMalloc ) {
						((CompoundStatement)mallocPoint.getParent()).addStatementBefore(mallocPoint, malloc_stmt);
						if( firstCudaStmt2 == null ) {
							firstCudaStmt2 = malloc_stmt;
						}
						if( mallocPoint == region ) {
							num_cudastmts-=1;
						}
					}

					/*
					 * Insert cudaFree() to deallocate device memory.
					 * Because cuda-related statements are added in reverse order,
					 * this function call is added first.
					 */
					if( insertFree ) {
						if( opt_addSafetyCheckingCode  ) {
							// Insert "gpuGmemSize -= gpuBytes;" statement
							((CompoundStatement)mallocPoint.getParent()).addStatementAfter(mallocPoint,
									(Statement)gMemSub_stmt.clone());
							if( mallocPoint == region ) {
								num_cudastmts-=1;
							}
						}
						// Insert "CUDA_SAFE_CALL(cudaFree(gpu_a));"
						FunctionCall cudaFree_call = new FunctionCall(new NameID("cudaFree"));
						cudaFree_call.addArgument((Identifier)gpu_var.clone());
						safe_call = new FunctionCall( new NameID("CUDA_SAFE_CALL"));
						safe_call.addArgument(cudaFree_call);
						ExpressionStatement cudaFree_stmt = new ExpressionStatement(safe_call);
						((CompoundStatement)mallocPoint.getParent()).addStatementAfter(mallocPoint, cudaFree_stmt);
						// Remove mapping from shared_var to gpu_declarator
						//c2gMap.remove(shared_var);
						if( mallocPoint == region ) {
							num_cudastmts-=1;
						}
						if( mallocPoint != refstmt ) {
							((CompoundStatement)mallocPoint.getParent()).addStatementAfter(mallocPoint,
									(Statement)gpuBytes_stmt.clone());
							if( mallocPoint == region ) {
								num_cudastmts-=1;
							}
						}
					}

					if( insertC2GMemTr ) {
						/* Insert memory copy function from CPU to GPU */
						// Ex: CUDA_SAFE_CALL(cudaMemcpy(gpu_b, b, gpuBytes, cudaMemcpyHostToDevice));
						if( !insertMalloc || (mallocPoint != refstmt) ) {
							Statement gpuBytesStmt = (Statement)gpuBytes_stmt.clone();
							((CompoundStatement)refstmt.getParent()).addStatementBefore(refstmt,
									gpuBytesStmt);
							if( firstCudaStmt2 == null ) {
								firstCudaStmt2 = gpuBytesStmt;
							}
							if( refstmt == region ) {
								num_cudastmts-=1;
							}
						}
						FunctionCall memCopy_call = new FunctionCall(new NameID("cudaMemcpy"));
						List<Expression> arg_list2 = new ArrayList<Expression>();
						arg_list2.add((Identifier)gpu_var.clone());
						if( SymbolTools.isScalar(fpriv_var)) {
							arg_list2.add( new UnaryExpression(UnaryOperator.ADDRESS_OF,
									new Identifier((VariableDeclarator)fpriv_var)));
						} else {
							arg_list2.add(new Identifier((VariableDeclarator)fpriv_var));
						}
						arg_list2.add((Identifier)cloned_bytes.clone());
						arg_list2.add(new NameID("cudaMemcpyHostToDevice"));
						memCopy_call.setArguments(arg_list2);
						safe_call = new FunctionCall( new NameID("CUDA_SAFE_CALL"));
						safe_call.addArgument(memCopy_call);
						ExpressionStatement memCopy_stmt = new ExpressionStatement(safe_call);
						((CompoundStatement)refstmt.getParent()).addStatementBefore(refstmt, memCopy_stmt);
						if( firstCudaStmt2 == null ) {
							firstCudaStmt2 = memCopy_stmt;
						}
						if( refstmt == region ) {
							num_cudastmts-=1;
						}
					}
					if( insertG2CMemTr ) {
						/* Insert memory copy function from GPU to CPU */
						// Ex: gpuBytes = (4096 * sizeof(float));
						//     CUDA_SAFE_CALL(cudaMemcpy(a, gpu_a, gpuBytes, cudaMemcpyDeviceToHost));
						FunctionCall memCopy_call2 = new FunctionCall(new NameID("cudaMemcpy"));
						List<Expression> arg_list3 = new ArrayList<Expression>();
						if( SymbolTools.isScalar(fpriv_var)) {
							arg_list3.add( new UnaryExpression(UnaryOperator.ADDRESS_OF,
									new Identifier((VariableDeclarator)fpriv_var)));
						} else {
							arg_list3.add(new Identifier((VariableDeclarator)fpriv_var));
						}
						arg_list3.add((Identifier)gpu_var.clone());
						arg_list3.add((Identifier)cloned_bytes.clone());
						arg_list3.add(new NameID("cudaMemcpyDeviceToHost"));
						memCopy_call2.setArguments(arg_list3);
						safe_call = new FunctionCall( new NameID("CUDA_SAFE_CALL"));
						safe_call.addArgument(memCopy_call2);
						ExpressionStatement memCopy_stmt = new ExpressionStatement(safe_call);
						((CompoundStatement)refstmt.getParent()).addStatementAfter(refstmt, memCopy_stmt);
						((CompoundStatement)refstmt.getParent()).addStatementAfter(refstmt, gpuBytes_stmt);
						if( refstmt == region ) {
							num_cudastmts-=2;
						}
					}
				}
			}
		}

		/*
		 * If OMP Private/Firstprivate variable does not have declaration statement in the region,
		 * insert necessary declaration statement into the region.
		 */
		if ((OmpPrivSet != null) || (OmpFirstPrivSet != null)) {
			Statement body = null;
			Set<Symbol> localSymbols = null;
			HashSet<Symbol> PrivSet = new HashSet<Symbol>();
			if( OmpPrivSet != null ) {
				PrivSet.addAll(OmpPrivSet);
			}
			if( OmpFirstPrivSet != null ) {
				PrivSet.addAll(OmpFirstPrivSet);
			}
			if( region instanceof CompoundStatement ) {
				body = region;
				localSymbols = SymbolTools.getLocalSymbols((SymbolTable)body);
			} else if ( region instanceof ForLoop ) {
				body = ((ForLoop)region).getBody();
				localSymbols = SymbolTools.getLocalSymbols((SymbolTable)body);
				localSymbols.addAll(SymbolTools.getLocalSymbols((SymbolTable)region));
			}
			//System.out.println("[INFO] private symbols in procedure "+proc.getSymbolName() + ": "+AnalysisTools.symbolsToString(PrivSet, ","));
			for( Symbol priv_var : PrivSet ) {
				if( !localSymbols.contains(priv_var) ) {
					VariableDeclaration decl = (VariableDeclaration)SymbolTools.findSymbol(procbody,
							priv_var.getSymbolName());
					VariableDeclarator cloned_declarator = (VariableDeclarator)decl.getDeclarator(0).clone();
					cloned_declarator.setInitializer(null);
					/////////////////////////////////////////////////////////////////////////////////
					// __device__ and __global__ functions can not declare static variables inside //
					// their body.                                                                 //
					/////////////////////////////////////////////////////////////////////////////////
					List<Specifier> clonedspecs = new ChainedList<Specifier>();
					clonedspecs.addAll(decl.getSpecifiers());
					clonedspecs.remove(Specifier.STATIC);
					VariableDeclaration cloned_decl = new VariableDeclaration(clonedspecs, cloned_declarator);
					Identifier cloned_ID = new Identifier(cloned_declarator);
					//////////////////////////////////////////////////////////////////////////////
					// priv_var is an original symbol existing in the procbody, and thus, below //
					// change will change the original symbol in the procbody. priv_var should  //
					// be replaced by cloned_declarator; for now, leave it as it is.            //
					//////////////////////////////////////////////////////////////////////////////
					//((VariableDeclarator)priv_var).getSymbol().setSymbol(
					//		(VariableDeclarator)cloned_decl.getDeclarator(0));

					//////////////////////////////////////////////////////////////////////////////
					// Replace the symbol pointer of the private variable with this new symbol. //
					// This replacement must be done before inserting the new declaration;      //
					// IRTools.replaceAll() can not replace IDExpression in a declaration.        //
					//////////////////////////////////////////////////////////////////////////////
					IRTools.replaceAll(region, cloned_ID, cloned_ID);
					((CompoundStatement)body).addDeclaration(cloned_decl);
				}
			}
			privateVariableCachingOnSM(region, cudaSharedSet, OmpPrivSet);
		}

		/*
		 * If firstCudaStmt == null, it means that no cuda-related statement is added for
		 * handling shared data. Then, make firstCudaStmt point to refstmt1.
		 */
		if( firstCudaStmt == null ) {
			firstCudaStmt = refstmt1;
		}
/*		if( firstCudaStmt2 == null ) {
			firstCudaStmt2 = refstmt2;
		}*/
		/*
		 * If firstCudaStmt2 != null, cuda-related statements for threadprivate/reduction/firstprivate
		 *  clause are added.
		 *
		 * If firstCudaStmt2 != null,
		 *     - setKernelConfParameters() adds kernel-configuration-parameters before firstCudaStmt2.
		 * Otherwise,
		 *     - setKernelConfParameters() adds kernel-configuration-parameters before firstCudaStmt.
		 */
		if( firstCudaStmt2 != null) {
			firstCudaStmt = firstCudaStmt2;
		}

		/*
		 * Create expressions for calculating global GPU thread ID (_gtid) and
		 * thread block ID (_bid).
		 * Current assumption: threadBlock is an one-dimensional array of threads.
		 * Ex: _bid = (blockIdx.x + (blockIdx.y * gridDim.x));
		 * Ex: _gtid = (threadIdx.x + (_bid * blockDim.x));
		 * [CAUTION] threadIdx.x, blockIdx.x, and blockDim.x are CUDA-built-in
		 * variables, and thus they don't have any declarations; Range Analysis
		 * can not decide the types of these variables, and therefore, ignore these.
		 */
		BinaryExpression biexp1 = new BinaryExpression(new NameID("blockIdx.y"),
				BinaryOperator.MULTIPLY, new NameID("gridDim.x"));
		BinaryExpression biexp2 = new BinaryExpression(new NameID("blockIdx.x"),
				BinaryOperator.ADD, biexp1);
		VariableDeclarator bid_declarator = new VariableDeclarator(new NameID("_bid"));
		bid_declarator.setInitializer(new Initializer(biexp2));
		Declaration bid_decl = new VariableDeclaration(Specifier.INT, bid_declarator);
		Identifier bid = new Identifier(bid_declarator);
		biexp1 = new BinaryExpression((Identifier)bid.clone(),
				BinaryOperator.MULTIPLY, new NameID("blockDim.x"));
		biexp2 = new BinaryExpression(new NameID("threadIdx.x"),
				BinaryOperator.ADD, biexp1);
		VariableDeclarator gtid_declarator = new VariableDeclarator(new NameID("_gtid"));
		gtid_declarator.setInitializer(new Initializer(biexp2));
		Declaration gtid_decl = new VariableDeclaration(Specifier.INT, gtid_declarator);
		Identifier gtid = new Identifier(gtid_declarator);

		///////////////////////////////////////////////////////////////
		// Modify target region to be outlined as a kernel function  //
		//     - Remove the outmost OMP parallel for loop.           //
		//     - Add necessary GPU thread mapping statements.        //
		///////////////////////////////////////////////////////////////
		if( region instanceof ForLoop ) {
			ForLoop ploop = (ForLoop)region;
			// check for a canonical loop
			if ( !LoopTools.isCanonical(ploop) ) {
				Tools.exit(pass_name + "[Error in extractKernelRegion()] Parallel Loop is not a canonical loop; " +
						"compiler can not determine iteration space of the following loop: \n" +
						ploop);
			}
			// identify the loop index variable
			Expression ivar = LoopTools.getIndexVariable(ploop);
			Expression lb = LoopTools.getLowerBoundExpression(ploop);
			//Expression ub = LoopTools.getUpperBoundExpression(ploop);
			//Expression iterspace = Symbolic.add(Symbolic.subtract(ub,lb),new IntegerLiteral(1));
			Expression iterspace = (Expression)annot.remove("iterspace");
			if( iterspace == null ) {
				Tools.exit(pass_name + " Error in extractKernelRegion(): annotation map for an omp-for loop" +
						" does not contain itersapce.");
			}
			setKernelConfParameters(global_table, region, iterspace, call_to_new_proc, func_num, firstCudaStmt);
			//CompoundStatement loopbody = (CompoundStatement)ploop.getBody().clone();
			CompoundStatement loopbody = (CompoundStatement)ploop.getBody();

			////////////////////////////////////////////////////////////////////
			// Update the kernel region so that instances of Identifier _gtid //
			// and _bid, which are generated in the previous transformation   //
			// steps have correct pointers to their symbol.                   //
			////////////////////////////////////////////////////////////////////
			IRTools.replaceAll((Traversable) loopbody, gtid, gtid);
			IRTools.replaceAll((Traversable) loopbody, bid, bid);

			Set<Symbol> localSymbols = SymbolTools.getVariableSymbols(loopbody);
			for( Symbol sm : localSymbols ) {
				//
				// FIXME: CompoundStatement.clone() has a problem in copying a symbol table;
				// symbol table has IDExpression to Symbol Declarator mapping, and each IDExpression,
				// which is a key of the symbol table, has private symbol field pointing to corresponding
				// Symbol. When symbol table is cloned, this field is not corrected, and thus it points
				// to the old one.
				// Therefore, the following two commented ways returns ones in the original statement.
				//
				//Declaration lsm_decl = (VariableDeclaration)((VariableDeclarator)sm).getParent();
				//Declaration lsm_decl = SymbolTools.findSymbol((SymbolTable)loopbody, sm.getSymbolName());
				Declaration lsm_decl = SymbolTools.findSymbol((SymbolTable)loopbody,
						new Identifier((VariableDeclarator)sm));
				DeclarationStatement lsm_stmt = (DeclarationStatement)lsm_decl.getParent();
/*				try {loopbody.removeChild(lsm_stmt); }
				catch (Exception e) {System.out.println(loopbody.toAnnotatedString());
					System.out.println("Trouble symbol: " + sm);
					System.exit(1);}*/
				loopbody.removeChild(lsm_stmt);
				lsm_decl.setParent(null);
				kernelRegion.addDeclaration(lsm_decl);
			}
			/*
			 * Insert expressions for calculating global GPU thread ID (_gtid)
			 */
			kernelRegion.addDeclaration(bid_decl);
			kernelRegion.addDeclaration(gtid_decl);
			BinaryExpression biexp3 = new BinaryExpression(gtid, BinaryOperator.ADD, lb);
			AssignmentExpression assgn = new AssignmentExpression(ivar,
					AssignmentOperator.NORMAL, biexp3);
			Statement thrmapstmt = new ExpressionStatement(assgn);
			kernelRegion.addStatement(thrmapstmt);
			////////////////////////////////////////////////////
			// Insert caching loading statements if existing. //
			////////////////////////////////////////////////////
			if( cacheLoadingStmts.size() > 0 ) {
				for( Statement lstmt : cacheLoadingStmts ) {
					kernelRegion.addStatement(lstmt);
				}
				cacheLoadingStmts.clear();
			}
			/*
			 * Replace the omp-for loop with if-statement containing the loop body
			 */
			loopbody.setParent(null);
			IfStatement ifstmt = new IfStatement((Expression)ploop.getCondition().clone(),
					loopbody);
			kernelRegion.addStatement(ifstmt);
			ploop.swapWith(kernelRegion);
			deviceFuncTransform((CompoundStatement)loopbody, annot,
					(List<VariableDeclaration>)new_proc.getParameters());
			//SymbolTools.linkSymbol(kernelRegion);
		} else { //region is a CompoundStatement
			////////////////////////////////////////////////////////////////////
			// Update the kernel region so that instances of Identifier _gtid,//
			// which are generated in the previous transformation steps have  //
			// correct pointers to their symbol.                              //
			////////////////////////////////////////////////////////////////////
			IRTools.replaceAll((Traversable) region, gtid, gtid);
			IRTools.replaceAll((Traversable) region, bid, bid);
			/*
			 * Insert expressions for calculating global GPU thread ID (_gtid)
			 */
			((CompoundStatement)region).addDeclaration(bid_decl);
			((CompoundStatement)region).addDeclaration(gtid_decl);

			/*
			 * Transforms omp for-loops into if-statements
			 */
			omp_annots = IRTools.collectPragmas(region, OmpAnnotation.class, "for");
			List<Expression> ispaces = new LinkedList<Expression>();
			for ( OmpAnnotation fannot : omp_annots ) {
				Statement target_stmt = (Statement)fannot.getAnnotatable();
				if( target_stmt instanceof ForLoop ) {
					ForLoop ploop = (ForLoop)target_stmt;
					if ( !LoopTools.isCanonical(ploop) ) {
						Tools.exit(pass_name + "[Error in extractKernelRegion()] Parallel Loop is not a canonical loop; " +
						"compiler can not determine iteration space of the following loop: \n" +
						ploop);
					}
					Expression ivar = LoopTools.getIndexVariable(ploop);
					Expression lb = LoopTools.getLowerBoundExpression(ploop);
					//Expression ub = LoopTools.getUpperBoundExpression(ploop);
					//Expression iterspace = Symbolic.add(Symbolic.subtract(ub,lb),new IntegerLiteral(1));
					Expression iterspace = (Expression)fannot.remove("iterspace");
					if( iterspace == null ) {
						Tools.exit(pass_name + " Error in extractKernelRegion(): annotation map for an omp-for loop" +
						" does not contain itersapce.");
					}
					ispaces.add(iterspace);
					BinaryExpression biexp3 = new BinaryExpression((Identifier)gtid.clone(), BinaryOperator.ADD, lb);
					AssignmentExpression assgn = new AssignmentExpression(ivar,
							AssignmentOperator.NORMAL, biexp3);
					Statement thrmapstmt = new ExpressionStatement(assgn);
					CompoundStatement parentStmt = (CompoundStatement)target_stmt.getParent();
					parentStmt.addStatementBefore(ploop, thrmapstmt);
					/*
					 * Replace the omp-for loop with if-statement containing the loop body
			 		*/
/*					IfStatement ifstmt = new IfStatement((Expression)ploop.getCondition().clone(),
					(CompoundStatement)ploop.getBody().clone());*/
					CompoundStatement ifbody = new CompoundStatement();
					IfStatement ifstmt = new IfStatement((Expression)ploop.getCondition().clone(),
					ifbody);
					ifbody.swapWith(ploop.getBody());
					ploop.swapWith(ifstmt);
					///////////////////////////////////////////////////////////////////////////
					// CAUTION: Omp for annotation will be inserted to the new if statement, //
					// but this insertion violates OpenMP semantics.                         //
					///////////////////////////////////////////////////////////////////////////
					//an_stmt.attachStatement(ifstmt);
					ifstmt.annotate(fannot);
				} else if( target_stmt instanceof CompoundStatement ) {
					Expression iterspace = (Expression)fannot.remove("iterspace");
					if( iterspace == null ) {
						Tools.exit(pass_name + " Error in extractKernelRegion(): annotation map for an omp-for loop" +
						" does not contain itersapce.");
					}
					ispaces.add(iterspace);
				}
			}
			/*
			 * Transforms omp-for loops interprocedurally
			 */
			interProcLoopTransform((CompoundStatement)region, ispaces, annot,
					(List<VariableDeclaration>)new_proc.getParameters());

			Expression gridsize = null;
			/*
			 * Calculate the grid size of the kernel function for this parallel region.
			 * The passed parallel region should have at least one omp for loop.
			 */
			/////////////////////////////////////////////////////////////////////////////////////////
			// FIXME: If iteration size expression of an omp-for loop in a function called in the  //
			//        parallel region contains any local variable, the local variable should be    //
			//        passed to the code region where kernel configuration parameter is calculated //
			/////////////////////////////////////////////////////////////////////////////////////////
			if( ispaces.size() == 0 ) {
				Tools.exit(pass_name + "[Error in extractKernelRegion()]: Parallel region in " +
						proc.getName() +" does not contain any omp for loops!");
			} else if( ispaces.size() == 1 ) {
				gridsize = ispaces.get(0);
			} else {
				Expression maxExp = Symbolic.simplify(
						new MinMaxExpression(false, ispaces.get(0), ispaces.get(1)));
				for( int i=2; i<ispaces.size(); i++) {
					MinMaxExpression newMax = new MinMaxExpression(false, maxExp,
							ispaces.get(i));
					maxExp = Symbolic.simplify(newMax);
				}
				gridsize = maxExp;
			}
			//setKernelConfParameters(global_table, region, gridsize, call_to_new_proc, func_num);
			setKernelConfParameters(global_table, region, gridsize, call_to_new_proc, func_num, firstCudaStmt);
			kernelRegion = (CompoundStatement)region;
		} // end of "else { //region is a CompoundStatement"

		/*
		 * Put call_to_new_proc inside new_proc and then swap it with the
		 * region. The call will end up where the region was and the region will
		 * end up in the new procedure.
		 */
		new_proc.getBody().addStatement(kernelCall_stmt);
		kernelCall_stmt.swapWith((Statement) kernelRegion);
		kernelRegion.setParent(null);
		new_proc.setBody(kernelRegion);
		kernelCallStmtSet.add(kernelCall_stmt);

		/* put new_proc before the calling proc (avoids prototypes) */
		((TranslationUnit) proc.getParent()).addDeclarationBefore(proc,
				new_proc);
		Traversable parent = kernelCall_stmt.getParent();
		if( opt_forceSyncKernelCall ) {
			FunctionCall syncCall = new FunctionCall(new NameID("cudaThreadSynchronize"));
			if( parent instanceof CompoundStatement ) {
				((CompoundStatement)parent).addStatementAfter(kernelCall_stmt, new ExpressionStatement(syncCall));
				num_cudastmts-=1;
			} else {
				Tools.exit(pass_name + "[Error in extractKernelRegion()] Kernel call statement (" +
						kernelCall_stmt + ") does not have a parent!");
			}
		}
		if( opt_addCudaErrorCheckingCode ) {
			FunctionCall errorCheckCall = new FunctionCall(new NameID("CUT_CHECK_ERROR"));
			errorCheckCall.addArgument(new StringLiteral("ERROR in executing a kernel, "+ new_proc.getSymbolName()));
			if( parent instanceof CompoundStatement ) {
				((CompoundStatement)parent).addStatementAfter(kernelCall_stmt, new ExpressionStatement(errorCheckCall));
				num_cudastmts-=1;
			} else {
				Tools.exit(pass_name + "[Error in extractKernelRegion()] Kernel call statement (" +
						kernelCall_stmt + ") does not have a parent!");
			}
		}

		if( opt_addSafetyCheckingCode ) {
			/////////////////////////////////////////////
			// Add GPU global memory usage check code. //
			/////////////////////////////////////////////
			Expression MemCheckExp = new BinaryExpression((Identifier)gmemsize.clone(),
					BinaryOperator.COMPARE_GT, new NameID("MAX_GMSIZE"));
			FunctionCall MemWarningCall = new FunctionCall(new NameID("printf"));
			StringLiteral warningMsg = new StringLiteral("[WARNING] size of allocated GPU global memory" +
			" (%d) exceeds the given limit (%d)\\n");
			MemWarningCall.addArgument(warningMsg);
			MemWarningCall.addArgument((Identifier)gmemsize.clone());
			MemWarningCall.addArgument( new NameID("MAX_GMSIZE"));
			IfStatement gMemCheckStmt = new IfStatement(MemCheckExp,
					new ExpressionStatement(MemWarningCall));
			/////////////////////////////////////////////
			// Add GPU shared memory usage check code. //
			/////////////////////////////////////////////
			MemCheckExp = new BinaryExpression((Identifier)smemsize.clone(),
					BinaryOperator.COMPARE_GT, new NameID("MAX_SMSIZE"));
			MemWarningCall = new FunctionCall(new NameID("printf"));
			warningMsg = new StringLiteral("[WARNING] size of allocated GPU shared memory" +
			" (%d) exceeds the given limit (%d)\\n");
			MemWarningCall.addArgument(warningMsg);
			MemWarningCall.addArgument((Identifier)smemsize.clone());
			MemWarningCall.addArgument( new NameID("MAX_SMSIZE"));
			IfStatement sMemCheckStmt = new IfStatement(MemCheckExp,
					new ExpressionStatement(MemWarningCall));
			if( parent instanceof CompoundStatement ) {
				((CompoundStatement)parent).addStatementBefore(kernelCall_stmt, gMemCheckStmt);
				((CompoundStatement)parent).addStatementBefore(kernelCall_stmt, sMemCheckStmt);
				num_cudastmts-=2;
			} else {
				Tools.exit(pass_name + "[Error in extractKernelRegion()] Kernel call statement (" +
						kernelCall_stmt + ") does not have a parent!");
			}
		}
		/*
		 * The original omp annotation will be inserted into this new kernel function.
		 * [CAUTION] Symbols in the annotation are not the ones used in
		 * the kernel region; they refer to original CPU symbols, but not
		 * GPU device symbols.
		 * [CAUTION] This insertion violates OpenMP semantics.
		 */
		//annot_stmt.attachStatement(kernelRegion);
		region.removeAnnotations(OmpAnnotation.class);
		//new_proc.annotate(annot);
		kernelCall_stmt.annotate(annot);
		//If CudaAnnotations exist, copy them too. //
		List<CudaAnnotation> cuda_annots = region.getAnnotations(CudaAnnotation.class);
		if( cuda_annots != null ) {
			for(CudaAnnotation cuda_annot : cuda_annots) {
				kernelCall_stmt.annotate(cuda_annot);
			}
			region.removeAnnotations(CudaAnnotation.class);
		}
		//CommentAnnotation may exist for debugging; copy them too. //
		List<CommentAnnotation> comment_annots = region.getAnnotations(CommentAnnotation.class);
		if( comment_annots != null ) {
			for(CommentAnnotation comment_annot : comment_annots) {
				kernelCall_stmt.annotate(comment_annot);
			}
			region.removeAnnotations(CommentAnnotation.class);
		}
	}

	/**
	 * Reduction transformation pass.
	 * This version uses GPU shared memory for in-block reduction, but this may set a limit on
	 * the applicability of the reduction transformation.
	 * FIXME: current version can not handle the cases where reduction is used
	 * in a function called in a parallel region.
	 *
	 * @param proc Procedure that contains the kernel region to be transformed
	 * @param region a kernel region to be transformed into a kernel function
	 * @param redMap HashMap containing reduction variable to CUDA global variable mapping
	 * @param call_to_new_proc Kernelfunction call for the kernel region
	 * @param new_proc a new KernelFunction that the kernel region is transformed to
	 * @return
	 */
	private static List reductionTransformation(Procedure proc, Statement region,
			HashMap redMap, FunctionCall call_to_new_proc, Procedure new_proc,
			HashSet<String> cudaNoRedUnrollSet, Statement refstmt1, boolean gridSizeNotChanged) {
		CompoundStatement procbody = proc.getBody();
		SymbolTable global_table = (SymbolTable) proc.getParent();
		Statement firstCudaStmt2 = null;
		LinkedList<HashMap> redMapList = new LinkedList<HashMap>();
		HashSet<Symbol> redItemSet = new HashSet<Symbol>();
		LinkedList<Statement> targetList = new LinkedList<Statement>();
		ArrayList<VariableDeclarator> redArgSet = new ArrayList<VariableDeclarator>();
		ArrayList<VariableDeclarator> redParamSet = new ArrayList<VariableDeclarator>();
		ArrayList<BinaryOperator> redOpSet = new ArrayList<BinaryOperator>();
		List redData = new ArrayList(2);
		Statement mallocPoint = null;
		Statement refstmt2 = null;
		Statement targetRegion = null;

		// Auxiliary variables used for GPU kernel conversion
		VariableDeclaration bytes_decl = (VariableDeclaration)SymbolTools.findSymbol(global_table, "gpuBytes");
		Identifier cloned_bytes = new Identifier((VariableDeclarator)bytes_decl.getDeclarator(0));
		VariableDeclaration gmem_decl = (VariableDeclaration)SymbolTools.findSymbol(global_table, "gpuGmemSize");
		Identifier gmemsize = new Identifier((VariableDeclarator)gmem_decl.getDeclarator(0));
		VariableDeclaration smem_decl = (VariableDeclaration)SymbolTools.findSymbol(global_table, "gpuSmemSize");
		Identifier smemsize = new Identifier((VariableDeclarator)smem_decl.getDeclarator(0));
		VariableDeclaration numBlocks_decl = (VariableDeclaration)SymbolTools.findSymbol(global_table, "gpuNumBlocks");
		Identifier numBlocks = new Identifier((VariableDeclarator)numBlocks_decl.getDeclarator(0));
/*		VariableDeclaration numBlocks_decl1 = (VariableDeclaration)SymbolTools.findSymbol(global_table, "gpuNumBlocks1");
		Identifier numBlocks1 = new Identifier((VariableDeclarator)numBlocks_decl1.getDeclarator(0));
		VariableDeclaration numBlocks_decl2 = (VariableDeclaration)SymbolTools.findSymbol(global_table, "gpuNumBlocks2");
		Identifier numBlocks2 = new Identifier((VariableDeclarator)numBlocks_decl2.getDeclarator(0));					*/
		VariableDeclaration numThreads_decl = (VariableDeclaration)SymbolTools.findSymbol(global_table, "gpuNumThreads");
		Identifier numThreads = new Identifier((VariableDeclarator)numThreads_decl.getDeclarator(0));
		ExpressionStatement gpuBytes_stmt = null;
		ExpressionStatement gpuBytes_stmt2 = null;
		ExpressionStatement gMemAdd_stmt = new ExpressionStatement( new AssignmentExpression((Identifier)gmemsize.clone(),
				AssignmentOperator.ADD, (Identifier)cloned_bytes.clone()) );
		ExpressionStatement gMemSub_stmt = new ExpressionStatement( new AssignmentExpression((Identifier)gmemsize.clone(),
				AssignmentOperator.SUBTRACT, (Identifier)cloned_bytes.clone()) );

		/////////////////////////////////////////////////////////////////////////////////////
		// If grid-size is not changed within refstmt1, refstmt1 is used as a mallocPoint. //
		// Otherwise, memory for reduction variable is allocated for each parallel region. //
		// FIXME: currently, duplicate malloc is not checked.                              //
		/////////////////////////////////////////////////////////////////////////////////////
		if( gridSizeNotChanged ) {
			mallocPoint = refstmt1;
		} else {
			mallocPoint = region;
		}
		refstmt2 = region;
		/*
		 * If enclosed omp-for loop contains reduction clause, extract reduction map
		 * and affected code region information.
		 * FIXME: current version can not handle the cases where reduction is used
		 * in a function called in a parallel region.
		 */
		List<OmpAnnotation> omp_annots = IRTools.collectPragmas(region, OmpAnnotation.class, "reduction");
		for (OmpAnnotation annot : omp_annots)
		{
			Statement stmt = (Statement)annot.getAnnotatable();
			redMapList.add((HashMap)annot.get("reduction"));
			targetList.add(stmt);
		}
		///////////////////////////////////////////////////////////////////////////////////////
		// In the new Annotation scheme, omp_annots shown above include the reduction map of //
		// the kernel region, and thus we don't need to add again.                           //
		///////////////////////////////////////////////////////////////////////////////////////
/*		if( reduction_map != null ) {
			redMapList.add(reduction_map);
			targetList.add(region);
		}*/

		for( HashMap reductionMap : redMapList ) {
			targetRegion = targetList.removeFirst();
			for (String ikey : (Set<String>)(reductionMap.keySet())) {
				HashSet<Symbol> redSet = (HashSet<Symbol>)reductionMap.get(ikey);
				BinaryOperator redOp = BinaryOperator.fromString(ikey);
				if( redSet == null ) {
					continue;
				}
				redItemSet.addAll(redSet);
				for( Symbol redSym : redSet) {
					if( redSym instanceof VariableDeclarator ) {
						VariableDeclaration decl = (VariableDeclaration)((VariableDeclarator)redSym).getParent();
						/*
						 * Create a cloned Declaration of the threadprivate variable.
						 */
						VariableDeclarator cloned_declarator =
							(VariableDeclarator)((VariableDeclarator)redSym).clone();
						cloned_declarator.setInitializer(null);
						/////////////////////////////////////////////////////////////////////////////////
						// __device__ and __global__ functions can not declare static variables inside //
						// their body.                                                                 //
						/////////////////////////////////////////////////////////////////////////////////
						List<Specifier> clonedspecs = new ChainedList<Specifier>();
						clonedspecs.addAll(decl.getSpecifiers());
						clonedspecs.remove(Specifier.STATIC);
						VariableDeclaration cloned_decl = new VariableDeclaration(clonedspecs, cloned_declarator);
						Identifier cloned_ID = new Identifier(cloned_declarator);
						VariableDeclarator gpu_declarator = null;
						VariableDeclarator extended_declarator = null;
						Identifier gpu_var = null;
						Identifier extended_var = null;
						ArraySpecifier aspec = null;
						int dimsize = 0;

						/////////////////////////////////////////////////////////
						// redMap contains a mapping from a reduction variable //
						// to corresponding GPU variable.                      //
						/////////////////////////////////////////////////////////
						// FIXME: redMap may need to be reset for each procedure.
						if( redMap.containsKey(redSym)) {
							// clone GPU device symbol corresponding to redSym
							gpu_declarator = (VariableDeclarator)redMap.get(redSym);
							gpu_var = new Identifier(gpu_declarator);
						} else {
							//////////////////////////////////////////////////////////
							// Create a GPU device variable corresponding to redSym //
							// - To distinguish this from GPU variable for shared   //
							//   variable, different naming convention is used.     //
							// Ex: float * red__x; //GPU variable for reduction     //
							//    cf: float * gpu_x; //GPU variable for shared data //
							//////////////////////////////////////////////////////////
							// Give a new name for the device variable
							StringBuilder str = new StringBuilder(80);
							str.append("red__");
							str.append(cloned_ID.toString());
							// The type of the device symbol should be a pointer type
							gpu_declarator = new VariableDeclarator(PointerSpecifier.UNQUALIFIED,
									new NameID(str.toString()));
							VariableDeclaration gpu_decl = new VariableDeclaration(cloned_decl.getSpecifiers(),
									gpu_declarator);
							gpu_var = new Identifier(gpu_declarator);
							procbody.addDeclaration(gpu_decl);
							// Add mapping from redSym to gpu_declarator
							redMap.put(redSym, gpu_declarator);
							//insertMalloc = true;
						}

						/////////////////////////////
						// Check duplicate Malloc. //
						/////////////////////////////
						boolean insertMalloc = false;
						HashSet<String> memTrSet = null;
						StringBuilder str = new StringBuilder(80);
						str.append("malloc_");
						str.append(gpu_var.getName());
						if( c2gMemTr.containsKey(mallocPoint) ) {
							memTrSet = (HashSet<String>)c2gMemTr.get(mallocPoint);
							if( !memTrSet.contains(str.toString()) ) {
								memTrSet.add(str.toString());
								insertMalloc = true;
							}
						} else {
							memTrSet = new HashSet<String>();
							memTrSet.add(str.toString());
							c2gMemTr.put(mallocPoint, memTrSet);
							insertMalloc = true;
						}

						/////////////////////////////////////////////////////////////////////////
						// Memory allocation for the device variable                           //
						//  - Insert cudaMalloc() function before the region                   //
						// Ex: CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu_x)), gpuBytes)); //
						/////////////////////////////////////////////////////////////////////////
						FunctionCall malloc_call = null;
						List<Specifier> specs = new ArrayList<Specifier>(4);
						specs.add(Specifier.VOID);
						specs.add(PointerSpecifier.UNQUALIFIED);
						specs.add(PointerSpecifier.UNQUALIFIED);
						List<Expression> arg_list = new ArrayList<Expression>();
						arg_list.add(new Typecast(specs, new UnaryExpression(UnaryOperator.ADDRESS_OF,
								(Identifier)gpu_var.clone())));
						SizeofExpression sizeof_expr = new SizeofExpression(cloned_decl.getSpecifiers());
						if( SymbolTools.isScalar(redSym) ) {
							malloc_call = new FunctionCall(new NameID("cudaMalloc"));
							// Insert "gpuBytes = gpuNumBlocks * sizeof(varType);" statement.
							AssignmentExpression assignex = new AssignmentExpression((Identifier)cloned_bytes.clone(),
									AssignmentOperator.NORMAL, new BinaryExpression((Expression)numBlocks.clone(),
											BinaryOperator.MULTIPLY, sizeof_expr));
							// Insert "gpuBytes = gpuNumBlocks1 * gpuNumBlocks2 * sizeof(varType);" statement.
/*							AssignmentExpression assignex = new AssignmentExpression((Identifier)cloned_bytes.clone(),
									AssignmentOperator.NORMAL, new BinaryExpression(new BinaryExpression(
											(Expression)numBlocks1.clone(), BinaryOperator.MULTIPLY,
											(Expression)numBlocks2.clone()), BinaryOperator.MULTIPLY, sizeof_expr));*/
							ExpressionStatement estmt = new ExpressionStatement(assignex);
							gpuBytes_stmt = (ExpressionStatement)estmt.clone();
							assignex = new AssignmentExpression((Identifier)cloned_bytes.clone(),
									AssignmentOperator.NORMAL, (Expression)sizeof_expr.clone());
							gpuBytes_stmt2 = new ExpressionStatement(assignex);
							// Create "gpuSmemSize += gpuNumThreads * sizeof(varType);" statement.
							assignex = new AssignmentExpression((Identifier)smemsize.clone(),
									AssignmentOperator.ADD, new BinaryExpression((Identifier)numThreads.clone(),
											BinaryOperator.MULTIPLY, (Expression)sizeof_expr.clone()));
							ExpressionStatement sMemAddStmt = new ExpressionStatement(assignex);
							// Create "gpuSmemSize -= gpuNumThreads * sizeof(varType);" statement.
							assignex = new AssignmentExpression((Identifier)smemsize.clone(),
									AssignmentOperator.SUBTRACT, new BinaryExpression((Identifier)numThreads.clone(),
											BinaryOperator.MULTIPLY, (Expression)sizeof_expr.clone()));
							ExpressionStatement sMemSubStmt = new ExpressionStatement(assignex);
							if( insertMalloc ) {
								((CompoundStatement)mallocPoint.getParent()).addStatementBefore(mallocPoint, estmt);
								if( firstCudaStmt2 == null ) {
									firstCudaStmt2 = estmt;
								}
								if( mallocPoint == region ) {
									num_cudastmts-=1;
								}
								if( opt_addSafetyCheckingCode ) {
									// Insert "gpuGmemSize += gpuBytes;" statement
									((CompoundStatement)mallocPoint.getParent()).addStatementBefore(mallocPoint,
											(Statement)gMemAdd_stmt.clone());
									// Insert "gpuSmemSize += gpuNumThreads * sizeof(varType);" statement.
									((CompoundStatement)mallocPoint.getParent()).addStatementBefore(mallocPoint,
											sMemAddStmt);
									// Insert "gpuSmemSize -= gpuNumThreads * sizeof(varType);" statement.
									((CompoundStatement)mallocPoint.getParent()).addStatementAfter(mallocPoint,
											sMemSubStmt);
									if( mallocPoint == region ) {
										num_cudastmts-=3;
									}
								}
							}

							/////////////////////////////////////////////////////////////
							// Create a parameter Declaration for the kernel function  //
							//  - Change the scalar variable to a pointer type         //
							/////////////////////////////////////////////////////////////
							//VariableDeclarator pointerV_declarator =
							//	scalarReductionConv(redOp, cloned_declarator, new_proc, targetRegion);
							VariableDeclarator pointerV_declarator =
								scalarReductionConv2(redOp, cloned_declarator, new_proc, targetRegion,
										redArgSet, redParamSet);
							redOpSet.add(redOp);
							Identifier pointer_var = new Identifier(pointerV_declarator);

							// Insert argument to the kernel function call
							call_to_new_proc.addArgument((Identifier)gpu_var.clone());

							///////////////////////////////////////////////////////////////////
							// Insert "gpuSmemSize += BLOCK_SIZE * sizeof(float);" statement //
							// before a kernel call site.                                    //
							///////////////////////////////////////////////////////////////////
							///////////////////////////////////////////////////////////////////
							// Insert "gpuSmemSize -= BLOCK_SIZE * sizeof(float);" statement //
							// after a kernel call site.                                     //
							///////////////////////////////////////////////////////////////////
						} else if( SymbolTools.isArray(redSym) ) {
							//////////////////////////////////////////////////////////////////////
							// Insert "gpuBytes = gpuNumBlocks * (dimension1 * dimension2 * ..) //
							// * sizeof(varType);" statement                                    //
							//////////////////////////////////////////////////////////////////////
							List aspecs = redSym.getArraySpecifiers();
							aspec = (ArraySpecifier)aspecs.get(0);
							dimsize = aspec.getNumDimensions();
							//////////////////////////////////////////////////////
							//  Create cudaMalloc() statement for GPU variable. //
							//////////////////////////////////////////////////////
							malloc_call = new FunctionCall(new NameID("cudaMalloc"));
							/////////////////////////////////////////////////////////////////////////////////////
							// Add malloc size (gpuBytes) statement                                            //
						    // Ex: gpuBytes= gpuNumBlocks1 * gpuNumBlocks2 * (SIZE1 * SIZE2 * sizeof (float)); //
							/////////////////////////////////////////////////////////////////////////////////////
							Object o1 = aspec.getDimension(0).clone();
							if( o1 == null ) {
								Tools.exit(pass_name + " [Error in transforming a parallel region in a function, " +
										proc.getSymbolName() + "()] the first dimension of a reduction array, "
										+ redSym + ", is missing; for the O2G translator " +
										"to allocate GPU memory for this array, the exact dimension size of accessed" +
										" array section should be specified." );
							}
							Expression biexp = (Expression)o1;
							for( int i=1; i<dimsize; i++ )
							{
								Object o2 = aspec.getDimension(i).clone();
								if (o2 instanceof Expression)
									biexp = new BinaryExpression((Expression)o1, BinaryOperator.MULTIPLY, (Expression)o2);
								else
									throw new IllegalArgumentException("all list items must be Expressions; found a "
											+ o2.getClass().getName() + " instead");
								o1 = biexp;
							}
							BinaryExpression biexp2 = new BinaryExpression(biexp, BinaryOperator.MULTIPLY, sizeof_expr);
							biexp = new BinaryExpression((Expression)numBlocks.clone(),
									BinaryOperator.MULTIPLY, biexp2);
/*							biexp = new BinaryExpression(new BinaryExpression((Expression)numBlocks1.clone(),
									BinaryOperator.MULTIPLY, (Expression)numBlocks2.clone()),
									BinaryOperator.MULTIPLY, biexp2);*/
							AssignmentExpression assignex = new AssignmentExpression((Expression)cloned_bytes.clone(),
									AssignmentOperator.NORMAL, biexp);
							ExpressionStatement estmt = new ExpressionStatement(assignex);
							assignex = new AssignmentExpression((Expression)cloned_bytes.clone(),
									AssignmentOperator.NORMAL, (Expression)biexp2.clone());
							gpuBytes_stmt2 = new ExpressionStatement(assignex);
							// Create "gpuSmemSize += gpuNumThreads * SIZE1 * SIZE2 * sizeof(varType);" statement.
							assignex = new AssignmentExpression((Identifier)smemsize.clone(),
									AssignmentOperator.ADD, new BinaryExpression((Identifier)numThreads.clone(),
											BinaryOperator.MULTIPLY, (Expression)biexp2.clone()));
							ExpressionStatement sMemAddStmt = new ExpressionStatement(assignex);
							// Create "gpuSmemSize -= gpuNumThreads * SIZE1 * SIZE2 * sizeof(varType);" statement.
							assignex = new AssignmentExpression((Identifier)smemsize.clone(),
									AssignmentOperator.SUBTRACT, new BinaryExpression((Identifier)numThreads.clone(),
											BinaryOperator.MULTIPLY, (Expression)biexp2.clone()));
							ExpressionStatement sMemSubStmt = new ExpressionStatement(assignex);
							gpuBytes_stmt = (ExpressionStatement)estmt.clone();
							if( insertMalloc ) {
								((CompoundStatement)mallocPoint.getParent()).addStatementBefore(mallocPoint, estmt);
								if( firstCudaStmt2 == null ) {
									firstCudaStmt2 = estmt;
								}
								if( mallocPoint == region ) {
									num_cudastmts-=1;
								}
								if( opt_addSafetyCheckingCode ) {
									// Insert "gpuGmemSize += gpuBytes;" statement
									((CompoundStatement)mallocPoint.getParent()).addStatementBefore(mallocPoint,
											(Statement)gMemAdd_stmt.clone());
									// Insert "gpuSmemSize += gpuNumThreads * SIZE1 * SIZE2 * sizeof(varType);" statement.
									((CompoundStatement)mallocPoint.getParent()).addStatementBefore(mallocPoint,
											sMemAddStmt);
									// Insert "gpuSmemSize -= gpuNumThreads * SIZE1 * SIZE2 * sizeof(varType);" statement.
									((CompoundStatement)mallocPoint.getParent()).addStatementAfter(mallocPoint,
											sMemSubStmt);
									if( mallocPoint == region ) {
										num_cudastmts-=3;
									}
								}
							}

							////////////////////////////////////////////////////////////
							// Create a parameter Declaration for the kernel function //
							////////////////////////////////////////////////////////////
							// Create an extended array type                          //
							// Ex1: "float x[][SIZE1]"                                //
							// Ex2: "float x[][SIZE1][SIZE2]"                         //
							////////////////////////////////////////////////////////////
							//VariableDeclarator arrayV_declarator = arrayReductionConv(redOp,
							//		cloned_declarator, new_proc, targetRegion);
							VariableDeclarator arrayV_declarator = arrayReductionConv2(redOp,
									cloned_declarator, new_proc, targetRegion, redArgSet, redParamSet);
							redOpSet.add(redOp);

							Identifier array_var = new Identifier(arrayV_declarator);

							///////////////////////////////////////////////////////
							// Insert argument to the kernel function call       //
							//  - Cast the gpu variable to pointer-to-array type //
							//  Ex: (float (*)[SIZE1][SIZE2]) gpu_x              //
							///////////////////////////////////////////////////////
							List castspecs = new LinkedList();
							castspecs.addAll(cloned_decl.getSpecifiers());
							/*
							 * FIXME: ArrayAccess was used for (*)[SIZE1][SIZE2],
							 * but this may not be semantically correct
							 * way to represent (*)[SIZE1][SIZE2] in IR.
							 */
							List tindices = new LinkedList();
							for( int i=0; i<dimsize; i++) {
								tindices.add(aspec.getDimension(i).clone());
							}
							ArrayAccess castArray = new ArrayAccess(new NameID("(*)"), tindices);
							castspecs.add(castArray);
							call_to_new_proc.addArgument(new Typecast(castspecs,
									(Identifier)gpu_var.clone()));
							/////////////////////////////////////////////////////////////////////////
							// Insert "gpuSmemSize += BLOCK_SIZE * SIZE1 * SIZE2 * sizeof(float);" //
							// statement before a kernel call site.                                //
							/////////////////////////////////////////////////////////////////////////
							/////////////////////////////////////////////////////////////////////////
							// Insert "gpuSmemSize -= BLOCK_SIZE * SIZE1 * SIZE2 * sizeof(float);" //
							// statement after a kernel call site.                                 //
							/////////////////////////////////////////////////////////////////////////
						} else if( SymbolTools.isPointer(redSym) ) {
							Tools.exit(pass_name + "[ERROR] extractKernelRegion() needs to support Pointer type " +
									"threadprivate variable: "
									+ redSym.toString());
						} else {
							Tools.exit(pass_name + "[ERROR] extractKernelRegion() found unsupported " +
									"threadprivate symbols." + redSym.toString());
						}

						List<Specifier> castspecs = null;
						AssignmentExpression assignex = null;
						BinaryExpression hostWidthBytes = null;
						// Add gpuBytes argument to cudaMalloc() call
						arg_list.add((Identifier)cloned_bytes.clone());
						malloc_call.setArguments(arg_list);
						FunctionCall safe_call = new FunctionCall( new NameID("CUDA_SAFE_CALL") );
						safe_call.addArgument(malloc_call);
						ExpressionStatement malloc_stmt = new ExpressionStatement(safe_call);
						if( insertMalloc ) {
							((CompoundStatement)mallocPoint.getParent()).addStatementBefore(
									mallocPoint, malloc_stmt);
							if( mallocPoint == region ) {
								num_cudastmts--;
							}
							if( firstCudaStmt2 == null ) {
								firstCudaStmt2 = malloc_stmt;
							}
						}

						///////////////////////////////////////////////////////////////////////////////////////
						// Create a temporary array that is an extended version of the reduction variable.   //
						// - The extended array is used for final reduction across thread blocks on the CPU. //
						///////////////////////////////////////////////////////////////////////////////////////
						if( insertMalloc ) {
							/*
							 * redMap also contains reduction variable to extended variable mapping.
							 */
							if( redMap.containsKey(gpu_declarator)) {
								// clone GPU device symbol corresponding to threadPriv_var
								extended_declarator = (VariableDeclarator)redMap.get(gpu_declarator);
								extended_var = new Identifier(extended_declarator);
							} else {
								//////////////////////////////////////////////////////////////////////////
								// Create a temporary pointer variable pointing to the temporary array. //
								// Ex: float * x__extended;                                              //
								//////////////////////////////////////////////////////////////////////////
								str = new StringBuilder(80);
								str.append(cloned_ID.toString());
								str.append("__extended");
								// The type of the device symbol should be a pointer type
								extended_declarator = new VariableDeclarator(PointerSpecifier.UNQUALIFIED,
										new NameID(str.toString()));
								VariableDeclaration extended_decl = new VariableDeclaration(
										cloned_decl.getSpecifiers(), extended_declarator);
								extended_var = new Identifier(extended_declarator);
								procbody.addDeclaration(extended_decl);
								// Add mapping from gpu_declarator to extended_declarator
								redMap.put(gpu_declarator, extended_declarator);
								}
							}
						///////////////////////////////////////////////////////////////////////////
						// Create malloc() statement, "x__extended = (float *)malloc(gpuBytes);" //
						///////////////////////////////////////////////////////////////////////////
						FunctionCall tempMalloc_call = new FunctionCall(new NameID("malloc"));
						tempMalloc_call.addArgument((Expression)cloned_bytes.clone());
						castspecs = new LinkedList<Specifier>();
						castspecs.addAll(cloned_decl.getSpecifiers());
						castspecs.add(PointerSpecifier.UNQUALIFIED);
						assignex = new AssignmentExpression((Identifier)extended_var.clone(),
								AssignmentOperator.NORMAL, new Typecast(castspecs, tempMalloc_call));
						ExpressionStatement eMallocStmt = new ExpressionStatement(assignex);
						((CompoundStatement)mallocPoint.getParent()).addStatementBefore(
								mallocPoint, eMallocStmt);
						if( mallocPoint == region ) {
							num_cudastmts--;
						}
						if( firstCudaStmt2 == null ) {
							firstCudaStmt2 = eMallocStmt;
						}

						/////////////////////////////////////////////////////////////////
						// Insert cudaFree() to deallocate device memory.              //
						// Because cuda-related statements are added in reverse order, //
						// this function call is added first.                          //
						/////////////////////////////////////////////////////////////////
						if( insertMalloc ) {
							if( opt_addSafetyCheckingCode ) {
								// Insert "gpuGmemSize -= gpuBytes;" statement.
								((CompoundStatement)mallocPoint.getParent()).addStatementAfter(mallocPoint,
										(Statement)gMemSub_stmt.clone());
								if( mallocPoint == region ) {
									num_cudastmts-=1;
								}
							}
							// Insert "CUDA_SAFE_CALL(cudaFree(gpu_x));" statement.
							FunctionCall cudaFree_call = new FunctionCall(new NameID("cudaFree"));
							cudaFree_call.addArgument((Identifier)gpu_var.clone());
							safe_call = new FunctionCall( new NameID("CUDA_SAFE_CALL"));
							safe_call.addArgument(cudaFree_call);
							ExpressionStatement free_stmt = new ExpressionStatement(safe_call);
							((CompoundStatement)mallocPoint.getParent()).addStatementAfter(mallocPoint, free_stmt);
							if( mallocPoint == region ) {
								num_cudastmts-=1;
							}

							// Insert free(x__extended);
							if( redMap.containsKey(gpu_declarator)) {
								// clone GPU device symbol corresponding to threadPriv_var
								extended_var = new Identifier((VariableDeclarator)
										redMap.get(gpu_declarator));
								FunctionCall free_call = new FunctionCall(new NameID("free"));
								free_call.addArgument(extended_var);
								free_stmt = new ExpressionStatement(free_call);
								((CompoundStatement)mallocPoint.getParent()).addStatementAfter(mallocPoint,
										free_stmt);
								if( mallocPoint == region ) {
									num_cudastmts--;
								}
							}
							// Remove mapping from shared_var to gpu_declarator
							//redMap.remove(redSym);
							// Remove mapping from gpu_declarator to extended_declarator
							//redMap.remove(gpu_declarator);
						}

						/*
						 * Create or find temporaray pointers that are used to pointer calculation.
						 * redMap also contains extended variable to row_temps list mapping.
						 */
						List<Identifier> row_temps = null;
						if( redMap.containsKey(extended_declarator)) {
							row_temps = (List<Identifier>)
									redMap.get(extended_declarator);
						} else {
							row_temps = new ArrayList<Identifier>(dimsize+1);
							for( int i=0; i<dimsize; i++ ) {
								row_temps.add(SymbolTools.getPointerTemp(procbody,
										clonedspecs, "row_temp"));
							}
							row_temps.add((Identifier)extended_var.clone());
							// Add mapping from extended_declarator to row_temps list
							redMap.put(extended_declarator, row_temps);
						}
						////////////////////////////////////////////////////
						// Insert codes for final reduction on the CPU.   //
						////////////////////////////////////////////////////
						// Create or find temporary index variables.
						List<Identifier> index_vars = new LinkedList<Identifier>();
						for( int i=0; i<=dimsize; i++ ) {
							index_vars.add(TransformTools.getTempIndex(procbody, i));
						}
						List<Expression> edimensions = new LinkedList<Expression>();
						if( aspec == null ) {
							edimensions.add((Expression)numBlocks.clone());
						} else {
							edimensions.add((Expression)numBlocks.clone());
							for( int i=0; i<dimsize; i++ )
							{
								edimensions.add((Expression)aspec.getDimension(i).clone());
							}
						}
						Identifier index_var = null;
						Statement loop_init = null;
						Expression condition = null;
						Expression step = null;
						CompoundStatement loop_body = null;
						ForLoop innerLoop = null;
						//////////////////////////////////////////////////////////////////////////////////
						// Insert codes for final reduction on the CPU.                                 //
						//////////////////////////////////////////////////////////////////////////////////
						// Ex: for(i=0; i<gpuNumBlocks; i++) {                                          //
						// 		row_temp1 = (float*)((char*)x__extended + i*SIZE1*SIZE2*sizeof(float)); //
						//         for(k=0; k<SIZE1; k++) {                                             //
						// 			row_temp0 = (float*)((char*)row_temp1 + k*SIZE2*sizeof(float));     //
						//             for(m=0; m<SIZE2; m++) {                                         //
						//                x[k][m] += row_temp0[m];                                      //
						//             }                                                                //
						//         }                                                                    //
						//      }                                                                       //
						//////////////////////////////////////////////////////////////////////////////////
						// Create the nested loops.
						for( int i=0; i<=dimsize; i++ ) {
							index_var = index_vars.get(i);
							assignex = new AssignmentExpression((Identifier)index_var.clone(),
									AssignmentOperator.NORMAL, new IntegerLiteral(0));
							loop_init = new ExpressionStatement(assignex);
							if( i<dimsize ) {
								condition = new BinaryExpression((Identifier)index_var.clone(),
										BinaryOperator.COMPARE_LT,
										(Expression)aspec.getDimension(dimsize-1-i).clone());
							} else {
								condition = new BinaryExpression((Identifier)index_var.clone(),
										BinaryOperator.COMPARE_LT, (Identifier)numBlocks.clone());
							}
							step = new UnaryExpression(UnaryOperator.POST_INCREMENT,
									(Identifier)index_var.clone());
							loop_body = new CompoundStatement();
							if( i==0  ) {
								if( dimsize == 0 ) {
									assignex = TransformTools.RedExpression((Identifier)cloned_ID.clone(),
											redOp, new ArrayAccess(
													(Identifier)row_temps.get(0).clone(),
													(Identifier)index_var.clone()));
								} else {
								List<Expression> indices = new LinkedList<Expression>();
								for( int k=dimsize-1; k>=0; k-- ) {
									indices.add((Expression)index_vars.get(k).clone());
								}
									assignex = TransformTools.RedExpression(new ArrayAccess(
											(Identifier)cloned_ID.clone(), indices),
											redOp, new ArrayAccess(
													(Identifier)row_temps.get(0).clone(),
													(Identifier)index_var.clone()));
								}
							} else {
								castspecs = new ArrayList<Specifier>(2);
								castspecs.add(Specifier.CHAR);
								castspecs.add(PointerSpecifier.UNQUALIFIED);
								Typecast tcast1 = new Typecast(castspecs, (Identifier)row_temps.get(i).clone());
								BinaryExpression biexp1 = new BinaryExpression((Expression)sizeof_expr.clone(),
										BinaryOperator.MULTIPLY, (Expression)aspec.getDimension(dimsize-1).clone());
								BinaryExpression biexp2 = null;
								for( int k=1; k<i; k++ ) {
									biexp2 = new BinaryExpression(biexp1, BinaryOperator.MULTIPLY,
											(Expression)aspec.getDimension(dimsize-1-k).clone());
									biexp1 = biexp2;
								}
								biexp2 = new BinaryExpression((Expression)index_var.clone(),
										BinaryOperator.MULTIPLY, biexp1);
								biexp1 = new BinaryExpression(tcast1, BinaryOperator.ADD, biexp2);
								castspecs = new ArrayList<Specifier>();
								castspecs.addAll(cloned_decl.getSpecifiers());
								castspecs.add(PointerSpecifier.UNQUALIFIED);
								tcast1 = new Typecast(castspecs, biexp1);
								assignex = new AssignmentExpression((Identifier)row_temps.get(i-1).clone(),
										AssignmentOperator.NORMAL, tcast1);
							}
							loop_body.addStatement(new ExpressionStatement(assignex));
							if( innerLoop != null ) {
								loop_body.addStatement(innerLoop);
							}
							innerLoop = new ForLoop(loop_init, condition, step, loop_body);
						}
						((CompoundStatement)refstmt2.getParent()).addStatementAfter(refstmt2, innerLoop);
						if( refstmt2 == region ) {
							num_cudastmts-=1;
						}

						////////////////////////////////////////////////////////////////////
						// Insert memory copy function from GPU to CPU.                   //
						////////////////////////////////////////////////////////////////////
						// Ex: gpuBytes= gpuNumBlocks * (SIZE1 * SIZE2 * sizeof (float)); //
						//     CUDA_SAFE_CALL(cudaMemcpy(x__extended, gpu_x, gpuBytes,    //
						//     cudaMemcpyDeviceToHost));                                  //
						////////////////////////////////////////////////////////////////////
						FunctionCall memCopy_call2 = new FunctionCall(new NameID("cudaMemcpy"));
						List<Expression> arg_list3 = new ArrayList<Expression>();
						arg_list3.add((Identifier)extended_var.clone());
						arg_list3.add((Identifier)gpu_var.clone());
						arg_list3.add((Identifier)cloned_bytes.clone());
						arg_list3.add(new NameID("cudaMemcpyDeviceToHost"));
						memCopy_call2.setArguments(arg_list3);
						safe_call = new FunctionCall( new NameID("CUDA_SAFE_CALL"));
						safe_call.addArgument(memCopy_call2);
						ExpressionStatement memCopy_stmt = new ExpressionStatement(safe_call);
						((CompoundStatement)refstmt2.getParent()).addStatementAfter(refstmt2, memCopy_stmt);
						((CompoundStatement)refstmt2.getParent()).addStatementAfter(refstmt2, gpuBytes_stmt);
						if( refstmt2 == region ) {
							num_cudastmts-=2;
						}
					} else {
						Tools.exit(pass_name + "[ERROR] reductionTransformation() supports VariableDeclarator reduction symbols only; " +
							"current procedure = " + proc.getSymbolName() + " current symbol = " + redSym);
					}
				}
			} // end of  for (String ikey : (Set<String>)(reductionMap.keySet())) {
		} // end of for( HashMap reductionMap : redMapList ) {

		if( redArgSet.size() > 0 ) {
			reductionConv(redArgSet, redParamSet, redOpSet, region, cudaNoRedUnrollSet);
		}

		redData.add(firstCudaStmt2);
		redData.add(redItemSet);
		return redData;
	}

	/**
	 * Add necessary memory transfer calls for Omp parallel regions that will be executed on the CPU.
	 *
	 * @param proc Procedure that contains the kernel region to be transformed
	 * @param annot OmpAnnotation attached to the parallel region that will be executed on the CPU.
	 */
	private static void handleOtherOmpParallelRegion(Procedure proc, OmpAnnotation annot) {

		PrintTools.println(pass_name + " handles other Omp parallel region.", 2);

		Statement region = (Statement)annot.getAnnotatable();
		SymbolTable global_table = (SymbolTable) proc.getParent();
		TranslationUnit tu = (TranslationUnit)proc.getParent();
		CompoundStatement procbody = proc.getBody();
		boolean use_MallocPitch = opt_MallocPitch;

		///////////////////////////////////////////////////////////////
		// Extract Cuda directives attached to this parallel region. //
		///////////////////////////////////////////////////////////////
		HashSet<String> cudaC2GMemTrSet = new HashSet<String>();
		HashSet<String> cudaNoC2GMemTrSet = new HashSet<String>();
		HashSet<String> cudaG2CMemTrSet = new HashSet<String>();
		HashSet<String> cudaNoG2CMemTrSet = new HashSet<String>();
		List<CudaAnnotation> cudaAnnots = region.getAnnotations(CudaAnnotation.class);
		if( cudaAnnots != null ) {
			for( CudaAnnotation cannot : cudaAnnots ) {
				HashSet<String> dataSet = (HashSet<String>)cannot.get("c2gmemtr");
				if( dataSet != null ) {
					cudaC2GMemTrSet.addAll(dataSet);
				}
				dataSet = (HashSet<String>)cannot.get("noc2gmemtr");
				if( dataSet != null ) {
					cudaNoC2GMemTrSet.addAll(dataSet);
				}
				dataSet = (HashSet<String>)cannot.get("g2cmemtr");
				if( dataSet != null ) {
					cudaG2CMemTrSet.addAll(dataSet);
				}
				dataSet = (HashSet<String>)cannot.get("nog2cmemtr");
				if( dataSet != null ) {
					cudaNoG2CMemTrSet.addAll(dataSet);
				}
			}
	 	}

		/* Extract data sharing attributes from OpenMP pragma */
		HashSet<Symbol> OmpSharedSet = null;
		HashSet<Symbol> OmpROSharedSet = null;
		HashSet<Symbol> OmpPrivSet = null;
		HashSet<Symbol> OmpThreadPrivSet = null;
		HashSet<Symbol> OmpCopyinSet = null;
		if (annot.keySet().contains("shared"))
			OmpSharedSet = (HashSet<Symbol>) annot.get("shared");
		if (annot.keySet().contains("private"))
			OmpPrivSet = (HashSet<Symbol>) annot.get("private");
		if (annot.keySet().contains("threadprivate"))
			OmpThreadPrivSet = (HashSet<Symbol>) annot.get("threadprivate");
		if (annot.keySet().contains("copyin"))
			OmpCopyinSet = (HashSet<Symbol>) annot.get("copyin");

		CompoundStatement parentRegion = (CompoundStatement)region.getParent();
		Statement refstmt = region;

		// Auxiliary variables used for GPU kernel conversion
		VariableDeclaration bytes_decl = (VariableDeclaration)SymbolTools.findSymbol(global_table, "gpuBytes");
		Identifier cloned_bytes = new Identifier((VariableDeclarator)bytes_decl.getDeclarator(0));
		ExpressionStatement gpuBytes_stmt = null;

		/////////////////////////////
		// Handle OMP Shared data. //
		/////////////////////////////
		if (OmpSharedSet != null) {
			for( Symbol shared_var : OmpSharedSet ) {
				if( shared_var instanceof VariableDeclarator ) {
					VariableDeclaration decl = (VariableDeclaration)((VariableDeclarator)shared_var).getParent();
					/*
					 * Create a cloned Declaration of the shared variable.
					 */
					VariableDeclarator cloned_declarator =
						(VariableDeclarator)((VariableDeclarator)shared_var).clone();
					cloned_declarator.setInitializer(null);
					/////////////////////////////////////////////////////////////////////////////////
					// __device__ and __global__ functions can not declare static variables inside //
					// their body.                                                                 //
					/////////////////////////////////////////////////////////////////////////////////
					List<Specifier> clonedspecs = new ChainedList<Specifier>();
					clonedspecs.addAll(decl.getSpecifiers());
					clonedspecs.remove(Specifier.STATIC);
					VariableDeclaration cloned_decl = new VariableDeclaration(clonedspecs, cloned_declarator);
					Identifier cloned_ID = new Identifier(cloned_declarator);

					Identifier gpu_var = null;
					Identifier pitch_var = null;
					Identifier pointer_var = null;


					/*
					 * c2gMap contains a mapping from a shared/threadprivate variable to corresponding GPU variable.
					 */
					if( c2gMap.containsKey(shared_var)) {
						// clone GPU device symbol corresponding to shared_var
						gpu_var = new Identifier((VariableDeclarator)c2gMap.get(shared_var));
					} else {
						// This shared variable has not been used by GPU yet.
						////////////////////////////////////////////////////////////////////////////////////
						// If a kernel function, which is called  after this parallel region, modify this //
						// shared_var, and both regions are in the same loop body, the kernel function    //
						// should move this shared_var back to CPU.                                       //
						////////////////////////////////////////////////////////////////////////////////////
						Traversable t = region.getParent().getParent();
						if( t instanceof Loop) {
							List<Traversable> children = parentRegion.getChildren();
							int refIndex = Tools.indexByReference(children, refstmt);
							int currIndex = children.size() - 1;
							while( currIndex > refIndex ) {
								Statement currStmt = (Statement)children.get(currIndex);
								if( currStmt.containsAnnotation(OmpAnnotation.class, "parallel") ) {
									if( AnalysisTools.checkKernelEligibility(currStmt) != 0 ) {
										//currStmt will be executed by CPU.
										currIndex--;
										continue;
									}
									Set<Symbol> DefSet = DataFlowTools.getDefSymbol(currStmt);
									if( DefSet.contains(shared_var) ) {
										OmpAnnotation oannot = currStmt.getAnnotation(OmpAnnotation.class,
												"reduction");
										boolean usedAsReduction = false;
										if( oannot != null ) {
											HashMap reductionMap = oannot.get("reduction");
											for (String ikey : (Set<String>)(reductionMap.keySet())) {
												HashSet<Symbol> redSet = (HashSet<Symbol>)reductionMap.get(ikey);
												if( redSet.contains(shared_var) ) {
													usedAsReduction = true;
													break;
												}
											}
										}
										if( !usedAsReduction ) {
											CudaAnnotation cannot = currStmt.getAnnotation(CudaAnnotation.class, "g2cmemtr");
											if( cannot == null ) {
												cannot = new CudaAnnotation("gpurun", "true");
												cannot.put("g2cmemtr", new HashSet<String>());
												currStmt.annotate(cannot);
											}
											Set<String> hSet = cannot.get("g2cmemtr");
											hSet.add(shared_var.getSymbolName());
											break;
										}
									}
								}
								currIndex--;
							}
						}
						continue;
					}
					StringBuilder str = new StringBuilder(80);

					HashSet<String> memTrSet = null;
					/*
					 * Check duplicate GPU to CPU memory transfers
					 * Currently, simple name-only analysis is conducted; if the same array
					 * is transferred multiply at the same program point, insert only one memory transfer.
					 */
					boolean insertG2CMemTr = false;
					List<Traversable> children = parentRegion.getChildren();
					int currIndex = Tools.indexByReference(children, refstmt);
					while( currIndex >= 0 ) {
						Statement currStmt = (Statement)children.get(currIndex);
						if( g2cMemTr.containsKey(currStmt) ) {
							memTrSet = (HashSet<String>)g2cMemTr.get(currStmt);
							if( memTrSet.contains(gpu_var.getName()) ) {
								insertG2CMemTr = false;
								break;
							} else if( kernelCallStmtSet.contains(currStmt) ) {
								OmpAnnotation omp_annot = currStmt.getAnnotation(OmpAnnotation.class, "parallel");
								if( omp_annot == null ) {
									//We don't have information about this kernel statement.
									insertG2CMemTr = true;
									break;
								} else {
									HashSet<Symbol> hSet = (HashSet<Symbol>)omp_annot.get("shared");
									if( AnalysisTools.containsSymbol(hSet, cloned_ID.getName()) ) {
										insertG2CMemTr = true;
										break;
									}
								}
							}
						} else if( kernelCallStmtSet.contains(currStmt)) {
							OmpAnnotation omp_annot = currStmt.getAnnotation(OmpAnnotation.class, "parallel");
							if( omp_annot == null ) {
								//We don't have information about this kernel statement.
								insertG2CMemTr = true;
								break;
							} else {
								HashSet<Symbol> hSet = (HashSet<Symbol>)omp_annot.get("shared");
								if( AnalysisTools.containsSymbol(hSet, cloned_ID.getName()) ) {
									insertG2CMemTr = true;
									break;
								}
							}
						}
						currIndex--;
					}
					if( (currIndex < 0) && (!insertG2CMemTr) ) {
						Traversable t = region.getParent().getParent();
						if( t instanceof Loop) {
							int refIndex = Tools.indexByReference(children, refstmt);
							currIndex = children.size() - 1;
							while( currIndex > refIndex ) {
								Statement currStmt = (Statement)children.get(currIndex);
								if( currStmt.containsAnnotation(OmpAnnotation.class, "parallel") ) {
									if( AnalysisTools.checkKernelEligibility(currStmt) != 0 ) {
										//currStmt will be executed by CPU.
										currIndex--;
										continue;
									}
									Set<Symbol> DefSet = DataFlowTools.getDefSymbol(currStmt);
									if( DefSet.contains(shared_var) ) {
										OmpAnnotation oannot = currStmt.getAnnotation(OmpAnnotation.class,
												"reduction");
										boolean usedAsReduction = false;
										if( oannot != null ) {
											HashMap reductionMap = oannot.get("reduction");
											for (String ikey : (Set<String>)(reductionMap.keySet())) {
												HashSet<Symbol> redSet = (HashSet<Symbol>)reductionMap.get(ikey);
												if( redSet.contains(shared_var) ) {
													usedAsReduction = true;
													break;
												}
											}
										}
										if( !usedAsReduction ) {
											CudaAnnotation cannot = currStmt.getAnnotation(CudaAnnotation.class, "g2cmemtr");
											if( cannot == null ) {
												cannot = new CudaAnnotation("gpurun", "true");
												cannot.put("g2cmemtr", new HashSet<String>());
												currStmt.annotate(cannot);
											}
											Set<String> hSet = cannot.get("g2cmemtr");
											hSet.add(shared_var.getSymbolName());
											break;
										}
									}
								}
								currIndex--;
							}
						}
					}
					/////////////////////////////////////////////////////////////////////////
					// If Cuda g2cmemtr clause contains this symbol, insertG2CMemTr should //
					// set to true.                                                        //
					/////////////////////////////////////////////////////////////////////////
					if( cudaG2CMemTrSet.contains(cloned_ID.getName()) ) {
						insertG2CMemTr = true;
					}
					///////////////////////////////////////////////////////////////////////////
					// If Cuda nog2cmemtr clause contains this symbol, insertG2CMemTr should //
					// set to false.                                                         //
					///////////////////////////////////////////////////////////////////////////
					if( cudaNoG2CMemTrSet.contains(cloned_ID.getName()) ) {
						insertG2CMemTr = false;
					}
					if( insertG2CMemTr ) {
						if( g2cMemTr.containsKey(refstmt) ) {
							memTrSet = (HashSet<String>)g2cMemTr.get(refstmt);
							if( !memTrSet.contains(gpu_var.getName()) ) {
								memTrSet.add(gpu_var.getName());
							}
						} else {
							memTrSet = new HashSet<String>();
							memTrSet.add(gpu_var.getName());
							g2cMemTr.put(refstmt, memTrSet);
						}
					}

					/*
					 * Check duplicate CPU to GPU memory transfers.
					 * Currently, simple name-only analysis is conducted; if the same array
					 * is transferred multiply at the same program point, insert only one memory transfer.
					 */
					boolean insertC2GMemTr = false;
					Set<Symbol> DefSet = DataFlowTools.getDefSymbol(region);
					if( DefSet.contains(shared_var) ) {
						insertC2GMemTr = true;
					}
					/////////////////////////////////////////////////////////////////////////
					// If Cuda c2gmemtr clause contains this symbol, insertC2GMemTr should //
					// set to true.                                                        //
					/////////////////////////////////////////////////////////////////////////
					if( cudaC2GMemTrSet.contains(cloned_ID.getName()) ) {
						insertC2GMemTr = true;
					}
					///////////////////////////////////////////////////////////////////////////
					// If Cuda noc2gmemtr clause contains this symbol, insertC2GMemTr should //
					// set to false.                                                         //
					///////////////////////////////////////////////////////////////////////////
					if( cudaNoC2GMemTrSet.contains(cloned_ID.getName()) ) {
						insertC2GMemTr = false;
					}
					if( insertC2GMemTr ) {
						if( c2gMemTr.containsKey(refstmt) ) {
							memTrSet = (HashSet<String>)c2gMemTr.get(refstmt);
							if( !memTrSet.contains(gpu_var.getName()) ) {
								memTrSet.add(gpu_var.getName());
							}
						} else {
							memTrSet = new HashSet<String>();
							memTrSet.add(gpu_var.getName());
							c2gMemTr.put(refstmt, memTrSet);
						}
					}

					SizeofExpression sizeof_expr = new SizeofExpression(cloned_decl.getSpecifiers());
					if( SymbolTools.isScalar(shared_var) ) {
						use_MallocPitch = false;
						// Insert "gpuBytes = sizeof(varType);" statement
						AssignmentExpression assignex = new AssignmentExpression(cloned_bytes,AssignmentOperator.NORMAL,
								sizeof_expr);
						ExpressionStatement estmt = new ExpressionStatement(assignex);
						gpuBytes_stmt = (ExpressionStatement)estmt.clone();
					} else if( SymbolTools.isArray(shared_var) ) {
						// Insert "gpuBytes = (dimension1 * dimension2 * ..) * sizeof(varType);" statement
						List aspecs = shared_var.getArraySpecifiers();
						ArraySpecifier aspec = (ArraySpecifier)aspecs.get(0);
						int dimsize = aspec.getNumDimensions();
						VariableDeclaration pitch_decl = null;
						VariableDeclarator pointerV_declarator =  null;
						// cudaMallocPitch() is used only for 2 dimensional arrays.
						if( dimsize == 2 ) {
							use_MallocPitch = opt_MallocPitch;
						}
						else {
							use_MallocPitch = false;
						}
						if( aspec.getDimension(0) == null ) {
							Tools.exit(pass_name + " [Error in transforming a parallel region in a function, " +
									proc.getSymbolName() + "()] the first dimension of a shared array, "
									+ shared_var + ", is missing; for the O2G translator " +
									"to allocate GPU memory for this array, the exact dimension size of accessed" +
									" array section should be specified." );
						}
						if( use_MallocPitch ) {
							// Give a new name for a new pitch variable
							str = new StringBuilder(80);
							str.append("pitch_");
							str.append(cloned_ID.toString());
							pitch_decl = (VariableDeclaration)SymbolTools.findSymbol(procbody, str.toString());
							pitch_var = new Identifier((VariableDeclarator)pitch_decl.getDeclarator(0));
							if( insertC2GMemTr ) {
								BinaryExpression biexp = new BinaryExpression((Expression)pitch_var.clone(),
										BinaryOperator.MULTIPLY, (Expression)aspec.getDimension(0).clone());
								AssignmentExpression assignex = new AssignmentExpression(cloned_bytes,AssignmentOperator.NORMAL,
										biexp);
								gpuBytes_stmt = new ExpressionStatement(assignex);
							}
						} else {
							// Add malloc size (gpuBytes) statement
							// Ex: gpuBytes=(((2048+2)*(2048+2))*sizeof (float));
							Object o1 = aspec.getDimension(0).clone();
							Expression biexp = (Expression)o1;
							for( int i=1; i<dimsize; i++ )
							{
								Object o2 = aspec.getDimension(i).clone();
								if (o2 instanceof Expression)
									biexp = new BinaryExpression((Expression)o1, BinaryOperator.MULTIPLY, (Expression)o2);
								else
									throw new IllegalArgumentException("all list items must be Expressions; found a "
											+ o2.getClass().getName() + " instead");
								o1 = biexp;
							}
							BinaryExpression biexp2 = new BinaryExpression(biexp, BinaryOperator.MULTIPLY, sizeof_expr);
							AssignmentExpression assignex = new AssignmentExpression(cloned_bytes,AssignmentOperator.NORMAL,
									biexp2);
							ExpressionStatement estmt = new ExpressionStatement(assignex);
							gpuBytes_stmt = (ExpressionStatement)estmt.clone();
						}
					} else if( SymbolTools.isPointer(shared_var) ) {
						Tools.exit(pass_name + "[ERROR] extractKernelRegion() needs to support Pointer type shared variable: "
								+ shared_var.toString());
					} else {
						Tools.exit(pass_name + "[ERROR] extractKernelRegion() found unsupported shared symbols."
								+ shared_var.toString());
					}
					FunctionCall safe_call = null;
					BinaryExpression hostWidthBytes = null;
					ArraySpecifier aspec = null;
					if( use_MallocPitch ) {
						List aspecs = shared_var.getArraySpecifiers();
						aspec = (ArraySpecifier)aspecs.get(0);
						hostWidthBytes = new BinaryExpression((Expression)aspec.getDimension(1).clone(),
								BinaryOperator.MULTIPLY, (Expression)sizeof_expr.clone());
					}

					if( use_MallocPitch ) {
						if( insertC2GMemTr ) {
							/* Insert memory copy function from CPU to GPU */
							// Ex: CUDA_SAFE_CALL(cudaMemcpy2D(gpu_b, pitch_b, b, width*sizeof(float),
							// width*sizeof(float), height, cudaMemcpyHostToDevice));
							FunctionCall memCopy_call = new FunctionCall(new NameID("cudaMemcpy2D"));
							List<Expression> arg_list2 = new ArrayList<Expression>();
							arg_list2.add((Identifier)gpu_var.clone());
							arg_list2.add((Identifier)pitch_var.clone());
							arg_list2.add(new Identifier((VariableDeclarator)shared_var));
							arg_list2.add((Expression)hostWidthBytes.clone());
							arg_list2.add((Expression)hostWidthBytes.clone());
							arg_list2.add((Expression)aspec.getDimension(0).clone());
							arg_list2.add(new NameID("cudaMemcpyHostToDevice"));
							memCopy_call.setArguments(arg_list2);
							safe_call = new FunctionCall( new NameID("CUDA_SAFE_CALL"));
							safe_call.addArgument(memCopy_call);
							ExpressionStatement memCopy_stmt = new ExpressionStatement(safe_call);
							((CompoundStatement)refstmt.getParent()).addStatementAfter(refstmt, memCopy_stmt);
							((CompoundStatement)refstmt.getParent()).addStatementAfter(refstmt,
									(Statement)gpuBytes_stmt.clone());
							if( refstmt == region ) {
								num_cudastmts-=2;
							}
						}
						if( insertG2CMemTr ) {
							/* Insert memory copy function from GPU to CPU */
							// Ex: CUDA_SAFE_CALL(cudaMemcpy2D(b, width*sizeof(float), gpu_b, pitch_b,
							// width*sizeof(float), height, cudaMemcpyDeviceToHost));
							FunctionCall memCopy_call2 = new FunctionCall(new NameID("cudaMemcpy2D"));
							List<Expression> arg_list3 = new ArrayList<Expression>();
							arg_list3.add(new Identifier((VariableDeclarator)shared_var));
							arg_list3.add((Expression)hostWidthBytes.clone());
							arg_list3.add((Identifier)gpu_var.clone());
							arg_list3.add((Identifier)pitch_var.clone());
							arg_list3.add((Expression)hostWidthBytes.clone());
							arg_list3.add((Expression)aspec.getDimension(0).clone());
							arg_list3.add(new NameID("cudaMemcpyDeviceToHost"));
							memCopy_call2.setArguments(arg_list3);
							safe_call = new FunctionCall( new NameID("CUDA_SAFE_CALL"));
							safe_call.addArgument(memCopy_call2);
							ExpressionStatement memCopy_stmt = new ExpressionStatement(safe_call);
							((CompoundStatement)refstmt.getParent()).addStatementBefore(refstmt, gpuBytes_stmt);
							((CompoundStatement)refstmt.getParent()).addStatementBefore(refstmt, memCopy_stmt);
							if( refstmt == region ) {
								num_cudastmts-=2;
							}
						}
					} else {
						if( insertC2GMemTr ) {
							/* Insert memory copy function from CPU to GPU */
							// Ex: CUDA_SAFE_CALL(cudaMemcpy(gpu_b, b, gpuBytes, cudaMemcpyHostToDevice));
							FunctionCall memCopy_call = new FunctionCall(new NameID("cudaMemcpy"));
							List<Expression> arg_list2 = new ArrayList<Expression>();
							arg_list2.add((Identifier)gpu_var.clone());
							if( SymbolTools.isScalar(shared_var)) {
								arg_list2.add( new UnaryExpression(UnaryOperator.ADDRESS_OF,
												new Identifier((VariableDeclarator)shared_var)));
							} else {
								arg_list2.add(new Identifier((VariableDeclarator)shared_var));
							}
							arg_list2.add((Identifier)cloned_bytes.clone());
							arg_list2.add(new NameID("cudaMemcpyHostToDevice"));
							memCopy_call.setArguments(arg_list2);
							safe_call = new FunctionCall( new NameID("CUDA_SAFE_CALL"));
							safe_call.addArgument(memCopy_call);
							ExpressionStatement memCopy_stmt = new ExpressionStatement(safe_call);
							((CompoundStatement)refstmt.getParent()).addStatementAfter(refstmt, memCopy_stmt);
							((CompoundStatement)refstmt.getParent()).addStatementAfter(refstmt,
									(Statement)gpuBytes_stmt.clone());
							if( refstmt == region ) {
								num_cudastmts-=2;
							}
						}
						if( insertG2CMemTr ) {
							/* Insert memory copy function from GPU to CPU */
							// Ex: gpuBytes = (4096 * sizeof(float));
							//     CUDA_SAFE_CALL(cudaMemcpy(a, gpu_a, gpuBytes, cudaMemcpyDeviceToHost));
							FunctionCall memCopy_call2 = new FunctionCall(new NameID("cudaMemcpy"));
							List<Expression> arg_list3 = new ArrayList<Expression>();
							if( SymbolTools.isScalar(shared_var)) {
								arg_list3.add( new UnaryExpression(UnaryOperator.ADDRESS_OF,
												new Identifier((VariableDeclarator)shared_var)));
							} else {
								arg_list3.add(new Identifier((VariableDeclarator)shared_var));
							}
							arg_list3.add((Identifier)gpu_var.clone());
							arg_list3.add((Identifier)cloned_bytes.clone());
							arg_list3.add(new NameID("cudaMemcpyDeviceToHost"));
							memCopy_call2.setArguments(arg_list3);
							safe_call = new FunctionCall( new NameID("CUDA_SAFE_CALL"));
							safe_call.addArgument(memCopy_call2);
							ExpressionStatement memCopy_stmt = new ExpressionStatement(safe_call);
							((CompoundStatement)refstmt.getParent()).addStatementBefore(refstmt, gpuBytes_stmt);
							((CompoundStatement)refstmt.getParent()).addStatementBefore(refstmt, memCopy_stmt);
							if( refstmt == region ) {
								num_cudastmts-=2;
							}
						}
					}
				} else {
					Tools.exit(pass_name + "[ERROR] handleOtherOmpParallel() supports VariableDeclarator shared symbols only; " +
							"current procedure = " + proc.getSymbolName() + " current symbol = " + shared_var);
				}
			}
		}
	}

	/**
	 * Checks whether CompoundStatement parent contains kernel regions only.
	 * If so, return true.
	 */
	private static Boolean containsKernelRegionsOnly(CompoundStatement parent, boolean isParent) {
	    FlatIterator iter = new FlatIterator(parent);
	    int num_stmts = 0;
	    if( isParent ) {
	    	num_stmts = num_cudastmts;
	    }

	    for (;;)
	    {
	      Statement stmt = null;

	      try {
	        stmt = (Statement)iter.next(Statement.class);
	      } catch (NoSuchElementException e) {
	        break;
	      }

	      if (stmt instanceof DeclarationStatement)
	      {
	    	  Object o = ((DeclarationStatement)stmt).getDeclaration();
	    	  if( !(o instanceof AnnotationDeclaration) )
	    		  num_stmts++;
	      } else if( stmt instanceof AnnotationStatement ) {
	    	  List<CommentAnnotation> cannots = stmt.getAnnotations(CommentAnnotation.class);
	    	  if( (cannots != null) && (cannots.size() > 0) ) {
	    		  CommentAnnotation annot = cannots.get(0);
    			  // CommentAnnotation contains actual comment in the value field.
	    		  if( ((String)annot.get("comment")).contains("#pragma omp parallel") ) {
	    			  //This annotation is an old parallel pragma commented out
	    			  //by OmpAnalysis.splitParallelRegions().
	    			  num_stmts--;
	    		  }
	    	  }
	      } else if( stmt.containsAnnotation(OmpAnnotation.class, "parallel") ) {
	    	  ////////////////////////////////////////////////////////////////////////
	    	  // In the new Annotation scheme, Annotations attached to a statement  //
	    	  // are not visible in the IR tree.                                    //
	    	  ////////////////////////////////////////////////////////////////////////
	    	  if( stmt instanceof ExpressionStatement ) {
	    		  //////////////////////////////////////////////////////////////////////////////
	    		  // A KernelFunction call has the Omp parallel annotation of the orginal Omp //
	    		  // parallel region.                                                         //
	    		  //////////////////////////////////////////////////////////////////////////////
	    		  Expression expr = ((ExpressionStatement)stmt).getExpression();
	    		  if( !(expr instanceof KernelFunctionCall) ) {
						Tools.exit("[Error in containsKernelRegionsOnly()] Unexpected statement attached " +
						"to an Omp parallel annotation");
	    		  }
	    	  } else { //stmt is an instance of either CompoundStatement or ForLoop.
	    		  /////////////////////////////////////////////////////////////////////////////////
	    		  // In the new convParRegionToKernelFunc(), necessary memory transfer calls are //
	    		  // added for all omp parallel regions, and thus we can safely ignore these.    //
	    		  /////////////////////////////////////////////////////////////////////////////////
/*	    		  int eligibility = checkKernelEligibility(stmt);
	    		  if( eligibility != 0 ) {
	    			  if (eligibility == 3) {
	    				  // Check whether this parallel region is an omp-for loop.
	    				  if( stmt.containsAnnotation(OmpAnnotation.class, "for") ) {
	    					  // In the new annotation scheme, the above check is redundant.
	    					  eligibility = 0;
	    				  } else {
	    					  // Check whether called functions have any omp-for loop.
	    					  List<FunctionCall> funcCalls = IRTools.getFunctionCalls(stmt);
	    					  for( FunctionCall calledProc : funcCalls ) {
	    						  Procedure tProc = calledProc.getProcedure();
	    						  if( tProc != null ) {
	    							  eligibility = checkKernelEligibility(tProc.getBody());
	    							  if(  eligibility == 0 ) {
	    								  break;
	    							  }
	    						  }
	    					  }
	    				  }
	    				  if( eligibility != 0 ) {
	    					  num_stmts++;
	    				  }
	    			  } else if( eligibility == 5 ) {
	    				  //////////////////////////////////////////////////////////////////////////
	    				  // If "cudaMemTrOptLevel >= 2", O2GTranslator assumes that necessary    //
	    				  // information about shared variables needed to be transferred between  //
	    				  // CPU and GPU for executing single/master region exists in cuda        //
	    				  // directives, which can be provided either by user or by compiler      //
	    				  // analysis.                                                            //
	    				  //////////////////////////////////////////////////////////////////////////
	    				  if( MemTrOptLevel > 1 ) {
	    					  PrintTools.println("[WARNING] User may need to provide Cuda directives " +
	    							  "to inform the translator of the memory transfers needed " +
	    							  "for correct Omp Single/Master section execution.", 0);
	    				  } else {
	    					  num_stmts++;
	    				  }
	    			  } else {
	    				  num_stmts++;
	    			  }
	    		  }*/
	    	  }
	      } else {
	    	  num_stmts++;
	      }
	    }

	    PrintTools.println(pass_name + " The following CompoundStatement contains " + num_stmts
	    		+ " non-parallel-region statements: \n" + parent.toString() + "\n", 9);
	    if( isParent ) {
	    	PrintTools.println("num_stmts = " + num_stmts, 2);
	    }
	    if( num_stmts == 0)
	    	return true;
	    else
	    	return false;
	}

	/**
	 * Set kernel configuration parameters:
	 * Assumption: threadBlock is one-dimensional array of threads.
	 * - dim3 dimBlock(gpuNumThreads, 1, 1);
	 * - dim3 dimGrid(gpuNumBlocks1, gpuNumBlocks2, 1);
	 *
	 * @param global_table Global Symbol Table
	 * @param region Kernel region to be outlined
	 * @param iterspace Iteration Space size
	 * @param new_kernel_call function call to the newly created kernel
	 * @param func_num
	 */
	private static void setKernelConfParameters(SymbolTable global_table, Statement region, Expression iterspace,
			KernelFunctionCall new_kernel_call, int func_num, Statement firstCudaStmt) {
		//CompoundStatement parentStmt = (CompoundStatement)region.getParent();
		CompoundStatement parentStmt = (CompoundStatement)firstCudaStmt.getParent();
		/* Insert "dim3 dimBlock(gpuNumThreads, 1, 1);" statement into the procedure body */
		VariableDeclaration numThreads_decl = (VariableDeclaration)SymbolTools.findSymbol(global_table,
				"gpuNumThreads");
		Identifier cloned_numThreads = new Identifier((VariableDeclarator)numThreads_decl.getDeclarator(0));
		Dim3Specifier dim3Spec = new Dim3Specifier(cloned_numThreads, new IntegerLiteral(1),
				new IntegerLiteral(1));
		VariableDeclarator dimBlock_declarator = new VariableDeclarator(new NameID("dimBlock"+func_num), dim3Spec);
		Identifier dimBlock = new Identifier(dimBlock_declarator);
		Declaration dimBlock_decl = new VariableDeclaration(CUDASpecifier.CUDA_DIM3, dimBlock_declarator);
		//parentStmt.addStatementBefore(region, new DeclarationStatement(dimBlock_decl));
		//parentStmt.addStatementBefore(firstCudaStmt, new DeclarationStatement(dimBlock_decl));
		TransformTools.addStatementBefore(parentStmt, firstCudaStmt, new DeclarationStatement(dimBlock_decl));
		if( region.getParent() == firstCudaStmt.getParent() ) {
			num_cudastmts--; // This counter does not include statements generated by this O2G translator
		}

		/*
		 * Insert "gpuNumBlocks = iterspace/BLOCK_SIZE;
		 *         if (gpuNumBlocks > MAX_GDIMENSION) {
    	 *            gpuNumBlocks2 = ceilf( ((float)gpuNumBlocks)/((float)MAX_NDIMENSION) );
    	 *            gpuNumBlocks1 = MAX_NDIMENSION;
         *          }
         *          else {
         *              gpuNumBlocks2 = 1;
         *              gpuNumBlocks1 = gpuNumBlocks;
         *          }
		 *          dim3 dimGrid(gpuNumBlocks1, gpuNumBlocks2, 1);
		 *          gpuNumBlocks = gpuNumBlocks1 * gpuNumBlocks2;"
		 * statement into the procedure body
		 */
		VariableDeclaration numBlocks_decl = (VariableDeclaration)SymbolTools.findSymbol(global_table,
				"gpuNumBlocks");
		Identifier cloned_numBlocks = new Identifier((VariableDeclarator)numBlocks_decl.getDeclarator(0));
		VariableDeclaration numBlocks_decl1 = (VariableDeclaration)SymbolTools.findSymbol(global_table,
				"gpuNumBlocks1");
		Identifier cloned_numBlocks1 = new Identifier((VariableDeclarator)numBlocks_decl1.getDeclarator(0));
		VariableDeclaration numBlocks_decl2 = (VariableDeclaration)SymbolTools.findSymbol(global_table,
				"gpuNumBlocks2");
		Identifier cloned_numBlocks2 = new Identifier((VariableDeclarator)numBlocks_decl2.getDeclarator(0));
		Expression blockSize;
		if( iterspace instanceof IntegerLiteral ) {
			int gsize = (int)Math.ceil( ((double)((IntegerLiteral)iterspace).getValue())/
					((double)defaultBlockSize) );
			blockSize = new IntegerLiteral(gsize);
		} else {
			List<Specifier> specs = new ArrayList<Specifier>(1);
			specs.add(Specifier.FLOAT);
			Expression floatsize = new Typecast(specs, iterspace);
			FunctionCall ceilfunc = new FunctionCall(new NameID("ceil"));
			ceilfunc.addArgument(Symbolic.divide(floatsize,
					new FloatLiteral((double)defaultBlockSize, "F")));
			blockSize = ceilfunc;
		}
		AssignmentExpression asExp = new AssignmentExpression(cloned_numBlocks, AssignmentOperator.NORMAL,
				blockSize);
		Statement expStmt = new ExpressionStatement(asExp);
		//parentStmt.addStatementBefore(region, expStmt);
		parentStmt.addStatementBefore(firstCudaStmt, expStmt);
		Expression biExp1 = new BinaryExpression((Expression)cloned_numBlocks.clone(), BinaryOperator.COMPARE_GT,
				new NameID("MAX_GDIMENSION"));
		CompoundStatement ifBody = new CompoundStatement();
		CompoundStatement elseBody = new CompoundStatement();
		Statement ifStmt = new IfStatement(biExp1, ifBody, elseBody);
		Expression blockSize1, blockSize2;
		if( blockSize instanceof IntegerLiteral ) {
			int gsize2 = (int)Math.ceil( ((double)((IntegerLiteral)blockSize).getValue())/
					((double)defaultGridDimSize) );
			blockSize2 = new IntegerLiteral(gsize2);
		} else {
			List<Specifier> specs = new ArrayList<Specifier>(1);
			specs.add(Specifier.FLOAT);
			Expression floatsize = new Typecast(specs, (Expression)cloned_numBlocks.clone());
			FunctionCall ceilfunc = new FunctionCall(new NameID("ceil"));
			ceilfunc.addArgument(Symbolic.divide(floatsize,
					new FloatLiteral((double)defaultGridDimSize, "F")));
			blockSize2 = ceilfunc;
		}
		asExp = new AssignmentExpression((Expression)cloned_numBlocks2.clone(), AssignmentOperator.NORMAL,
				blockSize2);
		ifBody.addStatement(new ExpressionStatement(asExp));
		asExp = new AssignmentExpression((Expression)cloned_numBlocks1.clone(), AssignmentOperator.NORMAL,
				new NameID("MAX_NDIMENSION"));
		ifBody.addStatement(new ExpressionStatement(asExp));
		asExp = new AssignmentExpression((Expression)cloned_numBlocks2.clone(), AssignmentOperator.NORMAL,
				new IntegerLiteral(1));
		elseBody.addStatement(new ExpressionStatement(asExp));
		asExp = new AssignmentExpression((Expression)cloned_numBlocks1.clone(), AssignmentOperator.NORMAL,
				(Expression)cloned_numBlocks.clone());
		elseBody.addStatement(new ExpressionStatement(asExp));
		parentStmt.addStatementBefore(firstCudaStmt, ifStmt);
		if( region.getParent() == firstCudaStmt.getParent() ) {
			num_cudastmts-=2; // This counter does not include statements generated by this O2G translator
		}
		dim3Spec = new Dim3Specifier((Identifier)cloned_numBlocks1.clone(), (Identifier)cloned_numBlocks2.clone(),
				new IntegerLiteral(1));
		VariableDeclarator dimGrid_declarator = new VariableDeclarator(new NameID("dimGrid"+func_num), dim3Spec);
		Identifier dimGrid = new Identifier(dimGrid_declarator);
		Declaration dimGrid_decl = new VariableDeclaration(CUDASpecifier.CUDA_DIM3, dimGrid_declarator);
		//parentStmt.addStatementBefore(region, new DeclarationStatement(dimGrid_decl));
		//parentStmt.addStatementBefore(firstCudaStmt, new DeclarationStatement(dimGrid_decl));
		TransformTools.addStatementBefore(parentStmt, firstCudaStmt, new DeclarationStatement(dimGrid_decl));
		if( region.getParent() == firstCudaStmt.getParent() ) {
			num_cudastmts--; // This counter does not include statements generated by this O2G translator
		}
		asExp = new AssignmentExpression((Expression)cloned_numBlocks.clone(), AssignmentOperator.NORMAL,
				new BinaryExpression((Expression)cloned_numBlocks1.clone(), BinaryOperator.MULTIPLY,
						(Expression)cloned_numBlocks2.clone()));
		parentStmt.addStatementBefore(firstCudaStmt, new ExpressionStatement(asExp));
		if( region.getParent() == firstCudaStmt.getParent() ) {
			num_cudastmts--; // This counter does not include statements generated by this O2G translator
		}
		List<Expression> kernelConf = new ArrayList<Expression>();
		//the dimension of the grid is the first argument, and then that of thread block comes.
		kernelConf.add((Identifier)dimGrid.clone());
		kernelConf.add((Identifier)dimBlock.clone());
		kernelConf.add(new IntegerLiteral(0));
		kernelConf.add(new IntegerLiteral(0));
		new_kernel_call.setConfArguments(kernelConf);
		/*
		 * Insert "totalNumThreads = gpuNumThreads * gpuNumBlocks;"
		 * statement into the procedure body
		 */
		VariableDeclaration totalNumThreads_decl = (VariableDeclaration)SymbolTools.findSymbol(global_table,
				"totalNumThreads");
		Identifier totalNumThreads = new Identifier((VariableDeclarator)totalNumThreads_decl.getDeclarator(0));
		asExp = new AssignmentExpression(totalNumThreads, AssignmentOperator.NORMAL,
				new BinaryExpression((Expression)cloned_numBlocks.clone(), BinaryOperator.MULTIPLY,
						(Expression)cloned_numThreads.clone()));
		parentStmt.addStatementBefore(firstCudaStmt, new ExpressionStatement(asExp));
		if( region.getParent() == firstCudaStmt.getParent() ) {
			num_cudastmts--; // This counter does not include statements generated by this O2G translator
		}

		//////////////////////////////////////////////////////////////////////////////////
		//If extractTuningParameters option is on, insert iteration space infomation to //
		//the tuning-parameter file as a comment.                                       //
		//////////////////////////////////////////////////////////////////////////////////
		if( tuningParamFile != null ) {
			CudaAnnotation cAnnot = region.getAnnotation(CudaAnnotation.class, "ainfo");
			String kernelID;
			if( cAnnot != null ) {
				kernelID = cAnnot.toString();
			} else {
				kernelID = new_kernel_call.getName().toString();
			}
			try {
				BufferedWriter out = new BufferedWriter(new FileWriter(tuningParamFile, true));
				out.write(kernelID + " iterationspace=" + iterspace);
				out.newLine();
				out.close();
			} catch (Exception e) {
				PrintTools.println("[ERROR setKernelConfParameters()] writing to a file, "+ tuningParamFile +
						", failed.", 0);
			}
		}
	}

	/**
	 * Convert the access of a scalar shared variable into a pointer access expression.
	 * @param targetSym symbol of a target OpenMP shared variable
	 * @param new_proc a new function where the target symbol will be accessed
	 * @param region original code region, which will be transformed into the new function, new_proc
	 * @return
	 */
	private static VariableDeclarator scalarVariableConv( VariableDeclarator targetSym, Procedure new_proc,
			Statement region, boolean useRegister, boolean ROData) {
		// Create a parameter Declaration for the kernel function
		// Change the scalar variable to a pointer type
		Identifier cloned_ID = new Identifier(targetSym);
		VariableDeclarator pointerV_declarator = new VariableDeclarator(PointerSpecifier.UNQUALIFIED,
				new NameID(targetSym.getSymbolName()));
		List<Specifier> clonedspecs = new ChainedList<Specifier>();
		/////////////////////////////////////////////////////////////////////////////////////
		// CAUTION: VariableDeclarator.getTypeSpecifiers() returns both specifiers of      //
		// its parent VariableDeclaration and the VariableDeclarator's leading specifiers. //
		// Therefore, if VariableDeclarator is a pointer symbol, this method will return   //
		// pointer specifiers too.                                                         //
		/////////////////////////////////////////////////////////////////////////////////////
		clonedspecs.addAll(targetSym.getTypeSpecifiers());
		clonedspecs.remove(Specifier.STATIC);
		VariableDeclaration pointerV_decl = new VariableDeclaration(clonedspecs,
				pointerV_declarator);
		Identifier pointer_var = new Identifier(pointerV_declarator);
		new_proc.addDeclaration(pointerV_decl);

		CompoundStatement targetStmt = null;
		if( region instanceof CompoundStatement ) {
			targetStmt = (CompoundStatement)region;
		} else if( region instanceof ForLoop ) {
			targetStmt = (CompoundStatement)((ForLoop)region).getBody();
		} else {
			Tools.exit(pass_name + "[ERROR] Unknwon region in extractKernelRegion(): "
					+ region.toString());
		}
		if( useRegister ) {
			// Insert a statement to load the global variable to register at the beginning
			//and a statement to dump register value to the global variable at the end
			// SymbolTools.getTemp() inserts the new temp symbol to the symbol table of the closest parent
			// if region is a loop.
			Identifier local_var = SymbolTools.getTemp(targetStmt, (Identifier)cloned_ID);
			Statement estmt = new ExpressionStatement(new AssignmentExpression(local_var,
					AssignmentOperator.NORMAL,
					new UnaryExpression(UnaryOperator.DEREFERENCE, (Identifier)pointer_var.clone())));
			Statement astmt = new ExpressionStatement(new AssignmentExpression(
					new UnaryExpression(UnaryOperator.DEREFERENCE, (Identifier)pointer_var.clone()),
					AssignmentOperator.NORMAL,(Identifier)local_var.clone()));
			// Replace all instances of the shared variable to the local variable
			//IRTools.replaceAll((Traversable) targetStmt, cloned_ID, local_var);
			IRTools.replaceAll((Traversable) region, cloned_ID, local_var);
			/////////////////////////////////////////////////////////////////////////////////////////
			// If the address of the local variable is passed as an argument of a function called  //
			// in the parallel region, revert the instance of the local variable back to the       //
			// pointer variable; in CUDA, dereferencing of device variables is not allowed.        //
			/////////////////////////////////////////////////////////////////////////////////////////
			List<FunctionCall> funcCalls = IRTools.getFunctionCalls(region);
			for( FunctionCall calledProc : funcCalls ) {
				List<Expression> argList = (List<Expression>)calledProc.getArguments();
				List<Expression> newList = new LinkedList<Expression>();
				boolean foundArg = false;
				for( Expression arg : argList ) {
					if(arg instanceof UnaryExpression) {
						UnaryExpression uarg = (UnaryExpression)arg;
						if( uarg.getOperator().equals(UnaryOperator.ADDRESS_OF)
								&& uarg.getExpression().equals(local_var) ) {
							newList.add((Expression)pointer_var.clone());
							foundArg = true;
						} else {
							newList.add(arg);
						}
					} else {
						newList.add(arg);
					}
				}
				calledProc.setArguments(newList);
				if( !ROData ) {
					if( foundArg ) {
						targetStmt.addStatementBefore(calledProc.getStatement(),
								(Statement)astmt.clone());
						targetStmt.addStatementAfter(calledProc.getStatement(),
								(Statement)estmt.clone());
					} else {
						///////////////////////////////////////////////////////////////////////////
						// If the address of the shared variable is not passed as an argument    //
						// of a function called in the kernel region, but accessed in the called //
						// function, load&store statements should be inserted before&after the   //
						// function call site.                                                   //
						///////////////////////////////////////////////////////////////////////////
						Procedure proc = calledProc.getProcedure();
						if( proc != null ) {
							Statement body = proc.getBody();
							if( IRTools.containsSymbol(body, targetSym) ) {
								targetStmt.addStatementBefore(calledProc.getStatement(),
										(Statement)astmt.clone());
								targetStmt.addStatementAfter(calledProc.getStatement(),
										(Statement)estmt.clone());
							}
						}
					}
				}
			}

			if( region instanceof ForLoop ) {
				cacheLoadingStmts.add(estmt);
			} else {
				Statement last_decl_stmt;
				last_decl_stmt = IRTools.getLastDeclarationStatement(targetStmt);
				if( last_decl_stmt != null ) {
					targetStmt.addStatementAfter(last_decl_stmt,(Statement)estmt);
				} else {
					last_decl_stmt = (Statement)targetStmt.getChildren().get(0);
					targetStmt.addStatementBefore(last_decl_stmt,(Statement)estmt);
				}
			}
			if( !ROData ) {
				targetStmt.addStatement(astmt);
			}
		} else {
			Expression deref_expr = new UnaryExpression(UnaryOperator.DEREFERENCE,
					(Identifier)pointer_var.clone());
			// Replace all instances of the shared variable to a pointer-dereferencing expression (ex: *x).
			//IRTools.replaceAll((Traversable) targetStmt, cloned_ID, deref_expr);
			IRTools.replaceAll((Traversable) region, cloned_ID, deref_expr);
		}

		return pointerV_declarator;
	}

	/**
	 * Convert the access of a scalar threadprivate variable into an array access using array extension.
	 * @param targetSym symbol of a target threadprivate variable
	 * @param new_proc a new function where the target symbol will be accessed
	 * @param region original code region, which will be transformed into the new function, new_proc
	 * @return
	 */
	private static VariableDeclarator scalarVariableConv2(VariableDeclarator targetSym, Procedure new_proc,
			Statement region, boolean useRegister, boolean ROData) {
		// Create a parameter Declaration for the kernel function
		// Change the scalar variable to a pointer type
		Identifier cloned_ID = new Identifier(targetSym);
		VariableDeclarator pointerV_declarator = new VariableDeclarator(PointerSpecifier.UNQUALIFIED,
				new NameID(targetSym.getSymbolName()));
		List<Specifier> clonedspecs = new ChainedList<Specifier>();
		/////////////////////////////////////////////////////////////////////////////////////
		// CAUTION: VariableDeclarator.getTypeSpecifiers() returns both specifiers of      //
		// its parent VariableDeclaration and the VariableDeclarator's leading specifiers. //
		// Therefore, if VariableDeclarator is a pointer symbol, this method will return   //
		// pointer specifiers too.                                                         //
		/////////////////////////////////////////////////////////////////////////////////////
		clonedspecs.addAll(targetSym.getTypeSpecifiers());
		clonedspecs.remove(Specifier.STATIC);
		VariableDeclaration pointerV_decl = new VariableDeclaration(clonedspecs,
				pointerV_declarator);
		Identifier pointer_var = new Identifier(pointerV_declarator);
		new_proc.addDeclaration(pointerV_decl);

		CompoundStatement targetStmt = null;
		if( region instanceof CompoundStatement ) {
			targetStmt = (CompoundStatement)region;
			//targetStmt.addDeclaration(cloned_decl);
		} else if( region instanceof ForLoop ) {
			targetStmt = (CompoundStatement)((ForLoop)region).getBody();
			//targetStmt.addDeclaration(cloned_decl);
		} else {
			Tools.exit(pass_name + "[ERROR] Unknwon region in extractKernelRegion(): "
					+ region.toString());
		}
		if( useRegister ) {
			/////////////////////////////////////////////////////////////////////////////////
			// Insert a statement to load the global variable to register at the beginning //
			// and a statement to dump register value to the global variable at the end    //
			// SymbolTools.getTemp() inserts the new temp symbol to the symbol table of the      //
			// closest parent if region is a loop.                                         //
			/////////////////////////////////////////////////////////////////////////////////
			Identifier local_var = SymbolTools.getTemp(targetStmt, (Identifier)cloned_ID);
			// Identifier "_gtid" should be updated later so that it can point to a corresponding symbol.
			Statement estmt = new ExpressionStatement(new AssignmentExpression((Identifier)local_var.clone(),
					AssignmentOperator.NORMAL, new ArrayAccess((Identifier)pointer_var.clone(),
							SymbolTools.getOrphanID("_gtid"))));
			Statement astmt = new ExpressionStatement(new AssignmentExpression(
					new ArrayAccess((Identifier)pointer_var.clone(), SymbolTools.getOrphanID("_gtid")),
					AssignmentOperator.NORMAL,(Identifier)local_var.clone()));
			// Replace all instances of the shared variable to the local variable
			//IRTools.replaceAll((Traversable) targetStmt, cloned_ID, local_var);
			IRTools.replaceAll((Traversable) region, cloned_ID, local_var);
			/////////////////////////////////////////////////////////////////////////////////
			// If the address of the local variable is passed as an argument of a function //
			// called in the parallel region, revert the instance of the local variable    //
			// back to a pointer expression; in CUDA, dereferencing of device variables is //
			// not allowed.                                                                //
			/////////////////////////////////////////////////////////////////////////////////
			List<FunctionCall> funcCalls = IRTools.getFunctionCalls(region);
			for( FunctionCall calledProc : funcCalls ) {
				List<Expression> argList = (List<Expression>)calledProc.getArguments();
				List<Expression> newList = new LinkedList<Expression>();
				boolean foundArg = false;
				for( Expression arg : argList ) {
					if(arg instanceof UnaryExpression) {
						UnaryExpression uarg = (UnaryExpression)arg;
						if( uarg.getOperator().equals(UnaryOperator.ADDRESS_OF)
								&& uarg.getExpression().equals(local_var) ) {
							newList.add((Identifier)pointer_var.clone());
							foundArg = true;
						} else {
							newList.add(arg);
						}
					} else {
						newList.add(arg);
					}
				}
				calledProc.setArguments(newList);
				if( !ROData ) {
					if( foundArg ) {
						targetStmt.addStatementBefore(calledProc.getStatement(),
								(Statement)astmt.clone());
						targetStmt.addStatementAfter(calledProc.getStatement(),
								(Statement)estmt.clone());
					} else {
						///////////////////////////////////////////////////////////////////////////
						// If the address of the shared variable is not passed as an argument    //
						// of a function called in the kernel region, but accessed in the called //
						// function, load&store statements should be inserted before&after the   //
						// function call site.                                                   //
						///////////////////////////////////////////////////////////////////////////
						Procedure proc = calledProc.getProcedure();
						if( proc != null ) {
							Statement body = proc.getBody();
							if( IRTools.containsSymbol(body, targetSym) ) {
								targetStmt.addStatementBefore(calledProc.getStatement(),
										(Statement)astmt.clone());
								targetStmt.addStatementAfter(calledProc.getStatement(),
										(Statement)estmt.clone());
							}
						}
					}
				}
			}
			tempMap.put((Identifier)pointer_var.clone(),
					new BinaryExpression((Expression)pointer_var.clone(),
							BinaryOperator.ADD, SymbolTools.getOrphanID("_gtid")));

			Statement last_decl_stmt;
			last_decl_stmt = IRTools.getLastDeclarationStatement(targetStmt);
			if( last_decl_stmt != null ) {
				targetStmt.addStatementAfter(last_decl_stmt,(Statement)estmt);
			} else {
				last_decl_stmt = (Statement)targetStmt.getChildren().get(0);
				targetStmt.addStatementBefore(last_decl_stmt,(Statement)estmt);
			}
			if( !ROData ) {
				targetStmt.addStatement(astmt);
			}
		} else {
			Expression access_expr =  new ArrayAccess((Identifier)pointer_var.clone(), SymbolTools.getOrphanID("_gtid"));
			// Replace all instances of the threadprivate variable to an array  expression (ex: x[_gtid]).
			//IRTools.replaceAll((Traversable) targetStmt, cloned_ID, access_expr);
			IRTools.replaceAll((Traversable) region, cloned_ID, access_expr);
		}

		return pointerV_declarator;
	}

	/**
	 * Convert 2D array access expression (aAccess) to a pointer access expression using pitch
	 * Example: x[i][j] is converted into "*(((float *)((char *)x + i*pitch_x)) + j)".
	 * @param aAccess : 2D array
	 * @param pitch : pitch used in cudaMallocPitch() for the array aAccess
	 * @return : pointer access expression using pitch
	 */
	private static Expression convArray2Pointer( ArrayAccess aAccess, Identifier pitch ) {
		List<Specifier> specs = new ArrayList<Specifier>(2);
		specs.add(Specifier.CHAR);
		specs.add(PointerSpecifier.UNQUALIFIED);
		Typecast tcast1 = new Typecast(specs, (Expression)aAccess.getArrayName().clone());
		BinaryExpression biexp1 = new BinaryExpression((Expression)aAccess.getIndex(0).clone(),
				BinaryOperator.MULTIPLY, (Identifier)pitch.clone());
		BinaryExpression biexp2 = new BinaryExpression(tcast1, BinaryOperator.ADD, biexp1);
		List<Specifier> specs2 = new ArrayList<Specifier>();
		/////////////////////////////////////////////////////////////////////////////////////
		// CAUTION: VariableDeclarator.getTypeSpecifiers() returns both specifiers of      //
		// its parent VariableDeclaration and the VariableDeclarator's leading specifiers. //
		// Therefore, if VariableDeclarator is a pointer symbol, this method will return   //
		// pointer specifiers too.                                                         //
		/////////////////////////////////////////////////////////////////////////////////////
		specs2.addAll(((Identifier)aAccess.getArrayName()).getSymbol().getTypeSpecifiers());
		specs2.remove(Specifier.STATIC);
		specs2.add(PointerSpecifier.UNQUALIFIED);
		Typecast tcast2 = new Typecast(specs2, biexp2);
		BinaryExpression biexp3 = new BinaryExpression(tcast2, BinaryOperator.ADD,
				(Expression)aAccess.getIndex(1).clone());
		UnaryExpression uexp = new UnaryExpression(UnaryOperator.DEREFERENCE, biexp3);
		return uexp;
	}

	/**
	 * Convert 1D array access expression (aAccess) to a pointer access expression using pitch.
	 * This conversion is used for MatrixTranspose optimization on Threadprivate data.
	 * Example: x[i] is converted into "*((float *)((char *)x + i*pitch_x))".
	 * @param aAccess : 1D array
	 * @param pitch : pitch used in cudaMallocPitch() for the array aAccess
	 * @return : pointer access expression using pitch
	 */
	private static Expression convArray2Pointer2( ArrayAccess aAccess, Identifier pitch ) {
		List<Specifier> specs = new ArrayList<Specifier>(2);
		specs.add(Specifier.CHAR);
		specs.add(PointerSpecifier.UNQUALIFIED);
		Typecast tcast1 = new Typecast(specs, (Expression)aAccess.getArrayName().clone());
		BinaryExpression biexp1 = new BinaryExpression((Expression)aAccess.getIndex(0).clone(),
				BinaryOperator.MULTIPLY, (Identifier)pitch.clone());
		BinaryExpression biexp2 = new BinaryExpression(tcast1, BinaryOperator.ADD, biexp1);
		List<Specifier> specs2 = new ArrayList<Specifier>();
		/////////////////////////////////////////////////////////////////////////////////////
		// CAUTION: VariableDeclarator.getTypeSpecifiers() returns both specifiers of      //
		// its parent VariableDeclaration and the VariableDeclarator's leading specifiers. //
		// Therefore, if VariableDeclarator is a pointer symbol, this method will return   //
		// pointer specifiers too.                                                         //
		/////////////////////////////////////////////////////////////////////////////////////
		specs2.addAll(((Identifier)aAccess.getArrayName()).getSymbol().getTypeSpecifiers());
		specs2.remove(Specifier.STATIC);
		specs2.add(PointerSpecifier.UNQUALIFIED);
		Typecast tcast2 = new Typecast(specs2, biexp2);
		UnaryExpression uexp = new UnaryExpression(UnaryOperator.DEREFERENCE, tcast2);
		return uexp;
	}

	/**
	 * If an shared array element (ex: A[i]) is included in the cuda registerRO or
	 * cuda registerRW set, the element is cached in the GPU register.
	 *
	 * @param region
	 * @param cudaRegisterSet
	 * @param cudaRegisterROSet
	 * @param arraySymbol
	 */
	private static void arrayCachingOnRegister(Statement region, HashSet<String> cudaRegisterSet,
			HashSet<String> cudaRegisterROSet, VariableDeclarator arraySymbol) {
		Identifier array_ID = new Identifier(arraySymbol);
		CompoundStatement targetStmt = null;
		if( region instanceof CompoundStatement ) {
			targetStmt = (CompoundStatement)region;
		} else if( region instanceof ForLoop ) {
			targetStmt = (CompoundStatement)((ForLoop)region).getBody();
		} else {
			Tools.exit(pass_name + "[ERROR] Unknwon region in arrayCachingOnRegister(): "
					+ region.toString());
		}
		/////////////////////////////////////////////////////////////////////////
		// Find array access expressions that will be cached on the registers. //
		/////////////////////////////////////////////////////////////////////////
		HashMap<String, ArrayAccess> aAccessMap = new HashMap<String, ArrayAccess>();
		DepthFirstIterator iter = new DepthFirstIterator(targetStmt);
		for (;;)
		{
			ArrayAccess aAccess = null;

			try {
				aAccess = (ArrayAccess)iter.next(ArrayAccess.class);
			} catch (NoSuchElementException e) {
				break;
			}
			Identifier aID = (Identifier)aAccess.getArrayName();
			String aAccessString = aAccess.toString();
			////////////////////////////////////
			// Remove any '(', ')', or space. //
			////////////////////////////////////
			StringBuilder strB = new StringBuilder(aAccessString);
			int index = strB.toString().indexOf('(');
			while ( index != -1 ) {
				strB = strB.deleteCharAt(index);
				index = strB.toString().indexOf('(');
			}
			index = strB.toString().indexOf(')');
			while ( index != -1 ) {
				strB = strB.deleteCharAt(index);
				index = strB.toString().indexOf(')');
			}
			index = strB.toString().indexOf(' ');
			while ( index != -1 ) {
				strB = strB.deleteCharAt(index);
				index = strB.toString().indexOf(' ');
			}
			String aAccessString2 = strB.toString();

			if( array_ID.equals(aID) && (cudaRegisterSet.contains(aAccessString)
					|| cudaRegisterSet.contains(aAccessString2)) ) {
				aAccessMap.put(aAccessString, aAccess);
			}
		}
		for( ArrayAccess aAccess : aAccessMap.values() ) {
			String aAccessString = aAccess.toString();
			////////////////////////////////////
			// Remove any '(', ')', or space. //
			////////////////////////////////////
			StringBuilder strB = new StringBuilder(aAccessString);
			int index = strB.toString().indexOf('(');
			while ( index != -1 ) {
				strB = strB.deleteCharAt(index);
				index = strB.toString().indexOf('(');
			}
			index = strB.toString().indexOf(')');
			while ( index != -1 ) {
				strB = strB.deleteCharAt(index);
				index = strB.toString().indexOf(')');
			}
			index = strB.toString().indexOf(' ');
			while ( index != -1 ) {
				strB = strB.deleteCharAt(index);
				index = strB.toString().indexOf(' ');
			}
			String aAccessString2 = strB.toString();

			// Insert a statement to load the global variable to register at the beginning
			//and a statement to dump register value to the global variable at the end
			// SymbolTools.getTemp() inserts the new temp symbol to the symbol table of the closest parent
			// if region is a loop.
			Identifier local_var = SymbolTools.getTemp(targetStmt, array_ID);
			Statement estmt = new ExpressionStatement(new AssignmentExpression(local_var,
					AssignmentOperator.NORMAL, (ArrayAccess)aAccess.clone() ));
			Statement astmt = new ExpressionStatement(new AssignmentExpression(
					(ArrayAccess)aAccess.clone(), AssignmentOperator.NORMAL,
					(Identifier)local_var.clone()));
			///////////////////////////////////////////////////
			// Find the first instance of this array access. //
			///////////////////////////////////////////////////
			ArrayAccess firstAccess = null;
			ArrayAccess lastAccess = null;
			Statement firstAccessStmt = null;
			Statement lastAccessStmt = null;
			boolean foundFirstArrayAccess = false;
			iter = new DepthFirstIterator(targetStmt);
			for (;;)
			{
				ArrayAccess tAccess = null;

				try {
					tAccess = (ArrayAccess)iter.next(ArrayAccess.class);
				} catch (NoSuchElementException e) {
					break;
				}
				if( aAccess.equals(tAccess) ) {
					if( !foundFirstArrayAccess ) {
						firstAccess = tAccess;
						foundFirstArrayAccess = true;
					}
					lastAccess = tAccess;
				}
			}
			Traversable t = (Traversable)firstAccess;
			while( !(t instanceof Statement) ) {
				t = t.getParent();
			}
			if( t instanceof Statement ) {
				firstAccessStmt = (Statement)t;
			}
			t = (Traversable)lastAccess;
			while( !(t instanceof Statement) ) {
				t = t.getParent();
			}
			if( t instanceof Statement ) {
				lastAccessStmt = (Statement)t;
			}

			// Replace all instances of the shared variable to the local variable
			//IRTools.replaceAll((Traversable) targetStmt, cloned_ID, local_var);
			IRTools.replaceAll((Traversable) region, aAccess, local_var);
			/////////////////////////////////////////////////////////////////////////////////////////
			// If the address of the local variable is passed as an argument of a function called  //
			// in the parallel region, revert the instance of the local variable back to the       //
			// array variable; in CUDA, dereferencing of device variables is not allowed.        //
			/////////////////////////////////////////////////////////////////////////////////////////
			List<FunctionCall> funcCalls = IRTools.getFunctionCalls(region);
			for( FunctionCall calledProc : funcCalls ) {
				List<Expression> argList = (List<Expression>)calledProc.getArguments();
				boolean foundArg = false;
				for( Expression arg : argList ) {
					if(IRTools.containsSymbol(arg, arraySymbol) ) {
						foundArg = true;
					}
				}

				if( !cudaRegisterROSet.contains(aAccessString) &&
						!cudaRegisterROSet.contains(aAccessString2) ) {
					if( foundArg ) {
						targetStmt.addStatementBefore(calledProc.getStatement(),
								(Statement)astmt.clone());
						targetStmt.addStatementAfter(calledProc.getStatement(),
								(Statement)estmt.clone());
					} else {
						///////////////////////////////////////////////////////////////////////////
						// If the address of the shared variable is not passed as an argument    //
						// of a function called in the kernel region, but accessed in the called //
						// function, load&store statements should be inserted before&after the   //
						// function call site.                                                   //
						///////////////////////////////////////////////////////////////////////////
						Procedure proc = calledProc.getProcedure();
						if( proc != null ) {
							Statement body = proc.getBody();
							if( IRTools.containsSymbol(body, arraySymbol) ) {
								targetStmt.addStatementBefore(calledProc.getStatement(),
										(Statement)astmt.clone());
								targetStmt.addStatementAfter(calledProc.getStatement(),
										(Statement)estmt.clone());
							}
						}
					}
				}
			}
			if( firstAccessStmt != null ) {
				Traversable p = firstAccessStmt.getParent();
				while( !(p instanceof CompoundStatement) ) {
					p = p.getParent();
				}
				((CompoundStatement)p).addStatementBefore(
						firstAccessStmt, estmt);
			} else {
				Statement last_decl_stmt;
				last_decl_stmt = IRTools.getLastDeclarationStatement(targetStmt);
				if( last_decl_stmt != null ) {
					targetStmt.addStatementAfter(last_decl_stmt,(Statement)estmt);
				} else {
					last_decl_stmt = (Statement)targetStmt.getChildren().get(0);
					targetStmt.addStatementBefore(last_decl_stmt,(Statement)estmt);
				}
			}
			if( !cudaRegisterROSet.contains(aAccessString) &&
					!cudaRegisterROSet.contains(aAccessString2) ) {
				//Below may result in wrong result if control flows are involved.
/*				if( lastAccessStmt != null ) {
					((CompoundStatement)lastAccessStmt.getParent()).addStatementAfter(
							lastAccessStmt, astmt);
				} else {
					targetStmt.addStatement(astmt);
				}*/
				//////////////////////////////
				// Find enclosing For loop. //
				//////////////////////////////
				t = firstAccessStmt.getParent();
				Traversable parent_t = t.getParent();
				while( !(parent_t instanceof ForLoop) ) {
					t = parent_t;
					parent_t = t.getParent();
				}
				if( parent_t instanceof ForLoop ) {
					((CompoundStatement)t).addStatement(astmt);
				} else {
					targetStmt.addStatement(astmt);
				}
			}
		}
	}

	/**
	 * If cuda sharedRO() or sharedRW() set contains a private array, it is
	 * allocated on the shared memory using array expansion.
	 *
	 * @param region
	 * @param cudaSharedSet
	 * @param OmpPrivSet
	 */
	private static void privateVariableCachingOnSM(Statement region, HashSet<String> cudaSharedSet,
			HashSet<Symbol> OmpPrivSet) {
		CompoundStatement targetStmt = null;
		if( region instanceof CompoundStatement ) {
			targetStmt = (CompoundStatement)region;
		} else if( region instanceof ForLoop ) {
			targetStmt = (CompoundStatement)((ForLoop)region).getBody();
		} else {
			Tools.exit(pass_name + "[ERROR] Unknwon region in pVarCachingOnSM(): "
					+ region.toString());
		}

		///////////////////////////////////////////////////////////////////////////
		// Array expansion on SM for address-taken, private scalar variable is   //
		// mandatory; add the address-taken private scalar variables to the      //
		// cudaSharedSet.                                                        //
		// DEBUG: We don't need the above change; CUDA manual V1.1 says that     //
		// the address obtained by taking the address of a __device__,           //
		// __shared__, or __constant__ variable can only be used in device code. //
		// Below code is commented out.                                          //
		///////////////////////////////////////////////////////////////////////////
		/*
		List<UnaryExpression> uExpList = IRTools.getUnaryExpression(targetStmt,
				UnaryOperator.ADDRESS_OF);
		if( uExpList.size() > 0 ) {
			for( Symbol privSym : OmpPrivSet ) {
				if( SymbolTools.isScalar(privSym) ) {
					for( UnaryExpression uExp: uExpList ) {
						Expression exp = uExp.getExpression();
						if( IRTools.containsSymbol(exp, privSym) ) {
							cudaSharedSet.add(privSym.getSymbolName());
							PrintTools.println("[INFO] Address-taken private variable, " + exp +
									", will be expanded on the GPU shared memory", 1);
							continue;
						}
					}
				}
			}
		}
		 */

		for( Symbol privSym : OmpPrivSet ) {
			String symbolName = privSym.getSymbolName();
			if( cudaSharedSet.contains(symbolName) ) {
				if( SymbolTools.isScalar(privSym) ) {
					// Create an extended array type
					// Ex: "__shared__ float b[BLOCK_SIZE]"
					List edimensions = new LinkedList();
					edimensions.add(new NameID("BLOCK_SIZE"));
					ArraySpecifier easpec = new ArraySpecifier(edimensions);
					StringBuilder str = new StringBuilder(80);
					str.append("sh__");
					str.append(symbolName);
					VariableDeclarator arrayV_declarator = new VariableDeclarator(new NameID(str.toString()), easpec);
					List<Specifier> clonedspecs = new ChainedList<Specifier>();
					/////////////////////////////////////////////////////////////////////////////////////
					// CAUTION: VariableDeclarator.getTypeSpecifiers() returns both specifiers of      //
					// its parent VariableDeclaration and the VariableDeclarator's leading specifiers. //
					// Therefore, if VariableDeclarator is a pointer symbol, this method will return   //
					// pointer specifiers too.                                                         //
					/////////////////////////////////////////////////////////////////////////////////////
					clonedspecs.add(CUDASpecifier.CUDA_SHARED);
					clonedspecs.addAll(privSym.getTypeSpecifiers());
					clonedspecs.remove(Specifier.STATIC);
					VariableDeclaration arrayV_decl =
						new VariableDeclaration(clonedspecs, arrayV_declarator);
					Identifier array_var = new Identifier(arrayV_declarator);
					targetStmt.addDeclaration(arrayV_decl);
					/*
					 * Replace array access expression with extended access expression.
					 */
					Identifier cloned_ID = new Identifier((VariableDeclarator)privSym);

					// Remove the declaration of the private symbol.
					Traversable t = ((VariableDeclarator)privSym).getParent();
					while( !(t instanceof DeclarationStatement) ) {
						t = t.getParent();
					}
					if( t instanceof DeclarationStatement ) {
						try {
							targetStmt.removeChild(t);
						} catch(Exception e) {
							PrintTools.println("[WARNING in privateVariableCachingOnSM()] DeclarationStatement of the private symbol, "
									+ symbolName + ", does not exist in the enclosing CompoundStatement!", 1);
						}
					} else {
							PrintTools.println("[WARNING in privateVariableCachingOnSM()] Can not find the DeclarationStatement " +
									"of the private symbol, " + symbolName + "!", 1);

					}

					//replace b with b[tid]
					NameID tid = new NameID("threadIdx.x");
					List indices = new LinkedList();
					indices.add(tid);
					ArrayAccess extendedAccess = new ArrayAccess((IDExpression)array_var.clone(), indices);
					// Replace all instances of the shared variable to the parameter variable
					IRTools.replaceAll((Traversable) targetStmt, cloned_ID, extendedAccess);
				}
				else if( SymbolTools.isArray(privSym) ) {
					// Create an extended array type
					// Ex: "float b[SIZE1][SIZE2][BLOCK_SIZE]"
					List aspecs = privSym.getArraySpecifiers();
					ArraySpecifier aspec = (ArraySpecifier)aspecs.get(0);
					int dimsize = aspec.getNumDimensions();
					List edimensions = new LinkedList();
					for( int i=0; i<dimsize; i++ )
					{
						edimensions.add((Expression)aspec.getDimension(i).clone());
					}
					edimensions.add(new NameID("BLOCK_SIZE"));
					ArraySpecifier easpec = new ArraySpecifier(edimensions);
					StringBuilder str = new StringBuilder(80);
					str.append("sh__");
					str.append(symbolName);
					VariableDeclarator arrayV_declarator = new VariableDeclarator(new NameID(str.toString()), easpec);
					List<Specifier> clonedspecs = new ChainedList<Specifier>();
					/////////////////////////////////////////////////////////////////////////////////////
					// CAUTION: VariableDeclarator.getTypeSpecifiers() returns both specifiers of      //
					// its parent VariableDeclaration and the VariableDeclarator's leading specifiers. //
					// Therefore, if VariableDeclarator is a pointer symbol, this method will return   //
					// pointer specifiers too.                                                         //
					/////////////////////////////////////////////////////////////////////////////////////
					clonedspecs.add(CUDASpecifier.CUDA_SHARED);
					clonedspecs.addAll(privSym.getTypeSpecifiers());
					clonedspecs.remove(Specifier.STATIC);
					VariableDeclaration arrayV_decl =
						new VariableDeclaration(clonedspecs, arrayV_declarator);
					Identifier array_var = new Identifier(arrayV_declarator);
					targetStmt.addDeclaration(arrayV_decl);
					/*
					 * Replace array access expression with extended access expression.
					 */
					Identifier cloned_ID = new Identifier((VariableDeclarator)privSym);

					//replace b[k][m] with b[k][m][tid]
					NameID tid = new NameID("threadIdx.x");
					DepthFirstIterator iter = new DepthFirstIterator(region);
					for (;;)
					{
						ArrayAccess aAccess = null;

						try {
							aAccess = (ArrayAccess)iter.next(ArrayAccess.class);
						} catch (NoSuchElementException e) {
							break;
						}
						IDExpression arrayID = (IDExpression)aAccess.getArrayName();
						if( arrayID.equals(cloned_ID) ) {
							if( aAccess.getNumIndices() == dimsize ) {
								List indices = new LinkedList();
								///////////////////////
								// DEBUG: deprecated //
								///////////////////////
								//indices.addAll(aAccess.getIndices());
								for( Expression indx : aAccess.getIndices() ) {
									indices.add(indx.clone());
								}
								indices.add((NameID)tid.clone());
								ArrayAccess extendedAccess = new ArrayAccess((IDExpression)array_var.clone(), indices);
								aAccess.swapWith(extendedAccess);
							} else {
								Tools.exit(pass_name + "[ERROR in CreateExtendedArray()] Incorrect dimension of the array access :"
										+ aAccess.toString());
							}
						}
					}
					// Remove the declaration of the private symbol.
					Traversable t = ((VariableDeclarator)privSym).getParent();
					while( !(t instanceof DeclarationStatement) ) {
						t = t.getParent();
					}
					if( t instanceof DeclarationStatement ) {
						try {
							targetStmt.removeChild(t);
						} catch(Exception e) {
							PrintTools.println("[WARNING in pArrayCachingOnSM()] DeclarationStatement of the private symbol, "
									+ symbolName + ", does not exist in the enclosing CompoundStatement!", 2);
						}
					} else {
							PrintTools.println("[WARNING in pArrayCachingOnSM()] Can not find the DeclarationStatement " +
									"of the private symbol, " + symbolName + "!", 2);

					}
					// Replace all instances of the shared variable to the parameter variable
					IRTools.replaceAll((Traversable) targetStmt, cloned_ID, array_var);
				}
			}
		}
	}

	private static VariableDeclarator scalarReductionConv(BinaryOperator redOp,
			VariableDeclarator targetSym, Procedure new_proc, Statement region) {
		/////////////////////////////////////////////////////////////
		// Create a parameter Declaration for the kernel function. //
		// Change the scalar variable to a pointer type.           //
		/////////////////////////////////////////////////////////////
		Identifier cloned_ID = new Identifier(targetSym);
		StringBuilder str = new StringBuilder(80);
		str.append("red__");
		str.append(targetSym.getSymbolName());
		VariableDeclarator pointerV_declarator = new VariableDeclarator(
				PointerSpecifier.UNQUALIFIED, new NameID(str.toString()));
		List<Specifier> clonedspecs = new ChainedList<Specifier>();
		/////////////////////////////////////////////////////////////////////////////////////
		// CAUTION: VariableDeclarator.getTypeSpecifiers() returns both specifiers of      //
		// its parent VariableDeclaration and the VariableDeclarator's leading specifiers. //
		// Therefore, if VariableDeclarator is a pointer symbol, this method will return   //
		// pointer specifiers too.                                                         //
		/////////////////////////////////////////////////////////////////////////////////////
		clonedspecs.addAll(targetSym.getTypeSpecifiers());
		clonedspecs.remove(Specifier.STATIC);
		VariableDeclaration pointerV_decl =  new VariableDeclaration(
				clonedspecs, pointerV_declarator);
		Identifier pointer_var = new Identifier(pointerV_declarator);
		new_proc.addDeclaration(pointerV_decl);

		CompoundStatement targetRegion = null;
		CompoundStatement parallelRegion = null;
		if( region instanceof CompoundStatement ) {
			targetRegion = (CompoundStatement)region;
			parallelRegion = targetRegion;
		} else if( region instanceof ForLoop ) {
			targetRegion = (CompoundStatement)((ForLoop)region).getBody();
			parallelRegion = (CompoundStatement)region.getParent();
		} else {
			Tools.exit(pass_name + "[ERROR] Unknwon region in extractKernelRegion(): "
					+ region.toString());
		}
		/////////////////////////////////////////////////////////////////////////
		// Create an array on the GPU shared memory, which holds privatized    //
		// reduction variable.                                                 //
		// Assumption: GPU threadblock is an one-dimensional array of threads. //
		// Ex: __shared__ float sh__x[BLOCK_SIZE];                             //
		/////////////////////////////////////////////////////////////////////////
		NameID tid = new NameID("threadIdx.x");
		Identifier bid = SymbolTools.getOrphanID("_bid");
		ArraySpecifier aspec = new ArraySpecifier(new NameID("BLOCK_SIZE"));
		str = new StringBuilder(80);
		str.append("sh__");
		str.append(targetSym.getSymbolName());
		VariableDeclarator arrayV_declarator = new VariableDeclarator(new NameID(str.toString()), aspec);
		List specList = new LinkedList();
		specList.add(CUDASpecifier.CUDA_SHARED);
		specList.addAll(clonedspecs);
		VariableDeclaration arrayV_decl =
			new VariableDeclaration(specList, arrayV_declarator);
		Identifier array_var = new Identifier(arrayV_declarator);
		parallelRegion.addDeclaration(arrayV_decl);

		//////////////////////////////////////////////////////////////////////////
		// Replace all instances of the shared variable to an array expression. //
		// (ex: sh__x[threadIdx.x])                                             //
		//////////////////////////////////////////////////////////////////////////
		Expression access_expr =  new ArrayAccess((Identifier)array_var.clone(),
				(NameID)tid.clone());
		//IRTools.replaceAll((Traversable) targetRegion, cloned_ID, access_expr);
		IRTools.replaceAll((Traversable) region, cloned_ID, access_expr);

		///////////////////////////////////////////////////////
		// Add reduction variable initialization statements. //
		///////////////////////////////////////////////////////
		Expression initValue = TransformTools.getRInitValue(redOp, specList);
		Statement estmt = new ExpressionStatement(
				new AssignmentExpression((Expression)access_expr.clone(),
				AssignmentOperator.NORMAL, initValue));
		Statement last_decl_stmt;
		last_decl_stmt = IRTools.getLastDeclarationStatement(parallelRegion);
		if( last_decl_stmt != null ) {
			parallelRegion.addStatementAfter(last_decl_stmt,(Statement)estmt);
		} else {
			last_decl_stmt = (Statement)parallelRegion.getChildren().get(0);
			parallelRegion.addStatementBefore(last_decl_stmt,(Statement)estmt);
		}

		/////////////////////////////////////////////////////////////////////
		// Add in-block reduction codes at the end of the parallel region. //
		/////////////////////////////////////////////////////////////////////
		FunctionCall syncCall = new FunctionCall(new NameID("__syncthreads"));
		Statement syncCallStmt = new ExpressionStatement(syncCall);
		parallelRegion.addStatement(syncCallStmt);
		Expression condition = null;
		Expression assignex = null;
		Statement ifstmt = null;
		if( opt_UnrollingOnReduction ) {
			///////////////////////////////////////////////////
			// Version1:  reduction with loop unrolling code //
			// this version works only if BLOCK_SIZE = 2^m   //
			// and BLOCK_SIZE >= 64                          //
			///////////////////////////////////////////////////
			// Assume that BLOCK_SIZE = 512.
		    //if (tid < 256) {
		    //     sh__x[tid] += sh__x[tid + 256];
		    // }
		    // __syncthreads();
		    //if (tid < 128) {
		    //     sh__x[tid] += sh__x[tid + 128];
		    // }
		    // __syncthreads();
		    // if (tid < 64) {
		    //     sh__x[tid] += sh__x[tid + 64];
		    // }
		    // __syncthreads();
		    // if (tid < 32)
		    // {
		    //     sh__x[tid] += sh__x[tid + 32];
		    //     sh__x[tid] += sh__x[tid + 16];
		    //     sh__x[tid] += sh__x[tid + 8];
		    //     sh__x[tid] += sh__x[tid + 4];
		    //     sh__x[tid] += sh__x[tid + 2];
		    //     sh__x[tid] += sh__x[tid + 1];
		    // }
			///////////////////////////////////////////////////
			if( defaultBlockSize == 512 ) {
				condition = new BinaryExpression((NameID)tid.clone(),
						BinaryOperator.COMPARE_LT, new IntegerLiteral(256) );
				assignex = TransformTools.RedExpression((Expression)access_expr.clone(), redOp,
						new ArrayAccess((Identifier)array_var.clone(),
								new BinaryExpression((NameID)tid.clone(), BinaryOperator.ADD,
										new IntegerLiteral(256))));
				estmt = new ExpressionStatement(assignex);
				ifstmt = new IfStatement(condition, estmt);
				parallelRegion.addStatement(ifstmt);
				parallelRegion.addStatement((Statement)syncCallStmt.clone());
			}
			if( defaultBlockSize >= 256 ) {
				condition = new BinaryExpression((NameID)tid.clone(),
						BinaryOperator.COMPARE_LT, new IntegerLiteral(128) );
				assignex = TransformTools.RedExpression((Expression)access_expr.clone(), redOp,
						new ArrayAccess((Identifier)array_var.clone(),
								new BinaryExpression((NameID)tid.clone(), BinaryOperator.ADD,
										new IntegerLiteral(128))));
				estmt = new ExpressionStatement(assignex);
				ifstmt = new IfStatement(condition, estmt);
				parallelRegion.addStatement(ifstmt);
				parallelRegion.addStatement((Statement)syncCallStmt.clone());
			}
			if( defaultBlockSize >= 128 ) {
				condition = new BinaryExpression((NameID)tid.clone(),
						BinaryOperator.COMPARE_LT, new IntegerLiteral(64) );
				assignex = TransformTools.RedExpression((Expression)access_expr.clone(), redOp,
						new ArrayAccess((Identifier)array_var.clone(),
								new BinaryExpression((NameID)tid.clone(), BinaryOperator.ADD,
										new IntegerLiteral(64))));
				estmt = new ExpressionStatement(assignex);
				ifstmt = new IfStatement(condition, estmt);
				parallelRegion.addStatement(ifstmt);
				parallelRegion.addStatement((Statement)syncCallStmt.clone());
			}
			condition = new BinaryExpression((NameID)tid.clone(),
					BinaryOperator.COMPARE_LT, new IntegerLiteral(32) );
			CompoundStatement ifbody = new CompoundStatement();
			for( int s = 32; s > 0; s>>=1 ) {
				assignex = TransformTools.RedExpression((Expression)access_expr.clone(), redOp,
						new ArrayAccess((Identifier)array_var.clone(),
								new BinaryExpression((NameID)tid.clone(), BinaryOperator.ADD,
										new IntegerLiteral(s))));
				estmt = new ExpressionStatement(assignex);
				ifbody.addStatement(estmt);
			}
			ifstmt = new IfStatement(condition, ifbody);
			parallelRegion.addStatement(ifstmt);
		} else {
			///////////////////////////////////////////////////
			// Version2: Unoptimized reduction code          //
			///////////////////////////////////////////////////
			//     bsize = blockDim.x;
			//     tid = threadIdx.x;
			//     oldSize = bsize;
			//     for (s=(bsize>>1); s>0; s>>=1) {
			//         if(tid < s) {
			//             sh__x[tid] += sh__x[tid + s];
			//         }
			//         oddNum = oldSize & (0x01);
			//         if ( oddNum == 1 ) {
			//             if (tid == 0) {
			//                 sh__x[0] += sh__x[oldSize-1];
			//             }
			//         }
			//         oldSize = s;
			//         __syncthreads();
			//     }
			///////////////////////////////////////////////////
			Identifier index_var = TransformTools.getTempIndex(parallelRegion, 0);
			Identifier oldSize = TransformTools.getTempIndex(parallelRegion, 1);
			Identifier oddNum = TransformTools.getTempIndex(parallelRegion, 2);
			estmt = new ExpressionStatement(
					new AssignmentExpression((Expression)oldSize.clone(),
							AssignmentOperator.NORMAL, new NameID("BLOCK_SIZE")));
			parallelRegion.addStatement(estmt);
			assignex = new AssignmentExpression((Identifier)index_var.clone(),
					AssignmentOperator.NORMAL, new BinaryExpression(new NameID("BLOCK_SIZE"),
							BinaryOperator.SHIFT_RIGHT, new IntegerLiteral(1)));
			Statement loop_init = new ExpressionStatement(assignex);
			condition = new BinaryExpression((Identifier)index_var.clone(),
					BinaryOperator.COMPARE_GT, new IntegerLiteral(0));
			Expression step = new AssignmentExpression( (Identifier)index_var.clone(),
					AssignmentOperator.SHIFT_RIGHT, new IntegerLiteral(1));
			CompoundStatement loopbody = new CompoundStatement();
			ForLoop reductionLoop = new ForLoop(loop_init, condition, step, loopbody);
			condition = new BinaryExpression((NameID)tid.clone(), BinaryOperator.COMPARE_LT,
					(Identifier)index_var.clone());
			assignex = TransformTools.RedExpression((Expression)access_expr.clone(), redOp,
					new ArrayAccess((Identifier)array_var.clone(),
							new BinaryExpression((NameID)tid.clone(), BinaryOperator.ADD,
									(Identifier)index_var.clone())));
			estmt = new ExpressionStatement(assignex);
			loopbody.addStatement( new IfStatement(condition, estmt) );
			assignex = new AssignmentExpression( (Identifier)oddNum.clone(), AssignmentOperator.NORMAL,
					new BinaryExpression((Identifier)oldSize.clone(), BinaryOperator.BITWISE_AND,
							new IntegerLiteral(0x01)));
			loopbody.addStatement(new ExpressionStatement(assignex));
			condition = new BinaryExpression((NameID)tid.clone(), BinaryOperator.COMPARE_EQ,
					new IntegerLiteral(0));
			assignex = TransformTools.RedExpression( new ArrayAccess((Identifier)array_var.clone(),
					new IntegerLiteral(0)), redOp,
					new ArrayAccess((Identifier)array_var.clone(),
							new BinaryExpression( (Identifier)oldSize.clone(), BinaryOperator.SUBTRACT,
									new IntegerLiteral(1))));
			estmt = new ExpressionStatement(assignex);
			ifstmt = new IfStatement(condition, estmt);
			condition = new BinaryExpression( (Identifier)oddNum.clone(), BinaryOperator.COMPARE_EQ,
					new IntegerLiteral(1));
			loopbody.addStatement( new IfStatement(condition, ifstmt) );
			estmt = new ExpressionStatement(
					new AssignmentExpression((Expression)oldSize.clone(),
							AssignmentOperator.NORMAL, (Identifier)index_var.clone()));
			loopbody.addStatement( estmt );
			loopbody.addStatement((Statement)syncCallStmt.clone());
			parallelRegion.addStatement(reductionLoop);
		}
		////////////////////////////////////////////////////////////////////////////
		// Write the in-block reduction result back to the global reduction array //
		////////////////////////////////////////////////////////////////////////////
		//     bid = blockIdx.x + blockIdx.y * gridDim.x;
		//     tid = threadIdx.x;
		//     if( tid == 0 ) {
		//         x[bid] = sh__x[0];
		//     }
		////////////////////////////////////////////////////////////////////////////
		condition = new BinaryExpression((NameID)tid.clone(), BinaryOperator.COMPARE_EQ,
				new IntegerLiteral(0));
		assignex = new AssignmentExpression( new ArrayAccess((Identifier)pointer_var.clone(),
				(Identifier)bid.clone()), AssignmentOperator.NORMAL,
				new ArrayAccess((Identifier)array_var.clone(), new IntegerLiteral(0)));
		estmt = new ExpressionStatement(assignex);
		parallelRegion.addStatement( new IfStatement(condition, estmt) );

		return pointerV_declarator;
	}

	/**
	 * this method conducts the first part of scalarReductionConv() without generating
	 * the in-block reduction codes.
	 *
	 * @param redOp
	 * @param targetSym
	 * @param new_proc
	 * @param region
	 * @param redArgSet
	 * @return
	 */
	private static VariableDeclarator scalarReductionConv2(BinaryOperator redOp,
			VariableDeclarator targetSym, Procedure new_proc, Statement region,
			ArrayList<VariableDeclarator> redArgSet, ArrayList<VariableDeclarator> redParamSet) {
		/////////////////////////////////////////////////////////////
		// Create a parameter Declaration for the kernel function. //
		// Change the scalar variable to a pointer type.           //
		/////////////////////////////////////////////////////////////
		Identifier cloned_ID = new Identifier(targetSym);
		StringBuilder str = new StringBuilder(80);
		str.append("red__");
		str.append(targetSym.getSymbolName());
		VariableDeclarator pointerV_declarator = new VariableDeclarator(
				PointerSpecifier.UNQUALIFIED, new NameID(str.toString()));
		List<Specifier> clonedspecs = new ChainedList<Specifier>();
		/////////////////////////////////////////////////////////////////////////////////////
		// CAUTION: VariableDeclarator.getTypeSpecifiers() returns both specifiers of      //
		// its parent VariableDeclaration and the VariableDeclarator's leading specifiers. //
		// Therefore, if VariableDeclarator is a pointer symbol, this method will return   //
		// pointer specifiers too.                                                         //
		/////////////////////////////////////////////////////////////////////////////////////
		clonedspecs.addAll(targetSym.getTypeSpecifiers());
		clonedspecs.remove(Specifier.STATIC);
		VariableDeclaration pointerV_decl =  new VariableDeclaration(
				clonedspecs, pointerV_declarator);
		Identifier pointer_var = new Identifier(pointerV_declarator);
		new_proc.addDeclaration(pointerV_decl);

		redParamSet.add(pointerV_declarator);

		CompoundStatement targetRegion = null;
		CompoundStatement parallelRegion = null;
		if( region instanceof CompoundStatement ) {
			targetRegion = (CompoundStatement)region;
			parallelRegion = targetRegion;
		} else if( region instanceof ForLoop ) {
			targetRegion = (CompoundStatement)((ForLoop)region).getBody();
			parallelRegion = (CompoundStatement)region.getParent();
		} else {
			Tools.exit(pass_name + "[ERROR] Unknwon region in extractKernelRegion(): "
					+ region.toString());
		}
		/////////////////////////////////////////////////////////////////////////
		// Create an array on the GPU shared memory, which holds privatized    //
		// reduction variable.                                                 //
		// Assumption: GPU threadblock is an one-dimensional array of threads. //
		// Ex: __shared__ float sh__x[BLOCK_SIZE];                             //
		/////////////////////////////////////////////////////////////////////////
		NameID tid = new NameID("threadIdx.x");
		Identifier bid = SymbolTools.getOrphanID("_bid");
		ArraySpecifier aspec = new ArraySpecifier(new NameID("BLOCK_SIZE"));
		str = new StringBuilder(80);
		str.append("sh__");
		str.append(targetSym.getSymbolName());
		VariableDeclarator arrayV_declarator = new VariableDeclarator(new NameID(str.toString()), aspec);
		List specList = new LinkedList();
		specList.add(CUDASpecifier.CUDA_SHARED);
		specList.addAll(clonedspecs);
		VariableDeclaration arrayV_decl =
			new VariableDeclaration(specList, arrayV_declarator);
		Identifier array_var = new Identifier(arrayV_declarator);
		parallelRegion.addDeclaration(arrayV_decl);

		redArgSet.add(arrayV_declarator);

		//////////////////////////////////////////////////////////////////////////
		// Replace all instances of the shared variable to an array expression. //
		// (ex: sh__x[threadIdx.x])                                             //
		//////////////////////////////////////////////////////////////////////////
		Expression access_expr =  new ArrayAccess((Identifier)array_var.clone(),
				(NameID)tid.clone());
		//IRTools.replaceAll((Traversable) targetRegion, cloned_ID, access_expr);
		IRTools.replaceAll((Traversable) region, cloned_ID, access_expr);

		///////////////////////////////////////////////////////
		// Add reduction variable initialization statements. //
		///////////////////////////////////////////////////////
		Expression initValue = TransformTools.getRInitValue(redOp, specList);
		Statement estmt = new ExpressionStatement(
				new AssignmentExpression((Expression)access_expr.clone(),
				AssignmentOperator.NORMAL, initValue));
		Statement last_decl_stmt;
		last_decl_stmt = IRTools.getLastDeclarationStatement(parallelRegion);
		if( last_decl_stmt != null ) {
			parallelRegion.addStatementAfter(last_decl_stmt,(Statement)estmt);
		} else {
			last_decl_stmt = (Statement)parallelRegion.getChildren().get(0);
			parallelRegion.addStatementBefore(last_decl_stmt,(Statement)estmt);
		}

		return pointerV_declarator;
	}

	private static VariableDeclarator arrayReductionConv(BinaryOperator redOp,
			VariableDeclarator tarSym, Procedure new_proc, Statement region)  {
		Identifier tarSymID = new Identifier(tarSym);
		List aspecs = tarSym.getArraySpecifiers();
		ArraySpecifier aspec = (ArraySpecifier)aspecs.get(0);
		int dimsize = aspec.getNumDimensions();
		StringBuilder str = new StringBuilder(80);
		str.append("red__");
		str.append(tarSym.getSymbolName());

		////////////////////////////////////
		// Create an extended array type  //
		// Ex: "float b[][SIZE1][SIZE2] " //
		////////////////////////////////////
		List<Expression> edimensions = new LinkedList<Expression>();
		edimensions.add(null);
		for( int i=0; i<dimsize; i++ )
		{
			edimensions.add((Expression)aspec.getDimension(i).clone());
		}
		ArraySpecifier easpec = new ArraySpecifier(edimensions);
		VariableDeclarator redV_declarator = new VariableDeclarator(new NameID(str.toString()), easpec);
		List<Specifier> clonedspecs = new ChainedList<Specifier>();
		/////////////////////////////////////////////////////////////////////////////////////
		// CAUTION: VariableDeclarator.getTypeSpecifiers() returns both specifiers of      //
		// its parent VariableDeclaration and the VariableDeclarator's leading specifiers. //
		// Therefore, if VariableDeclarator is a pointer symbol, this method will return   //
		// pointer specifiers too.                                                         //
		/////////////////////////////////////////////////////////////////////////////////////
		clonedspecs.addAll(tarSym.getTypeSpecifiers());
		clonedspecs.remove(Specifier.STATIC);
		VariableDeclaration redV_decl =
			new VariableDeclaration(clonedspecs, redV_declarator);
		Identifier red_var = new Identifier(redV_declarator);
		// Insert function parameter.
		new_proc.addDeclaration(redV_decl);

		CompoundStatement targetRegion = null;
		CompoundStatement parallelRegion = null;
		if( region instanceof CompoundStatement ) {
			targetRegion = (CompoundStatement)region;
			parallelRegion = targetRegion;
		} else if( region instanceof ForLoop ) {
			targetRegion = (CompoundStatement)((ForLoop)region).getBody();
			parallelRegion = (CompoundStatement)region.getParent();
		} else {
			Tools.exit(pass_name + "[ERROR] Unknwon region in extractKernelRegion(): "
					+ region.toString());
		}
		/////////////////////////////////////////////////////////////////////////
		// Create an array on the GPU shared memory, which holds privatized    //
		// reduction variable.                                                 //
		// Assumption: GPU threadblock is an one-dimensional array of threads. //
		// Ex: __shared__ float sh__x[SIZE1][SIZE2][BLOCK_SIZE];               //
		/////////////////////////////////////////////////////////////////////////
		edimensions = new LinkedList<Expression>();
		for( int i=0; i<dimsize; i++ )
		{
			edimensions.add((Expression)aspec.getDimension(i).clone());
		}
		edimensions.add(new NameID("BLOCK_SIZE"));
		easpec = new ArraySpecifier(edimensions);
		str = new StringBuilder(80);
		str.append("sh__");
		str.append(tarSym.getSymbolName());
		VariableDeclarator arrayV_declarator = new VariableDeclarator(new NameID(str.toString()), easpec);
		List specList = new LinkedList();
		specList.add(CUDASpecifier.CUDA_SHARED);
		specList.addAll(clonedspecs);
		VariableDeclaration arrayV_decl =
			new VariableDeclaration(specList, arrayV_declarator);
		Identifier array_var = new Identifier(arrayV_declarator);
		parallelRegion.addDeclaration(arrayV_decl);

		/////////////////////////////////////////////////////////////////////////////////////
		// Replace all instances of the reduction array to the extended array expressions. //
		// (ex: x[i][k] => sh__x[i][k][threadIdx.x])                                       //
		/////////////////////////////////////////////////////////////////////////////////////
		NameID tid = new NameID("threadIdx.x");
		Identifier bid = SymbolTools.getOrphanID("_bid");
		DepthFirstIterator iter = new DepthFirstIterator(targetRegion);
		for (;;)
		{
			ArrayAccess aAccess = null;

			try {
				aAccess = (ArrayAccess)iter.next(ArrayAccess.class);
			} catch (NoSuchElementException e) {
				break;
			}
			IDExpression arrayID = (IDExpression)aAccess.getArrayName();
			if( arrayID.equals(tarSymID) ) {
				if( aAccess.getNumIndices() == dimsize ) {
					List indices = new LinkedList();
					///////////////////////
					// DEBUG: deprecated //
					///////////////////////
					//indices.addAll(aAccess.getIndices());
					for( Expression indx : aAccess.getIndices() ) {
						indices.add(indx.clone());
					}
					indices.add((NameID)tid.clone());
					ArrayAccess extendedAccess = new ArrayAccess((IDExpression)array_var.clone(), indices);
					aAccess.swapWith(extendedAccess);
				} else {
					Tools.exit(pass_name + "[ERROR in CreateExtendedArray()] Incorrect dimension of the array access :"
							+ aAccess.toString());
				}
			}
		}
		// Replace all instances of the reduction variable to the GPU shared variable
		IRTools.replaceAll((Traversable) targetRegion, tarSymID, array_var);

		///////////////////////////////////////////////////////
		// Add reduction variable initialization statements. //
		///////////////////////////////////////////////////////
		// Ex: for(i=0; i<SIZE1; i++) {                      //
		//         for(k=0; k<SIZE2; k++) {                  //
		//             sh__x[i][k][threadIdx.x] = initValue; //
		//         }                                         //
		//      }                                            //
		///////////////////////////////////////////////////////
		Expression initValue = TransformTools.getRInitValue(redOp, specList);
		//////////////////////////////////////// //////
		// Create or find temporary index variables. //
		//////////////////////////////////////// //////
		List<Identifier> index_vars = new LinkedList<Identifier>();
		for( int i=0; i<=dimsize; i++ ) {
			index_vars.add(TransformTools.getTempIndex(parallelRegion, i));
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
			condition = new BinaryExpression((Identifier)index_var.clone(),
					BinaryOperator.COMPARE_LT, (Expression)edimensions.get(i).clone());
			step = new UnaryExpression(UnaryOperator.POST_INCREMENT,
					(Identifier)index_var.clone());
			loop_body = new CompoundStatement();
			if( i == (dimsize-1) ) {
				List<Expression> indices = new LinkedList<Expression>();
				for( int k=0; k<dimsize; k++ ) {
					indices.add((Expression)index_vars.get(k).clone());
				}
				indices.add((NameID)tid.clone());
				assignex = new AssignmentExpression(new ArrayAccess(
						(Identifier)array_var.clone(), indices),
						AssignmentOperator.NORMAL, initValue);
				loop_body.addStatement(new ExpressionStatement(assignex));
			} else {
				loop_body.addStatement(innerLoop);
			}
			innerLoop = new ForLoop(loop_init, condition, step, loop_body);
		}
		Statement last_decl_stmt;
		last_decl_stmt = IRTools.getLastDeclarationStatement(parallelRegion);
		if( last_decl_stmt != null ) {
			parallelRegion.addStatementAfter(last_decl_stmt,(Statement)innerLoop);
		} else {
			last_decl_stmt = (Statement)parallelRegion.getChildren().get(0);
			parallelRegion.addStatementBefore(last_decl_stmt,(Statement)innerLoop);
		}
		/////////////////////////////////////////////////////////////////////
		// Add in-block reduction codes at the end of the parallel region. //
		/////////////////////////////////////////////////////////////////////
		Statement ifstmt = null;
		FunctionCall syncCall = new FunctionCall(new NameID("__syncthreads"));
		Statement syncCallStmt = new ExpressionStatement(syncCall);
		parallelRegion.addStatement(syncCallStmt);
		if( opt_UnrollingOnReduction ) {
			///////////////////////////////////////////////////
			// Version1:  reduction with loop unrolling code //
			///////////////////////////////////////////////////
			// Assume that BLOCK_SIZE = 512.
		    //if (tid < 256) {
			//    for (i=0; i<SIZE1; i++) {
			//        for (k=0; k<SIZE2; k++) {
			//            sh__x[i][k][tid] += sh__x[i][k][tid + 256];
			//        }
			//    }
		    // }
		    // __syncthreads();
		    //if (tid < 128) {
			//    for (i=0; i<SIZE1; i++) {
			//        for (k=0; k<SIZE2; k++) {
			//            sh__x[i][k][tid] += sh__x[i][k][tid + 128];
			//        }
			//    }
		    // }
		    // __syncthreads();
		    // if (tid < 64) {
			//    for (i=0; i<SIZE1; i++) {
			//        for (k=0; k<SIZE2; k++) {
			//            sh__x[i][k][tid] += sh__x[i][k][tid + 64];
			//        }
			//    }
		    // }
		    // __syncthreads();
		    // if (tid < 32)
		    // {
			//    for (i=0; i<SIZE1; i++) {
			//        for (k=0; k<SIZE2; k++) {
			//            sh__x[i][k][tid] += sh__x[i][k][tid + 32];
			//        }
			//    }
			//    for (i=0; i<SIZE1; i++) {
			//        for (k=0; k<SIZE2; k++) {
			//            sh__x[i][k][tid] += sh__x[i][k][tid + 16];
			//        }
			//    }
			//    for (i=0; i<SIZE1; i++) {
			//        for (k=0; k<SIZE2; k++) {
			//            sh__x[i][k][tid] += sh__x[i][k][tid + 8];
			//        }
			//    }
			//    for (i=0; i<SIZE1; i++) {
			//        for (k=0; k<SIZE2; k++) {
			//            sh__x[i][k][tid] += sh__x[i][k][tid + 4];
			//        }
			//    }
			//    for (i=0; i<SIZE1; i++) {
			//        for (k=0; k<SIZE2; k++) {
			//            sh__x[i][k][tid] += sh__x[i][k][tid + 2];
			//        }
			//    }
			//    for (i=0; i<SIZE1; i++) {
			//        for (k=0; k<SIZE2; k++) {
			//            sh__x[i][k][tid] += sh__x[i][k][tid + 1];
			//        }
			//    }
		    // }
			///////////////////////////////////////////////////
			if( defaultBlockSize == 512 ) {
				for( int i=dimsize-1; i>=0; i-- ) {
					index_var = index_vars.get(i);
					assignex = new AssignmentExpression((Identifier)index_var.clone(),
							AssignmentOperator.NORMAL, new IntegerLiteral(0));
					loop_init = new ExpressionStatement(assignex);
					condition = new BinaryExpression((Identifier)index_var.clone(),
							BinaryOperator.COMPARE_LT, (Expression)edimensions.get(i).clone());
					step = new UnaryExpression(UnaryOperator.POST_INCREMENT,
							(Identifier)index_var.clone());
					loop_body = new CompoundStatement();
					if( i == dimsize-1 ) {
						List<Expression> indices1 = new LinkedList<Expression>();
						List<Expression> indices2 = new LinkedList<Expression>();
						for( int k=0; k<dimsize; k++ ) {
							indices1.add((Expression)index_vars.get(k).clone());
							indices2.add((Expression)index_vars.get(k).clone());
						}
						indices1.add((NameID)tid.clone());
						indices2.add(new BinaryExpression((NameID)tid.clone(),
								BinaryOperator.ADD, new IntegerLiteral(256)));
						assignex = TransformTools.RedExpression(new ArrayAccess(
								(Identifier)array_var.clone(), indices1),
								redOp, new ArrayAccess(
										(Identifier)array_var.clone(), indices2));
						loop_body.addStatement(new ExpressionStatement(assignex));
					} else {
						loop_body.addStatement(innerLoop);
					}
					innerLoop = new ForLoop(loop_init, condition, step, loop_body);
				}
				condition = new BinaryExpression((NameID)tid.clone(),
						BinaryOperator.COMPARE_LT, new IntegerLiteral(256) );
				ifstmt = new IfStatement(condition, innerLoop);
				parallelRegion.addStatement(ifstmt);
				parallelRegion.addStatement((Statement)syncCallStmt.clone());
			}
			if( defaultBlockSize >= 256 ) {
				for( int i=dimsize-1; i>=0; i-- ) {
					index_var = index_vars.get(i);
					assignex = new AssignmentExpression((Identifier)index_var.clone(),
							AssignmentOperator.NORMAL, new IntegerLiteral(0));
					loop_init = new ExpressionStatement(assignex);
					condition = new BinaryExpression((Identifier)index_var.clone(),
							BinaryOperator.COMPARE_LT, (Expression)edimensions.get(i).clone());
					step = new UnaryExpression(UnaryOperator.POST_INCREMENT,
							(Identifier)index_var.clone());
					loop_body = new CompoundStatement();
					if( i == dimsize-1 ) {
						List<Expression> indices1 = new LinkedList<Expression>();
						List<Expression> indices2 = new LinkedList<Expression>();
						for( int k=0; k<dimsize; k++ ) {
							indices1.add((Expression)index_vars.get(k).clone());
							indices2.add((Expression)index_vars.get(k).clone());
						}
						indices1.add((NameID)tid.clone());
						indices2.add(new BinaryExpression((NameID)tid.clone(),
								BinaryOperator.ADD, new IntegerLiteral(128)));
						assignex = TransformTools.RedExpression(new ArrayAccess(
								(Identifier)array_var.clone(), indices1),
								redOp, new ArrayAccess(
										(Identifier)array_var.clone(), indices2));
						loop_body.addStatement(new ExpressionStatement(assignex));
					} else {
						loop_body.addStatement(innerLoop);
					}
					innerLoop = new ForLoop(loop_init, condition, step, loop_body);
				}
				condition = new BinaryExpression((NameID)tid.clone(),
						BinaryOperator.COMPARE_LT, new IntegerLiteral(128) );
				ifstmt = new IfStatement(condition, innerLoop);
				parallelRegion.addStatement(ifstmt);
				parallelRegion.addStatement((Statement)syncCallStmt.clone());
			}
			if( defaultBlockSize >= 128 ) {
				for( int i=dimsize-1; i>=0; i-- ) {
					index_var = index_vars.get(i);
					assignex = new AssignmentExpression((Identifier)index_var.clone(),
							AssignmentOperator.NORMAL, new IntegerLiteral(0));
					loop_init = new ExpressionStatement(assignex);
					condition = new BinaryExpression((Identifier)index_var.clone(),
							BinaryOperator.COMPARE_LT, (Expression)edimensions.get(i).clone());
					step = new UnaryExpression(UnaryOperator.POST_INCREMENT,
							(Identifier)index_var.clone());
					loop_body = new CompoundStatement();
					if( i == dimsize-1 ) {
						List<Expression> indices1 = new LinkedList<Expression>();
						List<Expression> indices2 = new LinkedList<Expression>();
						for( int k=0; k<dimsize; k++ ) {
							indices1.add((Expression)index_vars.get(k).clone());
							indices2.add((Expression)index_vars.get(k).clone());
						}
						indices1.add((NameID)tid.clone());
						indices2.add(new BinaryExpression((NameID)tid.clone(),
								BinaryOperator.ADD, new IntegerLiteral(64)));
						assignex = TransformTools.RedExpression(new ArrayAccess(
								(Identifier)array_var.clone(), indices1),
								redOp, new ArrayAccess(
										(Identifier)array_var.clone(), indices2));
						loop_body.addStatement(new ExpressionStatement(assignex));
					} else {
						loop_body.addStatement(innerLoop);
					}
					innerLoop = new ForLoop(loop_init, condition, step, loop_body);
				}
				condition = new BinaryExpression((NameID)tid.clone(),
						BinaryOperator.COMPARE_LT, new IntegerLiteral(64) );
				ifstmt = new IfStatement(condition, innerLoop);
				parallelRegion.addStatement(ifstmt);
				parallelRegion.addStatement((Statement)syncCallStmt.clone());
			}
			CompoundStatement ifbody = new CompoundStatement();
			for( int s = 32; s > 0; s>>=1 ) {
				for( int i=dimsize-1; i>=0; i-- ) {
					index_var = index_vars.get(i);
					assignex = new AssignmentExpression((Identifier)index_var.clone(),
							AssignmentOperator.NORMAL, new IntegerLiteral(0));
					loop_init = new ExpressionStatement(assignex);
					condition = new BinaryExpression((Identifier)index_var.clone(),
							BinaryOperator.COMPARE_LT, (Expression)edimensions.get(i).clone());
					step = new UnaryExpression(UnaryOperator.POST_INCREMENT,
							(Identifier)index_var.clone());
					loop_body = new CompoundStatement();
					if( i == dimsize-1 ) {
						List<Expression> indices1 = new LinkedList<Expression>();
						List<Expression> indices2 = new LinkedList<Expression>();
						for( int k=0; k<dimsize; k++ ) {
							indices1.add((Expression)index_vars.get(k).clone());
							indices2.add((Expression)index_vars.get(k).clone());
						}
						indices1.add((NameID)tid.clone());
						indices2.add(new BinaryExpression((NameID)tid.clone(),
								BinaryOperator.ADD, new IntegerLiteral(s)));
						assignex = TransformTools.RedExpression(new ArrayAccess(
								(Identifier)array_var.clone(), indices1),
								redOp, new ArrayAccess(
										(Identifier)array_var.clone(), indices2));
						loop_body.addStatement(new ExpressionStatement(assignex));
					} else {
						loop_body.addStatement(innerLoop);
					}
					innerLoop = new ForLoop(loop_init, condition, step, loop_body);
				}
				ifbody.addStatement(innerLoop);
			}
			condition = new BinaryExpression((NameID)tid.clone(),
					BinaryOperator.COMPARE_LT, new IntegerLiteral(32) );
			ifstmt = new IfStatement(condition, ifbody);
			parallelRegion.addStatement(ifstmt);
		} else {
			/////////////////////////////////////////////////////////////////////
			// Version2: Unoptimized reduction code                            //
			/////////////////////////////////////////////////////////////////////
			//     bsize = BLOCK_SIZE;
			//     tid = threadIdx.x;
			//     oldSize = bsize;
			//     for (s=(bsize>>1); s>0; s>>=1) {
			//         if(tid < s) {
			//             for (i=0; i<SIZE1; i++) {
			//                 for (k=0; k<SIZE2; k++) {
			//                     sh__x[i][k][tid] += sh__x[i][k][tid + s];
			//                 }
			//             }
			//         }
			//         oddNum = oldSize & (0x01);
			//         if ( oddNum == 1 ) {
			//             if (tid == 0) {
			//                 for (i=0; i<SIZE1; i++) {
			//                     for (k=0; k<SIZE2; k++) {
			//                         sh__x[i][k][0] += sh__x[i][k][oldSize-1];
			//                     }
			//                 }
			//             }
			//         }
			//         oldSize = s;
			//         __syncthreads();
			//     }
			/////////////////////////////////////////////////////////////////////
			Identifier index_var2 = TransformTools.getTempIndex(parallelRegion, dimsize);
			Identifier oldSize = TransformTools.getTempIndex(parallelRegion, dimsize+1);
			Identifier oddNum = TransformTools.getTempIndex(parallelRegion, dimsize+2);
			Statement estmt = new ExpressionStatement(
					new AssignmentExpression((Expression)oldSize.clone(),
							AssignmentOperator.NORMAL, new NameID("BLOCK_SIZE")));
			parallelRegion.addStatement(estmt);
			assignex = new AssignmentExpression((Identifier)index_var2.clone(),
					AssignmentOperator.NORMAL, new BinaryExpression(new NameID("BLOCK_SIZE"),
							BinaryOperator.SHIFT_RIGHT, new IntegerLiteral(1)));
			loop_init = new ExpressionStatement(assignex);
			condition = new BinaryExpression((Identifier)index_var2.clone(),
					BinaryOperator.COMPARE_GT, new IntegerLiteral(0));
			step = new AssignmentExpression( (Identifier)index_var2.clone(),
					AssignmentOperator.SHIFT_RIGHT, new IntegerLiteral(1));
			CompoundStatement loopbody = new CompoundStatement();
			ForLoop reductionLoop = new ForLoop(loop_init, condition, step, loopbody);
			for( int i=dimsize-1; i>=0; i-- ) {
				index_var = index_vars.get(i);
				assignex = new AssignmentExpression((Identifier)index_var.clone(),
						AssignmentOperator.NORMAL, new IntegerLiteral(0));
				loop_init = new ExpressionStatement(assignex);
				condition = new BinaryExpression((Identifier)index_var.clone(),
						BinaryOperator.COMPARE_LT, (Expression)edimensions.get(i).clone());
				step = new UnaryExpression(UnaryOperator.POST_INCREMENT,
						(Identifier)index_var.clone());
				loop_body = new CompoundStatement();
				if( i == dimsize-1 ) {
					List<Expression> indices1 = new LinkedList<Expression>();
					List<Expression> indices2 = new LinkedList<Expression>();
					for( int k=0; k<dimsize; k++ ) {
						indices1.add((Expression)index_vars.get(k).clone());
						indices2.add((Expression)index_vars.get(k).clone());
					}
					indices1.add((NameID)tid.clone());
					indices2.add(new BinaryExpression((NameID)tid.clone(),
							BinaryOperator.ADD, (Identifier)index_var2.clone()));
					assignex = TransformTools.RedExpression(new ArrayAccess(
							(Identifier)array_var.clone(), indices1),
							redOp, new ArrayAccess(
									(Identifier)array_var.clone(), indices2));
					loop_body.addStatement(new ExpressionStatement(assignex));
				} else {
					loop_body.addStatement(innerLoop);
				}
				innerLoop = new ForLoop(loop_init, condition, step, loop_body);
			}
			condition = new BinaryExpression((NameID)tid.clone(), BinaryOperator.COMPARE_LT,
					(Identifier)index_var2.clone());
			loopbody.addStatement( new IfStatement(condition, innerLoop) );
			assignex = new AssignmentExpression( (Identifier)oddNum.clone(), AssignmentOperator.NORMAL,
					new BinaryExpression((Identifier)oldSize.clone(), BinaryOperator.BITWISE_AND,
							new IntegerLiteral(0x01)));
			loopbody.addStatement(new ExpressionStatement(assignex));
			for( int i=dimsize-1; i>=0; i-- ) {
				index_var = index_vars.get(i);
				assignex = new AssignmentExpression((Identifier)index_var.clone(),
						AssignmentOperator.NORMAL, new IntegerLiteral(0));
				loop_init = new ExpressionStatement(assignex);
				condition = new BinaryExpression((Identifier)index_var.clone(),
						BinaryOperator.COMPARE_LT, (Expression)edimensions.get(i).clone());
				step = new UnaryExpression(UnaryOperator.POST_INCREMENT,
						(Identifier)index_var.clone());
				loop_body = new CompoundStatement();
				if( i == dimsize-1 ) {
					List<Expression> indices1 = new LinkedList<Expression>();
					List<Expression> indices2 = new LinkedList<Expression>();
					for( int k=0; k<dimsize; k++ ) {
						indices1.add((Expression)index_vars.get(k).clone());
						indices2.add((Expression)index_vars.get(k).clone());
					}
					indices1.add(new IntegerLiteral(0));
					indices2.add(new BinaryExpression( (Identifier)oldSize.clone(),
							BinaryOperator.SUBTRACT, new IntegerLiteral(1)));
					assignex = TransformTools.RedExpression(new ArrayAccess(
							(Identifier)array_var.clone(), indices1),
							redOp, new ArrayAccess(
									(Identifier)array_var.clone(), indices2));
					loop_body.addStatement(new ExpressionStatement(assignex));
				} else {
					loop_body.addStatement(innerLoop);
				}
				innerLoop = new ForLoop(loop_init, condition, step, loop_body);
			}
			condition = new BinaryExpression((NameID)tid.clone(), BinaryOperator.COMPARE_EQ,
					new IntegerLiteral(0));
			ifstmt = new IfStatement(condition, innerLoop);
			condition = new BinaryExpression( (Identifier)oddNum.clone(), BinaryOperator.COMPARE_EQ,
					new IntegerLiteral(1));
			loopbody.addStatement( new IfStatement(condition, ifstmt) );
			estmt = new ExpressionStatement(
					new AssignmentExpression((Expression)oldSize.clone(),
							AssignmentOperator.NORMAL, (Identifier)index_var2.clone()));
			loopbody.addStatement( estmt );
			loopbody.addStatement((Statement)syncCallStmt.clone());
			parallelRegion.addStatement(reductionLoop);
		}
		////////////////////////////////////////////////////////////////////////////
		// Write the in-block reduction result back to the global reduction array //
		////////////////////////////////////////////////////////////////////////////
		//     bid = blockIdx.x + blockIdx.y * gridDim.x;
		//     tid = threadIdx.x;
		//     if( tid == 0 ) {
		//         for( i=0; i<SIZE1; i++ ) {
		//             for( k=0; k<SIZE2; k++ ) {
		//                 x[bid][i][k] = sh__x[i][k][0];
		//             }
		//         }
		//     }
		////////////////////////////////////////////////////////////////////////////
		for( int i=dimsize-1; i>=0; i-- ) {
			index_var = index_vars.get(i);
			assignex = new AssignmentExpression((Identifier)index_var.clone(),
					AssignmentOperator.NORMAL, new IntegerLiteral(0));
			loop_init = new ExpressionStatement(assignex);
			condition = new BinaryExpression((Identifier)index_var.clone(),
					BinaryOperator.COMPARE_LT, (Expression)edimensions.get(i).clone());
			step = new UnaryExpression(UnaryOperator.POST_INCREMENT,
					(Identifier)index_var.clone());
			loop_body = new CompoundStatement();
			if( i == dimsize-1 ) {
				List<Expression> indices1 = new LinkedList<Expression>();
				List<Expression> indices2 = new LinkedList<Expression>();
				indices1.add((Identifier)bid.clone());
				for( int k=0; k<dimsize; k++ ) {
					indices1.add((Expression)index_vars.get(k).clone());
					indices2.add((Expression)index_vars.get(k).clone());
				}
				indices2.add(new IntegerLiteral(0));
				assignex = new AssignmentExpression(new ArrayAccess(
						(Identifier)red_var.clone(), indices1),
						AssignmentOperator.NORMAL, new ArrayAccess(
						(Identifier)array_var.clone(), indices2));
				loop_body.addStatement(new ExpressionStatement(assignex));
			} else {
				loop_body.addStatement(innerLoop);
			}
			innerLoop = new ForLoop(loop_init, condition, step, loop_body);
		}
		condition = new BinaryExpression((NameID)tid.clone(), BinaryOperator.COMPARE_EQ,
				new IntegerLiteral(0));
		parallelRegion.addStatement( new IfStatement(condition, innerLoop) );

		return redV_declarator;
	}

	/**
	 * This method conducts the first part of arrayReductionConv(), without generating
	 * the in-block reduction codes.
	 *
	 * @param redOp
	 * @param tarSym
	 * @param new_proc
	 * @param region
	 * @param redArgSet
	 * @return
	 */
	private static VariableDeclarator arrayReductionConv2(BinaryOperator redOp,
			VariableDeclarator tarSym, Procedure new_proc, Statement region,
			ArrayList<VariableDeclarator> redArgSet, ArrayList<VariableDeclarator> redParamSet)  {
		Identifier tarSymID = new Identifier(tarSym);
		List aspecs = tarSym.getArraySpecifiers();
		ArraySpecifier aspec = (ArraySpecifier)aspecs.get(0);
		int dimsize = aspec.getNumDimensions();
		StringBuilder str = new StringBuilder(80);
		str.append("red__");
		str.append(tarSym.getSymbolName());

		////////////////////////////////////
		// Create an extended array type  //
		// Ex: "float b[][SIZE1][SIZE2] " //
		////////////////////////////////////
		List<Expression> edimensions = new LinkedList<Expression>();
		edimensions.add(null);
		for( int i=0; i<dimsize; i++ )
		{
			edimensions.add((Expression)aspec.getDimension(i).clone());
		}
		ArraySpecifier easpec = new ArraySpecifier(edimensions);
		VariableDeclarator redV_declarator = new VariableDeclarator(new NameID(str.toString()), easpec);
		List<Specifier> clonedspecs = new ChainedList<Specifier>();
		/////////////////////////////////////////////////////////////////////////////////////
		// CAUTION: VariableDeclarator.getTypeSpecifiers() returns both specifiers of      //
		// its parent VariableDeclaration and the VariableDeclarator's leading specifiers. //
		// Therefore, if VariableDeclarator is a pointer symbol, this method will return   //
		// pointer specifiers too.                                                         //
		/////////////////////////////////////////////////////////////////////////////////////
		clonedspecs.addAll(tarSym.getTypeSpecifiers());
		clonedspecs.remove(Specifier.STATIC);
		VariableDeclaration redV_decl =
			new VariableDeclaration(clonedspecs, redV_declarator);
		Identifier red_var = new Identifier(redV_declarator);
		// Insert function parameter.
		new_proc.addDeclaration(redV_decl);

		redParamSet.add(redV_declarator);

		CompoundStatement targetRegion = null;
		CompoundStatement parallelRegion = null;
		if( region instanceof CompoundStatement ) {
			targetRegion = (CompoundStatement)region;
			parallelRegion = targetRegion;
		} else if( region instanceof ForLoop ) {
			targetRegion = (CompoundStatement)((ForLoop)region).getBody();
			parallelRegion = (CompoundStatement)region.getParent();
		} else {
			Tools.exit(pass_name + "[ERROR] Unknwon region in extractKernelRegion(): "
					+ region.toString());
		}
		/////////////////////////////////////////////////////////////////////////
		// Create an array on the GPU shared memory, which holds privatized    //
		// reduction variable.                                                 //
		// Assumption: GPU threadblock is an one-dimensional array of threads. //
		// Ex: __shared__ float sh__x[SIZE1][SIZE2][BLOCK_SIZE];               //
		/////////////////////////////////////////////////////////////////////////
		edimensions = new LinkedList<Expression>();
		for( int i=0; i<dimsize; i++ )
		{
			edimensions.add((Expression)aspec.getDimension(i).clone());
		}
		edimensions.add(new NameID("BLOCK_SIZE"));
		easpec = new ArraySpecifier(edimensions);
		str = new StringBuilder(80);
		str.append("sh__");
		str.append(tarSym.getSymbolName());
		VariableDeclarator arrayV_declarator = new VariableDeclarator(new NameID(str.toString()), easpec);
		List specList = new LinkedList();
		specList.add(CUDASpecifier.CUDA_SHARED);
		specList.addAll(clonedspecs);
		VariableDeclaration arrayV_decl =
			new VariableDeclaration(specList, arrayV_declarator);
		Identifier array_var = new Identifier(arrayV_declarator);
		parallelRegion.addDeclaration(arrayV_decl);

		redArgSet.add(arrayV_declarator);

		/////////////////////////////////////////////////////////////////////////////////////
		// Replace all instances of the reduction array to the extended array expressions. //
		// (ex: x[i][k] => sh__x[i][k][threadIdx.x])                                       //
		/////////////////////////////////////////////////////////////////////////////////////
		NameID tid = new NameID("threadIdx.x");
		Identifier bid = SymbolTools.getOrphanID("_bid");
		DepthFirstIterator iter = new DepthFirstIterator(targetRegion);
		for (;;)
		{
			ArrayAccess aAccess = null;

			try {
				aAccess = (ArrayAccess)iter.next(ArrayAccess.class);
			} catch (NoSuchElementException e) {
				break;
			}
			IDExpression arrayID = (IDExpression)aAccess.getArrayName();
			if( arrayID.equals(tarSymID) ) {
				if( aAccess.getNumIndices() == dimsize ) {
					List indices = new LinkedList();
					///////////////////////
					// DEBUG: deprecated //
					///////////////////////
					//indices.addAll(aAccess.getIndices());
					for( Expression indx : aAccess.getIndices() ) {
						indices.add(indx.clone());
					}
					indices.add((NameID)tid.clone());
					ArrayAccess extendedAccess = new ArrayAccess((IDExpression)array_var.clone(), indices);
					aAccess.swapWith(extendedAccess);
				} else {
					Tools.exit(pass_name + "[ERROR in CreateExtendedArray()] Incorrect dimension of the array access :"
							+ aAccess.toString());
				}
			}
		}
		// Replace all instances of the reduction variable to the GPU shared variable
		IRTools.replaceAll((Traversable) targetRegion, tarSymID, array_var);

		///////////////////////////////////////////////////////
		// Add reduction variable initialization statements. //
		///////////////////////////////////////////////////////
		// Ex: for(i=0; i<SIZE1; i++) {                      //
		//         for(k=0; k<SIZE2; k++) {                  //
		//             sh__x[i][k][threadIdx.x] = initValue; //
		//         }                                         //
		//      }                                            //
		///////////////////////////////////////////////////////
		Expression initValue = TransformTools.getRInitValue(redOp, specList);
		//////////////////////////////////////// //////
		// Create or find temporary index variables. //
		//////////////////////////////////////// //////
		List<Identifier> index_vars = new LinkedList<Identifier>();
		for( int i=0; i<=dimsize; i++ ) {
			index_vars.add(TransformTools.getTempIndex(parallelRegion, i));
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
			condition = new BinaryExpression((Identifier)index_var.clone(),
					BinaryOperator.COMPARE_LT, (Expression)edimensions.get(i).clone());
			step = new UnaryExpression(UnaryOperator.POST_INCREMENT,
					(Identifier)index_var.clone());
			loop_body = new CompoundStatement();
			if( i == (dimsize-1) ) {
				List<Expression> indices = new LinkedList<Expression>();
				for( int k=0; k<dimsize; k++ ) {
					indices.add((Expression)index_vars.get(k).clone());
				}
				indices.add((NameID)tid.clone());
				assignex = new AssignmentExpression(new ArrayAccess(
						(Identifier)array_var.clone(), indices),
						AssignmentOperator.NORMAL, initValue);
				loop_body.addStatement(new ExpressionStatement(assignex));
			} else {
				loop_body.addStatement(innerLoop);
			}
			innerLoop = new ForLoop(loop_init, condition, step, loop_body);
		}
		Statement last_decl_stmt;
		last_decl_stmt = IRTools.getLastDeclarationStatement(parallelRegion);
		if( last_decl_stmt != null ) {
			parallelRegion.addStatementAfter(last_decl_stmt,(Statement)innerLoop);
		} else {
			last_decl_stmt = (Statement)parallelRegion.getChildren().get(0);
			parallelRegion.addStatementBefore(last_decl_stmt,(Statement)innerLoop);
		}

		return redV_declarator;
	}

	private static void reductionConv(ArrayList<VariableDeclarator> redArgSet,
			ArrayList<VariableDeclarator> redParamSet, ArrayList<BinaryOperator> redOpSet,
			Statement region, HashSet<String> cudaNoRedUnrollSet)  {
		ArrayList<VariableDeclarator> redArgSet1 = new ArrayList<VariableDeclarator>();
		ArrayList<VariableDeclarator> redArgSet2 = new ArrayList<VariableDeclarator>();
		ArrayList<VariableDeclarator> redParamSet1 = new ArrayList<VariableDeclarator>();
		ArrayList<VariableDeclarator> redParamSet2 = new ArrayList<VariableDeclarator>();
		ArrayList<BinaryOperator> redOpSet1 = new ArrayList<BinaryOperator>();
		ArrayList<BinaryOperator> redOpSet2 = new ArrayList<BinaryOperator>();
		BinaryOperator redOp = null;
		LinkedList<Expression> edimensions = null;
		int _index = 0;
		Expression assignex = null;
		Statement loop_init = null;
		Expression condition = null;
		Expression step = null;
		CompoundStatement loop_body = null;
		ForLoop innerLoop = null;
		NameID tid = new NameID("threadIdx.x");
		Identifier bid = SymbolTools.getOrphanID("_bid");
		CompoundStatement targetRegion = null;
		CompoundStatement parallelRegion = null;
		if( region instanceof CompoundStatement ) {
			targetRegion = (CompoundStatement)region;
			parallelRegion = targetRegion;
		} else if( region instanceof ForLoop ) {
			targetRegion = (CompoundStatement)((ForLoop)region).getBody();
			parallelRegion = (CompoundStatement)region.getParent();
		} else {
			Tools.exit(pass_name + "[ERROR] Unknwon region in extractKernelRegion(): "
					+ region.toString());
		}

		/////////////////////////////////////////////////////////////////////
		// Add in-block reduction codes at the end of the parallel region. //
		/////////////////////////////////////////////////////////////////////
		Statement ifstmt = null;
		CompoundStatement ifBody = null;
		FunctionCall syncCall = new FunctionCall(new NameID("__syncthreads"));
		Statement syncCallStmt = new ExpressionStatement(syncCall);
		parallelRegion.addStatement(syncCallStmt);

		if( opt_UnrollingOnReduction ) {
			HashSet<String> cudaNoRedUnrollSet2 = new HashSet<String>();
			// Add prefix, "sh__", to each variable in the cudaNoRedUnrollSet.
			if( cudaNoRedUnrollSet.size() > 0) {
				for( String redVar : cudaNoRedUnrollSet ) {
					cudaNoRedUnrollSet2.add("sh__"+redVar);
				}
			}
			_index = 0;
			for( VariableDeclarator arrayV_declarator : redArgSet ) {
				redOp = redOpSet.get(_index);
				VariableDeclarator redV_declarator = redParamSet.get(_index);
				String redVarName = arrayV_declarator.getSymbolName();
				if( cudaNoRedUnrollSet2.contains(redVarName) ) {
					redArgSet2.add(arrayV_declarator);
					redParamSet2.add(redV_declarator);
					redOpSet2.add(redOp);
				} else {
					redArgSet1.add(arrayV_declarator);
					redParamSet1.add(redV_declarator);
					redOpSet1.add(redOp);
				}
				_index++;
			}
		} else {
			redArgSet2.addAll(redArgSet);
			redParamSet2.addAll(redParamSet);
			redOpSet2.addAll(redOpSet);

		}
		if( redArgSet1.size() > 0 ) {
			///////////////////////////////////////////////////
			// Version1:  reduction with loop unrolling code //
			// this version works only if BLOCK_SIZE = 2^m   //
			///////////////////////////////////////////////////
			// Case1:  reduction variable is scalar.         //
			///////////////////////////////////////////////////
			// Assume that BLOCK_SIZE = 512.
		    //if (tid < 256) {
		    //     sh__x[tid] += sh__x[tid + 256];
		    // }
		    // __syncthreads();
		    //if (tid < 128) {
		    //     sh__x[tid] += sh__x[tid + 128];
		    // }
		    // __syncthreads();
		    // if (tid < 64) {
		    //     sh__x[tid] += sh__x[tid + 64];
		    // }
		    // __syncthreads();
		    // if (tid < 32)
		    // {
		    //     sh__x[tid] += sh__x[tid + 32];
		    // }
		    // if (tid < 16)
		    // {
		    //     sh__x[tid] += sh__x[tid + 16];
		    // }
		    // if (tid < 8)
		    // {
		    //     sh__x[tid] += sh__x[tid + 8];
		    // }
		    // if (tid < 4)
		    // {
		    //     sh__x[tid] += sh__x[tid + 4];
		    // }
		    // if (tid < 2)
		    // {
		    //     sh__x[tid] += sh__x[tid + 2];
		    // }
		    // if (tid < 1)
		    // {
		    //     sh__x[tid] += sh__x[tid + 1];
		    // }
			///////////////////////////////////////////////////
			// Case2:  reduction variable is array.          //
			///////////////////////////////////////////////////
			// Assume that BLOCK_SIZE = 512.
		    //if (tid < 256) {
			//    for (i=0; i<SIZE1; i++) {
			//        for (k=0; k<SIZE2; k++) {
			//            sh__x[i][k][tid] += sh__x[i][k][tid + 256];
			//        }
			//    }
		    // }
		    // __syncthreads();
		    //if (tid < 128) {
			//    for (i=0; i<SIZE1; i++) {
			//        for (k=0; k<SIZE2; k++) {
			//            sh__x[i][k][tid] += sh__x[i][k][tid + 128];
			//        }
			//    }
		    // }
		    // __syncthreads();
		    // if (tid < 64) {
			//    for (i=0; i<SIZE1; i++) {
			//        for (k=0; k<SIZE2; k++) {
			//            sh__x[i][k][tid] += sh__x[i][k][tid + 64];
			//        }
			//    }
		    // }
		    // __syncthreads();
		    // if (tid < 32)
		    // {
			//    for (i=0; i<SIZE1; i++) {
			//        for (k=0; k<SIZE2; k++) {
			//            sh__x[i][k][tid] += sh__x[i][k][tid + 32];
			//        }
			//    }
		    // }
		    // if (tid < 16)
		    // {
			//    for (i=0; i<SIZE1; i++) {
			//        for (k=0; k<SIZE2; k++) {
			//            sh__x[i][k][tid] += sh__x[i][k][tid + 16];
			//        }
			//    }
		    // }
		    // if (tid < 8)
		    // {
			//    for (i=0; i<SIZE1; i++) {
			//        for (k=0; k<SIZE2; k++) {
			//            sh__x[i][k][tid] += sh__x[i][k][tid + 8];
			//        }
			//    }
		    // }
		    // if (tid < 4)
		    // {
			//    for (i=0; i<SIZE1; i++) {
			//        for (k=0; k<SIZE2; k++) {
			//            sh__x[i][k][tid] += sh__x[i][k][tid + 4];
			//        }
			//    }
		    // }
		    // if (tid < 2)
		    // {
			//    for (i=0; i<SIZE1; i++) {
			//        for (k=0; k<SIZE2; k++) {
			//            sh__x[i][k][tid] += sh__x[i][k][tid + 2];
			//        }
			//    }
		    // }
		    // if (tid < 1)
		    // {
			//    for (i=0; i<SIZE1; i++) {
			//        for (k=0; k<SIZE2; k++) {
			//            sh__x[i][k][tid] += sh__x[i][k][tid + 1];
			//        }
			//    }
		    // }
			///////////////////////////////////////////////////
			for( int _bsize_ = 256; _bsize_ > 0; _bsize_>>=1 ) {
				if( defaultBlockSize >= 2*_bsize_ ) {
					_index = 0;
					ifBody = new CompoundStatement();
					for( VariableDeclarator arrayV_declarator : redArgSet1 ) {
						redOp = redOpSet1.get(_index);
						Identifier array_var = new Identifier(arrayV_declarator);
						List aspecs = arrayV_declarator.getArraySpecifiers();
						ArraySpecifier aspec = (ArraySpecifier)aspecs.get(0);
						int dimsize = aspec.getNumDimensions();
						edimensions = new LinkedList<Expression>();
						if( dimsize == 1 ) {
							assignex = TransformTools.RedExpression(new ArrayAccess((Identifier)array_var.clone(),
									(NameID)tid.clone()) , redOp,
									new ArrayAccess((Identifier)array_var.clone(),
											new BinaryExpression((NameID)tid.clone(), BinaryOperator.ADD,
													new IntegerLiteral(_bsize_))));
							ifBody.addStatement(new ExpressionStatement(assignex));
						} else {
							for( int i=0; i<dimsize; i++ )
							{
								edimensions.add((Expression)aspec.getDimension(i).clone());
							}
							//////////////////////////////////////// //////
							// Create or find temporary index variables. //
							//////////////////////////////////////// //////
							List<Identifier> index_vars = new LinkedList<Identifier>();
							for( int i=0; i<dimsize-1; i++ ) {
								index_vars.add(TransformTools.getTempIndex(parallelRegion, i));
							}
							Identifier index_var = null;

							for( int i=dimsize-2; i>=0; i-- ) {
								index_var = index_vars.get(i);
								assignex = new AssignmentExpression((Identifier)index_var.clone(),
										AssignmentOperator.NORMAL, new IntegerLiteral(0));
								loop_init = new ExpressionStatement(assignex);
								condition = new BinaryExpression((Identifier)index_var.clone(),
										BinaryOperator.COMPARE_LT, (Expression)edimensions.get(i).clone());
								step = new UnaryExpression(UnaryOperator.POST_INCREMENT,
										(Identifier)index_var.clone());
								loop_body = new CompoundStatement();
								if( i == dimsize-2 ) {
									List<Expression> indices1 = new LinkedList<Expression>();
									List<Expression> indices2 = new LinkedList<Expression>();
									for( int k=0; k<dimsize-1; k++ ) {
										indices1.add((Expression)index_vars.get(k).clone());
										indices2.add((Expression)index_vars.get(k).clone());
									}
									indices1.add((NameID)tid.clone());
									indices2.add(new BinaryExpression((NameID)tid.clone(),
											BinaryOperator.ADD, new IntegerLiteral(_bsize_)));
									assignex = TransformTools.RedExpression(new ArrayAccess(
											(Identifier)array_var.clone(), indices1),
											redOp, new ArrayAccess(
													(Identifier)array_var.clone(), indices2));
									loop_body.addStatement(new ExpressionStatement(assignex));
								} else {
									loop_body.addStatement(innerLoop);
								}
								innerLoop = new ForLoop(loop_init, condition, step, loop_body);
							}
							ifBody.addStatement(innerLoop);
						}
						_index++;
					}
					condition = new BinaryExpression((NameID)tid.clone(),
							BinaryOperator.COMPARE_LT, new IntegerLiteral(_bsize_) );
					ifstmt = new IfStatement(condition, ifBody);
					parallelRegion.addStatement(ifstmt);
					if( _bsize_ > 32 ) {
						parallelRegion.addStatement((Statement)syncCallStmt.clone());
					}
				}
			}
		}
		if( redArgSet2.size() > 0 ) {
			/////////////////////////////////////////////////////////////////////
			// Version2: Unoptimized reduction code                            //
			/////////////////////////////////////////////////////////////////////
			// Case1: reduction variable is scalar.                            //
			/////////////////////////////////////////////////////////////////////
			//     bsize = blockDim.x;
			//     tid = threadIdx.x;
			//     oldSize = bsize;
			//     for (s=(bsize>>1); s>0; s>>=1) {
			//         if(tid < s) {
			//             sh__x[tid] += sh__x[tid + s];
			//         }
			//         oddNum = oldSize & (0x01);
			//         if ( oddNum == 1 ) {
			//             if (tid == 0) {
			//                 sh__x[0] += sh__x[oldSize-1];
			//             }
			//         }
			//         oldSize = s;
			//         __syncthreads();
			//     }
			/////////////////////////////////////////////////////////////////////
			// Case2: reduction variable is an array.                          //
			/////////////////////////////////////////////////////////////////////
			//     bsize = BLOCK_SIZE;
			//     tid = threadIdx.x;
			//     oldSize = bsize;
			//     for (s=(bsize>>1); s>0; s>>=1) {
			//         if(tid < s) {
			//             for (i=0; i<SIZE1; i++) {
			//                 for (k=0; k<SIZE2; k++) {
			//                     sh__x[i][k][tid] += sh__x[i][k][tid + s];
			//                 }
			//             }
			//         }
			//         oddNum = oldSize & (0x01);
			//         if ( oddNum == 1 ) {
			//             if (tid == 0) {
			//                 for (i=0; i<SIZE1; i++) {
			//                     for (k=0; k<SIZE2; k++) {
			//                         sh__x[i][k][0] += sh__x[i][k][oldSize-1];
			//                     }
			//                 }
			//             }
			//         }
			//         oldSize = s;
			//         __syncthreads();
			//     }
			/////////////////////////////////////////////////////////////////////
			// Find the max value of dimensions of reduction variables.
			int maxdimsize = 0;
			for( VariableDeclarator arrayV_declarator : redArgSet2 ) {
				List aspecs = arrayV_declarator.getArraySpecifiers();
				ArraySpecifier aspec = (ArraySpecifier)aspecs.get(0);
				int dimsize = aspec.getNumDimensions();
				if( dimsize > maxdimsize ) {
					maxdimsize = dimsize;
				}
			}
			Identifier index_var2 = TransformTools.getTempIndex(parallelRegion, maxdimsize-1);
			Identifier oldSize = TransformTools.getTempIndex(parallelRegion, maxdimsize);
			Identifier oddNum = TransformTools.getTempIndex(parallelRegion, maxdimsize+1);
			Statement estmt = new ExpressionStatement(
					new AssignmentExpression((Expression)oldSize.clone(),
							AssignmentOperator.NORMAL, new NameID("BLOCK_SIZE")));
			parallelRegion.addStatement(estmt);
			assignex = new AssignmentExpression((Identifier)index_var2.clone(),
					AssignmentOperator.NORMAL, new BinaryExpression(new NameID("BLOCK_SIZE"),
							BinaryOperator.SHIFT_RIGHT, new IntegerLiteral(1)));
			loop_init = new ExpressionStatement(assignex);
			condition = new BinaryExpression((Identifier)index_var2.clone(),
					BinaryOperator.COMPARE_GT, new IntegerLiteral(0));
			step = new AssignmentExpression( (Identifier)index_var2.clone(),
					AssignmentOperator.SHIFT_RIGHT, new IntegerLiteral(1));
			CompoundStatement loopbody = new CompoundStatement();
			ForLoop reductionLoop = new ForLoop(loop_init, condition, step, loopbody);
			ifBody = new CompoundStatement();
			_index = 0;
			for( VariableDeclarator arrayV_declarator : redArgSet2 ) {
				redOp = redOpSet2.get(_index);
				Identifier array_var = new Identifier(arrayV_declarator);
				List aspecs = arrayV_declarator.getArraySpecifiers();
				ArraySpecifier aspec = (ArraySpecifier)aspecs.get(0);
				int dimsize = aspec.getNumDimensions();
				edimensions = new LinkedList<Expression>();
				if( dimsize == 1 ) {
					assignex = TransformTools.RedExpression(new ArrayAccess((Identifier)array_var.clone(),
							(NameID)tid.clone()) , redOp,
							new ArrayAccess((Identifier)array_var.clone(),
									new BinaryExpression((NameID)tid.clone(), BinaryOperator.ADD,
											(Identifier)index_var2.clone())));
					ifBody.addStatement(new ExpressionStatement(assignex));
				} else {
					for( int i=0; i<dimsize; i++ )
					{
						edimensions.add((Expression)aspec.getDimension(i).clone());
					}
					//////////////////////////////////////// //////
					// Create or find temporary index variables. //
					//////////////////////////////////////// //////
					List<Identifier> index_vars = new LinkedList<Identifier>();
					for( int i=0; i<dimsize-1; i++ ) {
						index_vars.add(TransformTools.getTempIndex(parallelRegion, i));
					}
					Identifier index_var = null;

					for( int i=dimsize-2; i>=0; i-- ) {
						index_var = index_vars.get(i);
						assignex = new AssignmentExpression((Identifier)index_var.clone(),
								AssignmentOperator.NORMAL, new IntegerLiteral(0));
						loop_init = new ExpressionStatement(assignex);
						condition = new BinaryExpression((Identifier)index_var.clone(),
								BinaryOperator.COMPARE_LT, (Expression)edimensions.get(i).clone());
						step = new UnaryExpression(UnaryOperator.POST_INCREMENT,
								(Identifier)index_var.clone());
						loop_body = new CompoundStatement();
						if( i == dimsize-2 ) {
							List<Expression> indices1 = new LinkedList<Expression>();
							List<Expression> indices2 = new LinkedList<Expression>();
							for( int k=0; k<dimsize-1; k++ ) {
								indices1.add((Expression)index_vars.get(k).clone());
								indices2.add((Expression)index_vars.get(k).clone());
							}
							indices1.add((NameID)tid.clone());
							indices2.add(new BinaryExpression((NameID)tid.clone(),
									BinaryOperator.ADD, (Identifier)index_var2.clone()));
							assignex = TransformTools.RedExpression(new ArrayAccess(
									(Identifier)array_var.clone(), indices1),
									redOp, new ArrayAccess(
											(Identifier)array_var.clone(), indices2));
							loop_body.addStatement(new ExpressionStatement(assignex));
						} else {
							loop_body.addStatement(innerLoop);
						}
						innerLoop = new ForLoop(loop_init, condition, step, loop_body);
					}
					ifBody.addStatement(innerLoop);
				}
				_index++;
			}
			condition = new BinaryExpression((NameID)tid.clone(), BinaryOperator.COMPARE_LT,
					(Identifier)index_var2.clone());
			loopbody.addStatement( new IfStatement(condition, ifBody) );
			assignex = new AssignmentExpression( (Identifier)oddNum.clone(), AssignmentOperator.NORMAL,
					new BinaryExpression((Identifier)oldSize.clone(), BinaryOperator.BITWISE_AND,
							new IntegerLiteral(0x01)));
			loopbody.addStatement(new ExpressionStatement(assignex));
			ifBody = new CompoundStatement();
			_index = 0;
			for( VariableDeclarator arrayV_declarator : redArgSet2 ) {
				redOp = redOpSet2.get(_index);
				Identifier array_var = new Identifier(arrayV_declarator);
				List aspecs = arrayV_declarator.getArraySpecifiers();
				ArraySpecifier aspec = (ArraySpecifier)aspecs.get(0);
				int dimsize = aspec.getNumDimensions();
				edimensions = new LinkedList<Expression>();
				if( dimsize == 1 ) {
					assignex = TransformTools.RedExpression( new ArrayAccess((Identifier)array_var.clone(),
							new IntegerLiteral(0)), redOp,
							new ArrayAccess((Identifier)array_var.clone(),
									new BinaryExpression( (Identifier)oldSize.clone(), BinaryOperator.SUBTRACT,
											new IntegerLiteral(1))));
					ifBody.addStatement(new ExpressionStatement(assignex));
				} else {
					for( int i=0; i<dimsize; i++ )
					{
						edimensions.add((Expression)aspec.getDimension(i).clone());
					}
					//////////////////////////////////////// //////
					// Create or find temporary index variables. //
					//////////////////////////////////////// //////
					List<Identifier> index_vars = new LinkedList<Identifier>();
					for( int i=0; i<dimsize-1; i++ ) {
						index_vars.add(TransformTools.getTempIndex(parallelRegion, i));
					}
					Identifier index_var = null;

					for( int i=dimsize-2; i>=0; i-- ) {
						index_var = index_vars.get(i);
						assignex = new AssignmentExpression((Identifier)index_var.clone(),
								AssignmentOperator.NORMAL, new IntegerLiteral(0));
						loop_init = new ExpressionStatement(assignex);
						condition = new BinaryExpression((Identifier)index_var.clone(),
								BinaryOperator.COMPARE_LT, (Expression)edimensions.get(i).clone());
						step = new UnaryExpression(UnaryOperator.POST_INCREMENT,
								(Identifier)index_var.clone());
						loop_body = new CompoundStatement();
						if( i == dimsize-2 ) {
							List<Expression> indices1 = new LinkedList<Expression>();
							List<Expression> indices2 = new LinkedList<Expression>();
							for( int k=0; k<dimsize-1; k++ ) {
								indices1.add((Expression)index_vars.get(k).clone());
								indices2.add((Expression)index_vars.get(k).clone());
							}
							indices1.add(new IntegerLiteral(0));
							indices2.add(new BinaryExpression( (Identifier)oldSize.clone(),
									BinaryOperator.SUBTRACT, new IntegerLiteral(1)));
							assignex = TransformTools.RedExpression(new ArrayAccess(
									(Identifier)array_var.clone(), indices1),
									redOp, new ArrayAccess(
											(Identifier)array_var.clone(), indices2));
							loop_body.addStatement(new ExpressionStatement(assignex));
						} else {
							loop_body.addStatement(innerLoop);
						}
						innerLoop = new ForLoop(loop_init, condition, step, loop_body);
					}
					ifBody.addStatement(innerLoop);
				}
				_index++;
			}
			condition = new BinaryExpression((NameID)tid.clone(), BinaryOperator.COMPARE_EQ,
					new IntegerLiteral(0));
			ifstmt = new IfStatement(condition, ifBody);
			condition = new BinaryExpression( (Identifier)oddNum.clone(), BinaryOperator.COMPARE_EQ,
					new IntegerLiteral(1));
			loopbody.addStatement( new IfStatement(condition, ifstmt) );
			estmt = new ExpressionStatement(
					new AssignmentExpression((Expression)oldSize.clone(),
							AssignmentOperator.NORMAL, (Identifier)index_var2.clone()));
			loopbody.addStatement( estmt );
			loopbody.addStatement((Statement)syncCallStmt.clone());
			parallelRegion.addStatement(reductionLoop);
		}
		////////////////////////////////////////////////////////////////////////////
		// Write the in-block reduction result back to the global reduction array //
		////////////////////////////////////////////////////////////////////////////
		// Case1: Reduction variable is scalar.                                   //
		////////////////////////////////////////////////////////////////////////////
		//     bid = blockIdx.x + blockIdx.y * gridDim.x;
		//     tid = threadIdx.x;
		//     if( tid == 0 ) {
		//         x[bid] = sh__x[0];
		//     }
		////////////////////////////////////////////////////////////////////////////
		// Case1: Reduction variable is an array.                                 //
		////////////////////////////////////////////////////////////////////////////
		//     bid = blockIdx.x + blockIdx.y * gridDim.x;
		//     tid = threadIdx.x;
		//     if( tid == 0 ) {
		//         for( i=0; i<SIZE1; i++ ) {
		//             for( k=0; k<SIZE2; k++ ) {
		//                 x[bid][i][k] = sh__x[i][k][0];
		//             }
		//         }
		//     }
		////////////////////////////////////////////////////////////////////////////
		if( redArgSet.size() > 0 ) {
			ifBody = new CompoundStatement();
			_index = 0;
			for( VariableDeclarator arrayV_declarator : redArgSet ) {
				redOp = redOpSet.get(_index);
				Identifier array_var = new Identifier(arrayV_declarator);
				Identifier red_var = new Identifier(redParamSet.get(_index));
				List aspecs = arrayV_declarator.getArraySpecifiers();
				ArraySpecifier aspec = (ArraySpecifier)aspecs.get(0);
				int dimsize = aspec.getNumDimensions();
				edimensions = new LinkedList<Expression>();
				if( dimsize == 1 ) {
					assignex = new AssignmentExpression( new ArrayAccess((Identifier)red_var.clone(),
							(Identifier)bid.clone()), AssignmentOperator.NORMAL,
							new ArrayAccess((Identifier)array_var.clone(), new IntegerLiteral(0)));
					ifBody.addStatement(new ExpressionStatement(assignex));
				} else {
					for( int i=0; i<dimsize; i++ )
					{
						edimensions.add((Expression)aspec.getDimension(i).clone());
					}
					//////////////////////////////////////// //////
					// Create or find temporary index variables. //
					//////////////////////////////////////// //////
					List<Identifier> index_vars = new LinkedList<Identifier>();
					for( int i=0; i<dimsize-1; i++ ) {
						index_vars.add(TransformTools.getTempIndex(parallelRegion, i));
					}
					Identifier index_var = null;

					for( int i=dimsize-2; i>=0; i-- ) {
						index_var = index_vars.get(i);
						assignex = new AssignmentExpression((Identifier)index_var.clone(),
								AssignmentOperator.NORMAL, new IntegerLiteral(0));
						loop_init = new ExpressionStatement(assignex);
						condition = new BinaryExpression((Identifier)index_var.clone(),
								BinaryOperator.COMPARE_LT, (Expression)edimensions.get(i).clone());
						step = new UnaryExpression(UnaryOperator.POST_INCREMENT,
								(Identifier)index_var.clone());
						loop_body = new CompoundStatement();
						if( i == dimsize-2 ) {
							List<Expression> indices1 = new LinkedList<Expression>();
							List<Expression> indices2 = new LinkedList<Expression>();
							indices1.add((Identifier)bid.clone());
							for( int k=0; k<dimsize-1; k++ ) {
								indices1.add((Expression)index_vars.get(k).clone());
								indices2.add((Expression)index_vars.get(k).clone());
							}
							indices2.add(new IntegerLiteral(0));
							assignex = new AssignmentExpression(new ArrayAccess(
									(Identifier)red_var.clone(), indices1),
									AssignmentOperator.NORMAL, new ArrayAccess(
											(Identifier)array_var.clone(), indices2));
							loop_body.addStatement(new ExpressionStatement(assignex));
						} else {
							loop_body.addStatement(innerLoop);
						}
						innerLoop = new ForLoop(loop_init, condition, step, loop_body);
					}
					ifBody.addStatement(innerLoop);
				}
				_index++;
			}
			condition = new BinaryExpression((NameID)tid.clone(), BinaryOperator.COMPARE_EQ,
					new IntegerLiteral(0));
			parallelRegion.addStatement( new IfStatement(condition, ifBody) );
		}
	}

	/**
	 * For each parallel region, find all reduction variables used in the region and
	 * store them into the redVarMap, which has mapping of (a parent of a parallel region,
	 * set of reduction variables used in the parallel region).
	 * This mapping information is used to decide when shared data should be transferred
	 * between the CPU and the GPU; if any reduction variable is used, corresponding
	 * shared variable may need to be transferred at the end of the parallel region.
	 *
	 * @param omp_annots list of OpneMP annotations referring to parallel regions
	 */
	private static void findRedVarsPerPRegion(List<OmpAnnotation> omp_annots) {

		HashSet<Symbol> redVarSet = null;
		for (OmpAnnotation annot : (List<OmpAnnotation>)omp_annots)
		{
			Statement stmt = (Statement)annot.getAnnotatable();
			Statement parent = (Statement)stmt.getParent();
			if( redVarMap.containsKey(parent) ) {
				redVarSet = (HashSet<Symbol>)redVarMap.get(parent);
			} else {
				redVarSet = new HashSet<Symbol>();
				redVarMap.put(parent, redVarSet);
			}
			if( annot.keySet().contains("reduction") ) {
				HashMap reduction_map = (HashMap)annot.get("reduction");
				for (String ikey : (Set<String>)(reduction_map.keySet())) {
					redVarSet.addAll( (HashSet<Symbol>)reduction_map.get(ikey) );
				}
			}
			List<OmpAnnotation> ompfor_annots = IRTools.collectPragmas(stmt, OmpAnnotation.class, "for");
			for( OmpAnnotation fannot : (List<OmpAnnotation>)ompfor_annots ) {
				if( fannot != annot ) {
					if( fannot.keySet().contains("reduction") ) {
						HashMap reduction_map = (HashMap)fannot.get("reduction");
						for (String ikey : (Set<String>)(reduction_map.keySet())) {
							redVarSet.addAll( (HashSet<Symbol>)reduction_map.get(ikey) );
						}
					}
				}
			}
			// FIXME: if reduction clauses exist in a function called in a parallel region,
			// the reduction variables used in the function must be added to redVarSet too.
			// However, if a shared variable is passed to the function, and the shared variable
			// is used as reduction variable in the called function, the corresponding reduction
			// clause will contain corresponding formal parameter as a reduction variable.
			// The formal parameter should be replaced to the original shared variable to be
			// inserted into the redVarSet.
		}
	}

	/**
	 * Visit functions called in the parallel region recursively, and transform
	 * existing omp-for loops into if-statements.
	 * For each function, a new GPU function is created by replicating the function;
	 * the new GPU function will be called only by GPU.
	 * FIXME: if reduction clauses exist in a function called in the parallel region,
	 * necessary transformations should be conducted in this method, but not yet implemented.
	 *
	 * @param region the parallel region to be transformed to a kernel function.
	 * @param ispaces List containing the iteration-space sizes of omp-for loops
	 * @param parMap HashMap of the omp parallel annotation attached to the enclosing parallel region
	 * @param callerParamList Parameter list of a new kernel function that the current parallel region
	 *                          will be transformed to. This list contains all shared variables accessed
	 *                          within the parallel region.
	 */

	private static void interProcLoopTransform( CompoundStatement region, List<Expression> ispaces,
			HashMap parMap, List<VariableDeclaration> callerParamList) {
		//Set<Symbol> localSymbolsInPRegion = DataFlowTools.getDefSymbol(region);
		// DEBUG: it seems that the above statement is wrong; changed to the following.
		Set<Symbol> localSymbolsInPRegion = SymbolTools.getVariableSymbols(region);
		List<FunctionCall> funcCalls = IRTools.getFunctionCalls(region);
		boolean use_MallocPitch = opt_MallocPitch || opt_MatrixTranspose;
		TranslationUnit	currTU = IRTools.getParentTranslationUnit(region);
		//Procedure parentProc = IRTools.getParentProcedure(region);
		if( !(currTU instanceof TranslationUnit) ) {
			Tools.exit(pass_name + " Error in interProcLoopTransform(): Can not find the enclosing" +
					"TranslationUnit of the following kernel region: \n" + region + "\n");
		}
		///////////////////////////////////////////////////////////////
		// Extract Cuda directives attached to this parallel region. //
		///////////////////////////////////////////////////////////////
		HashSet<String> cudaRegisterSet = new HashSet<String>();
		HashSet<String> cudaRegisterROSet = new HashSet<String>();
		HashSet<String> cudaSharedSet = new HashSet<String>();
		HashSet<String> cudaSharedROSet = new HashSet<String>();
		HashSet<String> cudaTextureSet = new HashSet<String>();
		HashSet<String> cudaConstantSet = new HashSet<String>();
		List<CudaAnnotation> cudaAnnots = region.getAnnotations(CudaAnnotation.class);
		if( cudaAnnots != null ) {
			for( CudaAnnotation cannot : cudaAnnots ) {
				HashSet<String> dataSet = (HashSet<String>)cannot.get("registerRO");
				if( dataSet != null ) {
					cudaRegisterSet.addAll(dataSet);
					cudaRegisterROSet.addAll(dataSet);
				}
				dataSet = (HashSet<String>)cannot.get("registerRW");
				if( dataSet != null ) {
					cudaRegisterSet.addAll(dataSet);
				}
				dataSet = (HashSet<String>)cannot.get("sharedRO");
				if( dataSet != null ) {
					cudaSharedROSet.addAll(dataSet);
					cudaSharedSet.addAll(dataSet);
				}
				dataSet = (HashSet<String>)cannot.get("sharedRW");
				if( dataSet != null ) {
					cudaSharedSet.addAll(dataSet);
				}
				dataSet = (HashSet<String>)cannot.get("texture");
				if( dataSet != null ) {
					cudaTextureSet.addAll(dataSet);
				}
				dataSet = (HashSet<String>)cannot.get("constant");
				if( dataSet != null ) {
					cudaConstantSet.addAll(dataSet);
				}
			}

		}

		for( FunctionCall calledProc : funcCalls ) {
			// Skip any C standard library call.
			if( StandardLibrary.contains(calledProc) ) {
				if( !CudaStdLibrary.contains(calledProc) ) {
					PrintTools.println(pass_name + " WARNING: C standard library function ("+calledProc.getName()+
						"is called in a kernel region, but not supported by CUDA runtime system V1.1; " +
						"it may cause compilation error if not inlinable.", 0);
				}
				continue;
			}
			Procedure tProc = calledProc.getProcedure();
			List<Expression> argList = (List<Expression>)calledProc.getArguments();
			int list_size = argList.size();
			boolean gtidIsUsed = false;
			if( tProc != null ) {
				PrintTools.println(pass_name + "interProcLoopTransform() handles " + tProc.getSymbolName(), 9);
				/*
				 * Create a new device function by replicating the procedure, tProc.
				 */
				CompoundStatement body = tProc.getBody();
				////////////////////////////////////////////////////////////////////////////////
				// Check whether the function called in a kernel region contains static data. //
				// CUDA does not allow static data inside any kernel functions.               //
				////////////////////////////////////////////////////////////////////////////////
				if( AnalysisTools.ipaStaticDataCheck(body) ) {
					Tools.exit(pass_name + " [Error in interProcLoopTransform()] " +
					" a function, " + tProc.getName() + ", is called in a parallel region, which " +
					"will be transformed into a kernel function, but this function contains static " +
					"data, which are not allowd in the CUDA programming model.");
				}
				TranslationUnit tu = (TranslationUnit)tProc.getParent();
				//Annotation cudaAnnot = (Annotation)tu.getChildren().get(0);
				//HashMap c2gMap = (HashMap)cudaAnnot.getMap();
				//String new_func_name = "dev_"+tProc.getName().toString() + "_" + parentProc.getSymbolName();
				String new_func_name = "dev_"+tProc.getName().toString();
				//////////////////////////////////////////////////////////////////////////////////////
				// FIXME: If the new device function is defined in a file, which is not the current //
				// file, extern statement should be added for this new device function.             //
				// Because nvcc v1.1 does not support inlining of external functions, we don't have //
				// to implement this for now.                                                       //
				//////////////////////////////////////////////////////////////////////////////////////
				if( !tu.equals(currTU) ) {

				}
				/*
				 * Check whether this device function has cetus-for loops or not.
				 * If the function has any cetus-for loop, variable _gtid should be passed as an argument
				 * for cetus-for loop translation.
				 */
				List<OmpAnnotation> omp_annots = IRTools.collectPragmas(body, OmpAnnotation.class, "for");
				/*
				 * Create a functionCall for the new device function,
				 */
				FunctionCall gpu_funcCall = new FunctionCall(new NameID(new_func_name));
				//HashSet<Symbol> argSyms = new HashSet<Symbol>();
				HashSet<Symbol> OmpSharedSet = null;
				HashSet<Symbol> OmpThreadPrivSet = null;
				HashSet<Symbol> OmpSet = new HashSet<Symbol>();
				if (parMap.keySet().contains("shared")) {
					OmpSharedSet = (HashSet<Symbol>) parMap.get("shared");
					OmpSet.addAll(OmpSharedSet);
				}
				if (parMap.keySet().contains("threadprivate")) {
					OmpThreadPrivSet = (HashSet<Symbol>) parMap.get("threadprivate");
					OmpSet.addAll(OmpThreadPrivSet);
				}
				for(int i=0; i<list_size; i++) {
					boolean tempVarUsed = false;
					boolean isTPVar = false;
					Expression tempExp = null;
					Identifier pitch = null;
					Expression arg = (Expression)argList.get(i).clone();
					gpu_funcCall.addArgument(arg);
					Set<Expression> UseSet = DataFlowTools.getUseSet(arg);
					for( Expression exp : UseSet) {
						/////////////////////////////////////////////////////////////////
						// FIXME: DataFlowTools.getUseSet(arg) returns ArrayAccess and //
						// index expressions if are is ArrayAccess.                    //
						// Ex: DataFlowsTools.getUseSet(a[i]) returns a[i] and i.      //
						// Therefore, if the below statement assumes that array ID is  //
						// returned by getUseSet() method, the statement will not be   //
						// executed.                                                   //
						// CAUTION: AnalysisTools.containsSymbol() used below assumes  //
						// that exp is instanceof Identifier.                          //
						/////////////////////////////////////////////////////////////////
						if( exp instanceof Identifier ) {
							if( tempMap.containsKey((Identifier)exp) ) {
								tempExp = (Expression)tempMap.get(exp).clone();
								IRTools.replaceAll(arg, exp, tempExp);
								tempVarUsed = true;
							}
							Symbol sm = SymbolTools.getSymbolOf(exp);
							if( sm != null ) {
								//argSyms.add(sm);
								if( AnalysisTools.containsSymbol(OmpThreadPrivSet, exp.toString()) &&
										!AnalysisTools.containsSymbol(localSymbolsInPRegion, exp.toString()) ) {
									isTPVar = true;
									if(SymbolTools.isArray(sm)) {
										List aspecs = sm.getArraySpecifiers();
										ArraySpecifier aspec = (ArraySpecifier)aspecs.get(0);
										////////////////////////////////////////////////////////////////////////
										// Constant 2 should be used below because symbol sm refers to a      //
										// parameter of enclosing kernel function, which is already expanded. //
										////////////////////////////////////////////////////////////////////////
										if( aspec.getNumDimensions() > 2 ) {
											gtidIsUsed = true;
										}
									}
								}
								if( use_MallocPitch && pitchMap.containsKey(sm) ) {
									VariableDeclaration pitch_decl = pitchMap.get(sm);
									pitch = new Identifier((VariableDeclarator)pitch_decl.getDeclarator(0));
									gpu_funcCall.addArgument(pitch);
								}
							}
						}
					}
					//////////////////////////////////////////////////////////////////////////////////////
					// If an agrument, arg, contains one-dimensional Threadprivate array, x, and if the //
					// following conditins hold:                                                        //
					//     - Threadprivate array x is expanded using MatrixTranspose optimization.      //
					//     - the argument, arg, has the form of (x op exp)                              //
					// then, (x op exp) should be converted to the form of (float *)((char*)x_0 op      //
					//        exp*pitch_x))                                                             //
					//        where x_0 is a temporary variable generated in pitchedAccessConv2() and   //
					//        x_0 = (float*)(x + _gtid);                                                //
					//////////////////////////////////////////////////////////////////////////////////////
					// Below condition is true only if ThreadPrivate array x is convertedy by  //
					// pitchedAccessConv2() function.                                          //
					/////////////////////////////////////////////////////////////////////////////
					if( tempVarUsed && isTPVar && (pitch != null) ) {
						if( arg instanceof BinaryExpression ) {
							BinaryExpression biexp = (BinaryExpression)arg;
							BinaryOperator op = biexp.getOperator();
							Expression lhs = biexp.getLHS();
							Expression rhs = biexp.getRHS();
							Expression dispExp = null;
							if( tempExp.equals(lhs) ) {
								dispExp = rhs;
							} else if( tempExp.equals(rhs) ) {
								dispExp = lhs;
							}
							if( dispExp == null ) {
								PrintTools.println(pass_name+ " [WARNING] argument expression (" + arg + ") in " +
										"a function call of "+ calledProc + " is too complex to ananlyze;" +
										"the converted argument expression may be incorrect.",0);
							} else {
								List<Specifier> specs = new ArrayList<Specifier>(2);
								specs.add(Specifier.CHAR);
								specs.add(PointerSpecifier.UNQUALIFIED);
								Typecast tcast1 = new Typecast(specs, (Expression)tempExp.clone());
								BinaryExpression biexp1 = new BinaryExpression((Expression)dispExp.clone(),
										BinaryOperator.MULTIPLY, (Identifier)pitch.clone());
								BinaryExpression biexp2 = new BinaryExpression(tcast1, op, biexp1);
								List<Specifier> specs2 = new ArrayList<Specifier>();
								/////////////////////////////////////////////////////////////////////////////////////
								// CAUTION: VariableDeclarator.getTypeSpecifiers() returns both specifiers of      //
								// its parent VariableDeclaration and the VariableDeclarator's leading specifiers. //
								// Therefore, if VariableDeclarator is a pointer symbol, this method will return   //
								// pointer specifiers too.                                                         //
								/////////////////////////////////////////////////////////////////////////////////////
								specs2.addAll(((Identifier)tempExp).getSymbol().getTypeSpecifiers());
								specs2.remove(Specifier.STATIC);
								/////////////////////////////////////////////////////////////////////////
								// We don't need to add pointer specifier because tempExp.getSymbol(). //
								// getTypeSpecifiers() will include pointer specifier already.         //
								/////////////////////////////////////////////////////////////////////////
								//specs2.add(PointerSpecifier.UNQUALIFIED);
								Typecast tcast2 = new Typecast(specs2, biexp2);
								arg.swapWith(tcast2);
							}
						} else if( !(arg instanceof Identifier) ) {
							PrintTools.println(pass_name+ " [WARNING] argument expression (" + arg + ") in " +
									"a function call of "+ calledProc + " is too complex to ananlyze;" +
									"the converted argument expression may be incorrect.",0);
						}
					}

				}
				/*
				 * If this called function accesses shared/threadprivate variables that are not passed
				 * as function arguments, add those shared/threadprivate variables as arguments.
				 */
				HashSet<Symbol> callerParamSet = new HashSet<Symbol>();
				for( VariableDeclaration v_decl : callerParamList ) {
					callerParamSet.add((VariableDeclarator)v_decl.getDeclarator(0));
				}
				List<VariableDeclaration> oldParamList = (List<VariableDeclaration>)tProc.getParameters();
				HashSet<Symbol> oldParamSet = new HashSet<Symbol>();
				for( VariableDeclaration v_decl : oldParamList ) {
					oldParamSet.add((VariableDeclarator)v_decl.getDeclarator(0));
				}
				Set<Symbol> accessedSymbols = SymbolTools.getAccessedSymbols(tProc.getBody());
				Set<Symbol> localSymbols = DataFlowTools.getDefSymbol(tProc.getBody());
				for( Symbol ssm : OmpSet ) {
					VariableDeclarator symVar = (VariableDeclarator)ssm;
					Identifier ssmID = new Identifier(symVar);
					String symName = ssm.getSymbolName();
/*					if( !OmpAnalysis.containsSymbol(argSyms, symName) &&
							!OmpAnalysis.containsSymbol(oldParamSet, symName) &&*/
					if( !AnalysisTools.containsSymbol(oldParamSet, symName) &&
							!AnalysisTools.containsSymbol(localSymbols, symName) &&
							AnalysisTools.containsSymbol(accessedSymbols, symName) ) {
						if( use_MallocPitch && AnalysisTools.containsSymbol(pitchMap.keySet(),symName) ) {
							VariableDeclarator var_declarator =
								(VariableDeclarator)AnalysisTools.findsSymbol(pitchMap.keySet(), symName);
							Identifier var_ID = new Identifier(var_declarator);
							if( tempMap.containsKey(var_ID) ) {
								gpu_funcCall.addArgument((Expression)tempMap.get(var_ID).clone());
							} else {
								gpu_funcCall.addArgument((Expression)var_ID.clone());
							}
							VariableDeclaration pitch_decl = pitchMap.get(var_declarator);
							gpu_funcCall.addArgument(new Identifier((VariableDeclarator)pitch_decl.getDeclarator(0)));
						} else {
							VariableDeclarator cparam_declarator =
								(VariableDeclarator)AnalysisTools.findsSymbol(callerParamSet, symName);
							if( cparam_declarator == null ) {
								Tools.exit(pass_name + "[Error] "+symName.toString()+
										"does not exist in the kernel function parameters");
							} else {
								Identifier var_ID = new Identifier((VariableDeclarator)cparam_declarator);
								if( tempMap.containsKey(var_ID) ) {
									gpu_funcCall.addArgument((Expression)tempMap.get(var_ID).clone());
								} else {
									gpu_funcCall.addArgument((Expression)var_ID.clone());
								}
							}
						}
						if( AnalysisTools.containsSymbol(OmpThreadPrivSet, symName) ) {
							if(SymbolTools.isArray(ssm)) {
								List aspecs = ssm.getArraySpecifiers();
								ArraySpecifier aspec = (ArraySpecifier)aspecs.get(0);
								if( aspec.getNumDimensions() > 1 ) {
									gtidIsUsed = true;
								}
							} else if( SymbolTools.isScalar(ssm)) {
								gtidIsUsed = true;
							}
						}
					}
				}
				/*
				 * If ThreadPrivate array is used in the called function or if this called function
				 * contains any cetus-for loop, _gtid variable shoulde be
				 * passed from the parent kernel function to the child called function.
				 */
				if( gtidIsUsed || (omp_annots.size() > 0) ) {
					gpu_funcCall.addArgument( SymbolTools.getOrphanID("_gtid"));
				}

				//if( !c2gMap.containsKey(new_func_name) ) {
				//////////////////////////////////////////////////////////////
				// Check whether a function with the given name exists in a //
				// given translation unit.                                  //
				//////////////////////////////////////////////////////////////
				boolean func_exist = false;
				BreadthFirstIterator itr = new BreadthFirstIterator(tu);
				itr.pruneOn(Procedure.class);
				for (;;)
				{
					Procedure proc = null;

					try {
						proc = (Procedure)itr.next(Procedure.class);
					} catch (NoSuchElementException e) {
						break;
					}
					if( proc.getSymbolName().equals(new_func_name) ) {
						func_exist = true;
						break;
					}
				}
				if( !func_exist ) {
					/*
					 * Create a new device function by replicating the procedure, tProc.
					 */
					List<Specifier> new_proc_ret_type = new LinkedList<Specifier>();
					new_proc_ret_type.add(CUDASpecifier.CUDA_DEVICE);
					List<Specifier> return_types = tProc.getReturnType();
					new_proc_ret_type.addAll(return_types);
					body = (CompoundStatement)tProc.getBody().clone();
					TransformTools.updateAnnotationsInRegion(body);
					Procedure new_proc = new Procedure(new_proc_ret_type,
							new ProcedureDeclarator(new NameID(new_func_name),
									new LinkedList()), body);
					// Calculate iteration sizes of omp-for loops before they are modified.
					TransformTools.calcLoopItrSize(body, null);
					for(int i=0; i<list_size; i++) {
						VariableDeclaration param_decl = (VariableDeclaration)oldParamList.get(i);
						VariableDeclarator param_declarator = (VariableDeclarator)param_decl.getDeclarator(0);
						Identifier paramID = new Identifier(param_declarator);
						Expression arg = argList.get(i);
						if( arg instanceof Identifier ) {
							VariableDeclarator sm = (VariableDeclarator)SymbolTools.getSymbolOf(arg);
							if( sm == null ) {
								PrintTools.println(pass_name + "[WARNING] Unexpected argument, " + arg.toString() +
										" is used for function, " + calledProc.getName().toString() +
										"; translated new function, " + new_proc.getSymbolName() +
										", may be incorrect.", 0);
								VariableDeclaration cloned_decl = (VariableDeclaration)param_decl.clone();
								Identifier cloned_ID = new Identifier((VariableDeclarator)cloned_decl.getDeclarator(0));
								new_proc.addDeclaration(cloned_decl);
								// Replace all instances of the shared variable to the parameter variable
								IRTools.replaceAll((Traversable) body, paramID, cloned_ID);
							} else {
								boolean isSharedVar = AnalysisTools.containsSymbol(OmpSharedSet, sm.getSymbolName()) &&
								!AnalysisTools.containsSymbol(localSymbolsInPRegion, arg.toString());
								boolean isTPVar = AnalysisTools.containsSymbol(OmpThreadPrivSet, sm.getSymbolName()) &&
								!AnalysisTools.containsSymbol(localSymbolsInPRegion, arg.toString());
								if( use_MallocPitch && pitchMap.containsKey(sm) ) {
									VariableDeclaration pitch_decl = (VariableDeclaration)pitchMap.get(sm).clone();
									if( isSharedVar ) {
										pitchedAccessConv(param_declarator, new_proc, pitch_decl, body);
									} else if( isTPVar ){
										pitchedAccessConv3(param_declarator, new_proc, pitch_decl, body);
									} else {
										Tools.exit(pass_name + "[ERROR] interProcLoopTransform() found wrong symbol, "
												+ sm.toString() + " in the pitchMap.");
									}
								} else {
									if( isTPVar && SymbolTools.isArray(param_declarator) ) {
										CreateExtendedArray2(param_declarator, new_proc, body);
									} else {
										VariableDeclaration cloned_decl = (VariableDeclaration)param_decl.clone();
										Identifier cloned_ID = new Identifier((VariableDeclarator)cloned_decl.getDeclarator(0));
										new_proc.addDeclaration(cloned_decl);
										// Replace all instances of the shared variable to the parameter variable
										IRTools.replaceAll((Traversable) body, paramID, cloned_ID);
									}
								}
							}
						} else if ( arg instanceof BinaryExpression ) {
							Set<Expression> UseSet = DataFlowTools.getUseSet(arg);
							boolean isConverted = false;
							for( Expression exp : UseSet ) {
								if( exp instanceof Identifier ) {
									VariableDeclarator sm = (VariableDeclarator)SymbolTools.getSymbolOf(exp);
									if( sm == null ) {
											PrintTools.println(pass_name + "[WARNING] Unrecognizable expression, " + exp.toString() +
													", in " + arg.toString() + " is used for function, " +
													calledProc.getName().toString() +  "; translated new function, " +
													new_proc.getSymbolName() + ", may be incorrect.", 0);
									} else {
										boolean isSharedVar = AnalysisTools.containsSymbol(OmpSharedSet, sm.getSymbolName()) &&
										!AnalysisTools.containsSymbol(localSymbolsInPRegion, exp.toString());
										boolean isTPVar = AnalysisTools.containsSymbol(OmpThreadPrivSet, sm.getSymbolName()) &&
										!AnalysisTools.containsSymbol(localSymbolsInPRegion, exp.toString());
										if( use_MallocPitch && pitchMap.containsKey(sm) ) {
											VariableDeclaration pitch_decl = (VariableDeclaration)pitchMap.get(sm).clone();
											if( isSharedVar ) {
												pitchedAccessConv(param_declarator, new_proc, pitch_decl, body);
												isConverted = true;
												break;
											} else if( isTPVar ){
												pitchedAccessConv3(param_declarator, new_proc, pitch_decl, body);
												isConverted = true;
												break;
											} else {
												Tools.exit(pass_name + "[ERROR] interProcLoopTransform() found wrong symbol, "
														+ sm.toString() + " in the pitchMap.");
											}
										} else {
											if( isTPVar && SymbolTools.isArray(param_declarator) ) {
												CreateExtendedArray2(param_declarator, new_proc, body);
												isConverted = true;
												break;
											}
										}
									}
								}
							}
							if( !isConverted ) {
								VariableDeclaration cloned_decl = (VariableDeclaration)param_decl.clone();
								Identifier cloned_ID = new Identifier((VariableDeclarator)cloned_decl.getDeclarator(0));
								new_proc.addDeclaration(cloned_decl);
								// Replace all instances of the shared variable to the parameter variable
								IRTools.replaceAll((Traversable) body, paramID, cloned_ID);
							}
						} else {
							if( !((arg instanceof Literal) || (arg instanceof UnaryExpression)
									|| (arg instanceof ArrayAccess)) ) {
								PrintTools.println(pass_name + "[WARNING] Unrecognizable argument, " + arg.toString() +
										" is used for function, " + calledProc.getName().toString() +
										"; translated new function, " + new_proc.getSymbolName() +
										", may be incorrect.", 0);
							}
							VariableDeclaration cloned_decl = (VariableDeclaration)param_decl.clone();
							Identifier cloned_ID = new Identifier((VariableDeclarator)cloned_decl.getDeclarator(0));
							new_proc.addDeclaration(cloned_decl);
							// Replace all instances of the shared variable to the parameter variable
							IRTools.replaceAll((Traversable) body, paramID, cloned_ID);
						}
					}
				/*
					 * If this called function accesses shared variables that are not passed as
					 * function arguments, add those shared variables as function parameters.
					 */
					for( Symbol ssm : OmpSet ) {
						VariableDeclarator symVar = (VariableDeclarator)ssm;
						Identifier ssmID = new Identifier(symVar);
						String symName = ssm.getSymbolName();
/*						if( !OmpAnalysis.containsSymbol(argSyms, symName) &&
							!OmpAnalysis.containsSymbol(oldParamSet, symName) &&*/
						if( !AnalysisTools.containsSymbol(oldParamSet, symName) &&
							!AnalysisTools.containsSymbol(localSymbols, symName) &&
								AnalysisTools.containsSymbol(accessedSymbols, symName) ) {
							boolean isSharedVar = AnalysisTools.containsSymbol(OmpSharedSet, symName);
							boolean isTPVar = AnalysisTools.containsSymbol(OmpThreadPrivSet, symName);
							if( use_MallocPitch && AnalysisTools.containsSymbol(pitchMap.keySet(),symName) ) {
								VariableDeclarator var_declarator =
									(VariableDeclarator)AnalysisTools.findsSymbol(pitchMap.keySet(), symName);
								VariableDeclaration pitch_decl =
									(VariableDeclaration)pitchMap.get(var_declarator).clone();
								if( isSharedVar ) {
									pitchedAccessConv(symVar, new_proc, pitch_decl, body);
								} else if( isTPVar ) {
									pitchedAccessConv3(symVar, new_proc, pitch_decl, body);
								}
							} else {
								if( SymbolTools.isScalar(ssm) ) {
									// Create a parameter Declaration for the device function
									// Change the scalar variable to a pointer type
									boolean useRegister = false;
									boolean ROData = false;
									if( cudaRegisterSet.contains(symName)) {
										useRegister = true;
									}
									if( cudaRegisterROSet.contains(symName)) {
										ROData = true;
									}
									if( isSharedVar ) {
										scalarVariableConv(symVar, new_proc, body, useRegister, ROData);
									}
									else if( isTPVar ) {
										scalarVariableConv2(symVar, new_proc, body, useRegister, ROData);
									} else {
										Tools.exit(pass_name + "[ERROR] interProcLoopTransform() found unknown symbol, "
												+ ssm.toString());
									}
								} else if( SymbolTools.isArray(ssm) ) {
									if( isSharedVar ) {
										VariableDeclaration decl =
											(VariableDeclaration)symVar.getParent();
										VariableDeclarator cloned_declarator =
											(VariableDeclarator)((VariableDeclarator)symVar).clone();
										cloned_declarator.setInitializer(null);
										//////////////////////////////////////////////////
										// Kernel function parameter can not be static. //
										//////////////////////////////////////////////////
										List<Specifier> clonedspecs = new ChainedList<Specifier>();
										clonedspecs.addAll(decl.getSpecifiers());
										clonedspecs.remove(Specifier.STATIC);
										VariableDeclaration cloned_decl = new VariableDeclaration(clonedspecs, cloned_declarator);
										Identifier clonedID = new Identifier(cloned_declarator);
										new_proc.addDeclaration(cloned_decl);
										// Replace all instances of the shared variable to the local variable
										IRTools.replaceAll((Traversable) body, ssmID, clonedID);
									} else if ( isTPVar ) {
										CreateExtendedArray2(symVar, new_proc, body);
									} else {
										Tools.exit(pass_name + "[ERROR] interProcLoopTransform() found unknown symbol, "
												+ ssm.toString());
									}
								} else {
									Tools.exit(pass_name + "[ERROR] interProcLoopTransform() found unsupported shared symbol, "
											+ ssm.toString());
								}
							}
						}
					}
					/*
					 * If ThreadPrivate array is used in the called function or if this called function
					 * contains any cetus-for loop, _gtid variable shoulde be
					 * passed from the parent kernel function to the child called function.
					 */
					omp_annots = IRTools.collectPragmas(body, OmpAnnotation.class, "for");
					Identifier gtid = null;
					if( gtidIsUsed || (omp_annots.size() > 0) ) {
						VariableDeclarator gtid_declarator = new VariableDeclarator(new NameID("_gtid"));
						gtid = new Identifier(gtid_declarator);
						VariableDeclaration gtid_decl = new VariableDeclaration(Specifier.INT, gtid_declarator);
						new_proc.addDeclaration(gtid_decl);
					}

					/* put new_proc before the procedure, tProc */
					tu.addDeclarationBefore(tProc, new_proc);

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
						if( procDeclr.getID().equals(tProc.getName()) ) {
							//Found function declaration.
							Declaration procDecl = (Declaration)procDeclr.getParent();
							//Create a new function declaration.
							VariableDeclaration newProcDecl =
								new VariableDeclaration(new_proc.getReturnType(), new_proc.getDeclarator().clone());
							//Insert the new function declaration.
							tu.addDeclarationAfter(procDecl, newProcDecl);
							break;
						}
					}

					///////////////////////////////////////////////////////////////////////////////////
					// FIXME: The below method does not update symbol pointers of Identifiers in the //
					// new procedure, and thus any symbol-related operation may fail.                //
					///////////////////////////////////////////////////////////////////////////////////
					TransformTools.updateAnnotationsInRegion(new_proc);
					c2gMap.put(new_func_name, new_proc);
					if( omp_annots.size() > 0 ) {
						for ( OmpAnnotation fannot : omp_annots ) {
							Statement target_stmt = (Statement)fannot.getAnnotatable();
							if( target_stmt instanceof ForLoop ) {
								ForLoop ploop = (ForLoop)target_stmt;
								if ( !LoopTools.isCanonical(ploop) ) {
									Tools.exit(pass_name + "[Error in interProcLoopTransformation()] Parallel Loop is not " +
											"a canonical loop; compiler can not determine iteration space of " +
											"the following loop: \n" + ploop);
								}
								Expression ivar = LoopTools.getIndexVariable(ploop);
								Expression lb = LoopTools.getLowerBoundExpression(ploop);
								//Expression ub = LoopTools.getUpperBoundExpression(ploop);
								//Expression iterspace =
								//	Symbolic.add(Symbolic.subtract(ub,lb),new IntegerLiteral(1));
								Expression iterspace = (Expression)fannot.remove("iterspace");
								if( iterspace == null ) {
									Tools.exit(pass_name + " Error in interProcLoopTransform(): annotation map for " +
											"an omp-for loop does not contain itersapce.");
								}
								ispaces.add(iterspace);
								BinaryExpression biexp3 = new BinaryExpression(gtid, BinaryOperator.ADD, lb);
								AssignmentExpression assgn = new AssignmentExpression(ivar,
										AssignmentOperator.NORMAL, biexp3);
								Statement thrmapstmt = new ExpressionStatement(assgn);
								CompoundStatement parentStmt = (CompoundStatement)target_stmt.getParent();
								parentStmt.addStatementBefore(ploop, thrmapstmt);
								/*
								 * Replace the omp-for loop with if-statement containing the loop body
								 */
								IfStatement ifstmt = new IfStatement((Expression)ploop.getCondition().clone(),
										(CompoundStatement)ploop.getBody().clone());
								ploop.swapWith(ifstmt);
								///////////////////////////////////////////////////////////////////////////
								// CAUTION: Omp for annotation will be inserted to the new if statement, //
								// but this insertion violates OpenMP semantics.                         //
								///////////////////////////////////////////////////////////////////////////
								//an_stmt.attachStatement(ifstmt);
								ifstmt.annotate(fannot);
							}
						}
					}
				} else {
					//Just check iteration space sizes of included omp-for-loops.
					//Procedure new_proc = (Procedure)c2gMap.get(new_func_name);
					//body = new_proc.getBody();
					omp_annots = IRTools.collectPragmas(body, OmpAnnotation.class, "for");
					if( omp_annots.size() > 0 ) {
						for ( OmpAnnotation annot : omp_annots ) {
							Statement target_stmt = (Statement)annot.getAnnotatable();
							if( target_stmt instanceof ForLoop ) {
								ForLoop ploop = (ForLoop)target_stmt;
								if ( !LoopTools.isCanonical(ploop) ) {
									Tools.exit(pass_name + "[Error in interProcLoopTransformation()] Parallel Loop is not " +
											"a canonical loop; compiler can not determine iteration space of " +
											"the following loop: \n" + ploop);
								}
								Expression ivar = LoopTools.getIndexVariable(ploop);
								Expression lb = LoopTools.getLowerBoundExpression(ploop);
								Expression ub = LoopTools.getUpperBoundExpression(ploop);
								Expression iterspace =
									Symbolic.add(Symbolic.subtract(ub,lb),new IntegerLiteral(1));
								ispaces.add(iterspace);
							}
						}
					}
				}
				/*
				 * Swap old functionCall with the new device function.
				 */
				calledProc.swapWith(gpu_funcCall);
				//tempMap.clear();
				/*
				 *  FIXME: Below recursive call is incorrect.
				 */
				//interProcLoopTransform(body, ispaces, pitchMap, parMap, callerParamList);
			}
		}
	}

	/**
	 * Visit functions called in the omp parallel for body recursively, and create device functions
	 * equivalent to the called functions.
	 * FIXME: if reduction clauses exist in a function called in the parallel region,
	 * necessary transformations should be conducted in this method, but not yet implemented.
	 *
	 * @param loopbody the parallel region to be transformed to a kernel function.
	 * @param parMap HashMap of the omp parallel annotation attached to the enclosing parallel region
	 * @param callerParamList Parameter list of a new kernel function that the current parallel region
	 *                          will be transformed to. This list contains all shared variables accessed
	 *                          within the parallel region.
	 */
	private static void deviceFuncTransform( CompoundStatement loopbody,
			HashMap parMap, List<VariableDeclaration> callerParamList) {
		List<FunctionCall> funcCalls = IRTools.getFunctionCalls(loopbody);
		boolean use_MallocPitch = opt_MallocPitch || opt_MatrixTranspose;
		//Procedure parentProc = IRTools.getParentProcedure(loopbody);

		for( FunctionCall calledProc : funcCalls ) {
			// Skip any C standard library call.
			if( StandardLibrary.contains(calledProc) ) {
				if( !CudaStdLibrary.contains(calledProc) ) {
					PrintTools.println(pass_name + " WARNING: C standard library function ("+calledProc.getName()+
						"is called in a kernel region, but not supported by CUDA runtime system V1.1; " +
						"it may cause compilation error if not inlinable.", 0);
				}
				continue;
			}
			Procedure tProc = calledProc.getProcedure();
			List<Expression> argList = (List<Expression>)calledProc.getArguments();
			int list_size = argList.size();
			boolean gtidIsUsed = false;
			if( tProc != null ) {
				PrintTools.println(pass_name + " deviceFuncTransform() handles " + tProc.getSymbolName(), 9);
				/*
				 * Create a new device function by replicating the procedure, tProc.
				 */
				CompoundStatement body = tProc.getBody();
				////////////////////////////////////////////////////////////////////////////////
				// Check whether the function called in a kernel region contains static data. //
				// CUDA does not allow static data inside any kernel functions.               //
				////////////////////////////////////////////////////////////////////////////////
				if( AnalysisTools.ipaStaticDataCheck(body) ) {
					Tools.exit(pass_name + " [Error in deviceFuncTransform()] " +
					" a function, " + tProc.getName() + ", is called in a parallel region, which " +
					"will be transformed into a kernel function, but this function contains static " +
					"data, which are not allowd in the CUDA programming model.");
				}
				TranslationUnit tu = (TranslationUnit)tProc.getParent();
				//Annotation cudaAnnot = (Annotation)tu.getChildren().get(0);
				//HashMap c2gMap = (HashMap)cudaAnnot.getMap();
				//String new_func_name = "dev_"+tProc.getName().toString() + "_" + parentProc.getSymbolName();
				String new_func_name = "dev_"+tProc.getName().toString();
				//////////////////////////////////////////////////////////////////////////////////////
				// FIXME: If the new device function is defined in a file, which is not the current //
				// file, extern statement should be added for this new device function.             //
				// Because nvcc v1.1 does not support inlining of external functions, we don't have //
				// to implement this for now.                                                       //
				//////////////////////////////////////////////////////////////////////////////////////
				/*
				 * Check whether this device function has cetus-for loops or not.
				 * Because this device function is called within other cetus-for loop,
				 * this function should not have any cetus-for loop.
				 */
				List<OmpAnnotation> omp_annots =
					IRTools.collectPragmas(body, OmpAnnotation.class, "for");
				if( omp_annots.size() > 0 ) {
					Tools.exit(pass_name + "[Error in deviceFuncTransform()] Found omp-for loop(s) in a function, "
							+ tProc.getName().toString() + ", which is called in another omp-for loop.");
				}
				/*
				 * Create a functionCall for the new device function,
				 */
				FunctionCall gpu_funcCall = new FunctionCall(new NameID(new_func_name));
				HashSet<Symbol> argSyms = new HashSet<Symbol>();
				HashSet<Symbol> OmpSharedSet = null;
				HashSet<Symbol> OmpThreadPrivSet = null;
				HashSet<Symbol> OmpSet = new HashSet<Symbol>();
				if (parMap.keySet().contains("shared")) {
					OmpSharedSet = (HashSet<Symbol>) parMap.get("shared");
					OmpSet.addAll(OmpSharedSet);
				}
				if (parMap.keySet().contains("threadprivate")) {
					OmpThreadPrivSet = (HashSet<Symbol>) parMap.get("threadprivate");
					OmpSet.addAll(OmpThreadPrivSet);
				}
				for(int i=0; i<list_size; i++) {
					boolean tempVarUsed = false;
					boolean isTPVar = false;
					Expression tempExp = null;
					Identifier pitch = null;
					Expression arg = (Expression)argList.get(i).clone();
					gpu_funcCall.addArgument(arg);
					Set<Expression> UseSet = DataFlowTools.getUseSet(arg);
					for( Expression exp : UseSet) {
						/////////////////////////////////////////////////////////////////
						// FIXME: DataFlowTools.getUseSet(arg) returns ArrayAccess and //
						// index expressions if are is ArrayAccess.                    //
						// Ex: DataFlowsTools.getUseSet(a[i]) returns a[i] and i.      //
						// Therefore, if the below statement assumes that array ID is  //
						// returned by getUseSet() method, the statement will not be   //
						// executed.                                                   //
						// CAUTION: AnalysisTools.containsSymbol() used below assumes  //
						// that exp is instanceof Identifier.                          //
						/////////////////////////////////////////////////////////////////
						if( exp instanceof Identifier ) {
							if( tempMap.containsKey((Identifier)exp) ) {
								tempExp = (Expression)tempMap.get(exp).clone();
								IRTools.replaceAll(arg, exp, tempExp);
								tempVarUsed = true;
							}
							Symbol sm = SymbolTools.getSymbolOf(exp);
							if( sm != null ) {
								//argSyms.add(sm);
								if( AnalysisTools.containsSymbol(OmpThreadPrivSet, exp.toString()) ) {
									isTPVar = true;
									if(SymbolTools.isArray(sm)) {
										List aspecs = sm.getArraySpecifiers();
										ArraySpecifier aspec = (ArraySpecifier)aspecs.get(0);
										////////////////////////////////////////////////////////////////////////
										// Constant 2 should be used below because symbol sm refers to a      //
										// parameter of enclosing kernel function, which is already expanded. //
										////////////////////////////////////////////////////////////////////////
										if( aspec.getNumDimensions() > 2 ) {
											gtidIsUsed = true;
										}
									}
								}
								if( use_MallocPitch && pitchMap.containsKey(sm) ) {
									VariableDeclaration pitch_decl = pitchMap.get(sm);
									pitch = new Identifier((VariableDeclarator)pitch_decl.getDeclarator(0));
									gpu_funcCall.addArgument(pitch);
								}
							}
						}
					}
					//////////////////////////////////////////////////////////////////////////////////////
					// If an agrument, arg, contains one-dimensional Threadprivate array, x, and if the //
					// following conditins hold:                                                        //
					//     - Threadprivate array x is expanded using MatrixTranspose optimization.      //
					//     - the argument, arg, has the form of (x op exp)                              //
					// then, (x op exp) should be converted to the form of (float *)((char*)x_0 op      //
					//        exp*pitch_x))                                                             //
					//        where x_0 is a temporary variable generated in pitchedAccessConv2() and   //
					//        x_0 = (float*)(x + _gtid);                                                //
					//////////////////////////////////////////////////////////////////////////////////////
					// Below condition is true only if ThreadPrivate array x is convertedy by  //
					// pitchedAccessConv2() function.                                          //
					/////////////////////////////////////////////////////////////////////////////
					if( tempVarUsed && isTPVar && (pitch != null) ) {
						if( arg instanceof BinaryExpression ) {
							BinaryExpression biexp = (BinaryExpression)arg;
							BinaryOperator op = biexp.getOperator();
							Expression lhs = biexp.getLHS();
							Expression rhs = biexp.getRHS();
							Expression dispExp = null;
							if( tempExp.equals(lhs) ) {
								dispExp = rhs;
							} else if( tempExp.equals(rhs) ) {
								dispExp = lhs;
							}
							if( dispExp == null ) {
								PrintTools.println(pass_name+ " [WARNING] argument expression (" + arg + ") in " +
										"a function call of "+ calledProc + " is too complex to ananlyze;" +
										"the converted argument expression may be incorrect.",0);
							} else {
								List<Specifier> specs = new ArrayList<Specifier>(2);
								specs.add(Specifier.CHAR);
								specs.add(PointerSpecifier.UNQUALIFIED);
								Typecast tcast1 = new Typecast(specs, (Expression)tempExp.clone());
								BinaryExpression biexp1 = new BinaryExpression((Expression)dispExp.clone(),
										BinaryOperator.MULTIPLY, (Identifier)pitch.clone());
								BinaryExpression biexp2 = new BinaryExpression(tcast1, op, biexp1);
								List<Specifier> specs2 = new ArrayList<Specifier>();
								/////////////////////////////////////////////////////////////////////////////////////
								// CAUTION: VariableDeclarator.getTypeSpecifiers() returns both specifiers of      //
								// its parent VariableDeclaration and the VariableDeclarator's leading specifiers. //
								// Therefore, if VariableDeclarator is a pointer symbol, this method will return   //
								// pointer specifiers too.                                                         //
								/////////////////////////////////////////////////////////////////////////////////////
								specs2.addAll(((Identifier)tempExp).getSymbol().getTypeSpecifiers());
								specs2.remove(Specifier.STATIC);
								/////////////////////////////////////////////////////////////////////////
								// We don't need to add pointer specifier because tempExp.getSymbol(). //
								// getTypeSpecifiers() will include pointer specifier already.         //
								/////////////////////////////////////////////////////////////////////////
								//specs2.add(PointerSpecifier.UNQUALIFIED);
								Typecast tcast2 = new Typecast(specs2, biexp2);
								arg.swapWith(tcast2);
							}
						} else if( !(arg instanceof Identifier) ) {
							PrintTools.println(pass_name+ " [WARNING] argument expression (" + arg + ") in " +
									"a function call of "+ calledProc + " is too complex to ananlyze;" +
									"the converted argument expression may be incorrect.",0);
						}
					}
				}
				/*
				 * If this called function accesses shared/threadprivate variables that are not passed as
				 * function arguments, add those shared/threadprivate variables as arguments.
				 */
				HashSet<Symbol> callerParamSet = new HashSet<Symbol>();
				for( VariableDeclaration v_decl : callerParamList ) {
					callerParamSet.add((VariableDeclarator)v_decl.getDeclarator(0));
				}
				List<VariableDeclaration> oldParamList = (List<VariableDeclaration>)tProc.getParameters();
				HashSet<Symbol> oldParamSet = new HashSet<Symbol>();
				for( VariableDeclaration v_decl : oldParamList ) {
					oldParamSet.add((VariableDeclarator)v_decl.getDeclarator(0));
				}
				Set<Symbol> accessedSymbols = SymbolTools.getAccessedSymbols(tProc.getBody());
				Set<Symbol> localSymbols = DataFlowTools.getDefSymbol(tProc.getBody());
				for( Symbol ssm : OmpSet ) {
					VariableDeclarator symVar = (VariableDeclarator)ssm;
					Identifier ssmID = new Identifier(symVar);
					String symName = ssm.getSymbolName();
/*					if( !OmpAnalysis.containsSymbol(argSyms, symName) &&
							!OmpAnalysis.containsSymbol(oldParamSet, symName) &&*/
					if( !AnalysisTools.containsSymbol(oldParamSet, symName) &&
							!AnalysisTools.containsSymbol(localSymbols, symName) &&
							AnalysisTools.containsSymbol(accessedSymbols, symName) ) {
						if( use_MallocPitch && AnalysisTools.containsSymbol(pitchMap.keySet(),symName) ) {
							VariableDeclarator var_declarator =
								(VariableDeclarator)AnalysisTools.findsSymbol(pitchMap.keySet(), symName);
							Identifier var_ID = new Identifier(var_declarator);
							if( tempMap.containsKey(var_ID) ) {
								gpu_funcCall.addArgument((Expression)tempMap.get(var_ID).clone());
							} else {
								gpu_funcCall.addArgument((Expression)var_ID.clone());
							}
							VariableDeclaration pitch_decl = pitchMap.get(var_declarator);
							gpu_funcCall.addArgument(new Identifier((VariableDeclarator)pitch_decl.getDeclarator(0)));
						} else {
							VariableDeclarator cparam_declarator =
								(VariableDeclarator)AnalysisTools.findsSymbol(callerParamSet, symName);
							if( cparam_declarator == null ) {
								Tools.exit(pass_name + "[Error] "+symName.toString()+
										"does not exist in the kernel function parameters");
							} else {
								Identifier var_ID = new Identifier((VariableDeclarator)cparam_declarator);
								if( tempMap.containsKey(var_ID) ) {
									gpu_funcCall.addArgument((Expression)tempMap.get(var_ID).clone());
								} else {
									gpu_funcCall.addArgument((Expression)var_ID.clone());
								}
							}
						}
						if( AnalysisTools.containsSymbol(OmpThreadPrivSet, symName) ) {
							if(SymbolTools.isArray(ssm)) {
								List aspecs = ssm.getArraySpecifiers();
								ArraySpecifier aspec = (ArraySpecifier)aspecs.get(0);
								if( aspec.getNumDimensions() > 1 ) {
									gtidIsUsed = true;
								}
							} else if( SymbolTools.isScalar(ssm)) {
								gtidIsUsed = true;
							}
						}
					}
				}
				/*
				 * If ThreadPrivate array is used in the called function,  _gtid variable shoulde be
				 * passed from the parent kernel function to the child called function.
				 */
				if( gtidIsUsed ) {
					gpu_funcCall.addArgument( SymbolTools.getOrphanID("_gtid"));
				}

				//if( !c2gMap.containsKey(new_func_name) ) {
				//////////////////////////////////////////////////////////////
				// Check whether a function with the given name exists in a //
				// given translation unit.                                  //
				//////////////////////////////////////////////////////////////
				boolean func_exist = false;
				BreadthFirstIterator itr = new BreadthFirstIterator(tu);
				itr.pruneOn(Procedure.class);
				for (;;)
				{
					Procedure proc = null;

					try {
						proc = (Procedure)itr.next(Procedure.class);
					} catch (NoSuchElementException e) {
						break;
					}
					if( proc.getSymbolName().equals(new_func_name) ) {
						func_exist = true;
						break;
					}
				}
				if( !func_exist ) {
					/*
					 * Create a new device function by replicating the procedure, tProc.
					 */
					List<Specifier> new_proc_ret_type = new LinkedList<Specifier>();
					new_proc_ret_type.add(CUDASpecifier.CUDA_DEVICE);
					List<Specifier> return_types = tProc.getReturnType();
					new_proc_ret_type.addAll(return_types);
					body = (CompoundStatement)tProc.getBody().clone();
					TransformTools.updateAnnotationsInRegion(body);
					Procedure new_proc = new Procedure(new_proc_ret_type,
							new ProcedureDeclarator(new NameID(new_func_name),
									new LinkedList()), body);
					for(int i=0; i<list_size; i++) {
						VariableDeclaration param_decl = (VariableDeclaration)oldParamList.get(i);
						VariableDeclarator param_declarator = (VariableDeclarator)param_decl.getDeclarator(0);
						Identifier paramID = new Identifier(param_declarator);
						Expression arg = argList.get(i);
						if( arg instanceof Identifier ) {
							VariableDeclarator sm = (VariableDeclarator)SymbolTools.getSymbolOf(arg);
							if( sm == null ) {
								PrintTools.println(pass_name + "[WARNING] Unexpected argument, " + arg.toString() +
										" is used for function, " + calledProc.getName().toString() +
										"; translated new function, " + new_proc.getSymbolName() +
										", may be incorrect.", 0);
								VariableDeclaration cloned_decl = (VariableDeclaration)param_decl.clone();
								Identifier cloned_ID = new Identifier((VariableDeclarator)cloned_decl.getDeclarator(0));
								new_proc.addDeclaration(cloned_decl);
								// Replace all instances of the shared variable to the parameter variable
								IRTools.replaceAll((Traversable) body, paramID, cloned_ID);
							} else {
								boolean isSharedVar = AnalysisTools.containsSymbol(OmpSharedSet, sm.getSymbolName());
								boolean isTPVar = AnalysisTools.containsSymbol(OmpThreadPrivSet, sm.getSymbolName());
								if( use_MallocPitch && pitchMap.containsKey(sm) ) {
									VariableDeclaration pitch_decl = (VariableDeclaration)pitchMap.get(sm).clone();
									if( isSharedVar ) {
										pitchedAccessConv(param_declarator, new_proc, pitch_decl, body);
									} else if( isTPVar ){
										pitchedAccessConv3(param_declarator, new_proc, pitch_decl, body);
									} else {
										Tools.exit(pass_name + "[ERROR] deviceFuncTransform() found wrong symbol, "
												+ sm.toString() + " in the pitchMap.");
									}
								} else {
									if( isTPVar && SymbolTools.isArray(param_declarator) ) {
										CreateExtendedArray2(param_declarator, new_proc, body);
									} else {
										VariableDeclaration cloned_decl = (VariableDeclaration)param_decl.clone();
										Identifier cloned_ID = new Identifier((VariableDeclarator)cloned_decl.getDeclarator(0));
										new_proc.addDeclaration(cloned_decl);
										// Replace all instances of the shared variable to the parameter variable
										IRTools.replaceAll((Traversable) body, paramID, cloned_ID);
									}
								}
							}
						} else if ( arg instanceof BinaryExpression ) {
							Set<Expression> UseSet = DataFlowTools.getUseSet(arg);
							boolean isConverted = false;
							for( Expression exp : UseSet ) {
								if( exp instanceof Identifier ) {
									VariableDeclarator sm = (VariableDeclarator)SymbolTools.getSymbolOf(exp);
									if( sm == null ) {
											PrintTools.println(pass_name + "[WARNING] Unrecognizable expression, " + exp.toString() +
													", in " + arg.toString() + " is used for function, " +
													calledProc.getName().toString() +  "; translated new function, " +
													new_proc.getSymbolName() + ", may be incorrect.", 0);
									} else {
										boolean isSharedVar = AnalysisTools.containsSymbol(OmpSharedSet, sm.getSymbolName());
										boolean isTPVar = AnalysisTools.containsSymbol(OmpThreadPrivSet, sm.getSymbolName());
										if( use_MallocPitch && pitchMap.containsKey(sm) ) {
											VariableDeclaration pitch_decl = (VariableDeclaration)pitchMap.get(sm).clone();
											if( isSharedVar ) {
												pitchedAccessConv(param_declarator, new_proc, pitch_decl, body);
												isConverted = true;
												break;
											} else if( isTPVar ){
												pitchedAccessConv3(param_declarator, new_proc, pitch_decl, body);
												isConverted = true;
												break;
											} else {
												Tools.exit(pass_name + "[ERROR] deviceFuncTransform() found wrong symbol, "
														+ sm.toString() + " in the pitchMap.");
											}
										} else {
											if( isTPVar && SymbolTools.isArray(param_declarator) ) {
												CreateExtendedArray2(param_declarator, new_proc, body);
												isConverted = true;
												break;
											}
										}
									}
								}
							}
							if( !isConverted ) {
								VariableDeclaration cloned_decl = (VariableDeclaration)param_decl.clone();
								Identifier cloned_ID = new Identifier((VariableDeclarator)cloned_decl.getDeclarator(0));
								new_proc.addDeclaration(cloned_decl);
								// Replace all instances of the shared variable to the parameter variable
								IRTools.replaceAll((Traversable) body, paramID, cloned_ID);
							}
						} else {
							if( !((arg instanceof Literal) || (arg instanceof UnaryExpression)
									|| (arg instanceof ArrayAccess)) ) {
								PrintTools.println(pass_name + "[WARNING] Unrecognizable argument, " + arg.toString() +
										" is used for function, " + calledProc.getName().toString() +
										"; translated new function, " + new_proc.getSymbolName() +
										", may be incorrect.", 0);
							}
							VariableDeclaration cloned_decl = (VariableDeclaration)param_decl.clone();
							Identifier cloned_ID = new Identifier((VariableDeclarator)cloned_decl.getDeclarator(0));
							new_proc.addDeclaration(cloned_decl);
							// Replace all instances of the shared variable to the parameter variable
							IRTools.replaceAll((Traversable) body, paramID, cloned_ID);
						}
					}
					/*
					 * If this called function accesses shared variables that are not passed as
					 * function arguments, add those shared variables as function parameters.
					 */
					for( Symbol ssm : OmpSet ) {
						VariableDeclarator symVar = (VariableDeclarator)ssm;
						Identifier ssmID = new Identifier(symVar);
						String symName = ssm.getSymbolName();
/*						if( !OmpAnalysis.containsSymbol(argSyms, symName) &&
							!OmpAnalysis.containsSymbol(oldParamSet, symName) &&*/
						if( !AnalysisTools.containsSymbol(oldParamSet, symName) &&
							!AnalysisTools.containsSymbol(localSymbols, symName) &&
								AnalysisTools.containsSymbol(accessedSymbols, symName) ) {
							boolean isSharedVar = AnalysisTools.containsSymbol(OmpSharedSet, symName);
							boolean isTPVar = AnalysisTools.containsSymbol(OmpThreadPrivSet, symName);
							if( use_MallocPitch && AnalysisTools.containsSymbol(pitchMap.keySet(),symName) ) {
								VariableDeclarator var_declarator =
									(VariableDeclarator)AnalysisTools.findsSymbol(pitchMap.keySet(), symName);
								VariableDeclaration pitch_decl =
									(VariableDeclaration)pitchMap.get(var_declarator).clone();
								if( isSharedVar ) {
									pitchedAccessConv(symVar, new_proc, pitch_decl, body);
								} else if( isTPVar ) {
									pitchedAccessConv3(symVar, new_proc, pitch_decl, body);
								}
							} else {
								if( SymbolTools.isScalar(ssm) ) {
									// Create a parameter Declaration for the device function
									// Change the scalar variable to a pointer type
									if( isSharedVar ) {
										scalarVariableConv(symVar, new_proc, body, true, false);
									}
									else if( isTPVar ) {
										scalarVariableConv2(symVar, new_proc, body, true, false);
									} else {
										Tools.exit(pass_name + "[ERROR] interProcLoopTransform() found unknown symbol, "
												+ ssm.toString());
									}
								} else if( SymbolTools.isArray(ssm) ) {
									if( isSharedVar ) {
										VariableDeclaration decl =
											(VariableDeclaration)symVar.getParent();
										VariableDeclarator cloned_declarator =
											(VariableDeclarator)((VariableDeclarator)symVar).clone();
										cloned_declarator.setInitializer(null);
										//////////////////////////////////////////////////
										// Kernel function parameter can not be static. //
										//////////////////////////////////////////////////
										List<Specifier> clonedspecs = new ChainedList<Specifier>();
										clonedspecs.addAll(decl.getSpecifiers());
										clonedspecs.remove(Specifier.STATIC);
										VariableDeclaration cloned_decl = new VariableDeclaration(clonedspecs, cloned_declarator);
										Identifier clonedID = new Identifier((VariableDeclarator)cloned_declarator);
										new_proc.addDeclaration(cloned_decl);
										// Replace all instances of the shared variable to the local variable
										IRTools.replaceAll((Traversable) body, ssmID, clonedID);
									} else if ( isTPVar ) {
										CreateExtendedArray2(symVar, new_proc, body);
									} else {
										Tools.exit(pass_name + "[ERROR] interProcLoopTransform() found unknown symbol, "
												+ ssm.toString());
									}
								} else {
									Tools.exit(pass_name + "[ERROR] interProcLoopTransform() found unsupported shared symbol, "
											+ ssm.toString());
								}
							}
						}
					}
					/*
					 * If ThreadPrivate array is used in the called function, _gtid variable shoulde be
					 * passed from the parent kernel function to the child called function.
					 */
					Identifier gtid = null;
					if( gtidIsUsed ) {
						VariableDeclarator gtid_declarator = new VariableDeclarator(new NameID("_gtid"));
						gtid = new Identifier(gtid_declarator);
						VariableDeclaration gtid_decl = new VariableDeclaration(Specifier.INT, gtid_declarator);
						new_proc.addDeclaration(gtid_decl);
					}

					/* put new_proc before the procedure, tProc */
					tu.addDeclarationBefore(tProc, new_proc);

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
						if( procDeclr.getID().equals(tProc.getName()) ) {
							//Found function declaration.
							Declaration procDecl = (Declaration)procDeclr.getParent();
							//Create a new function declaration.
							VariableDeclaration newProcDecl =
								new VariableDeclaration(new_proc.getReturnType(), new_proc.getDeclarator().clone());
							//Insert the new function declaration.
							tu.addDeclarationAfter(procDecl, newProcDecl);
							break;
						}
					}

					///////////////////////////////////////////////////////////////////////////////////
					// FIXME: The below method does not update symbol pointers of Identifiers in the //
					// new procedure, and thus any symbol-related operation may fail.                //
					///////////////////////////////////////////////////////////////////////////////////
					TransformTools.updateAnnotationsInRegion(new_proc);
					c2gMap.put(new_func_name, new_proc);
				}
				/*
				 * Swap old functionCall with the new device function.
				 */
				calledProc.swapWith(gpu_funcCall);
				//tempMap.clear();
				/*
				 *  FIXME: Below recursive call is incorrect.
				 */
				//deviceFuncTransform(body, pitchMap, parMap, callerParamList);
			}
		}
	}

	/*
	 * Create pointer variable declaration for a function parameter,
	 * and convert existing array access into pitched access
	 * This method is used to create a pitched access of an OpenMP Shared array
	 * in a kernel region or the body of a device function called
	 * within a kernel region.
	 * Ex: replace gpu_a[i][k] with *((float *)((char *)gpu_a + i * pitch_a) + k)
	 */
	private static VariableDeclarator pitchedAccessConv(VariableDeclarator tarSym,
			Procedure new_proc, VariableDeclaration pitch_decl,
			Statement body)  {
		Identifier tarSymID = new Identifier(tarSym);
		// Change to a pointer type
		// Ex:  "float* b"
		VariableDeclarator pointerV_declarator =
			new VariableDeclarator(PointerSpecifier.UNQUALIFIED, new NameID(tarSym.getSymbolName()));
		List<Specifier> clonedspecs = new ChainedList<Specifier>();
		/////////////////////////////////////////////////////////////////////////////////////
		// CAUTION: VariableDeclarator.getTypeSpecifiers() returns both specifiers of      //
		// its parent VariableDeclaration and the VariableDeclarator's leading specifiers. //
		// Therefore, if VariableDeclarator is a pointer symbol, this method will return   //
		// pointer specifiers too.                                                         //
		/////////////////////////////////////////////////////////////////////////////////////
		clonedspecs.addAll(tarSym.getTypeSpecifiers());
		clonedspecs.remove(Specifier.STATIC);
		VariableDeclaration pointerV_decl = new VariableDeclaration(clonedspecs,
				pointerV_declarator);
		Identifier pointer_var = new Identifier(pointerV_declarator);
		/*
		 * Insert function parameters needed for the pitched access
		 */
		new_proc.addDeclaration(pointerV_decl);
		new_proc.addDeclaration(pitch_decl);
		/*
		 * If MallocPitch is used to allocate 2 dimensional array, gpu_a,
		 * replace array access expression with pointer access expression with pitch
		 * Ex: gpu_a[i][k] => *((float *)((char *)gpu_a + i * pitch_a) + k)
		 */
		DepthFirstIterator iter = new DepthFirstIterator(body);
		for (;;)
		{
			ArrayAccess aAccess = null;

			try {
				aAccess = (ArrayAccess)iter.next(ArrayAccess.class);
			} catch (NoSuchElementException e) {
				break;
			}
			IDExpression arrayID = (IDExpression)aAccess.getArrayName();
			if( arrayID.equals(tarSymID) ) {
				Expression pAccess = convArray2Pointer(aAccess,
						new Identifier((VariableDeclarator)pitch_decl.getDeclarator(0)));
				aAccess.swapWith(pAccess);
			}
		}
		// Replace all instances of the shared variable to the parameter variable
		IRTools.replaceAll((Traversable) body, tarSymID, pointer_var);

		return pointerV_declarator;
	}

	/*
	 * Create pointer variable declaration for a function parameter,
	 * and convert existing array access into pitched access with extention.
	 * This method is used to create pitched access of an OpenMP Threadprivate array
	 * in a kernel region.
	 * Ex: Add "float* x_0 = (float*)((float *)x + _gtid);"and
	 *     replace x[i] with *((float*)((char*)x_0 + i*pitch_x))
	 */
	private static VariableDeclarator pitchedAccessConv2(VariableDeclarator tarSym,
			Procedure new_proc, VariableDeclaration pitch_decl,
			Statement region)  {
		Identifier tarSymID = new Identifier(tarSym);
		// Change to a pointer type
		// Ex:  "float* b"
		VariableDeclarator pointerV_declarator =
			new VariableDeclarator(PointerSpecifier.UNQUALIFIED, new NameID(tarSym.getSymbolName()));
		List<Specifier> clonedspecs = new ChainedList<Specifier>();
		/////////////////////////////////////////////////////////////////////////////////////
		// CAUTION: VariableDeclarator.getTypeSpecifiers() returns both specifiers of      //
		// its parent VariableDeclaration and the VariableDeclarator's leading specifiers. //
		// Therefore, if VariableDeclarator is a pointer symbol, this method will return   //
		// pointer specifiers too.                                                         //
		/////////////////////////////////////////////////////////////////////////////////////
		clonedspecs.addAll(tarSym.getTypeSpecifiers());
		clonedspecs.remove(Specifier.STATIC);
		VariableDeclaration pointerV_decl = new VariableDeclaration(clonedspecs,
				pointerV_declarator);
		Identifier pointer_var = new Identifier(pointerV_declarator);
		/*
		 * Insert function parameters needed for the pitched access
		 */
		new_proc.addDeclaration(pointerV_decl);
		new_proc.addDeclaration(pitch_decl);
		/*
		 *  Convert existing array access into pitched access with extention.
		 * Ex: Add "float* x_0 = (float*)((float *)x + _gtid);"and
		 *     replace x[i] with *((float*)((char*)x_0 + i*pitch_x))
		 */
		Statement estmt = null;
		CompoundStatement targetStmt = null;
		if( region instanceof CompoundStatement ) {
			targetStmt = (CompoundStatement)region;
		} else if( region instanceof ForLoop ) {
			targetStmt = (CompoundStatement)((ForLoop)region).getBody();
		} else {
			Tools.exit(pass_name + "[ERROR] Unknwon region in extractKernelRegion(): "
					+ region.toString());
		}
		//Identifier local_var = SymbolTools.getPointerTemp(targetStmt, (Identifier)tarSymID);
		Identifier local_var = SymbolTools.getPointerTemp(targetStmt, clonedspecs, ((Identifier)tarSymID).getName());
		// Identifier "_gtid" should be updated later so that it can point to a corresponding symbol.
		BinaryExpression biexp = new BinaryExpression(new Typecast(pointerV_declarator.getTypeSpecifiers(),
				(Identifier)pointer_var.clone()), BinaryOperator.ADD, SymbolTools.getOrphanID("_gtid"));
		estmt = new ExpressionStatement(new AssignmentExpression((Identifier)local_var.clone(),
				AssignmentOperator.NORMAL, biexp));
		DepthFirstIterator iter = new DepthFirstIterator(region);
		for (;;)
		{
			ArrayAccess aAccess = null;

			try {
				aAccess = (ArrayAccess)iter.next(ArrayAccess.class);
			} catch (NoSuchElementException e) {
				break;
			}
			IDExpression arrayID = (IDExpression)aAccess.getArrayName();
			if( arrayID.equals(tarSymID) ) {
				Expression pAccess = convArray2Pointer2(aAccess,
						new Identifier((VariableDeclarator)pitch_decl.getDeclarator(0)));
				aAccess.swapWith(pAccess);
			}
		}
		// Replace all instances of the threadprivate variable to the parameter variable
		IRTools.replaceAll((Traversable) region, tarSymID, local_var);
		///////////////////////////////////////////////////////////////////////////////////
		// Revert instances of the local variable used in function calls to the original //
		// threadprivae variable; this revert is needed for interProcLoopTransform().    //
		///////////////////////////////////////////////////////////////////////////////////
		List<FunctionCall> funcCalls = IRTools.getFunctionCalls(region);
		for( FunctionCall calledProc : funcCalls ) {
			List<Expression> argList = (List<Expression>)calledProc.getArguments();
			for( Expression arg : argList ) {
				IRTools.replaceAll(arg, local_var, pointer_var);
			}
		}
		tempMap.put((Identifier)pointer_var.clone(), (Identifier)local_var.clone());
		//tempMap.put((Identifier)tarSymID.clone(), (Identifier)local_var.clone());
		// Add "float* x_0 = (float*)(x + _gtid);" statement at the beginning.
		Statement last_decl_stmt;
		last_decl_stmt = IRTools.getLastDeclarationStatement(targetStmt);
		if( last_decl_stmt != null ) {
			targetStmt.addStatementAfter(last_decl_stmt, estmt);
		} else {
			last_decl_stmt = (Statement)targetStmt.getChildren().get(0);
			targetStmt.addStatementBefore(last_decl_stmt, estmt);
		}

		return pointerV_declarator;
	}

	/*
	 * Create a pointer variable declaration for the parameter of the device function
	 * called in a kernel function, and convert existing array access into pitched
	 * access.
	 * This method is used to create pitched access of an OpenMP Threadprivate array
	 * in a device function called within a kernel region.
	 * Ex:    replace x[i] with *((float*)((char*)x + i*pitch_x))
	 */
	private static VariableDeclarator pitchedAccessConv3(VariableDeclarator tarSym,
			Procedure new_proc, VariableDeclaration pitch_decl,
			Statement region)  {
		Identifier tarSymID = new Identifier(tarSym);
		// Change to a pointer type
		// Ex:  "float* b"
		VariableDeclarator pointerV_declarator =
			new VariableDeclarator(PointerSpecifier.UNQUALIFIED, new NameID(tarSym.getSymbolName()));
		List<Specifier> clonedspecs = new ChainedList<Specifier>();
		/////////////////////////////////////////////////////////////////////////////////////
		// CAUTION: VariableDeclarator.getTypeSpecifiers() returns both specifiers of      //
		// its parent VariableDeclaration and the VariableDeclarator's leading specifiers. //
		// Therefore, if VariableDeclarator is a pointer symbol, this method will return   //
		// pointer specifiers too.                                                         //
		/////////////////////////////////////////////////////////////////////////////////////
		clonedspecs.addAll(tarSym.getTypeSpecifiers());
		clonedspecs.remove(Specifier.STATIC);
		VariableDeclaration pointerV_decl = new VariableDeclaration(clonedspecs,
				pointerV_declarator);
		Identifier pointer_var = new Identifier(pointerV_declarator);
		/*
		 * Insert function parameters needed for the pitched access
		 */
		new_proc.addDeclaration(pointerV_decl);
		new_proc.addDeclaration(pitch_decl);
		/*
		 *  Convert existing array access into pitched access.
		 * Ex: replace x[i] with *((float*)((char*)x + i*pitch_x))
		 */
		DepthFirstIterator iter = new DepthFirstIterator(region);
		for (;;)
		{
			ArrayAccess aAccess = null;

			try {
				aAccess = (ArrayAccess)iter.next(ArrayAccess.class);
			} catch (NoSuchElementException e) {
				break;
			}
			IDExpression arrayID = (IDExpression)aAccess.getArrayName();
			if( arrayID.equals(tarSymID) ) {
				Expression pAccess = convArray2Pointer2(aAccess,
						new Identifier((VariableDeclarator)pitch_decl.getDeclarator(0)));
				aAccess.swapWith(pAccess);
			}
		}
		// Replace all instances of the threadprivate variable to the parameter variable
		IRTools.replaceAll((Traversable) region, tarSymID, pointer_var);

		return pointerV_declarator;
	}

	private static VariableDeclarator CreateExtendedArray(Symbol threadPriv_var,
			Procedure new_proc, Statement region ) {
		// Create an extended array type
		// Ex: "float b[][SIZE1][SIZE2]"
		List aspecs = threadPriv_var.getArraySpecifiers();
		ArraySpecifier aspec = (ArraySpecifier)aspecs.get(0);
		int dimsize = aspec.getNumDimensions();
		List edimensions = new LinkedList();
		edimensions.add(null);
		for( int i=0; i<dimsize; i++ )
		{
			edimensions.add((Expression)aspec.getDimension(i).clone());
		}
		ArraySpecifier easpec = new ArraySpecifier(edimensions);
		VariableDeclarator arrayV_declarator = new VariableDeclarator(new NameID(threadPriv_var.getSymbolName()), easpec);
		List<Specifier> clonedspecs = new ChainedList<Specifier>();
		/////////////////////////////////////////////////////////////////////////////////////
		// CAUTION: VariableDeclarator.getTypeSpecifiers() returns both specifiers of      //
		// its parent VariableDeclaration and the VariableDeclarator's leading specifiers. //
		// Therefore, if VariableDeclarator is a pointer symbol, this method will return   //
		// pointer specifiers too.                                                         //
		/////////////////////////////////////////////////////////////////////////////////////
		clonedspecs.addAll(threadPriv_var.getTypeSpecifiers());
		clonedspecs.remove(Specifier.STATIC);
		VariableDeclaration arrayV_decl =
			new VariableDeclaration(clonedspecs, arrayV_declarator);
		Identifier array_var = new Identifier(arrayV_declarator);
		new_proc.addDeclaration(arrayV_decl);
		/*
		 * Replace array access expression with extended access expression.
		 */
		Identifier cloned_ID = new Identifier((VariableDeclarator)threadPriv_var);
		if( dimsize == 1 ) {
			// Insert "float* x_0 = (float *)((float *)x + _gtid * SIZE1);" and
			// replace x with x_0
			Statement estmt = null;
			CompoundStatement targetStmt = null;
			if( region instanceof CompoundStatement ) {
				targetStmt = (CompoundStatement)region;
			} else if( region instanceof ForLoop ) {
				targetStmt = (CompoundStatement)((ForLoop)region).getBody();
			} else {
				Tools.exit(pass_name + "[ERROR] Unknwon region in extractKernelRegion(): "
						+ region.toString());
			}
			//Identifier pointer_var = SymbolTools.getPointerTemp(targetStmt, cloned_ID);
			Identifier pointer_var = SymbolTools.getPointerTemp(targetStmt, clonedspecs, cloned_ID.getName());
			// Identifier "_gtid" should be updated later so that it can point to a corresponding symbol.
			List<Specifier> clonedPspecs = new ChainedList<Specifier>();
			clonedPspecs.addAll(clonedspecs);
			clonedPspecs.add(PointerSpecifier.UNQUALIFIED);
			BinaryExpression biexp = new BinaryExpression(new Typecast(clonedPspecs, (Identifier)array_var.clone()),
					BinaryOperator.ADD, new BinaryExpression(SymbolTools.getOrphanID("_gtid"),
							BinaryOperator.MULTIPLY, (Expression)aspec.getDimension(0).clone()));
			estmt = new ExpressionStatement(new AssignmentExpression((Identifier)pointer_var.clone(),
					AssignmentOperator.NORMAL, biexp));
			// Replace all instances of the shared variable to the local variable
			//IRTools.replaceAll((Traversable) targetStmt, cloned_ID, pointer_var);
			IRTools.replaceAll((Traversable) region, cloned_ID, pointer_var);
			// Revert instances of the local variable used in function calls to the original threadprivae
			// variable; this revert is needed for interProcLoopTransform().
			List<FunctionCall> funcCalls = IRTools.getFunctionCalls(region);
			for( FunctionCall calledProc : funcCalls ) {
				List<Expression> argList = (List<Expression>)calledProc.getArguments();
				for( Expression arg : argList ) {
					IRTools.replaceAll(arg, pointer_var, array_var);
				}
			}
			tempMap.put((Identifier)array_var.clone(), (Identifier)pointer_var.clone());
			//tempMap.put((Identifier)cloned_ID.clone(), (Identifier)pointer_var.clone());
			Statement last_decl_stmt;
			last_decl_stmt = IRTools.getLastDeclarationStatement(targetStmt);
			if( last_decl_stmt != null ) {
				targetStmt.addStatementAfter(last_decl_stmt, estmt);
			} else {
				last_decl_stmt = (Statement)targetStmt.getChildren().get(0);
				targetStmt.addStatementBefore(last_decl_stmt, estmt);
			}
		} else {
			//replace x[k][m] with x[_gtid][k][m]
			DepthFirstIterator iter = new DepthFirstIterator(region);
			for (;;)
			{
				ArrayAccess aAccess = null;

				try {
					aAccess = (ArrayAccess)iter.next(ArrayAccess.class);
				} catch (NoSuchElementException e) {
					break;
				}
				IDExpression arrayID = (IDExpression)aAccess.getArrayName();
				if( arrayID.equals(cloned_ID) ) {
					if( aAccess.getNumIndices() == dimsize ) {
						List indices = new LinkedList();
						indices.add(SymbolTools.getOrphanID("_gtid"));
						///////////////////////
						// DEBUG: deprecated //
						///////////////////////
						//indices.addAll(aAccess.getIndices());
						for( Expression indx : aAccess.getIndices() ) {
							indices.add(indx.clone());
						}
						ArrayAccess extendedAccess = new ArrayAccess((IDExpression)array_var.clone(), indices);
						aAccess.swapWith(extendedAccess);
					} else {
						Tools.exit(pass_name + "[ERROR in CreateExtendedArray()] Incorrect dimension of the array access :"
								+ aAccess.toString());
					}
				}
			}
			// Replace all instances of the shared variable to the parameter variable
			IRTools.replaceAll((Traversable) region, cloned_ID, array_var);
		}
		return arrayV_declarator;
	}

	/*
	 * Array conversion function that is used for extending and converting
	 * threadprivate array in a device function, which is called in a kernel function.
	 */
	private static VariableDeclarator CreateExtendedArray2(Symbol threadPriv_var,
			Procedure new_proc, Statement region ) {
		/*
		 * If the dimesion size of a threadprivate array is greater than 1,
		 *    - Create an extended array type for the threadprivate array
		 * Else,
		 *    - Use existing function parameter as it is.
		 * Ex: "float b[][SIZE1][SIZE2]"
		 */
		List aspecs = threadPriv_var.getArraySpecifiers();
		ArraySpecifier aspec = (ArraySpecifier)aspecs.get(0);
		int dimsize = aspec.getNumDimensions();
		VariableDeclarator arrayV_declarator = null;
		if( dimsize == 1 ) {
			VariableDeclarator cloned_declarator = (VariableDeclarator)((VariableDeclarator)threadPriv_var).clone();
			cloned_declarator.setInitializer(null);
			List<Specifier> clonedspecs = new ChainedList<Specifier>();
			/////////////////////////////////////////////////////////////////////////////////////
			// CAUTION: VariableDeclarator.getTypeSpecifiers() returns both specifiers of      //
			// its parent VariableDeclaration and the VariableDeclarator's leading specifiers. //
			// Therefore, if VariableDeclarator is a pointer symbol, this method will return   //
			// pointer specifiers too.                                                         //
			/////////////////////////////////////////////////////////////////////////////////////
			clonedspecs.addAll(threadPriv_var.getTypeSpecifiers());
			clonedspecs.remove(Specifier.STATIC);
			VariableDeclaration cloned_decl = new VariableDeclaration(clonedspecs, cloned_declarator);
			Identifier cloned_ID = new Identifier(cloned_declarator);
			new_proc.addDeclaration(cloned_decl);
			// Replace all instances of the threadprivate variable to the parameter variable
			try {
			IRTools.replaceAll((Traversable) region, cloned_ID, cloned_ID);
			} catch (Exception e) {
				PrintTools.println("====> symbol : " + threadPriv_var, 0);
				Tools.exit("Error in CreateExtendedArray2()");
			}
		} else {
			List edimensions = new LinkedList();
			edimensions.add(null);
			for( int i=0; i<dimsize; i++ )
			{
				edimensions.add((Expression)aspec.getDimension(i).clone());
			}
			ArraySpecifier easpec = new ArraySpecifier(edimensions);
			arrayV_declarator = new VariableDeclarator(new NameID(threadPriv_var.getSymbolName()), easpec);
			List<Specifier> clonedspecs = new ChainedList<Specifier>();
			clonedspecs.addAll(threadPriv_var.getTypeSpecifiers());
			clonedspecs.remove(Specifier.STATIC);
			VariableDeclaration arrayV_decl =
				new VariableDeclaration(clonedspecs, arrayV_declarator);
			Identifier array_var = new Identifier(arrayV_declarator);
			new_proc.addDeclaration(arrayV_decl);
			/*
			 * Replace array access expression with extended access expression.
			 */
			Identifier cloned_ID = new Identifier((VariableDeclarator)threadPriv_var);
			//replace x[k][m] with x[_gtid][k][m]
			DepthFirstIterator iter = new DepthFirstIterator(region);
			for (;;)
			{
				ArrayAccess aAccess = null;

				try {
					aAccess = (ArrayAccess)iter.next(ArrayAccess.class);
				} catch (NoSuchElementException e) {
					break;
				}
				IDExpression arrayID = (IDExpression)aAccess.getArrayName();
				if( arrayID.equals(cloned_ID) ) {
					if( aAccess.getNumIndices() == dimsize ) {
						List indices = new LinkedList();
						indices.add(SymbolTools.getOrphanID("_gtid"));
						///////////////////////
						// DEBUG: deprecated //
						///////////////////////
						//indices.addAll(aAccess.getIndices());
						for( Expression indx : aAccess.getIndices() ) {
							indices.add(indx.clone());
						}
						ArrayAccess extendedAccess = new ArrayAccess((IDExpression)array_var.clone(), indices);
						aAccess.swapWith(extendedAccess);
					} else {
						Tools.exit(pass_name + "[ERROR in CreateExtendedArray()] Incorrect dimension of the array access :"
								+ aAccess.toString());
					}
				}
			}
			// Replace all instances of the threadprivate variable to the parameter variable
			IRTools.replaceAll((Traversable) region, cloned_ID, array_var);
		}
		return arrayV_declarator;
	}

	public String getPassName() {
		return pass_name;
	}

}
