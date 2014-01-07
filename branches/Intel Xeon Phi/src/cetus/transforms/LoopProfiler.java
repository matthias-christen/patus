package cetus.transforms;

import java.io.*;
import java.util.*;

import cetus.hir.*;
import cetus.analysis.*;

/*
 * LoopProfile pass inserts profiling calls around all loops in the input
 * program. Loop invocation count and execution time are collected using
 * this insrumentation. This pass was implemented to demonstrate the tasks
 * that traverse, create, and modify the Cetus IR. Following tasks can be
 * found in the code.
 * (1) Traversing the IR in depth-first order.
 * (2) Optimizing IR traversal.
 * (3) Adding a new variable declaration.
 * (4) Creating an expression.
 * (5) Creating a statement.
 * (6) Inserting a statement to the IR.
 * (7) Creating an annotation.
 * (8) Adjusting print method of an annotation.
 * (9) Inserting an annotation to the IR.
 */
/**
 * LoopProfile inserts timers around loops following the selection strategy
 * specified by "select" field.
 */
public class LoopProfiler extends TransformPass
{
	/* The total number of loops */
	private int num_loops;

	/* Loop names */
	private List<String> loop_names = new ArrayList<String>();

	/* The "main" procedure */
	private Procedure main_proc;

	/* The translation unit containing the "main" procedure */
	private TranslationUnit main_tu;

	private int strategy;

	/* Insertion points of timing routine */
	private static final int ADD = 0;
	private static final int ADD_BEFORE = 1;
	private static final int ADD_AFTER = 2;

	/* Profiling strategies */
	private static final int EVERY = 1;
	private static final int OUTER = 2;
	private static final int EVERY_PAR = 3;
	private static final int OUTER_PAR = 4;
	private static final int EVERY_OMP = 5;
	private static final int OUTER_OMP = 6;

	/**
	 * Constructs a new LoopProfile object from the specified program and
	 * performs profiling. It collects information such as total number of
	 * procedures, maximum number of loops per procedure, main procedure, and
	 * main translation unit for code generation.
	 */
	public LoopProfiler(Program prog)
	{
		super(prog);
		num_loops = 0;
		main_proc = null;
		main_tu = null;
		strategy = Integer.valueOf(
			cetus.exec.Driver.getOptionValue("profile-loops")).intValue();
	}

	public String getPassName()
	{
		return "[Loop Profiler]";
	}

	public void start()
	{
		LoopTools.addLoopName(program);
		DepthFirstIterator iter = new DepthFirstIterator(program);
		iter.pruneOn(Procedure.class);   /* Does not traverse beneath procedure */
		iter.pruneOn(Declaration.class); /* Does not traverse beneath declaration */
		TranslationUnit tu = null;
		while ( iter.hasNext() )
		{
			Object o = iter.next();
			if ( o instanceof TranslationUnit )
				tu = (TranslationUnit)o;
			else if ( o instanceof Procedure )
			{
				Procedure proc = (Procedure)o;
				int num_loop = profileProcedure(proc);
				num_loops += num_loop;
				if ( proc.getSymbolName().equals("main") )
				{
					main_tu = tu;
					main_proc = proc;
				}
			}
		}
		if ( main_tu == null )
			main_tu = tu;

		addTimingLibrary();
	}

	/**
	 * Profiles the given procedure. It inserts timing calls and loop tags around
	 * each loops and returns the total number of profiled loops.
	 */
	private int profileProcedure(Procedure proc)
	{
		List<Statement> loops = new ArrayList<Statement>();
		collectLoops(proc, loops);

		for ( int i=0; i<loops.size(); i++ )
		{
			Statement loop = loops.get(i);
			/* Inserts cetus_tic call: "cetus_tic(loop_id);" */
			insertTic(loop, num_loops+i);
			/* Inserts cetus_toc call: "cetus_toc(proc_id, loop_id, cetus_t);" */
			insertToc(loop, num_loops+i, ADD_AFTER);
			/* Prepares comment that shows the identifying tag for the loop */
			String loop_name = LoopTools.getLoopName(loop);
			loop_names.add("\""+loop_name+"\"");
		}
		return loops.size();
	}

	// Collect candidate loops following the specified option.
	private void collectLoops(Traversable t, List<Statement> loops)
	{
		if ( t.getChildren() == null )
			return;

		FlatIterator iter = new FlatIterator(t);
		while ( iter.hasNext() )
		{
			Traversable tr = (Traversable)iter.next();
			if ( tr instanceof ForLoop )
			{
				ForLoop for_loop = (ForLoop)tr;
				boolean is_omp_par =
					for_loop.containsAnnotation(OmpAnnotation.class, "parallel");
				boolean is_cetus_par =
					for_loop.containsAnnotation(CetusAnnotation.class, "parallel");
				if ( strategy == EVERY || strategy == OUTER ||
				is_omp_par && (strategy == EVERY_OMP || strategy == OUTER_OMP) ||
				is_cetus_par && (strategy == EVERY_PAR || strategy == OUTER_PAR) )
					loops.add((Statement)tr);
				if ( strategy == OUTER ||
				is_omp_par && strategy == OUTER_OMP ||
				is_cetus_par && strategy == OUTER_PAR )
					continue;
			}
			if ( tr instanceof CompoundStatement )
				collectLoops(tr, loops);
		}
	}

	private static void insertTic(Statement ref, int id)
	{
		insertTiming("cetus_tic", ref, id, ADD_BEFORE);
	}

	// Inserts a cetus_tic call.
	private static void insertTic(Statement ref, int id, int where)
	{
		insertTiming("cetus_tic", ref, id, where);
	}

	// Inserts a cetus_toc call.
	private static void insertToc(Statement ref, int id, int where)
	{
		insertTiming("cetus_toc", ref, id, where);
	}

	// Inserts a timing routine with the given routine name, reference statement,
	// and target position.
	private static void
			insertTiming(String name, Statement ref, int id, int where)
	{
		FunctionCall fc = new FunctionCall(SymbolTools.getOrphanID(name));
		fc.addArgument(new IntegerLiteral(id));
		Statement stmt = new ExpressionStatement(fc);
		addConditionalCode(stmt);

		CompoundStatement parent = null;
		if ( where == ADD ) // What is the case?
			parent = (CompoundStatement)ref;
		else
			parent = (CompoundStatement)ref.getParent();

		if ( where == ADD )
			parent.addStatement(stmt);
		else if ( where == ADD_BEFORE )
			parent.addStatementBefore(ref, stmt);
		else if ( where == ADD_AFTER )
			parent.addStatementAfter(ref, stmt);
		else
			PrintTools.printlnStatus("[WARNING] unknown profiling code being inserted", 0);
	}

	private static void addConditionalCode(Statement stmt)
	{
		CodeAnnotation ifdef = new CodeAnnotation("#ifdef CETUS_TIMING");
		CodeAnnotation endif = new CodeAnnotation("#endif");
		stmt.annotateBefore(ifdef);
		stmt.annotateAfter(endif);
	}

	/* Adds timing library and "cetus_print_timer" calls in the main procedure */
	private void addTimingLibrary()
	{
		/* Adds cetus_print_timer calls */
		if ( main_proc == null )
		{
			System.err.println("[WARNING] main routine not found");
			System.err.println("[WARNING] add \"cetus_init_timer()\" manually");
			System.err.println("[WARNING] add \"cetus_print_timer()\" manually");
		}
		else
		{
			List<Statement> return_stmts = new ArrayList<Statement>();
			Statement first_stmt = null;
			DepthFirstIterator iter = new DepthFirstIterator(main_proc.getBody());

			/* Collect the program points to add "cetus_print_timer" call before */
			while ( iter.hasNext() )
			{
				Object o = iter.next();
				if ( o == main_proc.getBody() || !(o instanceof Statement) )
					continue;
				if ( o instanceof ReturnStatement )
					return_stmts.add((Statement)o);
				if ( first_stmt == null && !(o instanceof DeclarationStatement) )
					first_stmt = (Statement)o;
			}

			/* Inserts "cetus_init_timer" calls */
			Statement init_timer = new ExpressionStatement(
				new FunctionCall(SymbolTools.getOrphanID("cetus_init_timer")));
			addConditionalCode(init_timer);
			main_proc.getBody().addStatementBefore(first_stmt, init_timer);

			/* Timer for the application */
			insertTic(first_stmt, num_loops, ADD_BEFORE);

			/* Inserts "cetus_print_timer" calls */
			Statement print_timer = new ExpressionStatement(
				new FunctionCall(SymbolTools.getOrphanID("cetus_print_timer")));
			addConditionalCode(print_timer);

			if ( return_stmts.size() == 0 )
			{
				insertToc(main_proc.getBody(), num_loops++, ADD);
				main_proc.getBody().addStatement(print_timer.clone());
			}
			else
			{
				for ( Statement stmt : return_stmts )
				{
					insertToc(stmt, num_loops, ADD_BEFORE);
					CompoundStatement parent = (CompoundStatement)stmt.getParent();
					parent.addStatementBefore(stmt, print_timer.clone());
				}
				num_loops++;
			}
		}

		/* Adds timing library. We chose to use "raw" type of annotation which
		 * is printed as a raw string */
		String header =
			"/* Automatically inserted by Cetus */\n"+
			"#ifdef CETUS_TIMING\n"+
			"#include <sys/time.h>\n"+
			"#include <stdio.h>\n"+
			"int cetus_sec;\n"+
			"double cetus_tic_offset, cetus_toc_offset;\n"+
			"double cetus_since["+num_loops+"];\n"+
			"double cetus_timer["+num_loops+"];\n"+
			"long cetus_counter["+num_loops+"];\n"+
			"char cetus_loop_name["+num_loops+"][32] = {\n"+
			"  "+PrintTools.listToString(loop_names, ",\n  ")+"\n};\n\n"+

			"int cetus_wtimeu()\n"+
			"{\n"+
			"  struct timeval tv;\n"+
			"  double ret;\n"+
			"  gettimeofday(&tv, 0);\n"+
			"  return (tv.tv_sec-cetus_sec)*1e6 + tv.tv_usec;\n"+
			"}\n\n"+

			"double cetus_wtime()\n"+
			"{\n"+
			"  struct timeval tv;\n"+
			"  double ret;\n"+
			"  gettimeofday(&tv, 0);\n"+
			"  return (tv.tv_sec-cetus_sec)+1.0e-6*tv.tv_usec;\n"+
			"}\n\n"+

			"void cetus_tic(int id)\n"+
			"{\n"+
			"  cetus_since[id] = cetus_wtime();\n"+
			"}\n\n"+

			"void cetus_toc(int id)\n"+
			"{\n"+
			"  cetus_timer[id] += cetus_wtime()-cetus_since[id];\n"+
			"  cetus_counter[id]++;\n"+
			"}\n\n"+

			"void cetus_init_timer()\n"+
			"{\n"+
			"  int i, t;\n"+
			"  struct timeval tv;\n"+
			"  gettimeofday(&tv, 0);\n"+
			"  cetus_sec = tv.tv_sec;\n"+
			"  t = cetus_wtimeu();\n"+
			"  for ( i=0; i<1e6; i++ ) cetus_tic("+(num_loops-1)+");\n"+
			"  cetus_tic_offset = (cetus_wtimeu()-t)*1.0e-6;\n"+
			"  t = cetus_wtimeu();\n"+
			"  for ( i=0; i<1e6; i++ ) cetus_toc("+(num_loops-1)+");\n"+
			"  cetus_toc_offset = (cetus_wtimeu()-t)*1.0e-6;\n"+
			"  for ( i=0; i<"+num_loops+"; i++ ) {\n"+
			"    cetus_since[i] = 0.0;\n"+
			"    cetus_timer[i] = 0.0;\n"+
			"    cetus_counter[i] = 0;\n"+
			"  }\n"+
			"}\n\n"+

			"void cetus_print_timer()\n"+
			"{\n"+
			"  int i, j;\n"+
			"  long invcs=0;\n"+
			"  double prog_time, loop_sum=0.0;\n"+
			"  for ( i=0; i<"+(num_loops-1)+"; i++ ) {\n"+
			"    double loop_time = cetus_timer[i] -\n"+
			"      cetus_counter[i]*cetus_toc_offset*1.0e-6;\n"+
			"    printf(\"[CETUS] %32s : \", cetus_loop_name[i]);\n"+
			"    printf(\"invoked = %.1e \", (double)cetus_counter[i]);\n"+
			"    printf(\"elapsed = %.2f\\n\", loop_time);\n"+
			"    loop_sum += loop_time;\n"+
			"    invcs += cetus_counter[i];\n"+
			"  }\n"+
			"  prog_time = cetus_timer[i] -\n"+
			"    1.0e-6*(cetus_tic_offset+cetus_toc_offset)*invcs;\n"+
			"  printf(\"[CETUS] loop time = %.2f\", loop_sum);\n"+
			"  printf(\"  profiled = "+(num_loops-1)+"\");\n"+
			"  printf(\"  invoked = %.2e\\n\", (double)invcs);\n"+
			"  printf(\"[CETUS] prog time = %.2f\\n\", prog_time);\n"+
			"}\n"+
			"#endif /* CETUS_TIMING */\n"+
			"/* End of Cetus timing routines */\n";

		CodeAnnotation header_code = new CodeAnnotation(header);

		/* Puts this code section at the top of the "main_tu" translation unit */
		main_tu.addDeclarationFirst(new AnnotationDeclaration(header_code));
	}
}
