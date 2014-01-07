package cetus.exec;

import java.io.IOException;
import java.util.HashSet;

import cetus.analysis.AnalysisPass;
import cetus.analysis.ArrayPrivatization;
import cetus.analysis.CallGraph;
import cetus.analysis.DDTDriver;
import cetus.analysis.LoopParallelizationPass;
import cetus.analysis.Reduction;
import cetus.codegen.CodeGenPass;
import cetus.codegen.ompGen;
import cetus.hir.PrintTools;
import cetus.hir.Program;
import cetus.hir.SymbolTools;
import cetus.transforms.AnnotationParser;
import cetus.transforms.IVSubstitution;
import cetus.transforms.InlineExpansionPass;
import cetus.transforms.LoopNormalization;
import cetus.transforms.LoopProfiler;
import cetus.transforms.LoopTiling;
import cetus.transforms.NormalizeReturn;
import cetus.transforms.SingleCall;
import cetus.transforms.SingleDeclarator;
import cetus.transforms.SingleReturn;
import cetus.transforms.TransformPass;

/**
 * Implements the command line parser and controls pass ordering.
 * Users may extend this class by overriding runPasses
 * (which provides a default sequence of passes).  The derived
 * class should pass an instance of itself to the run method.
 * Derived classes have access to a protected {@link Program Program} object.
 */
public class Driver
{
  /**
   * A mapping from option names to option values.
   */
  protected static CommandLineOptionSet options = new CommandLineOptionSet();

  /**
   * Override runPasses to do something with this object.
   * It will contain a valid program when runPasses is called.
   */
  protected Program program;

  /** The filenames supplied on the command line. */
  protected String[] filenames;

  /**
   * Constructor used by derived classes.
   */
  protected Driver()
  {
    Driver.options.add(Driver.options.ANALYSIS, "callgraph",
                "Print the static call graph to stdout");
    Driver.options.add(Driver.options.UTILITY, "expand-user-header",
                "Expand user (non-standard) header file #includes into code");
    Driver.options.add(Driver.options.UTILITY, "expand-all-header",
                "Expand all header file #includes into code");
    Driver.options.add(Driver.options.UTILITY, "help",
                "Print this message");
    Driver.options.add(Driver.options.TRANSFORM, "induction",
                "Perform induction variable substitution");
    Driver.options.add(Driver.options.UTILITY, "outdir", "dirname",
                "Set the output directory name (default is cetus_output)");
    Driver.options.add(Driver.options.TRANSFORM, "normalize-loops",
                "Normalize for loops so they begin at 0 and have a step of 1");
    Driver.options.add(Driver.options.UTILITY, "preprocessor", "command",
                "Set the preprocessor command to use");
    Driver.options.add(Driver.options.ANALYSIS, "privatize",
                "Perform scalar/array privatization analysis");
    Driver.options.add(Driver.options.ANALYSIS, "reduction", "N",
                "Perform reduction variable analysis\n"
                + "      =1 enable only scalar reduction analysis (default)\n"
                + "      =2 enable array reduction analysis as well");
    Driver.options.add(Driver.options.UTILITY, "skip-procedures", "proc1,proc2,...",
                "Causes all passes that observe this flag to skip the listed procedures");
    Driver.options.add(Driver.options.TRANSFORM, "tsingle-call",
                "Transform all statements so they contain at most one function call");
    Driver.options.add(Driver.options.TRANSFORM, "tinline-expansion",
    			"(Experimental) Perform simple subroutine inline expansion tranformation");
    Driver.options.add(Driver.options.TRANSFORM, "tsingle-declarator",
                "Transform all variable declarations so they contain at most one declarator");
    Driver.options.add(Driver.options.TRANSFORM, "tsingle-return",
                "Transform all procedures so they have a single return statement");
    Driver.options.add(Driver.options.UTILITY, "verbosity", "N",
                "Degree of status messages (0-4) that you wish to see (default is 0)");
    Driver.options.add(Driver.options.UTILITY, "version",
                "Print the version information");

    Driver.options.add(Driver.options.ANALYSIS, "ddt", "N",
        "Perform Data Dependence Testing\n"
        + "      =1 banerjee-wolfe test (default)\n"
        + "      =2 range test\n"
        + "      =3 not used");

    Driver.options.add(Driver.options.ANALYSIS, "parallelize-loops",
          "Annotate loops with Parallelization decisions");

    Driver.options.add(Driver.options.CODEGEN, "ompGen", "N",
        "Generate OpenMP pragma\n"
        + "      =1 keep existing pragmas (default)\n"
        + "      =2 remove existing OpenMP pragma\n"
        + "      =3 remove cetus-internal pragma\n"
        + "      =4 remove both");

/*
    options.add(options.TRANSFORM, "loop-interchange",
                "Interchange loop to improve locality (This flag should be used with -ddt flag)");
*/

    Driver.options.add(Driver.options.TRANSFORM, "profile-loops", "N",
        "Inserts loop-profiling calls\n"
        + "      =1 every loop          =2 outer-most loop\n"
        + "      =3 auto-parallel loop  =4 outer-most auto-parallel loop\n"
        + "      =5 OpenMP loop         =6 outer-most OpenMP loop");

	Driver.options.add(Driver.options.UTILITY, "macro",
				"Sets macros for the specified names with comma-separated list (no space is allowed). e.g., -macro=ARCH=i686,OS=linux");

	Driver.options.add(Driver.options.ANALYSIS, "alias", "N",
		"Specify level of alias analysis\n"
		+ "      =0 disable alias analysis\n"
		+ "      =1 advanced interprocedural analysis (default)\n"
		+ "         Uses interprocedural points-to analysis"
		);
//*
    Driver.options.add(Driver.options.TRANSFORM, "loop-tiling",
                "Loop tiling");
//*/
    Driver.options.add(Driver.options.TRANSFORM, "normalize-return-stmt",
    	"Normalize return statements for all procedures");
    Driver.options.add(Driver.options.ANALYSIS, "range", "N",
      "Specifies the accuracy of symbolic analysis with value ranges\n"
      + "      =0 disable range computation (minimal symbolic analysis)\n"
      + "      =1 enable local range computation (default)\n"
      + "      =2 enable inter-procedural computation (experimental)");
    Driver.options.add(Driver.options.UTILITY, "preserve-KR-function",
        "Preserves K&R-style function declaration");
    }

  /**
   * Returns the value of the given key or null
   * if the value is not set.  Key values are
   * set on the command line as <b>-option_name=value</b>.
   *
   * @return the value of the given key or null if the
   *   value is not set.
   */
  public static String getOptionValue(String key)
  {
    return Driver.options.getValue(key);
  }

  /**
   * Returns the set a procedure names that should be
   * excluded from transformations.  These procedure
   * names are specified with the skip-procedures
   * command line option by providing a comma-separated
   * list of names. */
  public static HashSet getSkipProcedureSet()
  {
    HashSet<String> proc_skip_set = new HashSet<String>();

    String s = Driver.getOptionValue("skip-procedures");
    if (s != null)
    {
      String[] proc_names = s.split(",");
      for (String name : proc_names)
        proc_skip_set.add(name);
    }

    return proc_skip_set;
  }

  /**
   * Parses command line options to Cetus.
   *
   * @param args The String array passed to main by the system.
   */
  protected void parseCommandLine(String[] args)
  {
    /* print a useful message if there are no arguments */
    if (args.length == 0)
    {
      printUsage();
      System.exit(1);
    }

    /* set default option values */
    Driver.setOptionValue("outdir", "cetus_output");
    //Driver.setOptionValue("preprocessor", "cpp -C");
    Driver.setOptionValue("verbosity", "0");
    Driver.setOptionValue("alias", "1");

    int i; /* used after loop; don't put inside for loop */
    for (i = 0; i < args.length; ++i)
    {
      String opt = args[i];

      if (opt.charAt(0) != '-')
        /* not an option -- skip to handling options and filenames */
        break;

      int eq = opt.indexOf('=');

      if (eq == -1)
      {
        /* no value on the command line, so just set it to "1" */
        String option_name = opt.substring(1);

        if (Driver.options.contains(option_name))
          Driver.setOptionValue(option_name, "1");
        else
          System.err.println("ignoring unrecognized option " + option_name);
      }
      else
      {
        /* use the value from the command line */
        String option_name = opt.substring(1, eq);

        if (Driver.options.contains(option_name))
          Driver.setOptionValue(option_name, opt.substring(eq + 1));
        else
          System.err.println("ignoring unrecognized option " + option_name);
      }
    }

    if (Driver.getOptionValue("help") != null || Driver.getOptionValue("usage") != null)
    {
      printUsage();
      System.exit(0);
    }

    if (Driver.getOptionValue("version") != null)
    {
      printVersion();
      System.exit(0);
    }

    if (i >= args.length)
    {
      System.err.println("No input files!");
      System.exit(1);
    }

    filenames = new String[args.length - i];
    for (int j = 0; j < filenames.length; ++j)
      filenames[j] = args[i++];
  }

  /**
   * Parses all of the files listed in <var>filenames</var>
   * and creates a {@link Program Program} object.
   */
/*
  protected void parseFiles_old()
  {
    try {
      program = new Program(filenames);
      program.parse();
    } //catch (TreeWalkException e) {
      //System.err.println("failed to build IR from syntax tree");
      //System.err.println(e);
      //System.exit(1);
    //}
    catch (IOException e) {
      System.err.println("I/O error parsing files");
      System.err.println(e);
      System.exit(1);
    } catch (Exception e) {
      System.err.println("Miscellaneous exception while parsing files: " + e);
      e.printStackTrace();
      System.exit(1);
    }
  }
*/

  protected void parseFiles()
	{
	  throw new RuntimeException ("Not implemented");
	  
//    try {
//      program = new Program();
//      Parser parser = new Parser();
//      for(String file : filenames)
//      program.addTranslationUnit(parser.parse(file));
//    } //catch (TreeWalkException e) {
//      //System.err.println("failed to build IR from syntax tree");
//      //System.err.println(e);
//      //System.exit(1);
//    //}
//    catch (IOException e) {
//      System.err.println("I/O error parsing files");
//      System.err.println(e);
//      System.exit(1);
//    } catch (Exception e) {
//      System.err.println("Miscellaneous exception while parsing files: " + e);
//      e.printStackTrace();
//      System.exit(1);
//		}
	}
  /**
   * Prints the list of options that Cetus accepts.
   */
  public void printUsage()
  {
    String usage = "\ncetus.exec.Driver [option]... [file]...\n" +
		"------------------------------------------------------\n";
    usage += Driver.options.getUsage();
    System.err.println(usage);
  }

  /**
   * Prints the compiler version.
   */
  public void printVersion()
  {
    System.err.println("Cetus 1.2 - A Source-to-Source Compiler for C");
    System.err.println("http://cetus.ecn.purdue.edu");
    System.err.println("Copyright (C) 2002-2010 ParaMount Research Group");
    System.err.println("Purdue University - School of Electrical & Computer Engineering");
  }

  /**
   * Runs this driver with args as the command line.
   *
   * @param args The command line from main.
   */
  public void run(String[] args)
  {
    parseCommandLine(args);

    parseFiles();

    if (Driver.getOptionValue("parse-only") != null)
    {
      System.err.println("parsing finished and parse-only option set");
      System.exit(0);
    }

    runPasses();

    PrintTools.printlnStatus("Printing...", 1);

    try {
      program.print();
    } catch (IOException e) {
      System.err.println("could not write output files: " + e);
      System.exit(1);
    }
  }

  /**
   * Runs analysis and optimization passes on the program.
   */
  public void runPasses()
  {
    /* check for option dependences */

    /* in each set of option strings, the first option requires the
       rest of the options to be set for it to run effectively */
    String[][] pass_prerequisites = {
			{ "inline", "tsingle-call", "tsingle-return" },
			{ "parallelize-loops", "alias", "ddt", "privatize",
				"reduction", "induction", "ompGen" },
/*
			{ "loop-interchange", "ddt" }
*/
			};

    for (int i = 0; i < pass_prerequisites.length; ++i)
    {
      if (Driver.getOptionValue(pass_prerequisites[i][0]) != null)
      {
        for (int j = 1; j < pass_prerequisites[i].length; ++j)
        {
          if (Driver.getOptionValue(pass_prerequisites[i][j]) == null)
          {
            System.out.println("WARNING: " + pass_prerequisites[i][0] + " flag is set but " + pass_prerequisites[i][j] + " is not set");
            System.out.println("WARNING: turning on " + pass_prerequisites[i][j]);
            Driver.setOptionValue(pass_prerequisites[i][j], "1");
          }
        }
      }
    }

    /* Link IDExpression => Symbol object for faster future access. */
    SymbolTools.linkSymbol(program);

		/* Convert the IR to a new one with improved annotation support */
		TransformPass.run(new AnnotationParser(program));

    if (Driver.getOptionValue("callgraph") != null)
    {
      CallGraph cg = new CallGraph(program);
      cg.print(System.out);
    }

    if (Driver.getOptionValue("tsingle-declarator") != null)
    {
      SingleDeclarator.run(program);
    }

    if (Driver.getOptionValue("tsingle-call") != null)
    {
      TransformPass.run(new SingleCall(program));
    }

    if (Driver.getOptionValue("tsingle-return") != null)
    {
      TransformPass.run(new SingleReturn(program));
    }

    if (Driver.getOptionValue("tinline-expansion") != null)
    {
      TransformPass.run(new InlineExpansionPass(program));
    }

    if (Driver.getOptionValue("normalize-loops") != null)
    {
      TransformPass.run(new LoopNormalization(program));
    }

    if (Driver.getOptionValue("normalize-return-stmt") != null)
    {
    	TransformPass.run(new NormalizeReturn(program));
    }

    if (Driver.getOptionValue("induction") != null)
    {
      TransformPass.run(new IVSubstitution(program));
    }

    if (Driver.getOptionValue("privatize") != null)
    {
			AnalysisPass.run(new ArrayPrivatization(program));
    }

    if (Driver.getOptionValue("ddt") != null)
    {
      AnalysisPass.run(new DDTDriver(program));
    }

    if (Driver.getOptionValue("reduction") != null)
    {
      AnalysisPass.run(new Reduction(program));
    }

		// CHECK
		/*
    if (getOptionValue("openmp") != null)
    {
      AnalysisPass.run(new OmpAnalysis(program));
    }
		*/

    if (Driver.getOptionValue("parallelize-loops") != null)
    {
      AnalysisPass.run(new LoopParallelizationPass(program));
    }

    if (Driver.getOptionValue("ompGen") != null)
    {
      CodeGenPass.run(new ompGen(program));
    }

/*
    if (getOptionValue("loop-interchange") != null)
    {
      TransformPass.run(new LoopInterchange(program));
    }
*/

    //*
    if (Driver.getOptionValue("loop-tiling") != null)
    {
      AnalysisPass.run(new LoopTiling(program));
    }
    //*/


    if (Driver.getOptionValue("profile-loops") != null)
    {
    	TransformPass.run(new LoopProfiler(program));
    }
  }

  /**
   * Sets the value of the option represented by <i>key</i> to
   * <i>value</i>.
   *
   * @param key The option name.
   * @param value The option value.
   */
  protected static void setOptionValue(String key, String value)
  {
    Driver.options.setValue(key, value);
  }

  /**
   * Entry point for Cetus; creates a new Driver object,
   * and calls run on it with args.
   *
   * @param args Command line options.
   */
  public static void main(String[] args)
  {
    (new Driver()).run(args);
  }

}

