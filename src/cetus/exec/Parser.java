package cetus.exec;
import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.util.Vector;

import cetus.base.grammars.NewCLexer;
import cetus.base.grammars.NewCParser;
import cetus.base.grammars.PreCLexer;
import cetus.base.grammars.PreCParser;
import cetus.hir.Declaration;
import cetus.hir.TranslationUnit;
import cetus.treewalker.CCTreeWalker;
import cetus.treewalker.CTreeWalker;
import cetus.treewalker.DagParser;
import cetus.treewalker.TreeNode;

public class Parser{

  /** This nested class continuously reads from the output
   * or error stream of an external parser and reproduces
   * the data on the JVM's output or error stream.
   */
  private class PipeThread extends Thread
  {
    private BufferedReader source;
    private PrintStream dest;

    PipeThread(BufferedReader source, PrintStream dest)
    {
      this.source = source;
      this.dest = dest;
    }

    @Override
	public void run()
    {
      String s = null;
      try {
        while ((s = source.readLine()) != null)
          dest.println(s);
      } catch (IOException e) {
        dest.println("cetus: I/O error on redirection, " + e);
      }
    }
  }
  /**
   * Parse this translation unit.
   *
   * @throws IOException if there is a problem accessing any file.
   */
  public TranslationUnit parse(String input_filename) throws IOException
  {
		return parseAntlr(input_filename);
/**
	* If you want to add another parser other than antlr, then you need to modify
	* parseExternal routine below for your purpose and update the cetus manual
	*
    return parseExternal(input_filename);
*/
  }

  /**
   * Parse the associated input file using the Antlr
   * parser and create IR for this translation unit.
   *
   * @throws IOException if there is any problem accessing the file.
   */
  protected TranslationUnit parseAntlr(String input_filename)
  {
    String currfile = input_filename;
		TranslationUnit tu = new TranslationUnit(input_filename);
    String filename = null;
    File f = null,myf=null;
    byte[] barray = null;
      //InputStream source = null;
      // pre step to handle header files
      // Insert markers for start and end of a header file
    String prename = null;

    /* Create the Antlr-derived lexer and parser through the ClassLoader
       so antlr.jar will be required only if the Antlr parser is used. */

    Class[] params = new Class[1];
    Object[] args = new Object[1];

    try  {
      Class class_PreCLexer  = getClass().getClassLoader().loadClass("cetus.base.grammars.PreCLexer");
      params[0] = InputStream.class;
      args[0] = new DataInputStream(new FileInputStream(currfile));
      PreCLexer lexer = (PreCLexer)class_PreCLexer.getConstructor(params).newInstance(args);

      Class class_PreCParser = getClass().getClassLoader().loadClass("cetus.base.grammars.PreCParser");
      params[0] = getClass().getClassLoader().loadClass("antlr.TokenStream");
      args[0] = lexer;
      PreCParser parser = (PreCParser)class_PreCParser.getConstructor(params).newInstance(args);

      File dir = new File (".");
      String currdir = dir.getCanonicalPath();

      File ff = new File(currfile);
      filename = ff.getName();
      prename = "cppinput_" + filename;
      myf = new File(currdir, prename);
      myf.deleteOnExit();
      FileOutputStream fo = new FileOutputStream(myf);
      parser.programUnit(new PrintStream(fo));
      fo.close();
    } catch (ClassNotFoundException e) {
      System.err.println("cetus: could not load class " + e);
      System.exit(1);
    } catch (NoSuchMethodException e) {
      System.err.println("cetus: could not find constructor " + e);
      System.exit(1);
    } catch (IllegalAccessException e) {
      System.err.println("cetus: could not access constructor " + e);
      System.exit(1);
    } catch (InstantiationException e) {
      System.err.println("cetus: constructor failed " + e);
      System.exit(1);
    } catch (FileNotFoundException e) {
      System.err.println("cetus: could not read input file " + e);
      System.exit(1);
    } catch (IOException e) {
      System.err.println("cetus: could not create intermdiate output file " + e);
      System.exit(1);
    } catch (Exception e) {
      System.err.println("cetus: exception: " + e);
      e.printStackTrace();
      System.exit(1);
    }

    // Run cpp on the input file and output to a temporary file.
	  String cmd = Driver.getOptionValue("preprocessor");
	  InputStream inputstream = null;

	  if (cmd != null && !"".equals (cmd))
	  {
	    try {

	      ByteArrayOutputStream bo = new ByteArrayOutputStream(50000);
	      PrintStream outStream = new PrintStream(bo);

				cmd += Parser.getMacros() + " " + prename;
	      Process p = Runtime.getRuntime().exec(cmd);

	      BufferedReader inReader = new BufferedReader(new InputStreamReader(p.getInputStream()));
	      BufferedReader errReader = new BufferedReader(new InputStreamReader(p.getErrorStream()));

	      PipeThread out_pipe = new PipeThread(inReader, outStream);
	      PipeThread err_pipe = new PipeThread(errReader, System.err);

	      out_pipe.start();
	      err_pipe.start();

	      if (p.waitFor() != 0)
	      {
	        System.err.println("cetus: preprocessor terminated with exit code " + p.exitValue());
	        System.exit(1);
	      }

	      out_pipe.join();
	      err_pipe.join();

	      barray = bo.toByteArray();
	      inputstream = new ByteArrayInputStream(barray);

	      //----------------
	      //System.out.write(barray, 0, Array.getLength(barray));

	    } catch (java.io.IOException e) {
	      System.err.println("Fatal error creating temporary file: " + e);System.exit(1);
	    } catch (java.lang.InterruptedException e) {
	      System.err.println("Fatal error starting preprocessor: " + e);System.exit(1);
	    }
    }
    else
    {
    	try
		{
    		// [MCH] --> commented out and replaced filename by currfile
			//inputstream = new FileInputStream (filename);
    		inputstream = new FileInputStream (currfile);
		}
		catch (FileNotFoundException e)
		{
			System.err.println("Fatal error reading input file: " + e);
			System.exit(1);
		}
    }

    // Actual antlr parser is called

    try {
      Class class_NewCLexer = getClass().getClassLoader().loadClass("cetus.base.grammars.NewCLexer");
      params[0] = InputStream.class;
      args[0] = new DataInputStream(inputstream);
      NewCLexer lexer = (NewCLexer)class_NewCLexer.getConstructor(params).newInstance(args);

      lexer.setOriginalSource(filename);
      lexer.setTokenObjectClass("cetus.base.grammars.CToken");
      lexer.initialize();

      Class class_NewCParser = getClass().getClassLoader().loadClass("cetus.base.grammars.NewCParser");
      params[0] = getClass().getClassLoader().loadClass("antlr.TokenStream");
      args[0] = lexer;
      NewCParser parser = (NewCParser)class_NewCParser.getConstructor(params).newInstance(args);

      parser.getPreprocessorInfoChannel(lexer.getPreprocessorInfoChannel());
      parser.setLexer(lexer);
      parser.translationUnit(tu);
    } catch (Exception e) {
      System.err.println("Parse error: " + e);
      System.exit(1);
    }

/*
    try {
      Class[] pparams = new Class[2];
      pparams[0] = TranslationUnit.class;
      pparams[1] = OutputStream.class;
      pparams[1] = PrintWriter.class;
      tu.setPrintMethod(pparams[0].getMethod("defaultPrint2", pparams));
    } catch (NoSuchMethodException e) {
      throw new InternalError();
    }
*/
		return tu;
  }

  /**
   * Parse the associated input file using an external
   * parser and create IR for this translation unit.
   *
   * @throws IOException if there is any problem accessing the file.
   */
  protected TranslationUnit
      parseExternal(String input_filename) throws IOException
  {
    /* parser can be a symlink to any parser program that can dump
       parse trees in graphviz format.  If your parser does not
       accept the arguments below, we suggest that you make parser a
       symlink to an executable shell script that does what you want.
       (Or, you could just hack this code and recompile Cetus.) */

    Vector<String> cmd = new Vector<String>();

    cmd.add("./parser");
    cmd.add("--dump-parse-trees");

    if (Driver.getOptionValue("echo") != null)
      cmd.add("--echo");

    /* The preprocessor is either cpp -C or gcc -E but they are specified as
       simply "cpp" or "gcc".  Passing "--preprocessor='cpp -C'" on the
       command-line causes many programs to parse that as two arguments, with
       the second being -C'".  It's up to the parser program itself to add
       preprocessor flags when it invokes the preprocessor. */
    cmd.add("--preprocessor=" + Driver.getOptionValue("preprocessor"));
    cmd.add(input_filename);

    try {
      Process p = Runtime.getRuntime().exec(cmd.toArray(new String[1]));

      /* read the parser's output stream */
      /* Note: Yes, this is reading the output stream even though it says getInputStream.
         Sun's naming conventions are weird like that. */
      BufferedReader parser_stdout = new BufferedReader(new InputStreamReader(p.getInputStream()));

      /* read the parser's error stream */
      BufferedReader parser_stderr = new BufferedReader(new InputStreamReader(p.getErrorStream()));

      /* Redirect all output from the parser to the JVM's output and error streams. */
      PipeThread out_pipe = new PipeThread(parser_stdout, System.out);
      PipeThread err_pipe = new PipeThread(parser_stderr, System.err);

      out_pipe.start();
      err_pipe.start();

      /* Wait on the parser to finish. */
      if (p.waitFor() != 0)
      {
        System.err.println("cetus: parser terminated with exit code " + p.exitValue());
        System.exit(1);
      }

      out_pipe.join();
      err_pipe.join();
    } catch (InterruptedException e) {
      System.err.println("cetus: interrupted waiting for parser to finish");
    }

    TranslationUnit ret = new TranslationUnit(input_filename);

    if (input_filename.endsWith(".cc") || input_filename.endsWith(".cpp"))
    {
      /* C++ */

      DagParser dag_parser = new DagParser();
      TreeNode tree_root = dag_parser.run(input_filename + ".dag");

      CCTreeWalker cc_treewalk = new CCTreeWalker(input_filename);
      TranslationUnit tu = cc_treewalk.run(tree_root);

      for (int i=0; i<tu.getChildren().size(); i++) {
        tu.getChildren().get(i).setParent(null);
        ret.setChild(i, tu.getChildren().get(i));
      }

      for (Declaration decl : tu.getDeclarations()) {
        decl.detach();
        if ( !ret.containsDeclaration(decl) )
          ret.addDeclaration(decl);
      }
    }
    else if (input_filename.endsWith(".c"))
    {
      /* C */

      DagParser dag_parser = new DagParser();
      TreeNode tree_root = dag_parser.run(input_filename + ".dag");

      CTreeWalker c_treewalk = new CTreeWalker(input_filename);
      TranslationUnit tu = c_treewalk.run(tree_root);

      for (int i=0; i<tu.getChildren().size(); i++) {
        tu.getChildren().get(i).setParent(null);
        ret.setChild(i, tu.getChildren().get(i));
      }

      for (Declaration decl : tu.getDeclarations()) {
        decl.detach();
        if ( !ret.containsDeclaration(decl) )
          ret.addDeclaration(decl);
      }
    }
    else
    {
      System.out.println("File name \"" + input_filename + "\" is not a valid C or C++ filename. Quitting ...\n");
      System.exit(1);
    }

    return ret;
  }


	// Reads option value from -macro and returns a converted string to be added
	// in the preprocessor cmd line.
	protected static String getMacros()
	{
		String ret = " ";
		String macro = Driver.getOptionValue("macro");
		if ( macro == null )
			return ret;

		String[] macro_list = macro.split(",");
		for ( int i=0; i<macro_list.length; i++ )
			ret += (" -D"+macro_list[i]);

		return ret;
	}
}
