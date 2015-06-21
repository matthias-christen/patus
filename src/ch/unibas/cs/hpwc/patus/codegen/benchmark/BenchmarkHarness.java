/*******************************************************************************
 * Copyright (c) 2011 Matthias-M. Christen, University of Basel, Switzerland.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the GNU Lesser Public License v2.1
 * which accompanies this distribution, and is available at
 * http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
 *
 * Contributors:
 *     Matthias-M. Christen, University of Basel, Switzerland - initial API and implementation
 ******************************************************************************/
package ch.unibas.cs.hpwc.patus.codegen.benchmark;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.log4j.Logger;

import ch.unibas.cs.hpwc.patus.ast.StatementList;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;
import ch.unibas.cs.hpwc.patus.codegen.KernelSourceFile;
import ch.unibas.cs.hpwc.patus.util.FileUtil;
import ch.unibas.cs.hpwc.patus.util.IndentOutputStream;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 *
 * @author Matthias-M. Christen
 */
public class BenchmarkHarness
{
	///////////////////////////////////////////////////////////////////
	// Constants

	private final static Logger LOGGER = Logger.getLogger (BenchmarkHarness.class);

	/**
	 * Pattern for &quot;#pragma patus&quot; pragmas in C files
	 */
	private final static Pattern PATTERN_PRAGMA = Pattern.compile ("\\s*#pragma\\s+patus\\s+([A-Za-z0-9_]+)\\s*(\\((.+)\\))?");
	
	/**
	 * Pattern for &quot;PATUS_*&quot; variables in C files
	 */
	//private final static Pattern PATTERN_CVAR = Pattern.compile ("PATUS_([A-Za-z0-9_]+)\\s*(\\((.+)\\))?");
	private final static Pattern PATTERN_CVAR = Pattern.compile ("PATUS_([A-Za-z0-9_]+)");

	/**
	 * Pattern for &quot;PATUS_*&quot; variables in Makefiles
	 */
	private final static Pattern PATTERN_MAKEVAR = Pattern.compile ("\\$\\(PATUS_([A-Za-z0-9_]+)\\)");


	///////////////////////////////////////////////////////////////////
	// Member Variables

	private CodeGeneratorSharedObjects m_data;

	/**
	 * List of runtime C source files that need to be included in the benchmark project
	 */
	private List<String> m_listRuntimeFiles;


	///////////////////////////////////////////////////////////////////
	// Implementation

	public BenchmarkHarness (CodeGeneratorSharedObjects data)
	{
		m_data = data;
		m_listRuntimeFiles = new LinkedList<> ();
	}

	/**
	 *
	 * @param fileOutputDir
	 */
	public void generate (File fileOutputDir, KernelSourceFile out)
	{
		File f = null;
		File fileArchDescr = m_data.getArchitectureDescription ().getFile ();
		if (fileArchDescr != null)
		{
			f = new File (fileArchDescr.getParent (), m_data.getArchitectureDescription ().getBuild ().getHarnessTemplateDir ());
			if (!f.exists ())
				f = null;
		}

		if (f == null)
			f = FileUtil.getFileRelativeToJar (m_data.getArchitectureDescription ().getBuild ().getHarnessTemplateDir ());
		
		LOGGER.info (StringUtil.concat ("Using directory ", f.getPath (), " for benchmarking harness..."));
		if (f.equals (fileOutputDir))
			throw new RuntimeException ("Benchmark harness directory and output directory are equal. Aborting...");

		// clean the output directory
		//FileUtil.cleanOutputDirectory (fileOutputDir);

		// process the input files in the benchmark harness directory
		m_data.getCodeGenerators ().getBackendCodeGenerator ().setKernelSourceFile (out);
		listFiles (f, f, fileOutputDir);
	}

	/**
	 * Determines whether the file <code>f</code> is a C or C++ source file (by its extension).
	 * @param f The file to check
	 * @return <code>true</code> if the file <code>f</code> is a C/C++ source file
	 */
	private static boolean isCSourceFile (File f)
	{
		String strExtension = FileUtil.getExtension (f).toLowerCase ();
		return "c".equals (strExtension) || "cpp".equals (strExtension) || "cu".equals (strExtension);
	}

	/**
	 * Determines whether the file <code>f</code> is a Makefile.
	 * @param f The file to check
	 * @return <code>true</code> if the file <code>f</code> is a Makefile
	 */
	private static boolean isMakefile (File f)
	{
		return "makefile".equals (f.getName ().toLowerCase ());
	}

	/**
	 * Determines whether the file <code>f</code> is a build file for Ant.
	 * @param f The file to check
	 * @return <code>true</code> if the file <code>f</code> is a Ant build file
	 */
	private static boolean isAntBuildFile (File f)
	{
		return "build.xml".equals (f.getName ());
	}

	/**
	 * Determines whether the file <code>f</code> is a header file (by its extension).
	 * @param f The file to check
	 * @return <code>true</code> if the file <code>f</code> is a header file
	 */
	private static boolean isHeaderFile (File f)
	{
		String strExtension = FileUtil.getExtension (f);
		return "h".equals (strExtension);
	}

	/**
	 * Returns an iterable over of runtime source files
	 * @return
	 */
	private Iterable<String> getRuntimeSourceFiles ()
	{
		List<String> list = new ArrayList<> (m_listRuntimeFiles.size ());
		for (String strRuntimeFile : m_listRuntimeFiles)
		{
			int nIdx = strRuntimeFile.lastIndexOf (".c");
			if (nIdx >= 0)
				list.add (strRuntimeFile);
		}

		return list;
	}

	/**
	 * Returns an iterable over generated object file names of runtime files
	 * @return
	 */
	private Iterable<String> getRuntimeObjectFiles ()
	{
		List<String> list = new ArrayList<> (m_listRuntimeFiles.size ());
		for (String strRuntimeFile : m_listRuntimeFiles)
		{
			int nIdx = strRuntimeFile.lastIndexOf (".c");
			if (nIdx >= 0)
				list.add (StringUtil.concat (strRuntimeFile.substring (0, nIdx), ".o"));
		}

		return list;
	}

	/**
	 * Recursively lists all the files in the input directory <code>fileInputDir</code>
	 * and processes the files. Processed files are written to the output directory,
	 * <code>fileOutputDir</code>.
	 * @param fileInputDir The input directory to scan for files to process
	 * @param fileBaseDir The base directory (root for all input files)
	 * @param fileOutputDir The output directory (base) to which the processed files are written
	 */
	private void listFiles (File fileInputDir, File fileBaseDir, File fileOutputDir)
	{
		// skip hidden directories
		if (fileInputDir.getName ().startsWith ("."))
			return;

		String strRelativePathToOutputDir = FileUtil.relativeTo (fileBaseDir, fileInputDir);
		File fileRelOutputDir = new File (fileOutputDir, strRelativePathToOutputDir);

		// copy runtime files
		File fileRuntimeDir = FileUtil.getFileRelativeToJar ("runtime");
		LOGGER.info (StringUtil.concat ("Copying runtime files from ", fileRuntimeDir.getPath ()));
		File[] rgFiles = fileRuntimeDir.listFiles ();
		if (rgFiles != null)
		{
			for (File f : rgFiles)
			{
				boolean bIsCSourceFile = BenchmarkHarness.isCSourceFile (f);
				if (bIsCSourceFile || isHeaderFile (f))
				{
					try
					{
						File fileDest = new File (fileRelOutputDir, f.getName ());
						LOGGER.info (StringUtil.concat ("Copying file ", f.getPath (), " to ", fileDest.getPath ()));
						FileUtil.copy (f, fileDest);
					}
					catch (IOException e)
					{
						throw new RuntimeException (StringUtil.concat (
							"Copying of runtime file ", f.getName (), " to output directory failed: ", e.getMessage ()));
					}
				}

				if (bIsCSourceFile)
					m_listRuntimeFiles.add (f.getName ());
			}
		}

		// get all the files in the directory
		rgFiles = fileInputDir.listFiles ();
		if (rgFiles != null)
		{
			// process individual files
			for (File f : rgFiles)
			{
				if (f.isDirectory ())
					listFiles (f, fileBaseDir, fileOutputDir);
				else
				{
					try
					{
						processFile (f, new File (fileRelOutputDir, f.getName ()));
					}
					catch (IOException e)
					{
						LOGGER.error (StringUtil.concat ("Error while processing file ", f.getAbsolutePath (), " in benchmark harness creation."));
					}
				}
			}
		}
	}

	/**
	 * Processes the file <code>fileInput</code> and creates an output file,
	 * <code>fileOutput</code> from it.
	 * @param fileInput The input file to process
	 * @param fileOutput The file to which the processed output is written
	 * @throws IOException
	 */
	private void processFile (File fileInput, File fileOutput) throws IOException
	{
		LOGGER.info (StringUtil.concat ("Processing ", fileInput.getName ()));

		// skip hidden files
		if (fileInput.getName ().startsWith ("."))
			return;

		// make sure that the path exists
		fileOutput.getParentFile ().mkdirs ();

		if (BenchmarkHarness.isCSourceFile (fileInput))
			preprocessSourceFile (fileInput, fileOutput);
		else if (isMakefile (fileInput))
			processMakefile (fileInput, fileOutput);
		else
			FileUtil.copy (fileInput, fileOutput);
	}

	/**
	 * Processes a C source file.
	 * @param fileInput
	 * @param fileOutput
	 * @throws IOException
	 */
	private void preprocessSourceFile (File fileInput, File fileOutput) throws IOException
	{
		// create reader and writer
		BufferedReader in = new BufferedReader (new FileReader (fileInput));
		PrintWriter out = new PrintWriter (new IndentOutputStream (new FileOutputStream (fileOutput)));

		// process the file and replace Patus pragmas by the appropriate code
		Matcher matcherPragma = null;
		Matcher matcherVar = null;
		for (boolean bReadingIncludeFiles = true; ; )
		{
			String strLine = in.readLine ();
			if (strLine == null)
				break;

			// append the include files specified in the hardware description and the Patus runtime include (as last includes)
			if (bReadingIncludeFiles)
			{
				if (!strLine.startsWith ("#include ") && !strLine.trim ().equals (""))
				{
					// "hardware" includes
					for (String strIncludeFile : m_data.getArchitectureDescription ().getIncludeFiles ())
					{
						out.print ("#include <");
						out.print (strIncludeFile);
						out.println (">");
					}

					// runtime
					out.println ("#include \"patusrt.h\"\n");
					bReadingIncludeFiles = false;
				}
			}

			// create or reset the #pragma matcher and the variable matcher
			if (matcherPragma == null)
				matcherPragma = BenchmarkHarness.PATTERN_PRAGMA.matcher (strLine);
			else
				matcherPragma.reset (strLine);

			// if a Patus pragma has been found, replace it by code generated by the backend code generator
			if (matcherPragma.matches ())
			{
				String strPragma = matcherPragma.group (1);
				out.print ("// ");
				out.print (strPragma);
				out.println (" -->");

				generateCodeForPragma (strPragma, matcherPragma.group (3), out);

				out.print ("// <--\n\n");
			}
			else
			{
				if (matcherVar == null)
					matcherVar = BenchmarkHarness.PATTERN_CVAR.matcher (strLine);
				else
					matcherVar.reset (strLine);

				StringBuffer sbLine = new StringBuffer ();
				while (matcherVar.find ())
				{
					String strVar = matcherVar.group (1);
					String strCodeForVar = generateCodeForVar (strVar, null /*matcherVar.group (3)*/);
					if (strCodeForVar != null)
						matcherVar.appendReplacement (sbLine, strCodeForVar);
				}
				matcherVar.appendTail (sbLine);
				out.println (sbLine.toString ().trim ());
			}
		}

		in.close ();
		out.close ();
	}
	
	private static Object[] getArgs (String strArgList)
	{
		if (strArgList == null)
			return new Object[] { };
		
		int nStart = 0;
		int nEnd = strArgList.length () - 1;
		
		// trim leading and trailing brackets
		while (nStart < strArgList.length () && strArgList.charAt (nStart) == '(')
			nStart++;
		while (nEnd >= 0 && strArgList.charAt (nEnd) == ')')
			nEnd--;
		
		String[] rgStringArgs = strArgList.substring (nStart, nEnd + 1).split (",");
		Object[] rgArgs = new Object[rgStringArgs.length];
		for (int i = 0; i < rgStringArgs.length; i++)
		{
			// trim quotation marks
			nStart = 0;
			nEnd = rgStringArgs[i].length () - 1;
			
			while (nStart < rgStringArgs[i].length () && (rgStringArgs[i].charAt (nStart) == '"' || Character.isSpaceChar (rgStringArgs[i].charAt (nStart))))
				nStart++;
			while (nEnd >= 0 && (rgStringArgs[i].charAt (nEnd) == '"' || Character.isSpaceChar (rgStringArgs[i].charAt (nEnd))))
				nEnd--;
				
			rgArgs[i] = rgStringArgs[i].substring (nStart, nEnd + 1);
		}

		return rgArgs;
	}

	/**
	 * Generates code for the Patus pragma <code>strMethodName</code> and writes it to <code>out</code>.
	 * @param strMethodName The Patus pragma
	 * @param out The output print writer
	 */
	private void generateCodeForPragma (String strMethodName, String strArgList, PrintWriter out)
	{
		// try different naming conventions (with underscores: strMethodName) and with camel toes
		String strMethodName1 = StringUtil.toCamelToe (strMethodName);
		Object[] rgArgs = getArgs (strArgList);

		// try to find the methods
		Method method = findMethod (rgArgs.length, strMethodName, strMethodName1);
		if (method == null)
		{
			throw new RuntimeException (StringUtil.concat (
				"Can't find a code generator method for the pragma \"patus ", strMethodName,
				"\" on the backend ", m_data.getArchitectureDescription ().getBackend ()));
		}

		StatementList slResult = null;
		try
		{
			slResult = (StatementList) method.invoke (m_data.getCodeGenerators ().getBackendCodeGenerator (), rgArgs);
		}
		catch (IllegalArgumentException e)
		{
		}
		catch (IllegalAccessException e)
		{
			throw new RuntimeException (e);
		}
		catch (InvocationTargetException e)
		{
			e.getTargetException ().printStackTrace ();
		}
		catch (Exception e)
		{
			// other exceptions... (related to code generating)
			e.printStackTrace ();
		}

		if (slResult != null)
			out.println (slResult.toStringWithDeclarations ());
	}

	private String generateCodeForVar (String strVar, String strArgList)
	{
		Object[] rgArgs = getArgs (strArgList);
		
		String strMethodName = "get_" + strVar;
		String strMethodName1 = StringUtil.toCamelToe (strMethodName);
		String strMethodName2 = StringUtil.toCamelToe (strMethodName.toLowerCase ());
		Method method = findMethod (rgArgs.length, strMethodName, strMethodName1, strMethodName2);

		if (method == null)
		{
			throw new RuntimeException (StringUtil.concat (
				"Can't find a code generator method for the Patus variable \"PATUS_", strVar,
				"\" on the backend ", m_data.getArchitectureDescription ().getBackend ()));
		}

		try
		{
			return method.invoke (m_data.getCodeGenerators ().getBackendCodeGenerator (), rgArgs).toString ();
		}
		catch (IllegalArgumentException e)
		{
		}
		catch (IllegalAccessException e)
		{
			throw new RuntimeException (e);
		}
		catch (InvocationTargetException e)
		{
			e.getTargetException ().printStackTrace ();
		}
		catch (Exception e)
		{
			// other exceptions... (related to code generating)
			e.printStackTrace ();
		}

		return null;
	}

	/**
	 * Searches a method with a name contained in <code>rgMethodNames</code> on the backend
	 * code generator and returns the first one that is found.
	 * @param rgMethodNames An array of method names
	 * @return The first method that matches a name in <code>rgMethodNames</code>
	 */
	private Method findMethod (int nParamsCount, String... rgMethodNames)
	{
		Class<?>[] rgParamTypes = new Class<?>[nParamsCount];
		for (int i = 0; i < nParamsCount; i++)
			rgParamTypes[i] = String.class;
		
		for (String strMethodName : rgMethodNames)
		{
			try
			{
				return m_data.getCodeGenerators ().getBackendCodeGenerator ().getClass ().getMethod (strMethodName, rgParamTypes);
			}
			catch (SecurityException e)
			{
			}
			catch (NoSuchMethodException e)
			{
			}
		}

		return null;
	}

	/**
	 * Processes a Makefile.
	 * @param fileInput
	 * @param fileOutput
	 */
	private void processMakefile (File fileInput, File fileOutput) throws IOException
	{
		// create reader and writer
		BufferedReader in = new BufferedReader (new FileReader (fileInput));
		PrintWriter out = new PrintWriter (fileOutput);

		// process the file and replace Patus pragmas by the appropriate code
		Matcher matcher = null;
		for ( ; ; )
		{
			String strLine = in.readLine ();
			if (strLine == null)
				break;

			// create or reset the #pragma matcher
			if (matcher == null)
				matcher = BenchmarkHarness.PATTERN_MAKEVAR.matcher (strLine);
			else
				matcher.reset (strLine);

			// if a Patus pragma has been found, replace it by code generated by the backend code generator
			StringBuffer sb = new StringBuffer ();
			while (matcher.find ())
			{
				String strVar = matcher.group (1);

				// get the replacement string
				String strReplacement = null;
				if ("RUNTIME_FILES".equals (strVar))
					strReplacement = StringUtil.join (getRuntimeSourceFiles (), " ");
				else if ("RUNTIME_OBJECT_FILES".equals (strVar))
					strReplacement = StringUtil.join (getRuntimeObjectFiles (), " ");
				else
					strReplacement = generateCodeForVar (strVar, null);

				// do the replacements (and replace "$$" by "\$$" (we need a lot of backslashes because of Java and Regex escaping))
				if (strReplacement != null)
					matcher.appendReplacement (sb, Matcher.quoteReplacement (strReplacement.replaceAll ("\\$\\$", "\\\\\\$\\$")));
			}
			matcher.appendTail (sb);
			out.println (sb.toString ());
		}

		in.close ();
		out.close ();
	}
}
