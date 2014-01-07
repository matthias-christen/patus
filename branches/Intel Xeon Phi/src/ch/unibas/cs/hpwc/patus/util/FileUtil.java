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
package ch.unibas.cs.hpwc.patus.util;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;

import org.apache.log4j.Logger;

public class FileUtil
{
	///////////////////////////////////////////////////////////////////
	// Static Member Variables

	private static File BASE_DIR = null;
	
	private final static Logger LOGGER = Logger.getLogger (FileUtil.class);


	///////////////////////////////////////////////////////////////////
	// Inner Types

	/**
	 * this class provides functions used to generate a relative path
	 * from two absolute paths
	 * @author David M. Howard<br/>
	 * <a href="http://www.devx.com/tips/Tip/13737">http://www.devx.com/tips/Tip/13737</a>
	 */
	private static class RelativePath
	{
		/**
		 * Break a path down into individual elements and add to a list.
		 * Example:
		 * If a path is /a/b/c/d.txt, the breakdown will be [d.txt,c,b,a]
		 *
		 * @param f
		 *            input file
		 * @return a List collection with the individual elements of the path in
		 *         reverse order
		 */
		private static List<String> getPathList (File f)
		{
			List<String> l = new ArrayList<> ();
			File r = null;

			try
			{
				r = f.getCanonicalFile ();
				while (r != null)
				{
					l.add (r.getName ());
					r = r.getParentFile ();
				}
			}
			catch (IOException e)
			{
				e.printStackTrace ();
				l = null;
			}

			return l;
		}

		/**
		 * Figure out a string representing the relative path of 'f' with
		 * respect to 'r'.
		 *
		 * @param r
		 *            home path
		 * @param f
		 *            path of file
		 */
		private static String matchPathLists (List<String> r, List<String> f)
		{
			int i;
			int j;
			StringBuilder sb = new StringBuilder ();

			// start at the beginning of the lists
			// iterate while both lists are equal
			i = r.size () - 1;
			j = f.size () - 1;

			// first eliminate common root
			while ((i >= 0) && (j >= 0) && (r.get (i).equals (f.get (j))))
			{
				i--;
				j--;
			}

			// for each remaining level in the home path, add a ..
			for ( ; i >= 0; i--)
			{
				sb.append ("..");
				sb.append (File.separator);
			}

			// for each level in the file path, add the path
			for ( ; j >= 1; j--)
			{
				sb.append (f.get (j));
				sb.append (File.separator);
			}

			// file name
			sb.append (f.get (j));
			return sb.toString ();
		}

		/**
		 * Get relative path of File 'f' with respect to 'home' directory example:
		 * home = /a/b/c
		 * f = /a/d/e/x.txt
		 * s = getRelativePath(home,f) = ../../d/e/x.txt
		 *
		 * @param home
		 *            base path, should be a directory, not a file, or it
		 *            doesn't make sense
		 * @param f
		 *            file to generate path for
		 * @return path from home to f as a string
		 */
		public static String getRelativePath (File home, File f)
		{
			return RelativePath.matchPathLists (RelativePath.getPathList (home), RelativePath.getPathList (f));
		}
	}

	///////////////////////////////////////////////////////////////////
	// Implementation

	/**
	 * Returns a relative path for <code>file</code> with respect to the base path
	 * <code>fileBase</code>.
	 * @param fileBase The base path
	 * @param file The target path
	 * @return <code>file</code> relative to <code>fileBase</code>
	 */
	public static String relativeTo (File fileBase, File file)
	{
		if (fileBase.equals (file))
			return "";
		return RelativePath.getRelativePath (fileBase, file);
	}

	/**
	 * Retrieves the file name extension for the file <code>f</code>.
	 * @param f The file for which to get the file name extension
	 * @return The extension of the file name of <code>f</code>
	 */
	public static String getExtension (File f)
	{
		int nIdx = f.getName ().lastIndexOf ('.');
		if (nIdx >= 0)
			return f.getName ().substring (nIdx + 1);
		return "";
	}

	public static String getFilenameWithoutExtension (File f)
	{
		int nIdx = f.getPath ().lastIndexOf ('.');
		if (nIdx >= 0)
			return f.getPath ().substring (0, nIdx);
		return f.getPath ();
	}

	/**
	 * Copies the file <code>fileSource</code> to <code>fileDestination</code>
	 * @param fileSource
	 * @param fileDestination
	 */
	public static void copy (File fileSource, File fileDestination) throws IOException
	{
		if (fileDestination.getParentFile () != null)
			fileDestination.getParentFile ().mkdirs ();

		FileChannel fcSource = null;
		FileChannel fcDestination = null;
		try
		{
			fcSource = new FileInputStream (fileSource).getChannel ();
			fcDestination = new FileOutputStream (fileDestination).getChannel ();
			fcDestination.transferFrom (fcSource, 0, fcSource.size ());
		}
		finally
		{
			if (fcSource != null)
				fcSource.close ();
			if (fcDestination != null)
				fcDestination.close ();
		}
	}

	/**
	 * @param filePath
	 *            the name of the file to open. Not sure if it can accept URLs
	 *            or just filenames. Path handling could be better, and buffer
	 *            sizes are hardcoded
	 */
	public static String readFileToString (String filePath) throws IOException
	{
		StringBuffer fileData = new StringBuffer (1000);
		BufferedReader reader = new BufferedReader (new FileReader (filePath));
		char[] buf = new char[1024];
		int numRead = 0;
		while ((numRead = reader.read (buf)) != -1)
		{
			String readData = String.valueOf (buf, 0, numRead);
			fileData.append (readData);
			buf = new char[1024];
		}
		reader.close ();
		return fileData.toString ();
	}

	/**
	 * Determines whether <code>file</code> is an absolute file name.
	 * 
	 * @param file
	 *            The file descriptor to test
	 * @return <code>true</code> iff <code>file</code> is an absolute file name
	 */
	public static boolean isPathAbsolute (File file)
	{
		if (file == null)
			return false;
		return FileUtil.isPathAbsolute (file.getPath ());
	}

	/**
	 * Determines whether <code>strFile</code> is an absolute file name.
	 * 
	 * @param strFile
	 *            The name of the file to test
	 * @return <code>true</code> iff <code>strFile</code> is an absolute file
	 *         name
	 */
	public static boolean isPathAbsolute (String strFile)
	{
		if (strFile == null)
			return false;

		// special treatment on Windows
		if (System.getProperty ("os.name").indexOf ("Windows") > -1)
		{
			if (strFile.length () < 2)
				return false;
			return strFile.charAt (1) == ':';	// character after the drive
		}

		// linux, Mac
		return strFile.startsWith (File.separator);
	}

	/**
	 * Determines whether <code>strFile</code> is absolute, and if it is not,
	 * interprets the file name relative to the location of the Jar file/source
	 * code.
	 * 
	 * @param strFile
	 *            The name of the file whose path to retrieve
	 * @return A descriptor of the file <code>strFile</code>
	 */
	public static File getFileRelativeToJar (String strFile)
	{
		if (FileUtil.isPathAbsolute (strFile))
			return new File (strFile);

		return new File (getBaseDir (), strFile);
	}
	
	/**
	 * Returns the descriptor of the patus.jar file.
	 * 
	 * @return The descriptor of the patus.jar file
	 */
	public static String getJarFilePath ()
	{
		// get the class path
		String strMainClassPath = null;
		String strClassPath = System.getProperty ("java.class.path");
		if (strClassPath == null)
			return null;

		// find
		String[] rgPaths = strClassPath.split (File.pathSeparator);
		if (rgPaths.length == 0)
			return null;
		else if (rgPaths.length == 1)
			strMainClassPath = rgPaths[0];
		else
		{
			// find the part of the class path that isn't a JAR or is named "patus.jar" if there are multiple class paths
			for (String strLib : rgPaths)
			{
				if (!strLib.endsWith (".jar") || strLib.endsWith ("patus.jar"))
				{
					strMainClassPath = strLib;
					break;
				}
			}
		}

		return strMainClassPath; 
	}

	private static File getBaseDir ()
	{
		if (FileUtil.BASE_DIR != null)
			return FileUtil.BASE_DIR;

		String strMainClassPath = getJarFilePath ();
		if (strMainClassPath == null)
			return FileUtil.BASE_DIR = new File ("");
		
		// remove "patus.jar" if it ends with that
		String strJar = File.separator + "patus.jar";
		if (strMainClassPath.endsWith (strJar))
			strMainClassPath = StringUtil.trimRight (strMainClassPath.substring (0, strMainClassPath.length () - strJar.length ()), new char[] { File.separatorChar });
		
		LOGGER.debug (StringUtil.concat ("Main class path = ", strMainClassPath));

		// remove "/bin" if it ends with that
		String strBin = File.separator + "bin";
		if (strMainClassPath.endsWith (strBin))
			strMainClassPath = strMainClassPath.substring (0, strMainClassPath.length () - strBin.length ());

		// remove "/classes" if it ends with that
		String strClasses = File.separator + "classes";
		if (strMainClassPath.endsWith (strClasses))
			strMainClassPath = strMainClassPath.substring (0, strMainClassPath.length () - strClasses.length ());

		FileUtil.BASE_DIR = new File (strMainClassPath);
		if (strMainClassPath.endsWith (".jar"))
			FileUtil.BASE_DIR = FileUtil.BASE_DIR.getParentFile ();

		return FileUtil.BASE_DIR;
	}

	/**
	 * Recursively cleans the directory <code>fileDir</code>.
	 * @param fileDir The directory to clean
	 */
	public static void cleanDirectory (File fileDir)
	{
		File[] rgFiles = fileDir.listFiles ();
		if (rgFiles == null)
			return;
		for (File f : rgFiles)
		{
			if (f.isDirectory ())
				FileUtil.cleanDirectory (f);
			f.delete ();
		}
	}
}
