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
package ch.unibas.cs.hpwc.patus.symbolic;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.FutureTask;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Simple (and somewhat experimental) facade over {@link Runtime#exec(String[])}
 * that makes it reasonably easy to perform a "conversation" or "session" with
 * Maxima.
 *
 * <h2>Usage Notes</h2>
 *
 * <ul>
 * <li>
 * An instance of this class should only be used by one Thread at a time but is
 * serially reusable.</li>
 * <li>
 * You must ensure that a resource <tt>maxima.properties</tt> is in the
 * ClassPath when using this. A template for this called
 * <tt>maxima.properties.sample</tt> is provided at the top of the project. All
 * of the relevant Ant build targets will remind you of this.</li>
 * <li>
 * Call {@link #open()} to initiate the conversation with Maxima. This will
 * start up a Maxima process and perform all required initialisation.</li>
 * <li>
 * Call {@link #executeRaw(String)},
 * {@link #executeExpectingSingleOutput(String)} and
 * {@link #executeExpectingMultipleLabels(String)} to perform 1 or more calls to
 * Maxima. (I have provided 3 methods here which differ only in how they process
 * the output from Maxima.)</li>
 * <li>
 * Call {@link #close()} to close Maxima and tidy up afterwards. (You can call
 * {@link #open()} again if you want to and start a new session up.</li>
 * <li>
 * If Maxima takes too long to respond to a call, a
 * {@link MaximaTimeoutException} is thrown and the underlying session is
 * closed. You can control the timeout time via {@link #setTimeout(int)} or via
 * your {@link IMaximaConfiguration}.</li>
 * <li>
 * See the test suite for some examples, also RawMaximaSessionExample in the
 * MathAssessTools-Examples module.</li>
 * </ul>
 *
 * <h2>Bugs!</h2>
 *
 * <ul>
 * <li>
 * It's possible to confuse things if you ask Maxima to output something which
 * looks like an input prompt (e.g. "(%i1)" or "(%x1)").</li>
 * </ul>
 *
 * @author David McKain
 * @version $Revision$
 *
 * http://qtitools.svn.sourceforge.net/viewvc/qtitools/MathAssessTools/trunk/MathAssessTools-MaximaConnector/src/main/java/org/qtitools/mathassess/tools/maximaconnector/
 */
public class Maxima
{
	///////////////////////////////////////////////////////////////////
	// Singleton Pattern

	private static Maxima THIS = null;

	public final static Maxima getInstance ()
	{
		return getInstance (true);
	}

	private final static Maxima getInstance (boolean bPrintErrorMessages)
	{
		if (Maxima.THIS == null)
		{
			// create a new Maxima instance
			Maxima.THIS = new Maxima (new MaximaConfiguration ());

			// try to open the session
			try
			{
				Maxima.THIS.open ();
			}
			catch (MaximaTimeoutException e)
			{
				Maxima.THIS = null;
				if (bPrintErrorMessages)
					System.err.println ("Can't connect to Maxima. A timeout has occurred.");
			}
			catch (Exception e)
			{
				Maxima.THIS = null;
				if (bPrintErrorMessages)
					System.err.println ("Can't connect to Maxima. Possibly Maxima is not installed on the system.");
			}
		}

		return Maxima.THIS;
	}

	public final static boolean isInstalled ()
	{
		return getInstance (false) != null;
	}


	///////////////////////////////////////////////////////////////////
	// Member Variables

	/**
	 * Helper to manage asynchronous calls to Maxima process thread
	 */
	private final ExecutorService m_executorService;

	/**
	 * Configuration for this session
	 */
	private final IMaximaConfiguration m_configuration;

	/**
	 * Current Maxima process, or null if no session open
	 */
	private Process m_procMaxima;

	/**
	 * Writes to Maxima, or null if no session open
	 */
	private PrintWriter m_pwMaximaInput;

	/**
	 * Reads Maxima standard output, or null if no session open
	 */
	private BufferedReader m_brMaximaOutput;

	/**
	 * Reads Maxima standard error, or null if no session open
	 */
	private BufferedReader m_brMaximaErrorStream;

	/**
	 * Timeout in seconds to wait for response from Maxima before killing session
	 */
	private int m_nTimeout;

	/**
	 * Builds up standard output from each command
	 */
	private final StringBuilder m_sbOutput;

	/**
	 * Builds up error output from each command
	 */
	private final StringBuilder m_sbErrorOutput;


	///////////////////////////////////////////////////////////////////
	// Implementation

	private Maxima (IMaximaConfiguration maximaConfiguration)
	{
		Maxima.ensureNotNull (maximaConfiguration, "MaximaConfiguration");
		this.m_configuration = maximaConfiguration;
		this.m_executorService = Executors.newFixedThreadPool (1);
		this.m_nTimeout = maximaConfiguration.getDefaultCallTimeout ();
		this.m_sbOutput = new StringBuilder ();
		this.m_sbErrorOutput = new StringBuilder ();
	}

	public int getTimeout ()
	{
		return m_nTimeout;
	}

	public void setTimeout (int timeout)
	{
		this.m_nTimeout = timeout;
	}

	/**
	 * Starts a new Maxima session, creating the underlying Maxima process and
	 * making sure it is ready for input.
	 * <p>
	 * The session must not already be open, otherwise an
	 * {@link IllegalStateException} is thrown.
	 *
	 * @throws MaximaConfigurationException
	 *             if necessary Maxima configuration details were missing or
	 *             incorrect.
	 * @throws MaximaTimeoutException
	 *             if a timeout occurred waiting for Maxima to become ready for
	 *             input
	 * @throws MaximaRuntimeException
	 *             if Maxima could not be started or if a general problem
	 *             occurred communicating with Maxima
	 * @throws IllegalStateException
	 *             if a session is already open.
	 */
	public void open () throws MaximaTimeoutException
	{
		ensureNotStarted ();

		// Extract relevant configuration required to get Maxima running
		String maximaExecutablePath = m_configuration.getMaximaExecutablePath ();
		String[] maximaRuntimeEnvironment = m_configuration.getMaximaRuntimeEnvironment ();

		// Start up Maxima with the -q option (which suppresses the startup message)
		try
		{
			m_procMaxima = Runtime.getRuntime ().exec (new String[] { maximaExecutablePath, "-q" }, maximaRuntimeEnvironment);
		}
		catch (IOException e)
		{
			throw new MaximaRuntimeException ("Could not launch Maxima process", e);
		}

		// Get at input and outputs streams, wrapped up as ASCII readers/writers
		try
		{
			m_brMaximaOutput = new BufferedReader (new InputStreamReader (m_procMaxima.getInputStream (), "ASCII"));
			m_brMaximaErrorStream = new BufferedReader (new InputStreamReader (m_procMaxima.getErrorStream (), "ASCII"));
			m_pwMaximaInput = new PrintWriter (new OutputStreamWriter (m_procMaxima.getOutputStream (), "ASCII"));
		}
		catch (UnsupportedEncodingException e)
		{
			throw new MaximaRuntimeException ("Could not extract Maxima IO stream", e);
		}

		// Wait for first input prompt
		readUntilFirstInputPrompt ("%i");
	}

	/**
	 * Tests whether the Maxima session is open or not.
	 *
	 * @return true if the session is open, false otherwise.
	 */
	public boolean isOpen ()
	{
		return m_procMaxima != null;
	}

	private String readUntilFirstInputPrompt (String inchar) throws MaximaTimeoutException
	{
		Pattern promptPattern = Pattern.compile ("^\\(\\Q" + inchar + "\\E\\d+\\)\\s*\\z", Pattern.MULTILINE);
		FutureTask<String> maximaCall = new FutureTask<> (new MaximaCallable (promptPattern));

		m_executorService.execute (maximaCall);

		String result = null;
		try
		{
			if (m_nTimeout > 0)
			{
				// Wait until timeout
				result = maximaCall.get (m_nTimeout, TimeUnit.SECONDS);
			}
			else
			{
				// Wait indefinitely (this can be dangerous!)
				result = maximaCall.get ();
			}
		}
		catch (TimeoutException e)
		{
			close ();
			throw new MaximaTimeoutException (m_nTimeout);
		}
		catch (Exception e)
		{
			throw new MaximaRuntimeException ("Unexpected Exception", e);
		}

		return result;
	}

	/**
	 * Trivial implementation of {@link Callable} that does all of the work of
	 * reading Maxima output until the next input prompt.
	 */
	private class MaximaCallable implements Callable<String>
	{
		private final Pattern promptPattern;

		public MaximaCallable (Pattern promptPattern)
		{
			this.promptPattern = promptPattern;
		}

		@Override
		public String call ()
		{
			m_sbOutput.setLength (0);
			m_sbErrorOutput.setLength (0);
			int outChar;
			try
			{
				for ( ; ; )
				{
					// First absorb anything the error stream wants to say
					absorbErrors ();

					// Block on standard output
					outChar = m_brMaximaOutput.read ();
					if (outChar == -1)
					{
						// STDOUT has finished. See if there are more errors
						absorbErrors ();
						handleReadFailure ("Maxima STDOUT and STDERR closed before finding an input prompt");
					}
					m_sbOutput.append ((char) outChar);

					// If there's currently no more to read, see if we're now sitting at an input prompt.
					if (!m_brMaximaOutput.ready ())
					{
						Matcher promptMatcher = promptPattern.matcher (m_sbOutput);
						if (promptMatcher.find ())
						{
							// Success. Trim off the prompt and store all of the raw output
							String result = promptMatcher.replaceFirst ("");
							m_sbOutput.setLength (0);
							return result;
						}

						// If we're here then we're not at a prompt - Maxima must still be thinking so loop through again
						continue;
					}
				}
			}
			catch (MaximaRuntimeException e)
			{
				close ();
				throw e;
			}
			catch (IOException e)
			{
				// If anything has gone wrong, we'll close the Session
				throw new MaximaRuntimeException ("IOException occurred reading from Maxima", e);
			}
		}

		private void handleReadFailure (String message)
		{
			throw new MaximaRuntimeException (message + "\nOutput buffer at this time was '" + m_sbOutput.toString () + "'\nError buffer at this time was '" + m_sbErrorOutput.toString () + "'");
		}

		private boolean absorbErrors () throws IOException
		{
			int errorChar;
			while (m_brMaximaErrorStream.ready ())
			{
				errorChar = m_brMaximaErrorStream.read ();
				if (errorChar != -1)
					m_sbErrorOutput.append ((char) errorChar);
				else
				{
					/* STDERR has closed */
					return true;
				}
			}
			return false;
		}
	}

	private String doMaximaUntil (String input, String inchar) throws MaximaTimeoutException
	{
		ensureStarted ();
		m_pwMaximaInput.println (input);
		m_pwMaximaInput.flush ();
		if (m_pwMaximaInput.checkError ())
			throw new MaximaRuntimeException ("An error occurred sending input to Maxima");

		return readUntilFirstInputPrompt (inchar);
	}

	/**
	 * Sends the given input to Maxima and pulls out the complete response until
	 * Maxima is ready for the next input.
	 * <p>
	 * Any intermediate input prompts are stripped from the result. Output
	 * prompts are left in to help the caller parse the difference between
	 * "stdout" output and output results.
	 *
	 * @param maximaInput
	 *            Maxima input to execute
	 *
	 * @return raw Maxima output, as described above.
	 *
	 * @throws IllegalArgumentException
	 *             if the given Maxima input is null
	 * @throws MaximaTimeoutException
	 *             if a timeout occurs waiting for Maxima to respond
	 */
	public String executeRaw (String maximaInput) throws MaximaTimeoutException
	{
		Maxima.ensureNotNull (maximaInput, "maximaInput");

		// Do call, modifying the input prompt at the end so we know exactly when to stop reading output
		String rawOutput = doMaximaUntil (maximaInput + " inchar: %x$", "%x");

		// Reset the input prompt and do a slightly hacky kill on the history so
		// that the last output appears to be the result of the initial maximaInput
		doMaximaUntil ("block(inchar: %i, temp: %th(2), kill(3), temp);", "%i");

		// Strip out any intermediate input prompts
		rawOutput = rawOutput.replaceAll ("\\(%i\\d+\\)", "");

		return rawOutput;
	}

	/**
	 * Alternative version of {@link #executeRaw(String)} that assumes that
	 * Maxima is going to output a result on a single line. In this case, the
	 * output prompt is stripped off and leading/trailing whitespace of the
	 * result is removed.
	 * <p>
	 * The result will not make sense if Maxima outputs more than a single line
	 * or if there are any side effect results to stdout.
	 *
	 * @throws MaximaTimeoutException
	 * @throws MaximaRuntimeException
	 */
	public String executeExpectingSingleOutput (String maximaInput) throws MaximaTimeoutException
	{
		return executeRaw (maximaInput).
			replaceFirst ("\\(%o\\d+\\)\\s*", "").trim ().
			replaceAll ("\\\\\n", "");	// replace newline markers
	}

	/**
	 * Alternative version of {@link #executeRaw(String)} that assumes that
	 * Maxima is going to output a multiple single line results. In this case,
	 * the output prompts are stripped off and leading/trailing whitespace on
	 * each output line is removed.
	 * <p>
	 * The result will not make sense if any of the Maxima outputs use up
	 * multiple lines or if there are any side effect results to stdout.
	 *
	 * @throws MaximaTimeoutException
	 * @throws MaximaRuntimeException
	 */
	public String[] executeExpectingMultipleLabels (String maximaInput) throws MaximaTimeoutException
	{
		return executeExpectingSingleOutput (maximaInput).split ("(?s)\\s*\\(%o\\d+\\)\\s*");
	}

	/**
	 * Closes the Maxima session, forcibly if required.
	 * <p>
	 * It is legal to close a session which is already closed.
	 *
	 * @return underlying exit value for the Maxima process, or -1 if the
	 *         session was already closed.
	 */
	public int close ()
	{
		if (isOpen ())
		{
			try
			{
				// Close down executor
				m_executorService.shutdown ();

				// Ask Maxima to nicely close down by closing its input
				m_pwMaximaInput.close ();
				if (m_pwMaximaInput.checkError ())
				{
					m_procMaxima.destroy ();
					return m_procMaxima.exitValue ();
				}

				// Wait for Maxima to shut down
				try
				{
					return m_procMaxima.waitFor ();
				}
				catch (InterruptedException e)
				{
					m_procMaxima.destroy ();
					return m_procMaxima.exitValue ();
				}
			}
			finally
			{
				resetState ();
				Maxima.THIS = null;
			}
		}

		// If session is already closed, we'll return -1
		return -1;
	}

	private void resetState ()
	{
		m_procMaxima = null;
		m_pwMaximaInput = null;
		m_brMaximaOutput = null;
		m_brMaximaErrorStream = null;
		m_sbOutput.setLength (0);
	}

	private void ensureNotStarted ()
	{
		if (m_procMaxima != null)
			throw new IllegalStateException ("Session already opened");
	}

	private void ensureStarted ()
	{
		if (m_procMaxima == null)
			throw new IllegalStateException ("Session not open - call open()");
	}

	/**
	 * Checks that the given object is non-null, throwing an
	 * IllegalArgumentException if the check fails. If the check succeeds then
	 * nothing happens.
	 *
	 * @param value
	 *            object to test
	 * @param objectName
	 *            name to give to supplied Object when constructing Exception
	 *            message.
	 *
	 * @throws IllegalArgumentException
	 *             if an error occurs.
	 */
	private static void ensureNotNull (Object value, String objectName)
	{
		if (value == null)
			throw new IllegalArgumentException (objectName + " must not be null");
	}
}
