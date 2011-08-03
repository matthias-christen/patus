package ch.unibas.cs.hpwc.patus.codegen.unrollloop;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

import cetus.hir.CompoundStatement;
import cetus.hir.Expression;
import cetus.hir.IDExpression;

/**
 *
 * @author Matthias-M. Christen
 */
public class UnrollLoopSharedObjects
{
	///////////////////////////////////////////////////////////////////
	// Member Variables

	/**
	 * Map containing all the loop indices of the nest
	 */
	private Map<IDExpression, Boolean> m_mapLoopIndices;

	/**
	 * Cache for step expressions
	 */
	private Map<Integer, Expression> m_mapStepCache;

	/**
	 *
	 */
	private List<int[]> m_listUnrollingFactors;

	/**
	 * The compound statement to which all the unrolled loop code is added.
	 * Entry <i>i</i> in the list corresponds to the unrolling <i>i</i> in <code>m_listUnrollingFactors</code>.
	 */
	private List<CompoundStatement> m_listUnrolledStatements;

	/**
	 * Flag determining whether the unroll configurations have been fixed (and can't change anymore)
	 * or if they still are in the volatile phase (in which unrolling configurations can be discarded because
	 * they are specified multiple times or due to complete unrolling)
	 */
	private boolean m_bUnrollingConfigurationsFixed;

	/**
	 * Cache for loop unrolling configurations for a single loop
	 */
	private Map<Integer, int[]> m_mapUnrollingFactorsForLoopCache;

	/**
	 *
	 */
	private final boolean m_bCreateTemporariesForLoopIndex = false;


	///////////////////////////////////////////////////////////////////
	// Implementation

	/**
	 * Constructs the object containing the shared data.
	 */
	public UnrollLoopSharedObjects (List<int[]> listUnrollingFactors)
	{
		m_mapLoopIndices = new HashMap<IDExpression, Boolean> ();
		m_mapStepCache = new HashMap<Integer, Expression> ();
		m_mapUnrollingFactorsForLoopCache = new HashMap<Integer, int[]> ();

		m_bUnrollingConfigurationsFixed = false;

		// set the unrolling configurations
		setUnrollingFactors (listUnrollingFactors);
	}

	/**
	 *
	 * @param listUnrollingFactors
	 */
	public void setUnrollingFactors (List<int[]> listUnrollingFactors)
	{
		m_listUnrollingFactors = new ArrayList<int[]> (listUnrollingFactors.size ());

		// normalize the entries (each of the entries has the same number of elements in the array;
		// if shorter entries are encountered, they will be filled up with ones (no unrolling))
		int nMaxLen = 0;
		for (int[] rgFactor : listUnrollingFactors)
			nMaxLen = Math.max (nMaxLen, rgFactor.length);

		for (int[] rgFactor : listUnrollingFactors)
		{
			if (rgFactor.length == nMaxLen)
				m_listUnrollingFactors.add (rgFactor.clone ());
			else
			{
				int[] rgFactorNew = new int[nMaxLen];
				System.arraycopy (rgFactor, 0, rgFactorNew, 0, rgFactor.length);
				for (int i = rgFactor.length; i < nMaxLen; i++)
					rgFactorNew[i] = 1;
				m_listUnrollingFactors.add (rgFactorNew);
			}
		}
	}

	/**
	 * Returns a list of unrolling factors desired for the loop with number <code>nLoopNumber</code> counting from the
	 * outermost to the innermost loop in the loop nest.<br/>
	 * Note that this method discards duplicate unrolling factors.
	 * @param nLoopNumber The loop number counting from the outermost to the innermost loop in the loop nest
	 * @return An array of <code>int</code>s containing the desired loop unrollings for the loop with number <code>nLoopNumber</code>
	 * @throws ArrayIndexOutOfBoundsException if <code>nLoopNumber</code> is < 0 or exceeds the number of loops in the nest
	 */
	public int[] getUnrollingFactorsForLoop (int nLoopNumber)
	{
		int[] rgFactors = m_mapUnrollingFactorsForLoopCache.get (nLoopNumber);
		if (rgFactors == null || !m_bUnrollingConfigurationsFixed)
		{
			Set<Integer> setFactors = new TreeSet<Integer> ();
			for (int[] rgFactor : m_listUnrollingFactors)
				setFactors.add (rgFactor[nLoopNumber]);

			rgFactors = new int[setFactors.size ()];
			int i = 0;
			for (int nFactor : setFactors)
			{
				rgFactors[i] = nFactor;
				i++;
			}
		}

		if (m_bUnrollingConfigurationsFixed)
			m_mapUnrollingFactorsForLoopCache.put (nLoopNumber, rgFactors);

		return rgFactors;
	}

	/**
	 * Returns the number of unrolling configurations (not counting duplicate configurations
	 * after {@link UnrollLoopSharedObjects#reassessUnrollingFactors()} has been called).
	 * @return The number of unrolling configurations
	 */
	public int getUnrollFactorsCount ()
	{
		return m_listUnrollingFactors.size ();
	}

	/**
	 * Returns the loop nest depth to which unrolling is performed.
	 * @return
	 */
	public int getUnrollDepthCount ()
	{
		if (getUnrollFactorsCount () == 0)
			return 0;
		return m_listUnrollingFactors.get (0).length;
	}

	/**
	 * Restricts all the unrolling factors of the loop with number <code>nLoopNumber</code> to a
	 * single value, <code>nUnrollingFactor</code>.
	 * (Handle complete unrolling.)
	 * @param nLoopNumber The number of the loop
	 * @param nUnrollingFactor The unrolling factor to which to restrict to in the unrolling configurations
	 */
	public void restrictUnrollingFactorTo (int nLoopNumber, int nUnrollingFactor)
	{
		for (int[] rgFactor : m_listUnrollingFactors)
			rgFactor[nLoopNumber] = nUnrollingFactor;
	}

	/**
	 * Sorts the unrolling factors and removes duplicates.
	 * The method also allocates the internal data structures.<br/>
	 * It is essential to call this method. Do so after the analysis phase.
	 */
	public void reassessUnrollingFactors ()
	{
		// sort the unrolling factors
		Collections.sort (m_listUnrollingFactors, new Comparator<int[]> ()
		{
			@Override
			public int compare (int[] rgFactor1, int[] rgFactor2)
			{
				// Note: we assume that both arrays rgFactor1 and rgFactor2 have same length
				// (because they have been constructed that way)
				assert (rgFactor1 != null);
				assert (rgFactor2 != null);
				assert (rgFactor1.length == rgFactor2.length);

				for (int i = 0; i < rgFactor1.length; i++)
				{
					if (rgFactor1[i] < rgFactor2[i])
						return -1;
					if (rgFactor1[i] > rgFactor2[i])
						return 1;
				}

				return 0;
			}
		});

		// create the final unrolling factors list, throwing out any duplicates
		List<int[]> listTmp = new ArrayList<int[]> (m_listUnrollingFactors.size ());
		int[] rgFactorPrev = null;
		for (int[] rgFactor : m_listUnrollingFactors)
		{
			if (!Arrays.equals (rgFactor, rgFactorPrev))
				listTmp.add (rgFactor);
			rgFactorPrev = rgFactor;
		}

		m_listUnrollingFactors = listTmp;

		// create the statement to which the unrolled code will be added
		m_listUnrolledStatements = new ArrayList<CompoundStatement> (m_listUnrollingFactors.size ());
		for (int i = 0; i < m_listUnrollingFactors.size (); i++)
			m_listUnrolledStatements.add (new CompoundStatement ());

		// fix the unrolling configurations
		m_bUnrollingConfigurationsFixed = true;
	}

	/**
	 * Returns an iterable over all the unrolling factors.
	 * @return
	 */
	public Iterable<int[]> getUnrollingFactors ()
	{
		return m_listUnrollingFactors;
	}

	/**
	 * Returns a map containing all the loop indices in the loop nest.
	 * @return A map containing the loop indices in the loop nest
	 */
	public final Map<IDExpression, Boolean> getLoopIndices ()
	{
		return m_mapLoopIndices;
	}

	/**
	 * Returns a map containing a mapping from steps to the corresponding
	 * simplified expression used to calculate the unrolled increment expression.
	 * @return A mapping from a step number to the expression for the corresponding increment expression
	 */
	public final Map<Integer, Expression> getStepCache ()
	{
		return m_mapStepCache;
	}

	/**
	 * Returns the {@link CompoundStatement} to which the statements during the
	 * unrolling process are added.
	 * @return The statement to which the generated code is added
	 */
	public final CompoundStatement getUnrolledStatement (int i)
	{
		return m_listUnrolledStatements.get (i);
	}

	/**
	 * Returns the list of the compound statements to which the unrolled loops are added.
	 * @return
	 */
	public final List<CompoundStatement> getUnrolledStatements ()
	{
		return m_listUnrolledStatements;
	}

	/**
	 * Specifies whether temporary variables for unrolled loop indices are to be created.
	 * @return <code>true</code> if the generators are supposed to create temporary
	 * 	variables for the unrolled loop indices
	 */
	public final boolean isCreatingTemporariesForLoopIndices ()
	{
		return m_bCreateTemporariesForLoopIndex;
	}
}
