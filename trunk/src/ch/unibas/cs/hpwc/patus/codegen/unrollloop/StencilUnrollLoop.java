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
package ch.unibas.cs.hpwc.patus.codegen.unrollloop;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import cetus.hir.CompoundStatement;
import cetus.hir.ForLoop;
import cetus.hir.Statement;
import cetus.hir.Traversable;
import ch.unibas.cs.hpwc.patus.util.ASTUtil;
import ch.unibas.cs.hpwc.patus.util.IntArray;

/**
 * Helper class to unroll an innermost stencil loop nest.
 * @author Matthias-M. Christen
 * 
 * @deprecated
 */
public class StencilUnrollLoop
{
	///////////////////////////////////////////////////////////////////
	// Constants

	/**
	 * The maximum depth of a loop nest that will be register blocked
	 * (loop unrolled).
	 */
	public final static int REGISTERBLOCKING_LOOPNESTDEPTH = 3;

	/**
	 * The default unrolling configurations
	 */
	public final static List<int[]> DEFAULT_UNROLLING_CONFIGURATIONS = new ArrayList<int[]> ();

	private final static int[] DEFAULT_UNROLLING_FACTORS = new int[] { 1, 2, 4 };


	///////////////////////////////////////////////////////////////////
	// Static Initializers

	static
	{
		StencilUnrollLoop.addUnrollConfiguration (new ArrayList<Integer> (), 0);
	}

	/**
	 * Recursively creates the default unrolling configurations.
	 * @param listConfig The list of unrolling factors
	 * @param nLevel The current recursion level
	 */
	private static void addUnrollConfiguration (List<Integer> listConfig, int nLevel)
	{
		if (nLevel == StencilUnrollLoop.REGISTERBLOCKING_LOOPNESTDEPTH)
		{
			int[] rgConfig = new int[listConfig.size ()];
			int i = 0;
			for (int nIdx : listConfig)
				rgConfig[i++] = nIdx;
			StencilUnrollLoop.DEFAULT_UNROLLING_CONFIGURATIONS.add (rgConfig);
		}
		else
		{
			for (int nFactor : StencilUnrollLoop.DEFAULT_UNROLLING_FACTORS)
			{
				List<Integer> l = new ArrayList<Integer> (listConfig.size () + 1);
				l.addAll (listConfig);
				l.add (nFactor);

				StencilUnrollLoop.addUnrollConfiguration (l, nLevel + 1);
			}
		}
	}


	///////////////////////////////////////////////////////////////////
	// Member Variables

	private final static UnrollLoop UNROLL_LOOP = new UnrollLoop (UniformlyIncrementingLoopNestPart.class);


	///////////////////////////////////////////////////////////////////
	// Implementation

	/**
	 * Not instantiable...
	 */
	private StencilUnrollLoop ()
	{
	}

	/**
	 * Unrolls the loop <code>loop</code>.
	 * @param listLoops A list of loops to unroll. All the loops in the list will be replaced by the unrolled code.
	 * @param rgUnrollFactors The unroll factors for each loop
	 * @param nMaxLevels Specifies how many loops in the loop nest will be unrolled at the maximum
	 * @return A map with the unrolling configuration as key and a list of the loops that have been unrolled for that
	 * 	configuration. The list of unrolled loops corresponds to <code>listLoops</code>.
	 * @see UnrollLoop#unroll(ForLoop, int[], int)
	 */
	public static Map<IntArray, Statement> unroll (List<ForLoop> listLoops, List<int[]> listUnrollFactors, int nMaxLevels)
	{
		if (listLoops == null || listLoops.size () == 0)
			return null;

		// find the common root of the loops in the list and create the reference structure
		Traversable trvReference = null;
		for (ForLoop loop : listLoops)
		{
			if (trvReference == null)
				trvReference = ASTUtil.getRoot (loop);
			else if (trvReference != ASTUtil.getRoot (loop))
				throw new RuntimeException ("The loops don't have a common parent");
		}
		Statement stmtReference = (Statement) trvReference;

		// if there are no loops, there is nothing to be done...
		if (listLoops.size () == 0)
		{
			int[] rgNoUnrollingConfig = null;
			if (listUnrollFactors.size () > 0)
				rgNoUnrollingConfig = new int[listUnrollFactors.get (0).length];
			Map<IntArray, Statement> map = new HashMap<IntArray, Statement> ();
			map.put (new IntArray (rgNoUnrollingConfig), stmtReference);
			return map;
		}

		// memorize the locations of the loops to replace
		List<int[]> listIndices = new ArrayList<int[]> (listLoops.size ());
		for (ForLoop loop : listLoops)
			listIndices.add (ASTUtil.getIndex (loop));

		// unroll each loop in the list of loops and replace the original statements with the unrolled ones
		Map<IntArray, Statement> mapUnrolledStatements = null;
		int i = 0;
		for (ForLoop loop : listLoops)
		{
			// unroll the loops (all at once); unroll returns a list of statements with different unrollings according to the
			// list of unroll factors that has been provided
			Map<IntArray, CompoundStatement> mapUnrolledLoops = StencilUnrollLoop.UNROLL_LOOP.unroll (loop, listUnrollFactors, nMaxLevels);

			// create the list of statements if it doesn't exist yet
			if (mapUnrolledStatements == null)
			{
				mapUnrolledStatements = new HashMap<IntArray, Statement> (mapUnrolledLoops.size ());

				// create a copy of the reference statement for each unrolling configuration
				for (IntArray arrUnrollFactors : mapUnrolledLoops.keySet ())
					mapUnrolledStatements.put (arrUnrollFactors, stmtReference.clone ());
			}

			// replace the loops within the cloned code copy with the unrolled versions of the loops
			for (IntArray arrUnrollFactors : mapUnrolledLoops.keySet ())
				ASTUtil.getStatement (mapUnrolledStatements.get (arrUnrollFactors), listIndices.get (i)).swapWith (mapUnrolledLoops.get (arrUnrollFactors));

			i++;
		}

		return mapUnrolledStatements;
	}

	/**
	 *
	 * @param listLoops
	 * @return
	 */
	public static Map<IntArray, Statement> unroll (List<ForLoop> listLoops)
	{
		return StencilUnrollLoop.unroll (listLoops, StencilUnrollLoop.DEFAULT_UNROLLING_CONFIGURATIONS, StencilUnrollLoop.REGISTERBLOCKING_LOOPNESTDEPTH);
	}
}
