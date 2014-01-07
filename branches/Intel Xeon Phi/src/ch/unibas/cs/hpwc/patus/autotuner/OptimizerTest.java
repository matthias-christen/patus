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
package ch.unibas.cs.hpwc.patus.autotuner;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import cetus.hir.Expression;

public class OptimizerTest
{
	private static class Function extends AbstractRunExecutable
	{
		public Function (List<int[]> listParamSet, List<Expression> listConstraints)
		{
			super (listParamSet, listConstraints);
		}

		@Override
		protected double runPrograms (int[] rgParams, StringBuilder sbResult)
		{
			if (rgParams[4] == 4 || rgParams[5] == 2)
				return Double.MAX_VALUE;
			
			//*
			int[] rgMin = new int[] { 128, 128, 128, 4, 2, 8, 7 };
			boolean bIsAtMin = true;
			for (int i = 0; i < rgParams.length; i++)
				if (rgParams[i] != rgMin[i])
				{
					bIsAtMin = false;
					break;
				}

			if (bIsAtMin)
				return -1;
			//*/

			double fSum = 0;
			for (int nParam : rgParams)
				fSum += (nParam - 5) * (nParam - 5);// * nParam;//- 10 * Math.cos (nParam / 10.0);
			return fSum;
		}
	}

	/**
	 * @param args
	 */
	public static void main (String[] args)
	{
//		IOptimizer opt = OptimizerFactory.getOptimizer ("ch.unibas.cs.hpwc.patus.autotuner.DiRectOptimizer");
//		IOptimizer opt = OptimizerFactory.getOptimizer ("ch.unibas.cs.hpwc.patus.autotuner.HookeJeevesOptimizer");
//		IOptimizer opt = OptimizerFactory.getOptimizer ("ch.unibas.cs.hpwc.patus.autotuner.MetaHeuristicOptimizer");
//		IOptimizer opt = OptimizerFactory.getOptimizer ("ch.unibas.cs.hpwc.patus.autotuner.GeneralCombinedEliminationOptimizer");
		IOptimizer opt = OptimizerFactory.getOptimizer ("ch.unibas.cs.hpwc.patus.autotuner.SimplexSearchOptimizer");
//		IOptimizer opt = OptimizerFactory.getOptimizer ("ch.unibas.cs.hpwc.patus.autotuner.ExhaustiveSearchOptimizer");

		List<int[]> listParams = new ArrayList<> ();
		//listParams.add (new int[] { -5, -4, -3, -2, -1, 0, 1, 2, 3, 4 });
		listParams.add (new int[] { 128 });
		listParams.add (new int[] { 128 });
		listParams.add (new int[] { 128 });
		listParams.add (new int[] { 1, 2, 4, 8, 16, 32, 64, 128 });
		listParams.add (new int[] { 1, 2, 4, 8, 16, 32, 64, 128 });
		listParams.add (new int[] { 1, 2, 4, 8, 16, 32, 64, 128 });
		listParams.add (new int[] { 0, 1, 2, 3, 4, 5, 6, 7 });

		List<Expression> listConstraints = new ArrayList<> ();
//		listConstraints.add (new BinaryExpression (new BinaryExpression (new NameID ("$4"), BinaryOperator.MULTIPLY, new BinaryExpression (new NameID ("$5"), BinaryOperator.MULTIPLY, new NameID ("$6"))), BinaryOperator.COMPARE_LE, new IntegerLiteral (128)));

		opt.optimize (new Function (listParams, listConstraints));

		System.out.println (Arrays.toString (opt.getResultParameters ()));
		System.out.println (opt.getResultTiming ());
	}
}
