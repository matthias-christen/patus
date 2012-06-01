package ch.unibas.cs.hpwc.patus.codegen.backend.assembly.test;

import static org.junit.Assert.*;

import java.util.Arrays;

import lpsolve.LpSolve;
import lpsolve.LpSolveException;

import org.junit.Before;
import org.junit.Test;

import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.analyze.DAGraph;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

public class ILPTest
{

	@Before
	public void setUp () throws Exception
	{
	}

	@Test
	public void test1 ()
	{
		try
		{
			final int nVarsCount = 4;
			LpSolve solver = LpSolve.makeLp (0, nVarsCount);
			solver.setMaxim ();
						
			// add constraints
			double[] c1 = { -1,  5, 7, 4, 3 };
			solver.addConstraint (c1, LpSolve.LE, 14);
			//solver.strAddConstraint ("5 7 4 3", LpSolve.LE, 14);
			
			double[] c2 = { -1,  1, 1, 1, 1 };
			solver.addConstraint (c2, LpSolve.LE, 2);
			//solver.strAddConstraint ("1 1 1 1", LpSolve.LE, 3);
						
			for (int i = 1; i <= nVarsCount; i++)
				solver.setBinary (i, true);

			// set objective
			double[] rgObj = { -1,  8, 11, 3, 4 };
			solver.setObjFn (rgObj);
			//solver.strSetObjFn ("8 11 3 4");
			
for (int $y = 1; $y <= solver.getNrows (); $y++)
{
	for (int $x = 1; $x <= solver.getNcolumns (); $x++)
		System.out.printf ("%.0f ", solver.getMat ($y, $x));
	System.out.println ();
}
			
			int nResult = solver.solve ();
			System.out.println (StringUtil.concat ("Result=", nResult));
			
			// get solution
			double[] y = solver.getPtrVariables ();
			System.out.println (Arrays.toString (y));
			
			solver.printSolution (1);
			solver.printObjective ();
			
			solver.deleteLp ();
		}
		catch (LpSolveException e)
		{
			e.printStackTrace();
		}
	}

	@Test
	public void test2 ()
	{
		try
		{
			final int nVerticesCount = 4;
			final int nCyclesCount = 3;
			
			class IndexingHelper
			{
				public int idx (int i, int j)
				{
					return 1 + (j - 1) + (i - 1) * nCyclesCount + 1;
				}
			}
			IndexingHelper x = new IndexingHelper ();

			LpSolve solver = LpSolve.makeLp (0, nVerticesCount + 1);

			// add constraints			
			for (int i = 1; i <= nVerticesCount; i++)
			{
				double[] rgCoeffs = new double[nVerticesCount + 1];
				for (int j = 1; j <= nCyclesCount; j++)
					rgCoeffs[x.idx (i, j)] = 1;
				solver.addConstraint (rgCoeffs, LpSolve.EQ, 1);
			}
			for (int j = 1; j <= nCyclesCount; j++)
			{
				double[] rgCoeffs = new double[nVerticesCount + 1];
				for (int i = 1; i <= nVerticesCount; i++)
					rgCoeffs[x.idx (i, j)] = 1;
				solver.addConstraint (rgCoeffs, LpSolve.LE, 4);
			}
						
			for (int i = 2; i <= nVerticesCount; i++)
				solver.setBinary (i, true);

			// set objective
			double[] rgObj = { -1,  8, 11, 3, 4 };
			solver.setObjFn (rgObj);
			//solver.strSetObjFn ("8 11 3 4");
			
for (int $y = 1; $y <= solver.getNrows (); $y++)
{
	for (int $x = 1; $x <= solver.getNcolumns (); $x++)
		System.out.printf ("%.0f ", solver.getMat ($y, $x));
	System.out.println ();
}
			
			int nResult = solver.solve ();
			System.out.println (StringUtil.concat ("Result=", nResult));
			
			// get solution
			double[] y = solver.getPtrVariables ();
			System.out.println (Arrays.toString (y));
			
			solver.printSolution (1);
			solver.printObjective ();
			
			solver.deleteLp ();
		}
		catch (LpSolveException e)
		{
			e.printStackTrace();
		}
	}
}
