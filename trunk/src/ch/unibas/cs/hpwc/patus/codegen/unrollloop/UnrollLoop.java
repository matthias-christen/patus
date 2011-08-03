/**
 *
 */
package ch.unibas.cs.hpwc.patus.codegen.unrollloop;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.log4j.Logger;

import cetus.hir.AssignmentExpression;
import cetus.hir.AssignmentOperator;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.CompoundStatement;
import cetus.hir.DeclarationStatement;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.ForLoop;
import cetus.hir.FunctionCall;
import cetus.hir.Identifier;
import cetus.hir.IntegerLiteral;
import cetus.hir.NameID;
import cetus.hir.Specifier;
import cetus.hir.Statement;
import cetus.hir.UnaryExpression;
import cetus.hir.UnaryOperator;
import cetus.hir.ValueInitializer;
import cetus.hir.VariableDeclaration;
import cetus.hir.VariableDeclarator;
import ch.unibas.cs.hpwc.patus.symbolic.Maxima;
import ch.unibas.cs.hpwc.patus.util.IntArray;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 * This class performs unrolling of a <code>for</code> loop.
 * @author Matthias-M. Christen
 */
public class UnrollLoop
{
	///////////////////////////////////////////////////////////////////
	// Static Members

	private final static Logger LOGGER = Logger.getLogger (UnrollLoop.class);


	///////////////////////////////////////////////////////////////////
	// Inner Types

	/**
	 * The class that actually performs the unrolling work.
	 */
	private class Unroller
	{
		/**
		 * The data that is shared among the generating classes
		 */
		private UnrollLoopSharedObjects m_data;

		/**
		 * The list of loops to unroll
		 */
		private List<ILoopNestPart> m_listLoops;

		/**
		 * The maximum number of loop nesting levels to consider.
		 * Set to {@link Integer#MAX_VALUE} for an &quot;infinite&quot; number of nesting levels.
		 */
		private int m_nMaxLevels;

		/**
		 * <code>true</code> iff we are working on the first (the most unrolled) loop pass
		 */
		private boolean m_bIsFirstRun;


		/**
		 *
		 * @param loop The loop nest to unroll
		 * @param listUnrollFactors List of unroll configurations (each entry in the list is an unroll configuration;
		 * 	the j-th entry of the array (the unroll configuration) specifies how many times the j-th loop in the nest
		 * 	is unrolled)
		 * @param nMaxLevels The maximum number of loop nesting levels to consider.
		 * 	To consider arbitrarily many nesting levels, pass {@link Integer#MAX_VALUE}.
		 */
		public Unroller (ForLoop loop, List<int[]> listUnrollFactors, int nMaxLevels)
		{
			m_data = new UnrollLoopSharedObjects (listUnrollFactors);
			m_nMaxLevels = nMaxLevels;
			m_bIsFirstRun = true;

			m_mapReferencedStatementReplacements = new HashMap<Statement, List<Statement>> ();

			// descend the loop nest and gather data
			m_listLoops = new ArrayList<ILoopNestPart> ();
			findLoops (loop, null, 0);

			m_data.reassessUnrollingFactors ();

			// unroll the loop nest
			// create a list of loop nest objects with an entry for each unrolling configuration
			List<LoopNest> listLoopNests = new ArrayList<LoopNest> (m_data.getUnrollFactorsCount ());
			for (int k = 0; k < m_data.getUnrollFactorsCount (); k++)
				listLoopNests.add (new LoopNest ());
			unroll (m_listLoops.get (0), null, listLoopNests, new ArrayList<Boolean> (), 0);
		}

		public Map<IntArray, CompoundStatement> getUnrolledLoops ()
		{
			Map<IntArray, CompoundStatement> map = new HashMap<IntArray, CompoundStatement> ();
			int i = 0;
			for (int[] rgUnrollFactors : m_data.getUnrollingFactors ())
			{
				map.put (new IntArray (rgUnrollFactors), m_data.getUnrolledStatement (i));
				i++;
			}

			return map;
		}

		/**
		 *
		 * @param loop
		 * @param rgUnrollFactors
		 * @return
		 */
		private ILoopNestPart createLoopNestPart (ForLoop loop, int nLoopNumber)
		{
			ILoopNestPart lnp = null;
			try
			{
				// try to create a new instance of the loop nest part
				lnp = m_clsLoopNestParts.newInstance ();
			}
			catch (InstantiationException e)
			{
				// If the creation of the special instance failed, create a standard object
				lnp = new GeneralLoopNestPart ();
			}
			catch (IllegalAccessException e)
			{
				// If the creation of the special instance failed, create a standard object
				lnp = new GeneralLoopNestPart ();
			}

			lnp.init (m_data, loop, nLoopNumber);
			return lnp;
		}

		/**
		 * Recursively descends the loop nest and allocates the data structures
		 * to analyze the corresponding loop.
		 * @param loop
		 * @param listUnrollFactors A list of unrolling configurations. A list entry is an array whose elements specify the
		 * 	unrolling factors for each loop in the loop nest
		 * @param nDepth
		 */
		private ILoopNestPart findLoops (ForLoop loop, ILoopNestPart lnpParent, int nDepth)
		{
			if (nDepth < m_data.getUnrollDepthCount ())
			{
				ILoopNestPart lnp = createLoopNestPart (loop, nDepth);
				lnp.setParent (lnpParent);
				m_listLoops.add (lnp);

				// try to get the loop (if any) nested within this one
				ForLoop loopChild = lnp.getNestedLoop ();
				if (loopChild != null)
					lnp.setChild (findLoops (loopChild, lnp, nDepth + 1));

				return lnp;
			}

			return null;
		}


		///////////////////////////////////////////////////////////////////
		// Perform Loop Unrolling

		private void unroll (ILoopNestPart lnpCurrent, ILoopNestPart lnpParent, List<LoopNest> listLoopNest, List<Boolean> listIsUnrolled, int nLevel)
		{
			if (nLevel == 0)
				UnrollLoop.LOGGER.info (StringUtil.concat ("Unrolling loop ", lnpCurrent == null ? "[null]" : lnpCurrent.getLoopIndex ().toString ()));

			if (lnpCurrent == null || nLevel >= m_nMaxLevels)
				generateUnrolledCode (lnpParent, listLoopNest, listIsUnrolled);
			else
				generateUnrollLists (lnpCurrent, lnpParent, listLoopNest, listIsUnrolled, nLevel);
		}

		/**
		 * Recursively builds the data structures for the unrolled and cleanup loop nests.
		 * @param lnpCurrent The current template loop nest for which the structures are created
		 * @param lnpParent The parent template loop of <code>lnpCurrent</code>
		 * @param listLoopNest The list of loop nests; each entry in the list corresponds to one of the unrolling configurations
		 * @param listIsUnrolled List of flags specifying whether the i-th loop in the nest has been unrolled (<code>true</code>) or is a cleanup loop (<code>false</code>)
		 * 	In each recursion call an entry will be added to a copy of this list; the length of the list corresponds to the number of loops in the loop nest
		 * 	at the end of the recursion
		 * @param nLevel The recursion level (the number of the loop in the nest when starting to count at the outermost loop)
		 */
		private void generateUnrollLists (ILoopNestPart lnpCurrent, ILoopNestPart lnpParent, List<LoopNest> listLoopNest, List<Boolean> listIsUnrolled, int nLevel)
		{
			////////////////////////////////
			// unroll

			// loop over all the unrolling configurations
			List<LoopNest> listUnrolledNest = new ArrayList<LoopNest> (m_data.getUnrollFactorsCount ());
			int i = 0;
			for (int[] rgUnrollFactors : m_data.getUnrollingFactors ())
			{
				LoopNest lnOrig = listLoopNest.get (i);
				if (lnOrig == null)
					listUnrolledNest.add (null);
				else
				{
					LoopNest lnUnrolledNest = (LoopNest) lnOrig.clone ();
					lnUnrolledNest.append (lnpCurrent.getUnrolledLoopHead (rgUnrollFactors[nLevel]));
					listUnrolledNest.add (lnUnrolledNest);
				}

				i++;
			}

			// recursively descend down the loop nest
			List<Boolean> listIsUnrolled0 = new ArrayList<Boolean> (listIsUnrolled.size () + 1);
			listIsUnrolled0.add (true);
			listIsUnrolled0.addAll (listIsUnrolled);
			unroll (lnpCurrent.getChild (), lnpCurrent, listUnrolledNest, listIsUnrolled0, nLevel + 1);


			////////////////////////////////
			// cleanup

			List<LoopNest> listCleanupNest = new ArrayList<LoopNest> (m_data.getUnrollFactorsCount ());
			i = 0;
			for (int[] rgUnrollFactors : m_data.getUnrollingFactors ())
			{
				// get the head of the cleanup loop
				// might be null if no cleanup loop is required (if the loop is unrolled completely)
				LoopNest lnUnroll = lnpCurrent.getCleanupLoopHead (rgUnrollFactors[nLevel]);
				if (lnUnroll == null)
					listCleanupNest.add (null);
				else
				{
					LoopNest lnOrig = listLoopNest.get (i);
					if (lnOrig == null)
						listCleanupNest.add (null);
					else
					{
						LoopNest lnCleanupNest = (LoopNest) lnOrig.clone ();
						lnCleanupNest.append (lnUnroll);
						listCleanupNest.add (lnCleanupNest);
					}
				}

				i++;
			}

			// recursively descent down the loop nest
			List<Boolean> listIsUnrolled1 = new ArrayList<Boolean> (listIsUnrolled.size () + 1);
			listIsUnrolled1.add (false);
			listIsUnrolled1.addAll (listIsUnrolled);
			unroll (lnpCurrent.getChild (), lnpCurrent, listCleanupNest, listIsUnrolled1, nLevel + 1);
		}

		/**
		 *
		 * @param lnpParent
		 * @param listLoopNest
		 * @param listIsUnrolled
		 */
		private void generateUnrolledCode (ILoopNestPart lnpParent, List<LoopNest> listLoopNest, List<Boolean> listIsUnrolled)
		{
			// prepare a list of the loop bodies
			Statement stmtBody = lnpParent.getLoop ().getBody ();
			List<CompoundStatement> listBodyStatements = new ArrayList<CompoundStatement> (listLoopNest.size ());
			if (stmtBody instanceof CompoundStatement)
			{
				for (int k = 0; k < listLoopNest.size (); k++)
				{
					CompoundStatement cmpstmtNewBody = (CompoundStatement) stmtBody.clone ();
					updateReference (stmtBody, cmpstmtNewBody);
					listBodyStatements.add (cmpstmtNewBody);
				}
			}
			else
			{
				for (int k = 0; k < listLoopNest.size (); k++)
				{
					CompoundStatement cmpstmtBody = new CompoundStatement ();
					Statement stmtNewBody = stmtBody.clone ();
					updateReference (stmtBody, stmtNewBody);
					cmpstmtBody.addStatement (stmtNewBody);
					listBodyStatements.add (cmpstmtBody);
				}
			}

			// process the loop bodies
			int k = 0;
			for (int[] rgUnrollFactors : m_data.getUnrollingFactors ())
			{
				// check whether this unrolling configuration can be skipped;
				// if there is nothing to unroll for this configuration, there is a null entry in listLoopNest
				// (e.g. because of a missing cleanup loop)
				if (listLoopNest.get (k) != null)
				{
					ILoopNestPart lnpThis = lnpParent;
					int j = 0;
					for (boolean bDoUnroll : listIsUnrolled)
					{
						if (bDoUnroll)
							listBodyStatements.set (k, lnpThis.unrollBody (listBodyStatements.get (k), rgUnrollFactors[j]));
						lnpThis = lnpThis.getParent ();
						j++;
					}
				}

				k++;
			}

			// set the loop bodies in the loop nests
			for (k = 0; k < listLoopNest.size (); k++)
			{
				LoopNest loopNest = listLoopNest.get (k);
				if (loopNest != null)
					loopNest.setNestBody (listBodyStatements.get (k));
			}

			// we reached the bottom (no child loop anymore or maximum nesting level reached):
			// add the unrolled loop nest to the list of statements with which the loop nest will be replaced ultimately
			k = 0;
			for (CompoundStatement cmpstmtUnrolled : m_data.getUnrolledStatements ())
			{
				LoopNest loopNest = listLoopNest.get (k);
				if (loopNest != null)
				{
					ForLoop loopNewNest = loopNest.clone ();
					updateReference (loopNest, loopNewNest);
					cmpstmtUnrolled.addStatement (loopNewNest);
				}
				k++;
			}

			// save the value of the loop index variable after the first run
			if (m_bIsFirstRun)
			{
				int nLevel = 0;
				for (ILoopNestPart lnp : m_listLoops)
				{
					k = 0;
					for (int[] rgUnrollFactor : m_data.getUnrollingFactors ())
					{
						if (rgUnrollFactor[nLevel] > 1)
						{
							m_data.getUnrolledStatement (k).addStatement (
								new ExpressionStatement (new AssignmentExpression (
									lnp.getEndValueIdentifier (),
									AssignmentOperator.NORMAL,
									lnp.getLoopIndex ().clone ())));
						}
						k++;
					}

					nLevel++;
				}

				m_bIsFirstRun = false;
			}
		}
	}


	///////////////////////////////////////////////////////////////////
	// Member Variables

	/**
	 * The class that is instantiated as loop nest parts
	 */
	private Class<? extends ILoopNestPart> m_clsLoopNestParts;

	/**
	 * List of statements
	 */
	private List<Statement> m_listReferencedStatements;

	/**
	 * Contains a mapping from the statements in the referenced list passed to
	 * {@link UnrollLoop} to the statements by which they were replaced during unrolling.
	 * The map is only created if a non-<code>null</code> list is passed to
	 * {@link UnrollLoop#UnrollLoop(Class, List)}.
	 */
	private Map<Statement, List<Statement>> m_mapReferencedStatementReplacements;


	///////////////////////////////////////////////////////////////////
	// Implementation

	public UnrollLoop (Class<? extends ILoopNestPart> clsLoopNestPart)
	{
		this (clsLoopNestPart, null);
	}

	/**
	 *
	 * @param clsLoopNestPart
	 * @param listReferencedStatements List of statements that are memorized for further use.
	 * 	Unrolling might clone objects, so the structure might change, rendering the memorized
	 * 	pointers invalid. Passing a list of referenced objects as argument will cause the unroller
	 * 	to update the statements that are referenced in the list.
	 */
	public UnrollLoop (Class<? extends ILoopNestPart> clsLoopNestPart, List<Statement> listReferencedStatements)
	{
		m_clsLoopNestParts = clsLoopNestPart;

		m_listReferencedStatements = listReferencedStatements;
		m_mapReferencedStatementReplacements = new HashMap<Statement, List<Statement>> ();
		if (m_listReferencedStatements != null)
			for (Statement stmt : m_listReferencedStatements)
				m_mapReferencedStatementReplacements.put (stmt, new ArrayList<Statement> ());
	}

	/**
	 * Unrolls the loop <code>loop</code> and returns the statements that have to
	 * be inserted into the code instead of the loop that is passed to this method.
	 * @param loop The loop to unroll
	 * @param listUnrollFactors The list of loop unrolling factors
	 * @param nMaxLevels
	 * @return An array of statements that replace <code>loop</code>
	 */
	public Map<IntArray, CompoundStatement> unroll (ForLoop loop, List<int[]> listUnrollFactors, int nMaxLevels)
	{
		Unroller unroller = new Unroller (loop, listUnrollFactors, nMaxLevels);
		return unroller.getUnrolledLoops ();
	}

	/**
	 * Unrolls the loop nest in <code>loop</code> if constant trip count loops are encountered. Other (non-constant
	 * trip count loops) are not touched.
	 * @param loop The outer most loop of the nest to unroll
	 * @param nMaxLevels The maximum number of loop nesting levels to consider.
	 * 	To consider arbitrarily many nesting levels, pass {@link Integer#MAX_VALUE}.
	 * @return
	 */
	public CompoundStatement unroll (ForLoop loop, int nMaxLevels)
	{
		// prepare the unroll factors
		List<int[]> listUnrollFactors = new ArrayList<int[]> ();
		listUnrollFactors.add (new int[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 });

		// unroll
		Map<IntArray, CompoundStatement> mapUnrolled = new Unroller (loop, listUnrollFactors, nMaxLevels).getUnrolledLoops ();

		if (mapUnrolled.size () == 0)
			return new CompoundStatement ();

		// return the first entry in the map (whatever "first" means...)
		return mapUnrolled.get (mapUnrolled.keySet ().iterator ().next ());
	}

	/**
	 * Updates the list of referenced statements by adding a new statement,
	 * <code>stmtAdd</code>, to the list of statements corresponding the the
	 * referenced statement.
	 * @param stmtOrig The original statement
	 * @param stmtAdd The statement to
	 */
	private void updateReference (Statement stmtOrig, Statement stmtAdd)
	{
		List<Statement> list = m_mapReferencedStatementReplacements.get (stmtOrig);
		if (list != null)
			list.add (stmtAdd);
	}

	/**
	 * Returns the list of statements that replace the reference statement <code>stmtOrig</code>.
	 * The original statement <code>stmtOrig</code> must have been provided to the constructor,
	 * otherwise an empty list will be returned.
	 * @param stmtOrig
	 * @return
	 */
	public List<Statement> getReplacedReferenceStatements (Statement stmtOrig)
	{
		List<Statement> list = m_mapReferencedStatementReplacements.get (stmtOrig);
		if (list == null)
			return new ArrayList<Statement> ();
		return list;
	}


	///////////////////////////////////////////////////////////////////
	// Testing

	public static void main (String[] args)
	{
		///////////////////////////////////////////////////////////////////

		VariableDeclarator declX = new VariableDeclarator (new NameID ("X"));
		Identifier idX = new Identifier (declX);

		VariableDeclarator declJ = new VariableDeclarator (new NameID ("j"));
		declJ.setInitializer (new ValueInitializer (new IntegerLiteral (1)));
		Identifier idJ = new Identifier (declJ);

		VariableDeclarator declI = new VariableDeclarator (new NameID ("i"));
		declI.setInitializer (new ValueInitializer (new IntegerLiteral (1) /* idX */));
		Identifier idI = new Identifier (declI);

		VariableDeclarator declSum = new VariableDeclarator (new NameID ("sum"));
		Identifier idSum = new Identifier (declSum);

		CompoundStatement cmpstmtBlock = new CompoundStatement ();

		VariableDeclarator declTemp = new VariableDeclarator (new NameID ("tmp"));
		declTemp.setInitializer (new ValueInitializer (new BinaryExpression (idI.clone (), BinaryOperator.MULTIPLY, idJ.clone ())));
		Identifier idTemp = new Identifier (declTemp);
		cmpstmtBlock.addDeclaration (new VariableDeclaration (Specifier.INT, declTemp));
		cmpstmtBlock.addStatement (new ExpressionStatement (new AssignmentExpression (idSum.clone (), AssignmentOperator.ADD, new BinaryExpression (idTemp.clone (), BinaryOperator.MULTIPLY, idTemp.clone ()))));

		List<Expression> listArgsCalc = new ArrayList<Expression> (1);
		listArgsCalc.add (new BinaryExpression (idI.clone (), BinaryOperator.ADD, new IntegerLiteral (2)));
		listArgsCalc.add (new BinaryExpression (idI.clone (), BinaryOperator.SUBTRACT, idJ.clone ()));
		cmpstmtBlock.addStatement (new ExpressionStatement (new FunctionCall (new NameID ("calc"), listArgsCalc)));

		ForLoop loop = new ForLoop (
			new DeclarationStatement (new VariableDeclaration (Specifier.INT, declJ)),
			new BinaryExpression (idJ.clone (), BinaryOperator.COMPARE_LE, new IntegerLiteral (4) /* new Identifier ("A") */),
			new UnaryExpression (UnaryOperator.PRE_INCREMENT, idJ.clone ()),
			new ForLoop (
				new DeclarationStatement (new VariableDeclaration (Specifier.INT, declI)),
				new BinaryExpression (idI.clone (), BinaryOperator.COMPARE_LT, /*new IntegerLiteral (10)*/ new BinaryExpression (idX.clone (), BinaryOperator.ADD, new IntegerLiteral (5))),
				//new UnaryExpression (UnaryOperator.PRE_INCREMENT, idI),
				new AssignmentExpression (idI.clone (), AssignmentOperator.ADD, new IntegerLiteral (1)),
				cmpstmtBlock.clone ()));

		System.out.println ();
		System.out.println ("Original:");
		System.out.println (loop);

		System.out.println ();
		System.out.println ("Unrolled:\n");
		UnrollLoop ul = new UnrollLoop (UniformlyIncrementingLoopNestPart.class);

		List<int[]> listUnrollingConfigurations = new ArrayList<int[]> ();
		listUnrollingConfigurations.add (new int[] { 3, 2 });
		listUnrollingConfigurations.add (new int[] { 1, 3 });


		Map<IntArray, CompoundStatement> mapUnrolled = ul.unroll (loop, listUnrollingConfigurations, Integer.MAX_VALUE);

		for (IntArray arrUnrollFactor : mapUnrolled.keySet ())
			System.out.println ("Unrolling " + Arrays.toString (arrUnrollFactor.get ()) + ":\n----------------------------------------------------\n" + mapUnrolled.get (arrUnrollFactor) + "\n\n\n");

		Maxima.getInstance ().close ();
	}
}
