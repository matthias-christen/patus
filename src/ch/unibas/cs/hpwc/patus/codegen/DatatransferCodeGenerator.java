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
package ch.unibas.cs.hpwc.patus.codegen;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import cetus.hir.Expression;
import cetus.hir.IntegerLiteral;
import cetus.hir.Statement;
import ch.unibas.cs.hpwc.patus.ast.StatementListBundle;
import ch.unibas.cs.hpwc.patus.ast.SubdomainIdentifier;
import ch.unibas.cs.hpwc.patus.ast.SubdomainIterator;
import ch.unibas.cs.hpwc.patus.codegen.backend.IBackend;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;
import ch.unibas.cs.hpwc.patus.representation.StencilNode;
import ch.unibas.cs.hpwc.patus.util.CodeGeneratorUtil;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 *
 * @author Matthias-M. Christen
 */
public class DatatransferCodeGenerator
{
	///////////////////////////////////////////////////////////////////
	// Member Variables

	private CodeGeneratorSharedObjects m_data;


	///////////////////////////////////////////////////////////////////
	// Implementation

	public DatatransferCodeGenerator (CodeGeneratorSharedObjects data)
	{
		m_data = data;
	}

	/**
	 * Adds the code to allocate local data.
	 */
	public void allocateLocalMemoryObjects (SubdomainIterator sdIterator, CodeGeneratorRuntimeOptions options)
	{
		StatementListBundle slbLocalAlloc = new StatementListBundle (new ArrayList<Statement> (1));
		m_data.getData ().getMemoryObjectManager ().allocateMemoryObjects (sdIterator, slbLocalAlloc, options);
		List<Statement> listStmtLocalAlloc = slbLocalAlloc.getDefaultList ().getStatementsAsList ();
		Statement[] rgStmtLocalAlloc = new Statement[listStmtLocalAlloc.size ()];
		listStmtLocalAlloc.toArray (rgStmtLocalAlloc);
		m_data.getData ().addInitializationStatements (rgStmtLocalAlloc);
	}

	/**
	 *
	 * @param setMemoryObjects
	 * @param sdIterator
	 * @param slbOuter
	 */
	public void loadData (StencilNodeSet setMemoryObjects, SubdomainIterator sdIterator, StatementListBundle slbOuter, CodeGeneratorRuntimeOptions options)
	{
		// get data
		IBackend backend = m_data.getCodeGenerators ().getBackendCodeGenerator ();
		MemoryObjectManager mgr = m_data.getData ().getMemoryObjectManager ();
		SubdomainIdentifier sdid = sdIterator.getIterator ();
		SubdomainIdentifier sdidParent = m_data.getCodeGenerators ().getStrategyAnalyzer ().getParentGrid (sdid);
		int nParallelismLevel = sdIterator.getParallelismLevel ();

		// create the code
		generateTransferComment (setMemoryObjects, sdIterator, slbOuter, true);
		for (StencilNode n : setMemoryObjects)
		{
			MemoryObject mo = mgr.getMemoryObject (sdid, n, true);
//			backend.loadData (sgid,
//				n, mo, //mgr.getMemoryObjectExpression (sgid, n, null, false, false, slbOuter, options),
//				n, mgr.getParentMemoryObject (sgIterator, mo),
////				new UnaryExpression (
////					UnaryOperator.ADDRESS_OF,
////					mgr.getMemoryObjectExpression (sgidParent, getMinStencilNode (n, mo), null, true, true, slbOuter, options)),
//				nParallelismLevel, slbOuter);
			backend.loadData (getMinStencilNode (n, mo), sdid, mo, mgr.getParentMemoryObject (sdIterator, mo), nParallelismLevel, slbOuter, options);
		}
		backend.doLoadData (nParallelismLevel, slbOuter, options);
	}

	/**
	 *
	 * @param setMemoryObjects
	 * @param sdIterator
	 * @param slbGenerated
	 */
	public void storeData (StencilNodeSet setMemoryObjects, SubdomainIterator sdIterator, StatementListBundle slbGenerated, CodeGeneratorRuntimeOptions options)
	{
		// get data
		IBackend backend = m_data.getCodeGenerators ().getBackendCodeGenerator ();
		MemoryObjectManager mgr = m_data.getData ().getMemoryObjectManager ();
		SubdomainIdentifier sdid = sdIterator.getIterator ();
		SubdomainIdentifier sdidParent = m_data.getCodeGenerators ().getStrategyAnalyzer ().getParentGrid (sdid);
		int nParallelismLevel = sdIterator.getParallelismLevel ();

		// create the code
		generateTransferComment (setMemoryObjects, sdIterator, slbGenerated, false);
		for (StencilNode n : setMemoryObjects)
		{
			MemoryObject mo = mgr.getMemoryObject (sdid, n, true);
//			backend.storeData (sgidParent,
//				n, mgr.getParentMemoryObject (sgIterator, mo),
////				new UnaryExpression (
////					UnaryOperator.ADDRESS_OF,
////					mgr.getMemoryObjectExpression (sgidParent, getMinStencilNode (n, mo), null, true, true, slbGenerated, options)),
//				n, mo, //mgr.getMemoryObjectExpression (sgid, n, null, false, false, slbGenerated, options),
//				nParallelismLevel, slbGenerated, options);

			backend.storeData (getMinStencilNode (n, mo), sdid,
				mgr.getParentMemoryObject (sdIterator, mo), mo,
				nParallelismLevel, slbGenerated, options);
		}
		backend.doStoreData (nParallelismLevel, slbGenerated, options);
	}

	/**
	 *
	 * @param setMemoryObjects
	 * @param sdIterator
	 * @param slbGenerated
	 */
	public void waitFor (StencilNodeSet setMemoryObjects, SubdomainIterator sdIterator, StatementListBundle slbGenerated, CodeGeneratorRuntimeOptions options)
	{
		// get data
		IBackend backend = m_data.getCodeGenerators ().getBackendCodeGenerator ();
		MemoryObjectManager mgr = m_data.getData ().getMemoryObjectManager ();
		SubdomainIdentifier sdid = sdIterator.getIterator ();
		int nParallelismLevel = sdIterator.getParallelismLevel ();

		// create the code
		generateWaitForComment (setMemoryObjects, sdIterator, slbGenerated);
		for (StencilNode n : setMemoryObjects)
			backend.waitFor (n, mgr.getMemoryObject (sdid, n, true), nParallelismLevel, slbGenerated);
		backend.doWaitFor (nParallelismLevel, slbGenerated, options);
	}

	/**
	 *
	 * @param mo
	 * @return
	 */
	private StencilNode getMinStencilNode (StencilNode node, MemoryObject mo)
	{
		StencilNode nodeRef = new StencilNode (node);

		Expression[] rgExprBorderMin = mo.getBorder ().getMin ().getCoords ();
		int[] rgBorderMin = new int[rgExprBorderMin.length];

		int[] rgMask = new int[rgExprBorderMin.length];
		Arrays.fill (rgMask, 1);
		rgMask = mo.getProjectionMask ().apply (rgMask);

		for (int i = 0; i < rgExprBorderMin.length; i++)
		{
			if (rgMask[i] == 1)
			{
				if (!(rgExprBorderMin[i] instanceof IntegerLiteral))
					throw new RuntimeException ("Border sizes must be constant");
				rgBorderMin[i] = -(int) ((IntegerLiteral) rgExprBorderMin[i]).getValue ();
			}
			else
				rgBorderMin[i] = node.getSpaceIndex ()[i];	// TODO: this is inefficient
		}

		nodeRef.getIndex ().setSpaceIndex (rgBorderMin);
		return nodeRef;
	}

	/**
	 * Generates the comment for a data transfer.
	 * @param set
	 * @param slb
	 */
	private void generateTransferComment (StencilNodeSet set, SubdomainIterator sgIterator, StatementListBundle slb, boolean bIsDestinationSmall)
	{
		MemoryObjectManager mgr = m_data.getData ().getMemoryObjectManager ();
		SubdomainIdentifier sdid = sgIterator.getIterator ();

		StringBuilder sbSource = new StringBuilder ();
		StringBuilder sbDestination = new StringBuilder ();

		for (StencilNode n : set)
		{
			if (sbSource.length () > 0)
			{
				sbSource.append (", ");
				sbDestination.append (", ");
			}

			MemoryObject moChild = mgr.getMemoryObject (sdid, n, true);
			MemoryObject moParent = mgr.getParentMemoryObject (sgIterator, moChild);

			if (bIsDestinationSmall)
			{
				sbSource.append (moParent.getIdentifier ().getName ());
				sbDestination.append (moChild.getIdentifier ().getName ());
			}
			else
			{
				sbSource.append (moChild.getIdentifier ().getName ());
				sbDestination.append (moParent.getIdentifier ().getName ());
			}
		}

		slb.addStatement (CodeGeneratorUtil.createComment (StringUtil.concat ("Transferring data from ", sbSource, " to ", sbDestination), true));
	}

	/**
	 * Generates the comment for waitFor calls.
	 * @param set The memory objects for which to wait
	 * @param slb The statement list bundle to which the generated code is added
	 */
	private void generateWaitForComment (StencilNodeSet set, SubdomainIterator sgIterator, StatementListBundle slb)
	{
		MemoryObjectManager mgr = m_data.getData ().getMemoryObjectManager ();
		SubdomainIdentifier sdid = sgIterator.getIterator ();

		StringBuilder sb = new StringBuilder ();
		for (StencilNode n : set)
		{
			if (sb.length () > 0)
				sb.append (", ");

			MemoryObject mo = mgr.getMemoryObject (sdid, n, true);
			sb.append (mo.getIdentifier ().getName ());
		}

		slb.addStatement (CodeGeneratorUtil.createComment (StringUtil.concat ("Waiting for data ", sb), true));
	}
}
