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
package ch.unibas.cs.hpwc.patus.codegen.backend;

import cetus.hir.Expression;
import ch.unibas.cs.hpwc.patus.ast.StatementListBundle;
import ch.unibas.cs.hpwc.patus.ast.SubdomainIdentifier;
import ch.unibas.cs.hpwc.patus.codegen.MemoryObject;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;
import ch.unibas.cs.hpwc.patus.representation.StencilNode;

/**
 *
 * @author Matthias-M. Christen
 */
public interface IDataTransfer
{
	/**
	 * Allocates local storage for the memory object <code>mo</code>.
	 * @param mo The memory object for which to allocate data
	 * @param exprMemoryObject The memory object expression with appropriate indices
	 * @param slbCode The statement list bundle to which the code is appended
	 */
	public abstract void allocateData (
		StencilNode node, MemoryObject mo, Expression exprMemoryObject,
		int nParallelismLevel,
		StatementListBundle slbCode);

	/**
	 * Called after all the calls to {@link IDataTransfer#allocateData(MemoryObject, Expression, Expression, Expression, StatementListBundle)}
	 * within an allocate entity.
	 */
	public abstract void doAllocateData (int nParallelismLevel, StatementListBundle slbCode, CodeGeneratorRuntimeOptions options);

	/**
	 * Loads data from the memory object <code>exprSourceMemoryObject</code> into the memory object <code>moDestination</code>.
	 * @param sgidLargeLevel The iterator subdomain identifier used to calculate the index into the memory object on the level with
	 * 	the larger memory objects (the index into the parent memory object)
	 * @param moDestination The destination memory object
	 * @param exprSourceMemoryObject The source expression (an indexed memory object)
	 * @param slbCode The statement list bundle to which the code is appended
	 */
	public abstract void loadData (
		StencilNode node, SubdomainIdentifier sdidSourceIterator, MemoryObject moDestination, MemoryObject moSource,
		int nParallelismLevel,
		StatementListBundle slbCode, CodeGeneratorRuntimeOptions options);

	/**
	 * Called after all the calls to {@link IDataTransfer#loadData(MemoryObject, Expression, Expression, Expression, Expression, StatementListBundle)}
	 * within a load entity.
	 */
	public abstract void doLoadData (int nParallelismLevel, StatementListBundle slbCode, CodeGeneratorRuntimeOptions options);

	/**
	 * Stores the data in the source memory object back to the destination, <code>exprDestinationMemoryObject</code>.
	 * @param sgidLargeLevel The iterator subdomain identifier used to calculate the index into the memory object on the level with
	 * 	the larger memory objects (the index into the parent memory object)
	 * @param exprDestinationMemoryObject The destination expression (an indexed memory object)
	 * @param moSource The source memory object from which data is copied to the destination
	 * @param slbCode The statement list bundle to which the code is appended
	 */
	public abstract void storeData (
		StencilNode node, SubdomainIdentifier sdidSourceIterator, MemoryObject moDestination, MemoryObject moSource,
		int nParallelismLevel,
		StatementListBundle slbCode, CodeGeneratorRuntimeOptions options);

	/**
	 * Called after all the calls to {@link IDataTransfer#storeData(Expression, MemoryObject, Expression, Expression, Expression, StatementListBundle)}
	 * within a store entity.
	 */
	public abstract void doStoreData (int nParallelismLevel, StatementListBundle slbCode, CodeGeneratorRuntimeOptions options);

	/**
	 * Waits for loading the data from the memory object <code>mo</code> to complete.
	 * @param mo
	 * @param slbCode The statement list bundle to which the code is appended
	 */
	public abstract void waitFor (StencilNode node, MemoryObject mo, int nParallelismLevel, StatementListBundle slbCode);

	/**
	 * Called after all the calls to {@link IDataTransfer#waitFor(MemoryObject, StatementListBundle)} within a wait for entity.
	 */
	public abstract void doWaitFor (int nParallelismLevel, StatementListBundle slbCode, CodeGeneratorRuntimeOptions options);
}
