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

import ch.unibas.cs.hpwc.patus.codegen.KernelSourceFile;

/**
 * The backend code generator.
 * @author Matthias-M. Christen
 */
public interface IBackend extends IParallel, IDataTransfer, IIndexing, IArithmetic, IAdditionalKernelSpecific, INonKernelFunctions
{
	/**
	 * 
	 * @param ksf
	 */
	public abstract void setKernelSourceFile (KernelSourceFile ksf);
}
