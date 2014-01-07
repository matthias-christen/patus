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

public interface IAdditionalKernelSpecific
{
	/**
	 * Returns a string with additional code to be added after the includes and before the actual kernel code.
	 * @return Additional code
	 */
	public abstract String getAdditionalKernelSpecificCode ();
}
