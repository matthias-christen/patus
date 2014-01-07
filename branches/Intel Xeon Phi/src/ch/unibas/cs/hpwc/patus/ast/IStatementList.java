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
package ch.unibas.cs.hpwc.patus.ast;

import cetus.hir.Declaration;
import cetus.hir.Statement;

public interface IStatementList
{
	/**
	 *
	 * @param stmt
	 */
	public abstract void addStatement (Statement stmt);

	/**
	 *
	 * @param declaration
	 */
	public abstract void addDeclaration (Declaration declaration);

	public void addStatementAtTop (Statement stmt);


}
