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

import cetus.hir.Traversable;
import ch.unibas.cs.hpwc.patus.ast.StatementListBundle;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;

public interface ICodeGenerator
{
	/**
	 * Generates code for a particular type of input.
	 * @param trvInput The input structure
	 * @return A statement list bundle containing the generated output structure
	 */
	public abstract StatementListBundle generate (Traversable trvInput, CodeGeneratorRuntimeOptions options);
}
