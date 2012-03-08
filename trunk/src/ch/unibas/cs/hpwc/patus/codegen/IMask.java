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

import java.util.List;
import java.util.Map;

import ch.unibas.cs.hpwc.patus.representation.ISpaceIndexable;
import ch.unibas.cs.hpwc.patus.util.IntArray;

public interface IMask
{
	/**
	 * Projects the spatial index of <code>index</code> onto the quotient space.
	 * @param index
	 * @return
	 */
	public abstract int[] getEquivalenceClass (ISpaceIndexable index);

	/**
	 * Groups the items in <code>itInput</code> into equivalence classes according to the
	 * mask and returns the equivalence classes as a map from the spatial index to a list
	 * of items contained in the equivalence class. The spatial index, which is the key
	 * in the map, is a vector in the quotient space.
	 * @return
	 */
	public abstract Map<IntArray, List<ISpaceIndexable>> getEquivalenceClasses (Iterable<? extends ISpaceIndexable> itInput);

	/**
	 * Applies the mask to a vector, <code>rgVector</code>.
	 * @return The masked vector
	 */
	public abstract int[] apply (int[] rgVector);
}
