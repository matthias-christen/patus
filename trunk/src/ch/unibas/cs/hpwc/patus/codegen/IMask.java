package ch.unibas.cs.hpwc.patus.codegen;

import java.util.List;
import java.util.Map;

import ch.unibas.cs.hpwc.patus.representation.ISpaceIndexable;
import ch.unibas.cs.hpwc.patus.util.IntArray;

public interface IMask
{
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
