package ch.unibas.cs.hpwc.patus.representation;

public interface ISpaceIndexable
{
	/**
	 * Returns the space index, an array of integers corresponding to
	 * the spatial coordinates.
	 * @return The spatial coordinates as an <code>int</code> array
	 */
	public abstract int[] getSpaceIndex ();
}
