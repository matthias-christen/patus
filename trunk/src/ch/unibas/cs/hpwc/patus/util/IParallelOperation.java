package ch.unibas.cs.hpwc.patus.util;

public interface IParallelOperation<T>
{
	/**
	 * Performs an operation on the element <code>element</code>.
	 * 
	 * @param element
	 *            The element on which to perform the operation
	 */
	public abstract void perform (T element);
}
