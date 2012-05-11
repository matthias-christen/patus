package ch.unibas.cs.hpwc.patus.graph;

public interface IParametrizedEdge<V extends IVertex, T> extends /*IEdge<V>,*/ IParametrizedObject<T>
{
	/**
	 * @see IEdge#getHeadVertex()
	 */
	public abstract V getHeadVertex ();

	/**
	 * @see IEdge#getTailVertex()
	 */
	public abstract V getTailVertex ();
}
