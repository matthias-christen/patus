package ch.unibas.cs.hpwc.patus.graph;

public interface IEdge<V extends IVertex>
{
	public abstract V getHeadVertex ();
	
	public abstract V getTailVertex ();
}