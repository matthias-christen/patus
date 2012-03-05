package ch.unibas.cs.hpwc.patus.graph;

public interface IParametrizedObject<T>
{
	public abstract void setData (T data);
	
	public abstract T getData ();
}
