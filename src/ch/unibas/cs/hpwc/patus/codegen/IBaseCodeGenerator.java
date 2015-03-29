package ch.unibas.cs.hpwc.patus.codegen;

public interface IBaseCodeGenerator
{
	public String getFileHeader ();
	public String getIncludesAndDefines (boolean bIncludeAutotuneParameters);
}
