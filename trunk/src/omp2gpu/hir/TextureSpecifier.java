package omp2gpu.hir;

import java.io.*;
import java.lang.reflect.*;
import java.util.*;
import cetus.hir.*;

/**
 * @author Seyong Lee <lee222@purdue.edu>
 *         ParaMount Group 
 *         School of ECE, Purdue University
 *
 * Represents an Texture specifier in CUDA, for example 
 * texture<Type, Dim, ReadMode> texRef;
 */
public class TextureSpecifier extends CUDASpecifier
{
	private List<Specifier> specs;
	private IntegerLiteral dim;
	private NameID readMode;

	public TextureSpecifier()
	{
		specs = new LinkedList<Specifier>();
	}

	public TextureSpecifier(Specifier type)
	{
		specs = new LinkedList<Specifier>();
		specs.add(type);
		dim = new IntegerLiteral(1);
		readMode = new NameID("cudaReadModeElementType");
	}
	
	public TextureSpecifier(List<Specifier> ispecs)
	{
		specs = new LinkedList<Specifier>();
		specs.addAll(ispecs);
		dim = new IntegerLiteral(1);
		readMode = new NameID("cudaReadModeElementType");
	}

	public TextureSpecifier(List<Specifier> ispecs, int idim)
	{
		specs = new LinkedList<Specifier>();
		specs.addAll(ispecs);
		dim = new IntegerLiteral(idim);
		readMode = new NameID("cudaReadModeElementType");
	}

	public TextureSpecifier(List<Specifier> ispecs, int idim, String ireadMode)
	{
		specs = new LinkedList<Specifier>();
		specs.addAll(ispecs);
		dim = new IntegerLiteral(idim);
		readMode = new NameID(ireadMode);
	}

	/** Prints the specifier to the print writer. */
	public void print(PrintWriter o)
	{
		o.print("texture<");
		o.print(PrintTools.listToString(specs, " "));
		o.print(", " + dim + ", " + readMode);
		o.print(">");
	}

	/** Returns a string representation of the specifier. */
	@Override
	public String toString()
	{
		StringWriter sw = new StringWriter(16);
		print(new PrintWriter(sw));
		return sw.toString();
	}

}
