package cetus.hir;

import java.io.*;
import java.lang.reflect.*;
import java.util.*;

/**
 * BitfieldSpecifier represents the bit field declared in a structure.
 */
public class BitfieldSpecifier extends Specifier
{
	private Expression bit;

	/**
	 * Constructs a bit field specifier from the given bit expression.
	 */
	public BitfieldSpecifier(Expression e)
	{
		bit = e;
	}

  /** Prints the specifier on the specified print writer. */
  public void print(PrintWriter o)
  {
    o.print(" : ");
    bit.print(o);
  }

}
