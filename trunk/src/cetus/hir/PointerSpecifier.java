package cetus.hir;

import java.io.*;
import java.lang.reflect.*;
import java.util.*;

/** Represents a C or C++ pointer. */
public class PointerSpecifier extends Specifier
{
  /** * */
  public static final PointerSpecifier UNQUALIFIED = new PointerSpecifier();

  /** * const */
  public static final PointerSpecifier CONST =
    new PointerSpecifier((new ChainedList()).addLink(Specifier.CONST));

  /** * volatile */
  public static final PointerSpecifier VOLATILE =
    new PointerSpecifier((new ChainedList()).addLink(Specifier.VOLATILE));

  /** * const volatile */
  public static final PointerSpecifier CONST_VOLATILE =
    new PointerSpecifier((new ChainedList()).addLink(Specifier.CONST).addLink(Specifier.VOLATILE));

  private List qualifiers;

  private PointerSpecifier()
  {
    qualifiers = null;
  }

  private PointerSpecifier(List qualifiers)
  {
    this.qualifiers = qualifiers;
  }

  public void print(PrintWriter o)
  {
    o.print("*");
    PrintTools.printList(qualifiers, o);
    o.print(" ");
  }
}
