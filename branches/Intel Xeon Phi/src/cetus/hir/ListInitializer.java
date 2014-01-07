package cetus.hir;

import java.io.*;
import java.lang.reflect.*;
import java.util.*;

public class ListInitializer extends Initializer
{
  public ListInitializer(List values)
  {
    for (Object o : values)
    {
      children.add((Traversable)o);
      ((Traversable)o).setParent(this);
    }
  }

  @Override
  public void print(PrintWriter o)
  {
    o.print(" = { ");
    PrintTools.printListWithComma(children, o);
    o.print(" } ");
  }
}
