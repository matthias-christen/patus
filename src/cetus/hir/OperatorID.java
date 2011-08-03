package cetus.hir;

import java.io.*;
import java.lang.reflect.*;
import java.util.*;

/**
 * C++ overloaded operator IDs. This class is no longer supported.
 */
public class OperatorID extends IDExpression
{
  private String name;

  public OperatorID(String name)
  {
    super(false);
    this.name = name;
  }

  public void print(PrintWriter o)
  {
    o.print("operator " + name);
  }

  @Override
  public boolean equals(Object o)
  {
    return (
        super.equals(o) &&
        name.equals(((OperatorID)o).name)
    );
  }
}
