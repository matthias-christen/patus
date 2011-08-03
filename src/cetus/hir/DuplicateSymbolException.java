package cetus.hir;

import java.util.*;

/**
 * Thrown when a symbol is about to be entered into a symbol
 * table and there is already a symbol of the same name in
 * that same table.
 */
public class DuplicateSymbolException extends RuntimeException
{
  /**
   * Creates the exception.
   *
   * @param message The message should indicate the name
   *    of the offending symbol.
   */
  public DuplicateSymbolException(String message)
  {
    super(message);
  }
}
