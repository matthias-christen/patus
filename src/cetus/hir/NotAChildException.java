package cetus.hir;

import java.util.*;

/**
 * Thrown when an action is performed on an IR object and
 * that action requires the object to be the child of
 * another object, but that is not the case.
 */
public class NotAChildException extends RuntimeException
{
  public NotAChildException()
  {
    super();
  }

  public NotAChildException(String message)
  {
    super(message);
  }
}
