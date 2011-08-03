package cetus.hir;

import java.util.*;

/**
 * Thrown when an action is performed on an IR object
 * and that object is required to have no parent object,
 * but that is not the case.
 */
public class NotAnOrphanException extends RuntimeException
{
  public NotAnOrphanException()
  {
    super();
  }

  public NotAnOrphanException(String message)
  {
    super(message);
  }
}
