package cetus.hir;

import java.io.PrintWriter;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;

/**
* <b>PrintTools</b> provides tools that perform printing of collections of IR
* or debug messages.
*/
public final class PrintTools
{
  /** Global verbosity taken from the command-line option */
  private static final int verbosity = 0;
      //Integer.valueOf(Driver.getOptionValue("verbosity")).intValue();

  private PrintTools() {}

  /**
   * Prints a Printable object to System.err if the
   * verbosity level is greater than min_verbosity.
   *
   * @param p A Printable object.
   * @param min_verbosity An integer to compare with the value
   *   set by the -verbosity command-line flag.
   */
  public static void printlnStatus(Printable p, int min_verbosity)
  {
    if (verbosity >= min_verbosity)
      System.err.println(p+"");
  }

  /**
   * Prints a string to System.err if the
   * verbosity level is greater than min_verbosity.
   *
   * @param message The message to be printed.
   * @param min_verbosity An integer to compare with the value
   *   set by the -verbosity command-line flag.
   */
  public static void printlnStatus(String message, int min_verbosity)
  {
    if (verbosity >= min_verbosity)
      System.err.println(message);
  }

  /**
   * Prints a string to System.out if the
   * verbosity level is greater than min_verbosity.
   *
   * @param message The message to be printed.
   * @param min_verbosity An integer to compare with the value
   *   set by the -verbosity command-line flag.
   */
  public static void print(String message, int min_verbosity)
  {
    if (verbosity >= min_verbosity)
      System.out.print(message);
  }

  /**
   * Prints a string to System.out if the
   * verbosity level is greater than min_verbosity.
   *
   * @param message The message to be printed.
   * @param min_verbosity An integer to compare with the value
   *   set by the -verbosity command-line flag.
   */
  public static void println(String message, int min_verbosity)
  {
    if (verbosity >= min_verbosity)
      System.out.println(message);
  }

  /**
   * Prints a Printable object to System.err if the
   * verbosity level is greater than min_verbosity.
   *
   * @param p A Printable object.
   * @param min_verbosity An integer to compare with the value
   *   set by the -verbosity command-line flag.
   */
  public static void printStatus(Printable p, int min_verbosity)
  {
    if (verbosity >= min_verbosity)
      System.err.print(p.toString());
  }

  /**
   * Prints a string to System.err if the
   * verbosity level is greater than min_verbosity.
   *
   * @param message The message to be printed.
   * @param min_verbosity An integer to compare with the value
   *   set by the -verbosity command-line flag.
   */
  public static void printStatus(String message, int min_verbosity)
  {
    if (verbosity >= min_verbosity)
      System.err.print(message);
  }

  /**
  * Prints a list of printable object to the specified print writer with a
  * separating stirng. If the list contains an object not printable, this method
  * throws a cast exception.
  *
  * @param list the list of printable object.
  * @param w the target print writer.
  * @param sep the separating string.
  */
  // No generic method is possible only because of Specifier list.
  public static void
      printListWithSeparator(List list, PrintWriter w, String sep)
  {
    if (list==null)
      return;
    Iterator iter = list.iterator();
    if (iter.hasNext()) {
      ((Printable)iter.next()).print(w);
      while (iter.hasNext()) {
        w.print(sep);
        ((Printable)iter.next()).print(w);
      }
    }
  }

  /**
  * Prints a list of printable object to the specified print writer with a
  * separating comma.
  *
  * @param list the list of printable object.
  * @param w the target print writer.
  */
  public static void printListWithComma(List list, PrintWriter w)
  {
    printListWithSeparator(list, w, ", ");
  }

  /**
  * Prints a list of printable object to the specified print writer with a
  * separating white space.
  *
  * @param list the list of printable object.
  * @param w the target print writer.
  */
  public static void printListWithSpace(List list, PrintWriter w)
  {
    printListWithSeparator(list, w, " ");
  }

  /**
  * Prints a list of printable object to the specified print writer without
  * any separating string.
  *
  * @param list the list of printable object.
  * @param w the target print writer.
  */
  public static void printList(List list, PrintWriter w)
  {
    printListWithSeparator(list, w, "");
  }

  /**
  * Prints a list of printable object to the specified print writer with a
  * separating new line character.
  *
  * @param list the list of printable object.
  * @param w the target print writer.
  */
  public static void printlnList(List list, PrintWriter w)
  {
    printListWithSeparator(list, w, "\n");
    w.println("");
  }

  /** Returns the global verbosity level */
  public static int getVerbosity() { return verbosity; }

  /**
   * Converts a collection of objects to a string with the given separator.
   * By default, the element of the collections are sorted alphabetically, and
   * any {@code Symbol} object is printed with its name.
   *
   * @param coll the collection to be converted.
   * @param separator the separating string.
   * @return the converted string.
   */
  public static String collectionToString(Collection coll, String separator)
  {
    if ( coll == null || coll.size() == 0 )
      return "";

    // Sort the collection first.
    TreeSet<String> sorted = new TreeSet<String>();
    for ( Object o : coll )
    {
      if ( o instanceof Symbol )
        sorted.add(((Symbol)o).getSymbolName());
      else
        sorted.add(o.toString());
    }

    StringBuilder str = new StringBuilder(80);

    Iterator<String> iter = sorted.iterator();
    if ( iter.hasNext() )
    {
      str.append(iter.next());
      while ( iter.hasNext() )
      {
        str.append(separator);
        str.append(iter.next());
      }
    }

    return str.toString();
  }

  /**
   * Converts a list of objects to a string with the given separator.
   *
   * @param list the list to be converted.
   * @param separator the separating string.
   * @return the converted string.
   */
  public static String listToString(List list, String separator)
  {
    if ( list == null || list.size() == 0 )
      return "";

    StringBuilder str = new StringBuilder(80);

    Iterator iter = list.iterator();
    if ( iter.hasNext() )
    {
      str.append(iter.next().toString());
      while ( iter.hasNext() )
        str.append(separator+iter.next().toString());
    }

    return str.toString();
  }

  /**
  * Converts a list of objects to a string. The difference from
  * {@code listToString} is that this method inserts the separating string
  * only if the heading string length is non-zero.
  */
  public static String listToStringWithSkip(List list, String separator)
  {
    if (list == null)
      return "";
    StringBuilder sb = new StringBuilder(80);
    Iterator iter = list.iterator();
    if (iter.hasNext()) {
      String prev = iter.next().toString();
      sb.append(prev);
      while (iter.hasNext()) {
        if (prev.length() > 0)
          sb.append(separator);
        prev = iter.next().toString();
        sb.append(prev);
      }
    }
    return sb.toString();
  }

  /** Converts a map to a string. */
  public static String mapToString(Map map, String separator)
  {
    if ( map == null || map.size() == 0 )
      return "";

    StringBuilder str = new StringBuilder(80);

    Iterator iter = map.keySet().iterator();
    if ( iter.hasNext() )
    {
      Object key = iter.next();
      str.append(key.toString()+":"+map.get(key).toString());
      while ( iter.hasNext() )
      {
        key = iter.next();
        str.append(separator+key.toString()+":"+map.get(key).toString());
      }
    }
    return str.toString();
  }

}
