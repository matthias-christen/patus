package cetus.treewalker;

import java.io.*;
import java.lang.*;
import java.util.*;

/**
 * Parser for graphviz-formatted dag files.
 */
public class DagParser
{
  private Vector<String> input_buffer;

  private class InputBufferMonitor
  {
    private boolean running;

    public InputBufferMonitor()
    {
      running = true;
    }

    public synchronized void done()
    {
      running = false;
      notifyAll();
    }

    public synchronized String get()
    {
      if (running)
      {
        try {
          return input_buffer.remove(0);
        } catch (ArrayIndexOutOfBoundsException e) {
        }

        try {
          wait();
        } catch (InterruptedException e) {
        }
      }

      try {
        return input_buffer.remove(0);
      } catch (ArrayIndexOutOfBoundsException e) {
        return null;
      }
    }

    public synchronized void put(String s)
    {
      input_buffer.add(s);
      notifyAll();
    }
  };

  private InputBufferMonitor input_buffer_monitor;

  /**
   * I/O thread to keep filling a buffer of tokens from the
   * dag file.  This is basically a simple, hand-coded
   * lexer that will handle bare dag files without
   * comments or annotations.
   */
  private class DagReaderThread extends Thread
  {
    private BufferedReader source;

    public DagReaderThread(BufferedReader source)
    {
      this.source = source;
    }

    public void run()
    {
      String line = null;

      while (true)
      {
        /* We just want to keep the buffer non-empty; no need
           to do anything if there's plenty there. */
        if (input_buffer.size() > 100)
          continue;

        try {
          line = source.readLine();
        } catch (IOException e) {
          System.err.println("cetus: error reading dag file, " + e);
        }

        /* eof? (really should never return here; see the } below) */
        if (line == null)
          break;

        line = line.trim();

        if (line.equals("}"))
        {
          /* a } closes the digraph and ends the dag file */
          input_buffer_monitor.put("}");
          break;
        }
        else if (line.endsWith(";"))
        {
          /* we've got a line that defines a node */
          input_buffer_monitor.put(line.substring(0, line.length() - 1));
          input_buffer_monitor.put(";");
        }
        else if (line.endsWith("}"))
        {
          /* we've got a line that defines edges */
          String[] tokens = line.split(" -> ");

          String lhs = tokens[0].trim();
          input_buffer_monitor.put(lhs);

          input_buffer_monitor.put("->");

          String rhs = tokens[1].trim();

          /* remove the braces */
          rhs = rhs.substring(1, rhs.length() - 1).trim();

          /* Split on the gaps between the quoted nodes on the rhs.
             This works, but wipes out some quotes, so they get
             put back in the loop below. */
          tokens = rhs.split("\" \"");

          input_buffer_monitor.put("{");
          for (int i = 0; i < tokens.length; ++i)
          {
            String s = tokens[i];

            /* these strings must start and end with a quote */
            if (!s.startsWith("\""))
              s = "\"" + s;
            if (!s.endsWith("\""))
              s = s + "\"";

            input_buffer_monitor.put(s);
          }
          input_buffer_monitor.put("}");
        }
        else
        {
          /* This is really just for the first line of input, and
             at least will do something reasonably sane if it's
             reached otherwise. */
          String[] tokens = line.split(" ");
          for (int i = 0; i < tokens.length; ++i)
            input_buffer_monitor.put(tokens[i]);
        }
      }

      input_buffer_monitor.done();
    }
  };

  private DagReaderThread reader_thread;

  private Hashtable node_table;

  public DagParser()
  {
    input_buffer = new Vector<String>(128);
    input_buffer_monitor = new InputBufferMonitor();
    node_table = new Hashtable(500);
  }

  private TreeNode fetchNode(int target_id, String target_info)
  {
    TreeNode  _node = (TreeNode)(node_table.remove(new Integer(target_id)));
    if (_node == null)  
    {
      //this is a leaf node, just creat one leaf node and return
      TreeNode leaf_node = new TreeNode(target_id, null, target_info);
      return leaf_node;
    }
    else
    {
      //not leaf, just return the node found
      return _node;
    }
  }

  private void hashSyntaxNode(int node_id, TreeNode node_to_hash)
  {
    if (node_table.put(new Integer(node_id), node_to_hash) != null)  //means collision
    {
      System.out.println("DagParser Fatal Error: collision in hashtable !");
      System.exit(1);
    }
  }

  /** Matches a string token and reports an error
   * if the match failed.
   */
  private void match(String s)
  {
    String token = input_buffer_monitor.get();
    if (!s.equals(token))
      throw new RuntimeException("cetus: dag file parser expected " + s + ", got " + token + " instead");
  }

  public TreeNode run(String dag_file_name)
  {
    BufferedReader dag_file = null;
    TreeNode lhs_syntax_node = null;

    try {
      dag_file = new BufferedReader(new FileReader(dag_file_name));
    } catch (FileNotFoundException e) {
      System.err.println("cetus: could not open dag file " + dag_file_name);
      return null;
    }

    reader_thread = new DagReaderThread(dag_file);
    reader_thread.start();

    match("digraph");
    input_buffer_monitor.get();  /* skip filename */
    match("{");  /* skip opening brace */

    while (true)
    {
      String current = input_buffer_monitor.get();
      String next = input_buffer_monitor.get();

      if (current.equals("}") && next == null)
      {
        /* done */
        break;
      }
      else if (next.equals(";"))
      {
        /* current defines a node; skip */
      }
      else if (next.equals("->"))
      {
        /* current is the lhs of an edge definition */

        /* remove the enclosing quotes */
        current = current.substring(1, current.length() - 1);

        int first_space = current.indexOf(" ");

        int     lhs_node_id    = Integer.parseInt(current.substring(0, first_space));
        String  node_info  = current.substring(first_space + 1, current.length());

        //create a new node, and then add the RHS nodes (must be ready or leaves) into the children list
        lhs_syntax_node = new TreeNode(lhs_node_id, null, node_info);

        String s = null;

        match("{");
        while (!(s = input_buffer_monitor.get()).equals("}"))
        {
          /* remove the enclosing quotes */
          s = s.substring(1, s.length() - 1);

          first_space = s.indexOf(" ");

          int node_id = Integer.parseInt(s.substring(0, first_space));
          node_info  = s.substring(first_space + 1, s.length());

          TreeNode child_syntax_node = fetchNode(node_id, node_info);

          child_syntax_node.setParent(lhs_syntax_node);
          lhs_syntax_node.addChildLast(child_syntax_node);          
        }

        hashSyntaxNode(lhs_node_id, lhs_syntax_node);
      }
    }

    return lhs_syntax_node;
  }
}
