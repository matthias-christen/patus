package cetus.treewalker;

import java.lang.*;
import java.util.*;

public class TreeNode implements Iterable<TreeNode>
{
  private TreeNode parent;
  private ArrayList<TreeNode> children;

  private int id;
  private String text;

  public TreeNode(int node_id, TreeNode parent, String node_info)
  {
    this.parent = parent;
    this.children = null;
    this.id = node_id;
    this.text = node_info;
  }

  public void addChildLast(TreeNode child)
  {
    if (children == null)
    {
      children = new ArrayList();
    }

    children.add(child);
  }

  public TreeNode getChild(int n)
  {
    if (children == null)
      return null;
    else
      return children.get(n);
  }

  public int getChildCount()
  {
    if (children == null)
      return 0;
    else
      return children.size();
  }

  public int getID()
  {
    return id;
  }

  public TreeNode getParent()
  {
    return parent;
  }

  public String getText()
  {
    return text;
  }

  public boolean hasChildren()
  {
    return children != null;
  }

  public Iterator<TreeNode> iterator()
  {
    return children.iterator();
  } 

  public void printTree(int indent)
  {
    for (int i = 0; i < indent; ++i)
      System.out.print(" ");

    System.out.println(text);

    if (hasChildren())
    {
      for (TreeNode n : children)
        n.printTree(indent + 2);
    }
  }

  public void setParent(TreeNode parent)
  {
    this.parent = parent;
  }
}
