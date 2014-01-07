package cetus.analysis;

import java.util.*;

import cetus.hir.*;

public class BasicBlock
{
  ArrayList<Traversable> statements;
  ArrayList<BasicBlock> preds;
  ArrayList<BasicBlock> succs;

  boolean visited = false;
  boolean calladjusted = false;

  public BasicBlock()
  {
    statements = new ArrayList<Traversable>();
    preds = new ArrayList<BasicBlock>();
    succs = new ArrayList<BasicBlock>();
  }

  public void addPredecessor(BasicBlock bb)
  {
    preds.add(bb);
  }

  public void addSuccessor(BasicBlock bb)
  {
    succs.add(bb);
  }
}
