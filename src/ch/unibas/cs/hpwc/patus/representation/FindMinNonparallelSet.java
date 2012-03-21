package ch.unibas.cs.hpwc.patus.representation;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

import cetus.hir.Specifier;
import ch.unibas.cs.hpwc.patus.util.IntArray;
import ch.unibas.cs.hpwc.patus.util.MathUtil;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

public class FindMinNonparallelSet
{	
	/**
	 * Determines whether <code>node2</code> is parallel to <code>node1</code>.
	 * @param node1
	 * @param node2
	 * @return The scaling factor by which <code>node1</code> has to be scaled to equal <code>node2</code>
	 * 	or <code>null</code> if the vectors of the spatial indices of the nodes are not parallel
	 */
	private static int[] isParallel (StencilNode node1, StencilNode node2)
	{
		if (node1.getSpaceIndex ().length != node2.getSpaceIndex ().length)
			throw new RuntimeException ("Stencil nodes must have spatial indices of equal dimensions.");
		
		final int nDim = node1.getSpaceIndex ().length;
		if (nDim <= 0)
			throw new RuntimeException ("The spatial indices of stencil nodes must be at least one-dimensional.");
				
		int nIdx = 0;
		while (nIdx < nDim && node1.getSpaceIndex ()[nIdx] == 0 && node2.getSpaceIndex ()[nIdx] == 0)
			nIdx++;
		if (nIdx >= nDim)
			return new int[nDim];	// all entries are 0
		if (node1.getSpaceIndex ()[nIdx] == 0 || node2.getSpaceIndex ()[nIdx] == 0)
			return null;	// not parallel if component nIdx of one vector is 0 and the other isn't 
		
		int[] rgResult = new int[nDim];

		int nNumerator = node2.getSpaceIndex ()[nIdx];
		int nDenominator = node1.getSpaceIndex ()[nIdx];
		int nGCD = MathUtil.getGCD (nNumerator, nDenominator);
		rgResult[nIdx] = Math.min (Math.abs (nNumerator), Math.abs (nDenominator)) / nGCD;
		
		for (int i = nIdx + 1; i < nDim; i++)
		{
			if (node2.getSpaceIndex ()[i] * nDenominator != node1.getSpaceIndex ()[i] * nNumerator)
				return null;	// not parallel
			rgResult[i] = Math.min (node1.getSpaceIndex ()[i], node2.getSpaceIndex ()[i]) / nGCD;
		}
		
		return rgResult;
	}
	
	private static int[] getLower (int[] rgV1, int[] rgV2)
	{
		int nDim = rgV1.length;
		int nIdx = 0;
		while (nIdx < nDim && rgV1[nIdx] == 0 && rgV2[nIdx] == 0)
			nIdx++;
		if (nIdx >= nDim)
			return rgV1;
		return Math.abs (rgV1[nIdx]) < Math.abs (rgV2[nIdx]) ? rgV1 : rgV2;
	}
	
	public static Map<StencilNode, IntArray> find (Collection<StencilNode> nodes)
	{
		Map<StencilNode, IntArray> map = new HashMap<> ();
		Map<StencilNode, IntArray> mapTmp = new HashMap<> ();

		for (StencilNode node : nodes)
		{
			boolean bParallelVectorFound = false;
			mapTmp.clear ();
			
			for (StencilNode n1 : map.keySet ())
			{
				int[] rgMinVector = FindMinNonparallelSet.isParallel (n1, node);
				if (rgMinVector != null)
				{
					int[] rgLower = FindMinNonparallelSet.getLower (rgMinVector, map.get (n1).get ());
					mapTmp.put (n1, new IntArray (rgLower));
					mapTmp.put (node, new IntArray (rgLower));
					bParallelVectorFound = true;
				}
			}
			
			if (!bParallelVectorFound)
				map.put (node, new IntArray (node.getSpaceIndex ()));
			else
				map.putAll (mapTmp);
		}
		
		return map;
	}
	
	public static void main (String[] args)
	{
		Collection<StencilNode> nodes = new ArrayList<> ();
		nodes.add (new StencilNode ("n0", Specifier.FLOAT, new Index (0, new int[] { 0, 0, 0 }, 0)));
		nodes.add (new StencilNode ("n1", Specifier.FLOAT, new Index (0, new int[] { 3, 0, 0 }, 0)));
		nodes.add (new StencilNode ("n2", Specifier.FLOAT, new Index (0, new int[] { -2, 0, 0 }, 0)));
		nodes.add (new StencilNode ("n3", Specifier.FLOAT, new Index (0, new int[] { 2, 0, 0 }, 0)));
		nodes.add (new StencilNode ("n4", Specifier.FLOAT, new Index (0, new int[] { 0, 1, 0 }, 0)));
		nodes.add (new StencilNode ("n5", Specifier.FLOAT, new Index (0, new int[] { 1, 1, 0 }, 0)));
		
		Map<StencilNode, IntArray> map = find (nodes);
		
		for (StencilNode n : map.keySet ())
			System.out.println (StringUtil.concat (n.toString (), " : ", map.get (n).toString ()));
	}
}
