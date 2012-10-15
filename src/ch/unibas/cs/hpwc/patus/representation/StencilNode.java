/**
 *
 */
package ch.unibas.cs.hpwc.patus.representation;

import java.io.PrintWriter;

import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.Expression;
import cetus.hir.Identifier;
import cetus.hir.IntegerLiteral;
import cetus.hir.Specifier;
import ch.unibas.cs.hpwc.patus.util.StringUtil;


/**
 * @author Matthias-M. Christen
 */
public class StencilNode extends Identifier implements ISpaceIndexable
{
	///////////////////////////////////////////////////////////////////
	// Member Variables

	/**
	 * The data type of the stencil node
	 */
	private Specifier m_specType;

	/**
	 * The index indicating where (in space, in time, in which vector
	 * component) the node is located relative to the center node
	 * (0, 0, 0) of the stencil
	 */
	private Index m_index;
	
	/**
	 * A constraint expression for identifying points in the boundary/initialization specifications
	 */
	private Expression m_exprConstraint;


	///////////////////////////////////////////////////////////////////
	// Implementation

	/**
	 * Creates a new stencil node locate at index <code>index</code> (which
	 * comprises
	 * the spatial, temporal, and vectorial components.
	 * 
	 * @param strIdentifier
	 *            The identifier by which this node is referred to in code
	 * @param index
	 *            The node index
	 */
	public StencilNode (String strIdentifier, Specifier specType, Index index)
	{
		super (strIdentifier);
		
		m_specType = specType;
		m_index = index == null ? new Index () : new Index (index);
		m_exprConstraint = null;
		
		setDefaultPrintMethod ();
	}

	public StencilNode (StencilNode node)
	{
		this (node.getName (), node.getSpecifier (), new Index (node.getIndex ()));
	}
	
	public void setDefaultPrintMethod ()
	{
		Class<?>[] rgParams = new Class<?>[] { Identifier.class, PrintWriter.class };
		try
		{
			object_print_method = Identifier.class.getMethod ("defaultPrint", rgParams);
		}
		catch (Exception e)
		{
			object_print_method = null;
		}
	}
	
	public void setExpandedPrintMethod ()
	{
		Class<?>[] rgParams = new Class<?>[] { StencilNode.class, PrintWriter.class };
		try
		{
			object_print_method = StencilNode.class.getMethod ("expandedPrint", rgParams);
		}
		catch (Exception e)
		{
			object_print_method = null;
		}
	}
	
	public static void expandedPrint (StencilNode node, PrintWriter o)
	{
		o.print (node.getName ());
		o.print ("___");
		
		for (Expression exprIdx : node.getIndex ().getSpaceIndexEx ())
		{
			if (exprIdx instanceof IntegerLiteral)
				printNum (((IntegerLiteral) exprIdx).getValue (), o);
			else
				exprIdx.print (o);
			
			o.print ('_');
		}
		
		o.print ("__");
		o.print (StringUtil.num2IdStr (node.getIndex ().getTimeIndex ()));
		o.print ("___");
		o.print (StringUtil.num2IdStr (node.getIndex ().getVectorIndex ()));
	}
	
	private static void printNum (long n, PrintWriter o)
	{
		if (n >= 0)
			o.print (n);
		else
		{
			o.print ('m');
			o.print (-n);
		}
	}

	public Specifier getSpecifier ()
	{
		return m_specType;
	}

	/**
	 * Returns the stencil index consisting of the spatial, temporal, and
	 * vectorial index (relative to the center node (0, 0, 0) of the stencil.
	 * 
	 * @return The stencil node index
	 */
	public Index getIndex ()
	{
		return m_index;
	}

	@Override
	public int[] getSpaceIndex ()
	{
		return m_index.getSpaceIndex ();
	}

	/**
	 * Returns <code>true</code> iff this stencil node represents a scalar variable.
	 * @return
	 */
	public boolean isScalar ()
	{
		return m_index.getSpaceIndexEx ().length == 0 && m_index.getTimeIndex () == 0 && m_index.getVectorIndex () == 0;
	}
	
	public Expression getConstraint ()
	{
		return m_exprConstraint;
	}
	
	public void setConstraint (Expression exprConstraint)
	{
		m_exprConstraint = exprConstraint;
	}
	
	public void addConstraint (Expression exprConstraint)
	{
		if (m_exprConstraint == null)
			m_exprConstraint = exprConstraint;
		else
			m_exprConstraint = new BinaryExpression (m_exprConstraint, BinaryOperator.LOGICAL_AND, exprConstraint);
	}
	
	public StencilNode clone ()
	{
		return new StencilNode (this);
	}

	@Override
	public boolean equals (Object obj)
	{
		if (obj instanceof StencilNode)
		{
			if (m_index.getSpaceIndexEx ().length == 0)
			{
				if (((StencilNode) obj).getIndex ().getSpaceIndexEx ().length != 0)
					return false;
				return getName ().equals (((StencilNode) obj).getName ());
			}
			else
				return m_index.equals (((StencilNode) obj).getIndex ());
		}
		if (obj instanceof Index)
			return m_index.equals (obj);

		return false;
	}

	@Override
	public int hashCode ()
	{
		if (m_index.getSpaceIndexEx ().length == 0)
			return getName ().hashCode ();
		return m_index.hashCode ();
	}

	@Override
	public String toString ()
	{
		return StringUtil.concat (getName (), m_index.toString ());
	}
	
	public String toExpandedString ()
	{
		StringBuilder sb = new StringBuilder (getName ());
		sb.append ("___");
		
		for (Expression exprIdx : getIndex ().getSpaceIndexEx ())
		{
			if (exprIdx instanceof IntegerLiteral)
				sb.append (StringUtil.num2IdStr (((IntegerLiteral) exprIdx).getValue ()));
			else
				sb.append (exprIdx.toString ());
			
			sb.append ('_');
		}
		
		sb.append ("__");
		sb.append (StringUtil.num2IdStr (getIndex ().getTimeIndex ()));
		sb.append ("___");
		sb.append (StringUtil.num2IdStr (getIndex ().getVectorIndex ()));
		
		return sb.toString ();
	}


	///////////////////////////////////////////////////////////////////
	// Comparable Implementation

	@Override
	public int compareTo (Expression expr)
	{
		if (expr instanceof StencilNode)
			return m_index.compareTo (((StencilNode) expr).getIndex ());
		return super.compareTo (expr);
	}
}
