package cetus.hir;

import java.util.*;
import cetus.analysis.*;

/**
 * CetusAnnotation is used for internal annotations inserted by Cetus analysis and
 * transformation. Parallelization passes usually insert this type of annotations.
 */
public class CetusAnnotation extends PragmaAnnotation
{
	/**
	 * Constructs an empty cetus annotation.
	 */
	public CetusAnnotation()
	{
		super();
	}

	/**
	 * Constructs a cetus annotation with the given key-value pair.
	 */
	public CetusAnnotation(String key, Object value)
	{
		super();
		put(key, value);
	}

	/**
	 * Returns a string representation of this cetus annotation.
	 * @return a string representation.
	 */
	public String toString()
	{
		if ( skip_print )
			return "";

		StringBuilder str = new StringBuilder(80);
		str.append(super.toString()+"cetus ");

		if ( containsKey("parallel") )
			str.append("parallel ");

		for ( String key : keySet() )
		{
			if ( key.equals("parallel") )
				;
			else if ( key.equals("lastprivate") || key.equals("private") )
			{
				Set<Symbol> private_set = this.get(key);
        str.append(key);
        str.append("(");
				str.append(PrintTools.collectionToString(private_set, ", "));
        str.append(") ");
			}
			else if ( key.equals("reduction") )
			{
				Map<String, Set<Expression>> reduction_map = this.get(key);
				for ( String op : reduction_map.keySet() )
				{
					str.append("reduction("+op+": ");
					str.append(PrintTools.collectionToString(reduction_map.get(op),", "));
					str.append(") ");
				}
			}
			else if ( key.equals("use") || key.equals("def") )
			{
				str.append(key + " (");
				str.append(
            PrintTools.collectionToString((Set<Symbol>)this.get(key), ", "));
				str.append(") ");
			}
			else
				str.append(key+" "+get(key)+" ");

		}
		return str.toString();
	}

}
