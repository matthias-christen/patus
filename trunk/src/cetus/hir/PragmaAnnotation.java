package cetus.hir;

import java.util.*;

/**
 * PragmaAnnotation is used for annotations of pragma type.
 */
public class PragmaAnnotation extends Annotation
{
	/**
	 * Constructs an empty pragma annotation.
	 */
	public PragmaAnnotation()
	{
		super();
	}

	/**
	 * Constructs a simple pragma with raw string.
	 */
	public PragmaAnnotation(String pragma)
	{
		super();
		put("pragma", pragma);
	}

	/**
	 * Returns the name of this pragma annotation.
	 */
	public String getName()
	{
		return (String)get("pragma");
	}

	/**
	 * Checks if the specified keys all exist in the key set.
	 */
	public boolean containsKeys(Collection<String> keys)
	{
		for ( String key : keys )
			if ( !containsKey(key) )
				return false;
		return true;
	}

	/**
	 * Returns the string representation of this pragma annotation.
	 * @return the string.
	 */
	public String toString()
	{
		if ( skip_print )
			return "";
		String ret = "#pragma ";
		String pragma = get("pragma");
		if ( pragma != null )
			ret += pragma+" ";
		if ( this.getClass() == PragmaAnnotation.class ) // pure pragma printing
			for ( String key : keySet() )
				if ( !key.equals("pragma") )
					ret += key+" "+get(key)+" ";
		return ret;
	}
}
