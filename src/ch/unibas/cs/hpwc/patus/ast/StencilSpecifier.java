package ch.unibas.cs.hpwc.patus.ast;

import cetus.hir.NameID;
import cetus.hir.Specifier;
import cetus.hir.UserSpecifier;

/**
 *
 * @author Matthias-M. Christen
 */
public class StencilSpecifier
{
	///////////////////////////////////////////////////////////////////
	// Constants

	/**
	 * The "grid" specifier.
	 */
	public static final Specifier STENCIL_GRID = new UserSpecifier (new NameID ("grid"));

	/**
	 * The "param" specifier denoting that a variable declared in the stencil "operation" method head is
	 * a parameter, i.e. a scalar value without spatial and temporal subscripts.
	 * Parameters might have one or more extra subscripts, though.
	 */
	public static final Specifier STENCIL_PARAM = new UserSpecifier (new NameID ("param"));



	/**
	 * Strategy parameter that is subject to be determined by the autotuner.
	 */
	public static final Specifier STRATEGY_AUTO = new UserSpecifier (new NameID ("auto"));

	/**
	 * Strategy dimension parameter.
	 */
	public static final Specifier STRATEGY_DIM = new UserSpecifier (new NameID ("dim"));
}
