/*******************************************************************************
 * Copyright (c) 2011 Matthias-M. Christen, University of Basel, Switzerland.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the GNU Lesser Public License v2.1
 * which accompanies this distribution, and is available at
 * http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
 * 
 * Contributors:
 *     Matthias-M. Christen, University of Basel, Switzerland - initial API and implementation
 ******************************************************************************/
package ch.unibas.cs.hpwc.patus.codegen;

import java.io.ByteArrayInputStream;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import cetus.hir.CompoundStatement;
import cetus.hir.Declaration;
import cetus.hir.NameID;
import cetus.hir.VariableDeclaration;
import cetus.hir.VariableDeclarator;
import ch.unibas.cs.hpwc.patus.ast.SubdomainIdentifier;
import ch.unibas.cs.hpwc.patus.geometry.Subdomain;
import ch.unibas.cs.hpwc.patus.grammar.strategy.IAutotunerParam;
import ch.unibas.cs.hpwc.patus.grammar.strategy.Parser;
import ch.unibas.cs.hpwc.patus.grammar.strategy.Scanner;
import ch.unibas.cs.hpwc.patus.representation.StencilCalculation;

/**
 *
 * @author Matthias-M. Christen
 */
public class Strategy
{
	///////////////////////////////////////////////////////////////////
	// Static Members

	/**
	 * Loads and parses the strategy from file <code>strFilename</code>.
	 * @param strFilename The strategy file to load
	 * @return The strategy object described in <code>strFilename</code>
	 */
	public static Strategy load (String strFilename, StencilCalculation stencilCalculation)
	{
		Parser parser = new Parser (new Scanner (strFilename));
		parser.setStencilCalculation (stencilCalculation);
		parser.Parse ();
		if (parser.hasErrors ())
			throw new RuntimeException ("Parsing the Strategy failed.");

		Strategy strategy = parser.getStrategy ();
		strategy.setFilename (strFilename);
		return strategy;
	}
	
	public static Strategy parse (String strategyCode, StencilCalculation stencilCalculation)
	{
		Parser parser = new Parser (new Scanner (new ByteArrayInputStream (strategyCode.getBytes())));
		parser.setStencilCalculation (stencilCalculation);
		parser.Parse ();
		if (parser.hasErrors ())
			throw new RuntimeException ("Parsing the Strategy failed.");
		
		return parser.getStrategy ();
	}


	///////////////////////////////////////////////////////////////////
	// Member Variables

	private String m_strFilename;
	
	/**
	 * List of parameters to the strategy
	 */
	private List<Declaration> m_listParameters;
	
	private Map<String, IAutotunerParam> m_mapAutotuneParams;

	/**
	 * Maps subdomain identifiers to subdomain objects
	 */
	private Map<String, Subdomain> m_mapSubdomains;

	/**
	 * The base domain: the problem domain (or at least the domain on which
	 * the stencil operates)
	 */
	private SubdomainIdentifier m_sdidBaseDomain;

	/**
	 * The strategy body
	 */
	private CompoundStatement m_cmpstmtBody;


	///////////////////////////////////////////////////////////////////
	// Implementation

	public Strategy ()
	{
		m_strFilename = "";
		m_mapAutotuneParams = new HashMap<> ();
	}

	public void setFilename (String strFilename)
	{
		m_strFilename = strFilename;
	}
	
	public String getFilename ()
	{
		return m_strFilename;
	}

	/**
	 *
	 * @param listParameters
	 */
	public void setParameters (List<Declaration> listParameters)
	{
		m_listParameters = listParameters;
	}

	/**
	 *
	 * @return
	 */
	public List<Declaration> getParameters ()
	{
		return m_listParameters;
	}

	/**
	 * Determines whether the symbol declared by <code>declarator</code> is a strategy parameter.
	 * @param declarator The variable declarator to check
	 * @return
	 */
	public boolean isParameter (VariableDeclarator declarator)
	{
		for (Declaration d : m_listParameters)
		{
			if (d instanceof VariableDeclaration &&
				((VariableDeclarator) ((VariableDeclaration) d).getDeclarator (0)).getSymbolName ().equals (declarator.getSymbolName ()))
			{
				// the symbol declared by declarator has been found in the strategy's parameter list
				return true;
			}
		}

		return false;
	}

	public boolean isParameter (NameID nid)
	{
		for (Declaration d : m_listParameters)
		{
			if (d instanceof VariableDeclaration &&
				((VariableDeclarator) ((VariableDeclaration) d).getDeclarator (0)).getSymbolName ().equals (nid.getName ()))
			{
				// the symbol declared by declarator has been found in the strategy's parameter list
				return true;
			}
		}

		return false;
	}
	
	/**
	 * Sets the range specification for the parameter <code>strParamName</code>,
	 * which will be used by the auto-tuner.
	 * 
	 * @param strParamName
	 *            The name of the parameter for which to set the auto-tuner
	 *            value range
	 * @param value
	 *            The value range descriptor
	 */
	public void setAutotuneSpecification (String strParamName, IAutotunerParam value)
	{
		m_mapAutotuneParams.put (strParamName, value);
	}
	
	/**
	 * Returns the range specification for the parameter
	 * <code>strParamName</code>, which will be used by the auto-tuner.
	 * 
	 * @param strParamName
	 *            The name of the parameter for which to retrieve its auto-tuner
	 *            value range
	 * @return The value range for the parameter <code>strParamName</code> or
	 *         <code>null</code> if the parameter doesn't exist or no
	 *         specification has been provided
	 */
	public IAutotunerParam getAutotuneSpecification (String strParamName)
	{
		return m_mapAutotuneParams.get (strParamName);
	}

	/**
	 *
	 * @param cmpstmtBody
	 */
	public void setBody (CompoundStatement cmpstmtBody)
	{
		m_cmpstmtBody = cmpstmtBody;
	}

	/**
	 *
	 * @return
	 */
	public CompoundStatement getBody ()
	{
		return m_cmpstmtBody;
	}

	/**
	 * Sets the base domain, i.e., the grid on which the stencil operation
	 * is performed.<br/>
	 * Used in the parser. Don't use otherwise.
	 * @param sgBaseGrid The base grid
	 */
	public void setBaseDomain (SubdomainIdentifier sdidBaseDomain)
	{
		m_sdidBaseDomain = sdidBaseDomain;
	}

	/**
	 * Returns the base grid, i.e., the grid on which the stencil operates.
	 * @return The base grid
	 */
	public SubdomainIdentifier getBaseDomain ()
	{
		return m_sdidBaseDomain;
	}

	/**
	 * Sets the subdomain map.<br/>
	 * Used in the parser. Don't use otherwise.
	 * @param mapSubdomains The subdomain map
	 */
	public void setSubdomains (Map<String, Subdomain> mapSubdomains)
	{
		m_mapSubdomains = mapSubdomains;
	}

	/**
	 * Returns the subdomain corresponding to the subdomain's identifier <code>strIdentifier</code>.
	 * @param strIdentifier The subdomain identifier
	 * @return The subdomain object
	 */
	public Subdomain getSubdomain (String strIdentifier)
	{
		return m_mapSubdomains.get (strIdentifier);
	}

	@Override
	public String toString ()
	{
		return m_cmpstmtBody.toString ();
	}
}
