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

import java.util.HashMap;
import java.util.Map;

import cetus.hir.IDExpression;
import cetus.hir.Identifier;
import cetus.hir.NameID;
import cetus.hir.Specifier;
import cetus.hir.VariableDeclaration;
import cetus.hir.VariableDeclarator;
import ch.unibas.cs.hpwc.patus.ast.StatementListBundle;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;
import ch.unibas.cs.hpwc.patus.util.IntArray;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 *
 * @author Matthias-M. Christen
 */
public class UnrollGeneratedIdentifiers
{
	///////////////////////////////////////////////////////////////////
	// Member Variables

	private CodeGeneratorSharedObjects m_objects;
	
	/**
	 * The map of identifiers duplicated during the vectorization.<br/>
	 * Maps <code>IsVectorized</code> -&gt; <code>Identifier</code> -&gt; <code>Offset</code> -&gt; <code>Expression</code>
	 */
	private Map<Boolean, Map<String, Map<IntArray, IDExpression>>> m_mapIdentifiers;


	///////////////////////////////////////////////////////////////////
	// Implementation

	public UnrollGeneratedIdentifiers (CodeGeneratorSharedObjects objects)
	{
		m_objects = objects;
		m_mapIdentifiers = new HashMap<> ();
	}


	public IDExpression createIdentifier (IDExpression id, int[] rgOffset, Specifier specDatatype, StatementListBundle slGenerated, CodeGeneratorRuntimeOptions options)
	{
		// determine whether to create special vector identifiers:
		// if vectorization is turned on and no native vector datatypes are used and vectorization is enabled for the current code generation phase
		boolean bNoVectorize = options.getBooleanValue (CodeGeneratorRuntimeOptions.OPTION_NOVECTORIZE, false);
		boolean bUseNativeSIMD = m_objects.getOptions ().useNativeSIMDDatatypes ();
		boolean bCreateVectorizedIdentifier = m_objects.getArchitectureDescription ().useSIMD () && !bUseNativeSIMD && !bNoVectorize;

		Map<String, Map<IntArray, IDExpression>> mapIdentifiers = m_mapIdentifiers.get (bCreateVectorizedIdentifier);
		if (mapIdentifiers == null)
			m_mapIdentifiers.put (bCreateVectorizedIdentifier, mapIdentifiers = new HashMap<> ());

		Map<IntArray, IDExpression> map = mapIdentifiers.get (id.getName ());
		if (map == null)
		{
			mapIdentifiers.put (id.getName (), map = new HashMap<> ());

			if (!bCreateVectorizedIdentifier)
			{
				// no entries in the map yet for id
				// create the map entry, but don't duplicate the identifier
				map.put (new IntArray (rgOffset, true), id.clone ());
				return id;
			}
		}

		// there already are substitute identifiers for id; add a new substitute
		IDExpression idNew = map.get (new IntArray (rgOffset));
		if (idNew == null)
		{
			VariableDeclarator decl = new VariableDeclarator (
				new NameID (StringUtil.concat (id.getName (), bCreateVectorizedIdentifier ? "_vec_" : "_unroll_", map.size ())));

			m_objects.getData ().addDeclaration (new VariableDeclaration (specDatatype, decl));
			map.put (new IntArray (rgOffset, true), idNew = new Identifier (decl));
		}
		else
			idNew = idNew.clone ();

		return idNew;
	}

	public void reset ()
	{
		m_mapIdentifiers.clear ();
	}
}
