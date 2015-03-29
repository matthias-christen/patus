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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import cetus.hir.ArraySpecifier;
import cetus.hir.Declaration;
import cetus.hir.Identifier;
import cetus.hir.NameID;
import cetus.hir.Specifier;
import cetus.hir.VariableDeclaration;
import cetus.hir.VariableDeclarator;
import ch.unibas.cs.hpwc.patus.ast.SubdomainIdentifier;
import ch.unibas.cs.hpwc.patus.geometry.Point;
import ch.unibas.cs.hpwc.patus.util.CodeGeneratorUtil;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

public class SubdomainGeneratedIdentifiers
{
	///////////////////////////////////////////////////////////////////
	// Inner Types

	private class Identifiers
	{
		private SubdomainIdentifier m_sdid;

		private Identifier m_idIndexIdentifier;
		private Identifier[] m_rgDimensionIndexIdentifiers;
		private Identifier[] m_rgDimensionMinIdentifiers;
		private Identifier[] m_rgDimensionMaxIdentifiers;
		private Identifier[] m_rgDimensionBlockIndexIdentifiers;
		private Identifier m_idLoopCounter;
		private Identifier m_idNumBlocks;
		private Identifier m_idTimeIndexIdentifier;

		private Map<Integer, Identifier> m_mapMemoryObjectIndices;
		private Map<Integer, Identifier> m_mapMemoryObjectStartIndices;
		private Map<Integer, Identifier> m_mapMemoryObjectCounts;


		/**
		 *
		 * @param sdid
		 */
		public Identifiers (SubdomainIdentifier sdid)
		{
			m_sdid = sdid;

			m_idIndexIdentifier = null;
			m_rgDimensionIndexIdentifiers = null;
			m_rgDimensionMinIdentifiers = null;
			m_rgDimensionMaxIdentifiers = null;
			m_rgDimensionBlockIndexIdentifiers = null;
			m_idLoopCounter = null;
			m_idNumBlocks = null;
			m_idTimeIndexIdentifier = null;

			m_mapMemoryObjectIndices = new HashMap<> ();
			m_mapMemoryObjectStartIndices = new HashMap<> ();
			m_mapMemoryObjectCounts = new HashMap<> ();
		}

		/**
		 *
		 * @param strNameSuffix
		 * @return
		 */
		private Identifier createIdentifier (String strNameSuffix)
		{
//			VariableDeclarator decl = new VariableDeclarator (new NameID (StringUtil.concat (m_sdid.getName (), strNameSuffix)));
//			Identifier id = new Identifier (decl);
//			m_data.addDeclaration (new VariableDeclaration (Globals.SPECIFIER_INDEX, decl));
//			return id;

			return createIdentifier (strNameSuffix, 0, false);
		}

		/**
		 *
		 * @param strNameSuffix
		 * @param nDimensions
		 * @param bGlobal
		 * @return
		 */
		private Identifier createIdentifier (String strNameSuffix, int nDimensions, boolean bGlobal)
		{
			List<Specifier> listArraySpecifiers = new ArrayList<> (nDimensions);
			for (int i = 0; i < nDimensions; i++)
				listArraySpecifiers.add (ArraySpecifier.UNBOUNDED);

			VariableDeclarator decl = new VariableDeclarator (
				new NameID (StringUtil.concat (m_sdid.getName (), strNameSuffix)),
				listArraySpecifiers);

			Identifier id = new Identifier (decl);
			Declaration declaration = new VariableDeclaration (Globals.SPECIFIER_INDEX, decl);

			if (bGlobal)
				m_data.getData ().addGlobalDeclaration (declaration);
			else
				m_data.getData ().addDeclaration (declaration);

			return id;
		}

		/**
		 *
		 * @return
		 */
		public Point getIndexPoint ()
		{
			return new Point (getDimensionIndexIdentifiers ());
		}

		/**
		 *
		 * @return
		 */
		public Identifier getIndexIdentifier ()
		{
			if (m_idIndexIdentifier == null)
				m_idIndexIdentifier = createIdentifier ("_idx");
			return m_idIndexIdentifier;
		}

		/**
		 *
		 * @return
		 */
		public Identifier[] getDimensionIndexIdentifiers ()
		{
			if (m_rgDimensionIndexIdentifiers == null)
			{
				m_rgDimensionIndexIdentifiers = new Identifier[m_sdid.getDimensionality ()];
				for (int i = 0; i < m_sdid.getDimensionality (); i++)
				{
					m_rgDimensionIndexIdentifiers[i] = createIdentifier (StringUtil.concat (
						"_idx_", CodeGeneratorUtil.getDimensionName (i)));
				}
			}

			return m_rgDimensionIndexIdentifiers;
		}

		/**
		 *
		 * @param nDimension
		 * @return
		 */
		public Identifier getDimensionIndexIdentifier (int nDimension)
		{
			return getDimensionIndexIdentifiers ()[nDimension];
		}

		/**
		 *
		 * @return
		 */
		public Identifier[] getDimensionMinIdentifiers ()
		{
			if (m_rgDimensionMinIdentifiers == null)
			{
				m_rgDimensionMinIdentifiers = new Identifier[m_sdid.getDimensionality ()];
				for (int i = 0; i < m_sdid.getDimensionality (); i++)
				{
					m_rgDimensionMinIdentifiers[i] = createIdentifier (StringUtil.concat (
						"_idx_", CodeGeneratorUtil.getDimensionName (i), "_min"));
				}
			}

			return m_rgDimensionMinIdentifiers;
		}

		/**
		 *
		 * @param nDimension
		 * @return
		 */
		public Identifier getDimensionMinIdentifier (int nDimension)
		{
			return getDimensionMinIdentifiers ()[nDimension];
		}
		
		/**
		 *
		 * @return
		 */
		public Identifier[] getDimensionMaxIdentifiers ()
		{
			if (m_rgDimensionMaxIdentifiers == null)
			{
				m_rgDimensionMaxIdentifiers = new Identifier[m_sdid.getDimensionality ()];
				for (int i = 0; i < m_sdid.getDimensionality (); i++)
				{
					m_rgDimensionMaxIdentifiers[i] = createIdentifier (StringUtil.concat (
						"_idx_", CodeGeneratorUtil.getDimensionName (i), "_max"));
				}
			}

			return m_rgDimensionMaxIdentifiers;
		}

		/**
		 *
		 * @param nDimension
		 * @return
		 */
		public Identifier getDimensionMaxIdentifier (int nDimension)
		{
			return getDimensionMaxIdentifiers ()[nDimension];
		}

		/**
		 *
		 * @return
		 */
		public Identifier[] getDimensionBlockIndexIdentifiers ()
		{
			if (m_rgDimensionBlockIndexIdentifiers == null)
			{
				m_rgDimensionBlockIndexIdentifiers = new Identifier[m_sdid.getDimensionality ()];
				for (int i = 0; i < m_sdid.getDimensionality (); i++)
				{
					m_rgDimensionBlockIndexIdentifiers[i] = createIdentifier (StringUtil.concat (
						"_blkidx_", CodeGeneratorUtil.getDimensionName (i)));
				}
			}

			return m_rgDimensionBlockIndexIdentifiers;
		}

		public Identifier getDimensionBlockIndexIdentifier (int nDimension)
		{
			return getDimensionBlockIndexIdentifiers ()[nDimension];
		}

		/**
		 *
		 * @return
		 */
		public Identifier getLoopCounterIdentifier ()
		{
			if (m_idLoopCounter == null)
				m_idLoopCounter = createIdentifier ("_counter");
			return m_idLoopCounter;
		}

		/**
		 *
		 * @return
		 */
		public Identifier getNumBlocksIdentifier ()
		{
			if (m_idNumBlocks == null)
				m_idNumBlocks = createIdentifier ("_numblocks");
			return m_idNumBlocks;
		}

		/**
		 *
		 * @param nVectorIndex
		 * @return
		 */
		public Identifier getMemoryObjectIndexIdentifier (int nVectorIndex)
		{
			Identifier id = m_mapMemoryObjectIndices.get (nVectorIndex);
			if (id == null)
			{
				m_mapMemoryObjectIndices.put (
					nVectorIndex,
					id = createIdentifier (StringUtil.concat ("_MEMORYOBJECT_INDEX_", String.valueOf (nVectorIndex)), 1, true));
			}

			return id;
		}

		/**
		 *
		 * @param nVectorIndex
		 * @return
		 */
		public Identifier getMemoryObjectStartIndexIdentifier (int nVectorIndex)
		{
			Identifier id = m_mapMemoryObjectStartIndices.get (nVectorIndex);
			if (id == null)
			{
				m_mapMemoryObjectStartIndices.put (
					nVectorIndex,
					id = createIdentifier (StringUtil.concat ("_MEMORYOBJECT_STARTINDEX_", String.valueOf (nVectorIndex)), 1, true));
			}

			return id;
		}

		/**
		 *
		 * @param nVectorIndex
		 * @return
		 */
		public Identifier getMemoryObjectCountIdentifier (int nVectorIndex)
		{
			Identifier id = m_mapMemoryObjectCounts.get (nVectorIndex);
			if (id == null)
			{
				m_mapMemoryObjectCounts.put (
					nVectorIndex,
					id = createIdentifier (StringUtil.concat ("_MEMORYOBJECT_COUNT_", String.valueOf (nVectorIndex)), 1, true));
			}

			return id;
		}

		public Identifier getTimeIndexIdentifier ()
		{
			if (m_idTimeIndexIdentifier == null)
				m_idTimeIndexIdentifier = createIdentifier ("_timesteps");
			return m_idTimeIndexIdentifier;
		}

		public void reset ()
		{
			m_idIndexIdentifier = null;
			m_idLoopCounter = null;
			m_idNumBlocks = null;
			m_rgDimensionIndexIdentifiers = null;
			m_rgDimensionMinIdentifiers = null;
			m_rgDimensionMaxIdentifiers = null;
			m_rgDimensionBlockIndexIdentifiers = null;
		}
	}


	///////////////////////////////////////////////////////////////////
	// Member Variables

	/**
	 * The shared data object
	 */
	private CodeGeneratorSharedObjects m_data;

	/**
	 * The map from the subdomain identifier to the identifiers object
	 */
	private Map<SubdomainIdentifier, Identifiers> m_mapIdentifiers;


	///////////////////////////////////////////////////////////////////
	// Implementation

	public SubdomainGeneratedIdentifiers (CodeGeneratorSharedObjects data)
	{
		m_data = data;
		m_mapIdentifiers = new HashMap<> ();
	}

	private Identifiers getIdentifiers (SubdomainIdentifier sdid)
	{
		Identifiers ids = m_mapIdentifiers.get (sdid);
		if (ids == null)
			m_mapIdentifiers.put (sdid, ids = new Identifiers (sdid));
		return ids;
	}

	/**
	 *
	 * @param sdid
	 * @return
	 */
	public Point getIndexPoint (SubdomainIdentifier sdid)
	{
		return getIdentifiers (sdid).getIndexPoint ();
	}

	/**
	 * Returns an identifier that will be used as the loop index looping over the blocks of the loop created from the
	 * loop with iterator <code>sdid</code>.
	 * @param sdid
	 * @return
	 */
	public Identifier getIndexIdentifier (SubdomainIdentifier sdid)
	{
		return getIdentifiers (sdid).getIndexIdentifier ();
	}

	/**
	 *
	 * @param sdid
	 * @param nDimension
	 * @return
	 */
	public Identifier getDimensionIndexIdentifier (SubdomainIdentifier sdid, int nDimension)
	{
		return getIdentifiers (sdid).getDimensionIndexIdentifier (nDimension);
	}

	/**
	 *
	 * @param sdid
	 * @param nDimension
	 * @return
	 */
	public Identifier getDimensionMinIdentifier (SubdomainIdentifier sdid, int nDimension)
	{
		return getIdentifiers (sdid).getDimensionMinIdentifier (nDimension);
	}

	/**
	 *
	 * @param sdid
	 * @param nDimension
	 * @return
	 */
	public Identifier getDimensionMaxIdentifier (SubdomainIdentifier sdid, int nDimension)
	{
		return getIdentifiers (sdid).getDimensionMaxIdentifier (nDimension);
	}

	public Identifier getDimensionBlockIndexIdentifier (SubdomainIdentifier sdid, int nDimension)
	{
		return getIdentifiers (sdid).getDimensionBlockIndexIdentifier (nDimension);
	}

	/**
	 *
	 * @param sdid
	 * @return
	 */
	public Identifier getLoopCounterIdentifier (SubdomainIdentifier sdid)
	{
		return getIdentifiers (sdid).getLoopCounterIdentifier ();
	}

	/**
	 *
	 * @param sdid
	 * @return
	 */
	public Identifier getNumBlocksIdentifier (SubdomainIdentifier sdid)
	{
		return getIdentifiers (sdid).getNumBlocksIdentifier ();
	}

	/**
	 *
	 * @param sdid
	 * @param nVectorIndex
	 * @return
	 */
	public Identifier getMemoryObjectIndexIdentifier (SubdomainIdentifier sdid, int nVectorIndex)
	{
		return getIdentifiers (sdid).getMemoryObjectIndexIdentifier (nVectorIndex);
	}

	/**
	 *
	 * @param sdid
	 * @param nVectorIndex
	 * @return
	 */
	public Identifier getMemoryObjectStartIndexIdentifier (SubdomainIdentifier sdid, int nVectorIndex)
	{
		return getIdentifiers (sdid).getMemoryObjectStartIndexIdentifier (nVectorIndex);
	}

	/**
	 *
	 * @param sdid
	 * @param nVectorIndex
	 * @return
	 */
	public Identifier getMemoryObjectCountIdentifier (SubdomainIdentifier sdid, int nVectorIndex)
	{
		return getIdentifiers (sdid).getMemoryObjectCountIdentifier (nVectorIndex);
	}

	/**
	 *
	 * @param sdid
	 * @return
	 */
	public Identifier getTimeIndexIdentifier (SubdomainIdentifier sdid)
	{
		return getIdentifiers (sdid).getTimeIndexIdentifier ();
	}

	/**
	 * Resets the generated identifiers map.
	 */
	public void reset ()
	{
		for (Identifiers ids : m_mapIdentifiers.values ())
			ids.reset ();
		m_mapIdentifiers.clear ();
	}
}
