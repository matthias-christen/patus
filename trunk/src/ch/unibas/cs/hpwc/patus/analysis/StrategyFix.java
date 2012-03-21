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
package ch.unibas.cs.hpwc.patus.analysis;

import java.util.ArrayList;
import java.util.List;

import cetus.hir.DepthFirstIterator;
import cetus.hir.Expression;
import cetus.hir.NameID;
import cetus.hir.VariableDeclaration;
import cetus.hir.VariableDeclarator;
import ch.unibas.cs.hpwc.patus.arch.IArchitectureDescription;
import ch.unibas.cs.hpwc.patus.ast.RangeIterator;
import ch.unibas.cs.hpwc.patus.ast.StencilSpecifier;
import ch.unibas.cs.hpwc.patus.ast.SubdomainIdentifier;
import ch.unibas.cs.hpwc.patus.ast.SubdomainIterator;
import ch.unibas.cs.hpwc.patus.codegen.CodeGenerationOptions;
import ch.unibas.cs.hpwc.patus.codegen.Globals;
import ch.unibas.cs.hpwc.patus.codegen.Strategy;
import ch.unibas.cs.hpwc.patus.geometry.Border;
import ch.unibas.cs.hpwc.patus.geometry.Size;
import ch.unibas.cs.hpwc.patus.geometry.Subdomain;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

public class StrategyFix
{
	///////////////////////////////////////////////////////////////////
	// Member Variables

	///////////////////////////////////////////////////////////////////
	// Implementation


	/**
	 * If parallel point subdomain iterators are used and no-native
	 * SIMD types (and innermost iterators create prologue and epilogue
	 * loops to account for non-alignments), parallel point loops are
	 * split into {@link RangeIterator}s and therefore no prologue and
	 * epilogue loops are created.
	 * Insert an auxiliary subdomain iterator above the point loop so that
	 * the point loop is still a subdomain iterator:
	 *
	 * for point p in u(:; t) parallel ...
	 *     ...
	 *
	 *   --&gt;
	 *
	 * for subdomain p_line(u.max(1), 1 ...) in u(:; t) parallel ...
	 *     for point p in p_line(:; t)
	 *         ...
	 *
	 * @param strategy
	 */
	public static void fix (Strategy strategy, IArchitectureDescription arch, CodeGenerationOptions cgoptions)
	{
		if (!cgoptions.useNativeSIMDDatatypes () && arch.useSIMD ())
		{
			for (DepthFirstIterator it = new DepthFirstIterator (strategy.getBody ()); it.hasNext (); )
			{
				Object o = it.next ();
				if (o instanceof SubdomainIterator)
				{
					SubdomainIterator sgit = (SubdomainIterator) o;
					if (sgit.isParallel () && sgit.getIteratorSubdomain ().getBox ().isPoint ())
					{
						// create the subdomain and iterator subdomain identifier for the subdomain iterator in which the point iterator will be embedded
						Expression[] rgSize = new Expression[sgit.getDomainSubdomain ().getBox ().getDimensionality ()];
						rgSize[0] = sgit.getDomainSubdomain ().getSize ().getCoord (0);
						for (int i = 1; i < rgSize.length; i++)
							rgSize[i] = Globals.ONE;	// will be cloned in the constructor of Size

						Subdomain sgEmbed = new Subdomain (sgit.getDomainSubdomain (), Subdomain.ESubdomainType.SUBDOMAIN, sgit.getDomainSubdomain ().getLocalCoordinates (), new Size (rgSize));
						VariableDeclarator declEmbed = new VariableDeclarator (new NameID (StringUtil.concat (sgit.getIterator ().getName (), "_embed")));
						strategy.getBody ().addDeclaration (new VariableDeclaration (StencilSpecifier.STENCIL_GRID, declEmbed));
						SubdomainIdentifier sdidEmbed = new SubdomainIdentifier (declEmbed, sgEmbed);

						if (sgit.getIterator ().getSpatialOffset () != null)
						{
							Expression[] rgOffset = new Expression[sgit.getIterator ().getSpatialOffset ().length];
							for (int i = 0; i < rgOffset.length; i++)
								rgOffset[i] = sgit.getIterator ().getSpatialOffset ()[i].clone ();
							sdidEmbed.setSpatialOffset (rgOffset);
						}

						if (sgit.getIterator ().getTemporalIndex () != null)
							sdidEmbed.setTemporalIndex (sgit.getIterator ().getTemporalIndex ().clone ());

						if (sgit.getIterator ().getVectorIndices () != null)
						{
							List<Expression> listVectorIndices = new ArrayList<> (sgit.getIterator ().getVectorIndices ().size ());
							for (Expression exprVecIdx : sgit.getIterator ().getVectorIndices ())
								listVectorIndices.add (exprVecIdx.clone ());
							sdidEmbed.setVectorIndex (listVectorIndices);
						}

						// the new point subdomain iterator
						rgSize[0] = Globals.ONE;
						Subdomain sgPoint = new Subdomain (sgEmbed, Subdomain.ESubdomainType.POINT, new Size (rgSize));
						SubdomainIdentifier sdidPoint = new SubdomainIdentifier (sgit.getIterator ().getSymbol (), sgPoint);

						if (sgit.getIterator ().getTemporalIndex () != null)
							sdidPoint.setTemporalIndex (sgit.getIterator ().getTemporalIndex ().clone ());

						if (sgit.getIterator ().getVectorIndices () != null)
						{
							List<Expression> listVectorIndices = new ArrayList<> (sgit.getIterator ().getVectorIndices ().size ());
							for (Expression exprVecIdx : sgit.getIterator ().getVectorIndices ())
								listVectorIndices.add (exprVecIdx.clone ());
							sdidPoint.setVectorIndex (listVectorIndices);
						}

						SubdomainIterator sgitEmbedBody = new SubdomainIterator (
							sdidPoint,
							sdidEmbed,
							new Border (sgit.getDomainBorder ().getDimensionality ()),
							1,
							null,
							sgit.getLoopBody ().clone (),
							sgit.getParallelismLevel ());

						// create the new subdomain iterator in which the point iterator is embedded
						SubdomainIterator sgitEmbed = new SubdomainIterator (
							sdidEmbed,
							sgit.getDomainIdentifier (),
							sgit.getDomainBorder (),
							0,
							sgit.getChunkSizes (),
							sgitEmbedBody,
							sgit.getParallelismLevel ());
						sgitEmbed.setNumberOfThreads (sgit.getNumberOfThreads ().clone ());

						sgit.swapWith (sgitEmbed);
					}
				}
			}
		}
	}
}
