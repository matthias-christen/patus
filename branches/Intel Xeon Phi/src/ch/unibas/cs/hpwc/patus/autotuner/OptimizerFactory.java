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
package ch.unibas.cs.hpwc.patus.autotuner;

import java.util.List;

import ch.unibas.cs.hpwc.patus.config.AbstractConfigurable;
import ch.unibas.cs.hpwc.patus.config.ConfigurationProperty;


/**
 *
 * @author Matthias-M. Christen
 */
public class OptimizerFactory extends AbstractConfigurable
{
	///////////////////////////////////////////////////////////////////
	// Constants

	/**
	 * The default optimizer
	 */
	private static final Class<? extends IOptimizer> CLS_DEFAULT_OPTIMIZER =
		MetaHeuristicOptimizer.class;
		//SimplexSearchOptimizer.class;
		//ExhaustiveSearchOptimizer.class;
		//GreedyOptimizer.class;

	/**
	 * Property: which optimizer to use
	 */
	private static final ConfigurationProperty PROP_OPTIMIZER = new ConfigurationProperty (
		"Autotuner", "Optimizer", ConfigurationProperty.EPropertyType.LIST, CLS_DEFAULT_OPTIMIZER.getName (),
		SimplexSearchOptimizer.class.getName (),
		MetaHeuristicOptimizer.class.getName (),
		GeneralCombinedEliminationOptimizer.class.getName (),
		RandomSearchOptimizer.class.getName (),
		GreedyOptimizer.class.getName (),
		HookeJeevesOptimizer.class.getName (),
		DiRectOptimizer.class.getName (),
		ExhaustiveSearchOptimizer.class.getName ());


	///////////////////////////////////////////////////////////////////
	// Factory Pattern

	private static OptimizerFactory THIS = new OptimizerFactory ();

	public final static IOptimizer getOptimizer (String strOptimizerKey)
	{
		if (strOptimizerKey == null || "".equals (strOptimizerKey))
			return OptimizerFactory.getOptimizer ();
		return THIS.getOptimizerByKey (strOptimizerKey);
	}

	/**
	 * Returns the optimizer to use (according to the configuration).
	 * @return The optimizer
	 */
	public final static IOptimizer getOptimizer ()
	{
		return THIS.getCurrentOptimizer ();
	}

	public final static String[] getOptimizerKeys ()
	{
		return OptimizerFactory.getRegisteredOptimizerKeys ();
	}


	///////////////////////////////////////////////////////////////////
	// Implementation

	/**
	 * Creates a factory instance.
	 */
	private OptimizerFactory ()
	{
		super ();

		// add the configuration properties
		addConfigurationProperty (PROP_OPTIMIZER);
	}

	/**
	 * Creates an optimizer object and returns it.
	 * @return A newly created optimizer object
	 */
	private IOptimizer getCurrentOptimizer ()
	{
		return getOptimizerByKey (PROP_OPTIMIZER.getValue ());
	}

	private IOptimizer getOptimizerByKey (String strOptKey)
	{
		// register optimizers (if needed)
		ensureIsRegistered ();

		// create a new instance of the optimizer
		try
		{
			return (IOptimizer) Class.forName (strOptKey).newInstance ();
		}
		catch (Exception e)
		{
			// if something goes wrong, instantiate the default optimizer
			try
			{
				return CLS_DEFAULT_OPTIMIZER.newInstance ();
			}
			catch (InstantiationException e1)
			{
				e1.printStackTrace ();
			}
			catch (IllegalAccessException e1)
			{
				e1.printStackTrace ();
			}
		}

		return null;
	}

	private static String[] getRegisteredOptimizerKeys ()
	{
		List<?> listOptimizers = PROP_OPTIMIZER.getValues ();
		String[] rgOptimizers = new String[listOptimizers.size ()];
		int i = 0;
		for (Object objOpt : listOptimizers)
			rgOptimizers[i++] = (String) objOpt;
		return rgOptimizers;
	}
}
