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
package ch.unibas.cs.hpwc.patus.codegen.backend;

import java.lang.reflect.InvocationTargetException;

import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;
import ch.unibas.cs.hpwc.patus.codegen.backend.cuda.CUDA1DCodeGenerator;
import ch.unibas.cs.hpwc.patus.codegen.backend.cuda.CUDA4CodeGenerator;
import ch.unibas.cs.hpwc.patus.codegen.backend.cuda.CUDACodeGenerator;
import ch.unibas.cs.hpwc.patus.codegen.backend.openmp.OpenMPCodeGenerator;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 *
 * @author Matthias-M. Christen
 */
public class BackendFactory
{
	@SuppressWarnings("unchecked")
	public static IBackend create (String strBackend, CodeGeneratorSharedObjects data)
	{
		if (strBackend == null)
			throw new RuntimeException ("No backend provided in the configuration.");

		// try to instantiate by name
		if ("OpenMP".equals (strBackend))
			return new OpenMPCodeGenerator (data);
		if ("CUDA".equals (strBackend))
			return new CUDACodeGenerator (data);
		if ("CUDA4".equals (strBackend))
			return new CUDA4CodeGenerator (data);
		if ("CUDA1D".equals (strBackend))
			return new CUDA1DCodeGenerator (data);

		// interpret the string as class name; try to instantiate it
		try
		{
			// try to get the class
			Class<? extends IBackend> clsBackendCG = (Class<? extends IBackend>) Class.forName (strBackend);

			// find a constructor
			try
			{
				return clsBackendCG.getConstructor (CodeGeneratorSharedObjects.class).newInstance (data);
			}
			catch (SecurityException e)
			{
			}
			catch (NoSuchMethodException e)
			{
			}
			catch (IllegalArgumentException e)
			{
			}
			catch (InstantiationException e)
			{
			}
			catch (IllegalAccessException e)
			{
			}
			catch (InvocationTargetException e)
			{
			}

			// try the default constructor
			try
			{
				return clsBackendCG.newInstance ();
			}
			catch (InstantiationException e)
			{
			}
			catch (IllegalAccessException e)
			{
			}

			throw new RuntimeException (StringUtil.concat ("Could not instantiate backend '", strBackend, "'."));
		}
		catch (ClassNotFoundException e)
		{
			throw new RuntimeException (StringUtil.concat ("The backend '", strBackend, "' could not be found."));
		}
	}
}
