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
import ch.unibas.cs.hpwc.patus.codegen.IInnermostLoopCodeGenerator;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.x86_64.X86_64InnermostLoopCodeGenerator;
import ch.unibas.cs.hpwc.patus.codegen.backend.cuda.CUDA1DCodeGenerator;
import ch.unibas.cs.hpwc.patus.codegen.backend.cuda.CUDA4CodeGenerator;
import ch.unibas.cs.hpwc.patus.codegen.backend.cuda.CUDACodeGenerator;
import ch.unibas.cs.hpwc.patus.codegen.backend.intel.IntelXeonCodeGenerator;
import ch.unibas.cs.hpwc.patus.codegen.backend.mic.OpenMPMICCodeGenerator;
import ch.unibas.cs.hpwc.patus.codegen.backend.openmp.OpenMPAVXAsmCodeGenerator;
import ch.unibas.cs.hpwc.patus.codegen.backend.openmp.OpenMPAVXCodeGenerator;
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
		if ("OpenMP_AVX".equals (strBackend))
			return new OpenMPAVXCodeGenerator (data);
		if ("OpenMP_AVX_Asm".equals (strBackend))
			return new OpenMPAVXAsmCodeGenerator (data);
		if ("OpenMP_MIC".equals (strBackend))
			return new OpenMPMICCodeGenerator (data);
		if ("INTELXEONPHI".equals(strBackend))
			return new IntelXeonCodeGenerator(data);
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
	
	@SuppressWarnings ("unchecked")
	public static IInnermostLoopCodeGenerator createInnermostLoopCodeGenerator (String strBackend, CodeGeneratorSharedObjects data)
	{
		if (strBackend == null || "".equals (strBackend))
			return null;
		
		IInnermostLoopCodeGenerator cg = null;

		// try to instantiate by name
		if ("x86_64".equals (strBackend))
			cg = new X86_64InnermostLoopCodeGenerator (data);

				
		// interpret the string as class name; try to instantiate it
		if (cg == null)
		{
			try
			{
				// try to get the class
				Class<? extends IInnermostLoopCodeGenerator> clsBackendCG = (Class<? extends IInnermostLoopCodeGenerator>) Class.forName (strBackend);
	
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
					cg = clsBackendCG.newInstance ();
				}
				catch (InstantiationException e)
				{
				}
				catch (IllegalAccessException e)
				{
				}
	
				throw new RuntimeException (StringUtil.concat ("Could not instantiate assembly backend '", strBackend, "'."));
			}
			catch (ClassNotFoundException e)
			{
				throw new RuntimeException (StringUtil.concat ("The assembly backend '", strBackend, "' could not be found."));
			}
		}
		
		// check whether the code generator requires an assembly specification
		if (cg.requiresAssemblySection () && data.getArchitectureDescription ().getAssemblySpec () == null)
			return null;
		
		return cg;
	}
}
