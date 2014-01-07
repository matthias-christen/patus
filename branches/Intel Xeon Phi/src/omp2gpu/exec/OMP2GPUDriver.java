package omp2gpu.exec;

import java.io.*;
import java.util.*;

import cetus.analysis.*;
import cetus.hir.*;
import cetus.transforms.*;
import cetus.codegen.*;
import cetus.exec.*;
import omp2gpu.analysis.CudaParser;
import omp2gpu.codegen.*;

/**
 * @author Seyong Lee <lee222@purdue.edu>
 *         ParaMount Group 
 *         School of ECE, Purdue University
 *
 * Implements the command line parser and controls pass ordering.
 * Users may extend this class by overriding runPasses
 * (which provides a default sequence of passes).  The derived
 * class should pass an instance of itself to the run method.
 * Derived classes have access to a protected {@link Program Program} object.
 */
public class OMP2GPUDriver extends Driver
{

	protected OMP2GPUDriver() {
		super();
		
	    options.add("omp2gpu",
        "Generate CUDA program from OpenMP program");
	    
		//Add Cuda-Specific command-line options.
		options.add("useMallocPitch", 
		"Use cudaMallocPitch() in OMP2CUDA translation");

		options.add("useMatrixTranspose",
		"Apply MatrixTranspose optimization in OMP2CUDA translation");

		options.add("cudaConfFile", "filename",
				"Name of the file that contains CUDA configuration parameters. " + 
				"(Any valid OpenMP-to-GPU compiler flags can be put in the file.) " + 
				"The file should exist in the current directory.");

		options.add("cudaThreadBlockSize", "number",
		"Size of CUDA thread block (default value = 256)");

		options.add("cudaGlobalMemSize", "size in bytes",
		"Size of CUDA global memory in bytes (default value = 1600000000); used for debugging");

		options.add("nvccVersion", "1.1",
		"version number of used nvcc compiler");

		options.add("OmpAnalysisOnly",
		"Conduct OpenMP analysis only");

		options.add("OmpKernelSplitOnly",
		"Generate kernel-splitted OpenMP codes without CUDA translation");
		
		options.add("cudaMemTrOptLevel", "N",
        "CUDA CPU-GPU memory transfer optimization level (0-4) (default is 3);" +
        "if N > 3, aggressive optimizations such as array-name-only analysis will be applied.");
		
		options.add("useParallelLoopSwap",
		"Apply ParallelLoopSwap optimization in OMP2GPU translation");
		
		options.add("useLoopCollapse",
		"Apply LoopCollapse optimization in OMP2GPU translation");
		
		options.add("useUnrollingOnReduction",
		"Apply loop unrolling optimization for in-block reduction in OMP2GPU translation;" +
		"to apply this opt, thread block size, BLOCK_SIZE = 2^m.");
		
		options.add("addSafetyCheckingCode",
		"Add GPU-memory-usage-checking code just before each kernel call; used for debugging.");
		
		options.add("addCudaErrorCheckingCode",
		"Add CUDA-error-checking code right after each kernel call (If this option is on, forceSyncKernelCall" +
		"option is suppressed); used for debugging.");
		
		options.add("cudaMaxGridDimSize", "number",
		"Maximum size of each dimension of a grid of thread blocks ( System max = 65535)");
		
		options.add("cudaGridDimSize", "number",
		"Size of each dimension of a grid of thread blocks, when thread block ID is 2-dimensional array " +
		"(max = 65535, default value = 10000)");
		
		options.add("forceSyncKernelCall", 
		"If enabled, cudaThreadSynchronize() call is inserted right after each kernel call " +
		"to force explicit synchronization; useful for debugging");
		
		options.add("cudaMallocOptLevel", "N",
        "CUDA Malloc optimization level (0-1) (default is 0)");
		
		options.add("assumeNonZeroTripLoops",
		"Assume that all loops have non-zero iterations");
		
		options.add("shrdSclrCachingOnReg",
		"Cache shared scalar variables onto GPU registers");
		
		options.add("shrdArryElmtCachingOnReg",
		"Cache shared array elements onto GPU registers");
		
		options.add("shrdSclrCachingOnSM",
		"Cache shared scalar variables onto GPU shared memory");
		
		options.add("prvtArryCachingOnSM",
		"Cache private array variables onto GPU shared memory");
		
		options.add("shrdArryCachingOnTM",
		"Cache 1-dimensional, R/O shared array variables onto GPU texture memory");
		
		options.add("useGlobalGMalloc",
		"Allocate GPU variables as global variables to reduce memory transfers " +
		"between CPU and GPU");
		
		options.add("globalGMallocOpt",
		"Optimize global GPU variable allocation to reduce memory transfers; " +
		"to use this option, useGlobalGMalloc should be on. If cudaMemTrOptLevel > 3, more aggressive" +
		"optimizations are applied.");
		
		options.add("showGResidentGVars",
		"After each function call, show globally allocated GPU variables " +
		"that are still residing in GPU global memory; " +
		"this works only if globalGMallocOpt is on.");
		
		options.add("cudaUserDirectiveFile", "filename",
				"Name of the file that contains user directives. " + 
				"The file should exist in the current directory.");
		
		options.add("extractTuningParameters", "filename",
				"Extract tuning parameters; output will be stored in the specified file. " +
				"(Default is TuningOptions.txt)" +
				"The generated file contains information on tuning parameters applicable " +
				"to current input program.");
		
		options.add("genTuningConfFiles", "tuningdir",
				"Generate tuning configuration files and/or userdirective files; " +
				"output will be stored in the specified directory. " +
				"(Default is tuning_conf)");
		options.add("tuningLevel", "N",
				"Set tuning level when genTuningConfFiles is on; \n" +
				"N = 1 (exhaustive search on program-level tuning options, default), \n" +
				"N = 2 (exhaustive search on kernel-level tuning options)");
		
		options.add("defaultTuningConfFile", "filename",
				"Name of the file that contains default Cuda tuning configurations. " +
				"(Default is cudaTuning.config) If the file does not exist, system-default setting will be used. ");
		
		options.add("maxNumOfCudaThreadBlocks", "N",
				"Maximum number of Cuda ThreadBlocks; if this option is on, tiling transformation code " +
				"is added to fit work partition into the thread batching specified by this option and cudaThreadBlockSize");
		
		options.add("shrdCachingOnConst",
		"Cache R/O shared variables onto GPU constant memory");
		
		options.add("disableCritical2ReductionConv",
		"Disable Critical-to-reduction conversion pass.");
	}

	/**
	 * Runs this driver with args as the command line.
	 *
	 * @param args The command line from main.
	 */
	public void run(String[] args)
	{
		parseCommandLine(args);
		parseCudaConfFile();
		HashMap<String, HashMap<String,Object>> userDirectives = parseCudaUserDirectiveFile();
		HashMap<String, Object> tuningConfigs = parseTuningConfig();

		parseFiles();

		if (getOptionValue("parse-only") != null)
		{
			System.err.println("parsing finished and parse-only option set");
			System.exit(0);
		}
		
		if(getOptionValue("genTuningConfFiles") != null)
		{
			if(getOptionValue("extractTuningParameters") == null) {
				setOptionValue("extractTuningParameters", "TuningOptions.txt");
			}
		}
		
		if(getOptionValue("extractTuningParameters") != null)
		{
			setOptionValue("useParallelLoopSwap", "1");
		}
		
		if(getOptionValue("useParallelLoopSwap") != null)
		{
			setOptionValue("ddt", "1");
		}

		runPasses();
		
	    if (getOptionValue("omp2gpu") != null)
	    {
	    	CodeGenPass.run(new omp2gpu(program, userDirectives, tuningConfigs));
	    }

		PrintTools.printlnStatus("Printing...", 1);

		try {
			program.print();
		} catch (IOException e) {
			System.err.println("could not write output files: " + e);
			System.exit(1);
		}
	}
	/**
	 * Entry point for Cetus; creates a new Driver object,
	 * and calls run on it with args.
	 *
	 * @param args Command line options.
	 */
	public static void main(String[] args)
	{
		/* Set default options for omp2gpu translator. */
		OMP2GPUDriver O2GDriver = new OMP2GPUDriver();
		setOptionValue("omp2gpu", "1");
		O2GDriver.run(args);
	}
	
	protected void parseCudaConfFile() {
		String value = getOptionValue("cudaConfFile");
		if( value != null ) {
			if( value.equals("1") ) {
				PrintTools.println("[WARNING] no Cuda-configuration file is specified; " +
						"cudaConfFile option will be ignored.", 0);
				return;
			}
			//Read contents of the file and parse configuration parameters.
			try {
				FileReader fr = new FileReader(value);
				BufferedReader br = new BufferedReader(fr);
				String inputLine = null;
				while( (inputLine = br.readLine()) != null ) {
					String opt = inputLine.trim();
					if( opt.length() == 0 ) {
						continue;
					}
					if( opt.charAt(0) == '#' ) {
						//Ignore comment line.
						continue;
					}
					int eq = opt.indexOf('=');

					if (eq == -1)
					{
						/* no value on the command line, so just set it to "1" */
						String option_name = opt.substring(0);

						if (options.contains(option_name)) {
							// Commandline input has higher priority than this configuarion input.
							if( getOptionValue(option_name) == null ) {
								setOptionValue(option_name, "1");
							} else {
								continue;
							}
						} else {
							System.err.println("ignoring unrecognized option " + option_name);
						}
					}
					else
					{
						/* use the value from the command line */
						String option_name = opt.substring(0, eq);

						if (options.contains(option_name)) {
							// Commandline input has higher priority than this configuarion input.
							if( getOptionValue(option_name) == null ) {
								setOptionValue(option_name, opt.substring(eq + 1));
							} else {
								if( option_name.equals("verbosity") || option_name.equals("outdir") ) {
									setOptionValue(option_name, opt.substring(eq + 1));
								} else {
									continue;
								}
							}
						} else {
							System.err.println("ignoring unrecognized option " + option_name);
						}
					}
				}
				br.close();
				fr.close();
			} catch (Exception e) {
				PrintTools.println("Error in readling cudaConfFile!!", 0);
			}
		}
	}
	
	protected HashMap<String, HashMap<String, Object>> parseCudaUserDirectiveFile() {
		HashMap<String, HashMap<String, Object>> userDirectiveMap = 
			new HashMap<String, HashMap<String, Object>>();
		String value = getOptionValue("cudaUserDirectiveFile");
		if( value != null ) {
			if( value.equals("1") ) {
				PrintTools.println("[WARNING] no Cuda-User-Directive file is specified; " +
						"cudaUserDirectiveFile option will be ignored.", 0);
				return userDirectiveMap;
			}
			//Read contents of the file and parse configuration parameters.
			try {
				FileReader fr = new FileReader(value);
				BufferedReader br = new BufferedReader(fr);
				String inputLine = null;
				HashMap<String, HashMap<String, Object>> uDirectives = 
					new HashMap<String, HashMap<String, Object>>();
				while( (inputLine = br.readLine()) != null ) {
					String opt = inputLine.trim();
					if( opt.length() == 0 ) {
						continue;
					}
					if( opt.charAt(0) == '#' ) {
						//Ignore comment line.
						continue;
					}
					/////////////////////////////////////////////////////////
					//Input string, opt, should have spaces before and     //
					//after the following tokens: '(', ')', ','            //
					/////////////////////////////////////////////////////////
					opt = opt.replaceAll("\\(", " ( ");
					opt = opt.replaceAll("\\)", " ) ");
					opt = opt.replaceAll(",", " , ");
					String[] token_array = opt.split("\\s+");
					uDirectives = CudaParser.parse_cuda_userdirective(token_array);
					if( uDirectives == null ) {
						continue;
					} else {
						Set<String> fKeySet = userDirectiveMap.keySet();
						Set<String> tKeySet = uDirectives.keySet();
						for( String tKey : tKeySet ) {
							HashMap<String, Object> tMap = uDirectives.get(tKey);
							if( fKeySet.contains(tKey) ) {
								HashMap<String, Object> fMap = userDirectiveMap.get(tKey);
								Set<String> ffKeySet = fMap.keySet();
								Set<String> ttKeySet = tMap.keySet();
								for( String ttKey : ttKeySet ) {
									Object ttObj = tMap.get(ttKey);
									if( ffKeySet.contains(ttKey) ) {
										Object ffObj = fMap.get(ttKey);
										if( ffObj instanceof Set ) {
											Set<String> ffSet = (Set<String>)ffObj;
											ffSet.addAll((Set<String>)ttObj);
										} else {
											fMap.put(ttKey, ttObj);
										}
									} else {
										fMap.put(ttKey, ttObj);
									}
									
								}
								
							} else {
								userDirectiveMap.put(tKey, tMap);
							}
						}
					}
				}
				br.close();
				fr.close();
			} catch (Exception e) {
				PrintTools.println("Error in readling cuda user-directive file!!", 0);
			}
		} 
		return userDirectiveMap;
	}
	
	protected HashMap<String, Object> parseTuningConfig() {
		HashMap<String, Object> tuningConfigMap = 
			new HashMap<String, Object>();
		String value = getOptionValue("defaultTuningConfFile");
		if( value != null ) {
			if( value.equals("1") ) {
				PrintTools.println("[INFO] no Cuda Tuning Config file is specified; " +
						"default configuration file (cudaTuning.config) will be used.", 0);
				value = "cudaTuning.config";
			}
			//Read contents of the file and parse configuration parameters.
			FileReader fr = null;
			try {
				fr = new FileReader(value);
			} catch (Exception e) {
				PrintTools.println("[INFO] no Cuda tuning configuration file is found;" +
						" default configuration will be used.", 0);
				return tuningConfigMap;
			}
			try {
				BufferedReader br = new BufferedReader(fr);
				String inputLine = null;
				HashMap<String, Object> tConfigMap = null;
				while( (inputLine = br.readLine()) != null ) {
					String opt = inputLine.trim();
					if( opt.length() == 0 ) {
						continue;
					}
					if( opt.charAt(0) == '#' ) {
						//Ignore comment line.
						continue;
					}
					/////////////////////////////////////////////////////////
					//Input string, opt, should have spaces before and     //
					//after the following tokens: '(', ')', ',', '='       //
					/////////////////////////////////////////////////////////
					opt = opt.replaceAll("\\(", " ( ");
					opt = opt.replaceAll("\\)", " ) ");
					opt = opt.replaceAll(",", " , ");
					opt = opt.replaceAll("=", " = ");
					String[] token_array = opt.split("\\s+");
					tConfigMap = CudaParser.parse_cuda_tuningconfig(token_array);
					if( tConfigMap == null ) {
						continue;
					} else {
						Set<String> fKeySet = tuningConfigMap.keySet();
						Set<String> tKeySet = tConfigMap.keySet();
						for( String tKey : tKeySet ) {
							Object tObj = tConfigMap.get(tKey);
							if( fKeySet.contains(tKey) ) {
								Object fObj = tuningConfigMap.get(tKey);
								if( fObj instanceof Set && tObj instanceof Set ) {
									((Set)fObj).addAll((Set)tObj);
								} else if( tObj instanceof String ) {
									//Overwrite value.
									tuningConfigMap.put(tKey, tObj);
								} else {
									PrintTools.println("[ERROR in parseTuningConfig()] unsuppored input found; " +
											"remaining input configuraiton will be ignored.", 0);
									return tuningConfigMap;
								}
							} else {
								tuningConfigMap.put(tKey, tObj);
							}
						}
					}
				}
				br.close();
				fr.close();
			} catch (Exception e) {
				PrintTools.println("Error in readling Cuda tuning configuration file!!", 0);
			}
		} 
		return tuningConfigMap;
	}

}

