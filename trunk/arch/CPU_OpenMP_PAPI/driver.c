#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <papi.h>
#include <papiStdEventDefs.h>


#pragma patus forward_decls


/* Define PAPI events to measure (http://icl.cs.utk.edu/projects/papi/presets.html) */
//int PAPI_EVENTS[] = { PAPI_FP_INS };
int PAPI_EVENTS[] = { PAPI_L1_DCA, PAPI_L1_DCM, PAPI_L2_DCA, PAPI_L2_DCM  };


long unsigned int thdid ()
{
    return omp_get_thread_num ();
}

int main (int argc, char** argv)
{
	int i, j;

    int nEventSet = PAPI_NULL;
    long_long* pAllValues = NULL;
    long_long* pLocalValues = NULL;
    const int nEventsCount = sizeof (PAPI_EVENTS) / sizeof (int);
    int bPAPIError = 0;
    
    // initialize PAPI
    if (PAPI_library_init (PAPI_VER_CURRENT) != PAPI_VER_CURRENT)
    {
        printf ("Failed to initialize PAPI library.\n");
        return -1;
    }
    
    if (PAPI_thread_init (thdid) != PAPI_OK)
    {
        printf ("PAPI thread init failed.\n");
        return -1;
    }
    
    if (PAPI_set_debug (PAPI_VERB_ECONT) != PAPI_OK)
    {
        printf ("Setting PAPI debug level failed.\n");
        return -1;
    }

	// prepare grids
	#pragma patus declare_grids
	#pragma patus allocate_grids

	// initialize
	#pragma omp parallel
	{
		#pragma patus initialize_grids
	}

	long nFlopsPerStencil = PATUS_FLOPS_PER_STENCIL;
	long nGridPointsCount = 5 * PATUS_GRID_POINTS_COUNT;
	long nBytesTransferred = 5 * PATUS_BYTES_TRANSFERRED;

	// warm up
	#pragma omp parallel
	{
		#pragma patus compute_stencil
	}
	
	// run the benchmark
	tic ();
	#pragma omp parallel firstprivate(nEventSet) private(pLocalValues, i, j)
	{
        // create PAPI event set
        if (PAPI_create_eventset (&nEventSet) != PAPI_OK)
        {
            printf ("Thread %d: Failed to create PAPI event set.\n", omp_get_thread_num ());
            bPAPIError = 1;
        }

        // add events one by one
        if (!bPAPIError)
        {
            for (j = 0; j < nEventsCount; j++)
            {
                if (PAPI_add_event (nEventSet, PAPI_EVENTS[j]) != PAPI_OK)
                {
                    char szBuf[200];
                    PAPI_event_code_to_name (PAPI_EVENTS[j], szBuf);
                    printf ("Thread %d: Failed to add PAPI event %s.\n", omp_get_thread_num (), szBuf);
                    bPAPIError = 1;
                }
            }
        }
        
        if (!bPAPIError && PAPI_start (nEventSet) != PAPI_OK)
        {
            printf ("Thread %d: Failed to start PAPI counters.\n", omp_get_thread_num ());
            bPAPIError = 1;
        }

        // wait till all threads have set up PAPI
        #pragma omp barrier

        if (!bPAPIError)
        {
            /* Create the values array; wait for it to be created before threads can grab their part */
            #pragma omp master
            pAllValues = (long_long*) malloc (omp_get_num_threads () * nEventsCount * sizeof (long_long));
            #pragma omp barrier

            pLocalValues = &pAllValues[omp_get_thread_num () * nEventsCount];
            for (j = 0; j < nEventsCount; j++)
                pLocalValues[j] = 0L;

			for (i = 0; i < 5; i++)
			{
				PAPI_reset (nEventSet);
				#pragma patus compute_stencil
				PAPI_accum (nEventSet, pLocalValues);
				
				#pragma omp barrier
			}

            #pragma omp master
            {
                // print per-thread statistics and accumulate counts on individual threads   
                for (i = 0; i < omp_get_num_threads (); i++)
                {
                    // display
                    printf ("Thread %d: ", i);
                    for (j = 0; j < nEventsCount; j++)
                    {
                        char szBuf[200];
                        PAPI_event_code_to_name (PAPI_EVENTS[j], szBuf);
                        printf ("* %s: %lld ", szBuf, pAllValues[i * nEventsCount + j]);
                    }
                    printf ("\n");

                    // accumulate
                    if (i >= 1)
                    {
                        for (j = 0; j < nEventsCount; j++)
                            pAllValues[j] += pAllValues[i * nEventsCount + j];
                    }
                }

                // print global statistics
                for (i = 0; i < nEventsCount; i++)
                {
                    char szBuf[200];
                    PAPI_event_code_to_name (PAPI_EVENTS[i], szBuf);
                    printf ("%s: %lld\n", szBuf, pAllValues[i]);
                }
            }
		}

        // wait till master thread is done printing the output
        #pragma omp barrier

        // stop PAPI and wait till all the threads have stopped
        PAPI_stop (nEventSet, pLocalValues);
        #pragma omp barrier

        // free the values array
        #pragma omp master
        free (pAllValues);
	}
	
	toc (nFlopsPerStencil, nGridPointsCount, nBytesTransferred);
	
	// validate
	if (PATUS_DO_VALIDATION)
	{
		#pragma omp parallel
		{
			#pragma patus initialize_grids
			#pragma omp barrier
			#pragma patus compute_stencil
		}
		#pragma patus validate_computation
		if (PATUS_VALIDATES)
			puts ("Validation OK.");
		else
		{
			#pragma patus deallocate_grids
			puts ("Validation failed.");
			return -1;
		}
	}	

	// free memory
	#pragma patus deallocate_grids

	return 0;
}
