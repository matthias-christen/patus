#include "ch_unibas_cs_hpwc_patus_ilp_ILPModel.h"
#include "ch_unibas_cs_hpwc_patus_ilp_ILPSolver.h"
#include "ch_unibas_cs_hpwc_patus_ilp_ILPSolution.h"

#include <CoinModel.hpp>
#include <CbcModel.hpp>
#include <OsiClpSolverInterface.hpp>

#include <string.h>
#include <math.h>
#include <omp.h>


/**
 * Returns a pointer to the CoinModel stored in the ILPModel class instance (pThis).
 */
CoinModel* getModel (JNIEnv* pEnv, jobject pModel)
{
	jclass cls = pEnv->GetObjectClass (pModel);
	jfieldID fid = pEnv->GetFieldID (cls, "m_ptrModel", "J");
	jlong nPtrModel = pEnv->GetLongField (pModel, fid);
	return reinterpret_cast<CoinModel*> (nPtrModel);
}

/**
 * Utility function that translates a jstring to a locale-specific native C string.
 * See Liang, The Java Native Interface, p. 100
 */
char* getStringNativeChars (JNIEnv* pEnv, jstring jstr)
{
	char* szResult = NULL;

	if (pEnv->EnsureLocalCapacity (2) < 0)
	{
		// throws exception
		return 0;
	}
	
	jclass clsString = pEnv->FindClass ("java/lang/String");
	jmethodID mid = pEnv->GetMethodID (clsString, "getBytes", "()[B");
	jbyteArray bytes = (jbyteArray) pEnv->CallObjectMethod (jstr, mid);
	jthrowable ex = pEnv->ExceptionOccurred ();

	if (!ex)
	{
		jint nLen = pEnv->GetArrayLength (bytes);
		szResult = new char[nLen + 1];
		pEnv->GetByteArrayRegion (bytes, 0, nLen, (jbyte*) szResult);
		szResult[nLen] = 0;
	}
	else
		pEnv->DeleteLocalRef (ex);

	pEnv->DeleteLocalRef (bytes);
	return szResult;
}


JNIEXPORT void JNICALL Java_ch_unibas_cs_hpwc_patus_ilp_ILPModel_createModelInternal (JNIEnv* pEnv, jobject pThis, jint nVariablesCount)
{
	// create the new CoinModel
	CoinModel* pModel = new CoinModel ();
	
	// store a pointer to the model in the ILPModel class instance (pThis)
	jclass cls = pEnv->GetObjectClass (pThis);
	jfieldID fid = pEnv->GetFieldID (cls, "m_ptrModel", "J");
	pEnv->SetLongField (pThis, fid, reinterpret_cast<jlong> (pModel));
}

JNIEXPORT void JNICALL Java_ch_unibas_cs_hpwc_patus_ilp_ILPModel_setVariableTypeInternal (JNIEnv* pEnv, jobject pThis,
	jint nVariableIdx, jboolean bIsLowerBoundSet, jdouble fLowerBound, jboolean bIsUpperBoundSet, jdouble fUpperBound, jboolean bIsInteger)
{
	CoinModel* pModel = getModel (pEnv, pThis);
	
	// set bounds and variable type
	pModel->setColumnBounds (nVariableIdx,
		bIsLowerBoundSet == JNI_TRUE ? fLowerBound : -COIN_DBL_MAX, 
		bIsUpperBoundSet == JNI_TRUE ? fUpperBound : COIN_DBL_MAX);
	pModel->setColumnIsInteger (nVariableIdx, bIsInteger == JNI_TRUE ? true : false);
}

JNIEXPORT void JNICALL Java_ch_unibas_cs_hpwc_patus_ilp_ILPModel_addConstraintInternal (JNIEnv* pEnv, jobject pThis,
	jdoubleArray rgCoeffs, jboolean bIsLowerBoundSet, jdouble fLowerBound, jboolean bIsUpperBoundSet, jdouble fUpperBound)
{
	CoinModel* pModel = getModel (pEnv, pThis);
	
	int nNumNzElts = 0;
	jdouble* pCoeffs = pEnv->GetDoubleArrayElements (rgCoeffs, NULL);
	jsize nEltsCount = pEnv->GetArrayLength (rgCoeffs);
	
	// count the number of nonzero elements
//#pragma omp parallel for reduction(+: nNumNzElts)
	for (int i = 0; i < nEltsCount; i++)
		if (pCoeffs[i] != 0.0)
			nNumNzElts++;
			
	// allocate memory for the values and the column pointers
	double* pVals = new double[nNumNzElts];
	int* pCols = new int[nNumNzElts];
	
	// copy the data
	int j = 0;
	
//#pragma omp parallel for
	for (int i = 0; i < nEltsCount; i++)
	{
		if (pCoeffs[i] != 0.0)
		{
			pVals[j] = (double) pCoeffs[i];
			pCols[j] = i;
			j++;
		}
	}
	
	// we don't need the JNI array data anymore now
	pEnv->ReleaseDoubleArrayElements (rgCoeffs, pCoeffs, 0);
	
	// add a row to the CoinModel
	pModel->addRow (nNumNzElts, pCols, pVals,
		bIsLowerBoundSet == JNI_TRUE ? fLowerBound : -COIN_DBL_MAX, 
		bIsUpperBoundSet == JNI_TRUE ? fUpperBound : COIN_DBL_MAX
	);
	
	// clean up
	delete[] pVals;
	delete[] pCols;
}

JNIEXPORT void JNICALL Java_ch_unibas_cs_hpwc_patus_ilp_ILPModel_setObjective (JNIEnv* pEnv, jobject pThis, jdoubleArray rgCoeffs)
{
	CoinModel* pModel = getModel (pEnv, pThis);
	
	jdouble* pCoeffs = pEnv->GetDoubleArrayElements (rgCoeffs, NULL);
	pModel->setObjective (pEnv->GetArrayLength (rgCoeffs), pCoeffs);
	pEnv->ReleaseDoubleArrayElements (rgCoeffs, pCoeffs, 0);
}

JNIEXPORT void JNICALL Java_ch_unibas_cs_hpwc_patus_ilp_ILPModel_writeMPS (JNIEnv* pEnv, jobject pThis, jstring strFilename)
{
	CoinModel* pModel = getModel (pEnv, pThis);
	const char* pszFilename = getStringNativeChars (pEnv, strFilename);
	
	pModel->writeMps (pszFilename);

	delete (char*) pszFilename;
}

JNIEXPORT void JNICALL Java_ch_unibas_cs_hpwc_patus_ilp_ILPModel_delete (JNIEnv* pEnv, jobject pThis)
{
	CoinModel* pModel = getModel (pEnv, pThis);
	delete pModel;
}


/**
 * cf. examples/driver4.cpp
 */
static int callback (CbcModel* pModel, int nWhereFrom)
{
	int nReturnCode = 0;
	
	switch (nWhereFrom)
	{
	case 1:
	case 2:
		if (!pModel->status () && pModel->secondaryStatus ())
			nReturnCode = 1;
		break;
		
	case 3:
		break;
		
	case 4:
		// if not good enough could skip postprocessing
		break;
		
	case 5:
		break;
		
	default:
		abort ();
	}
  
	return nReturnCode;
}

JNIEXPORT jint JNICALL Java_ch_unibas_cs_hpwc_patus_ilp_ILPSolver_solveInternal (JNIEnv* pEnv, jobject pThis,
	jobject pILPModel, jdoubleArray rgSolution, jdoubleArray rgObjective, jint nTimeLimit)
{
	CoinModel* pModel = getModel (pEnv, pILPModel);

	// load the model
	OsiClpSolverInterface solver;
	solver.loadFromCoinModel (*pModel, true);
	
	// get number of cores
	char szNumThds[5];
	int nNumCores = omp_get_num_procs ();
	sprintf (szNumThds, "%d", nNumCores);
	
	// format the time limit
	char szTimeLimit[8];
	memset (szTimeLimit, 0, 8);
	if (nTimeLimit < 0)
		strcpy (szTimeLimit, "1e10");
	else
	{
		// Cbc sets accumulated time limit, so multiply by the number of threads
		// to account for wallclock time (approximately)
		sprintf (szTimeLimit, "%d", nTimeLimit * nNumCores);
	}
	
	// create the Cbc model and invoke the solver
	CbcModel model (solver);
	CbcMain0 (model);
    const char* argv2[] = { "CoinInterface", "-threads", szNumThds, "-sec", szTimeLimit, "-solve", "-quit" };
    CbcMain1 (7, argv2, model, callback);	

	fflush (stdout);

	// check whether a problem has occurred during solving, and if yes, return immediately	
	if (model.isProvenInfeasible ())
		return ch_unibas_cs_hpwc_patus_ilp_ILPSolution_STATUS_INFEASIBLE;
	if (model.isSolutionLimitReached ())
		return ch_unibas_cs_hpwc_patus_ilp_ILPSolution_STATUS_LIMIT_REACHED;
	if (model.isAbandoned ())
		return ch_unibas_cs_hpwc_patus_ilp_ILPSolution_STATUS_ABANDONED;
	
	// otherwise copy the solver's solution into the solution array
	jdouble fObjective = (jdouble) model.getObjValue ();
	pEnv->SetDoubleArrayRegion (rgObjective, 0, 1, &fObjective);

	if (fObjective < 1e10)
	{
		const double* pSolution = model.bestSolution ();
		int nLen = model.solver ()->getNumCols ();
		jsize nArrLen = pEnv->GetArrayLength (rgSolution);
		if (nArrLen < nLen)
			nLen = nArrLen;
		pEnv->SetDoubleArrayRegion (rgSolution, 0, nLen, pSolution);
	}
	else
		return ch_unibas_cs_hpwc_patus_ilp_ILPSolution_STATUS_NOSOLUTIONFOUND;
	   
	return ch_unibas_cs_hpwc_patus_ilp_ILPSolution_STATUS_OPTIMAL;
}

/*
int main (int argc, char** argv)
{
	return 0;
}*/
