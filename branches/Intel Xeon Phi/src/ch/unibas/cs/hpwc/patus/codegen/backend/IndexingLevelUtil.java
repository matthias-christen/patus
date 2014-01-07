package ch.unibas.cs.hpwc.patus.codegen.backend;

import ch.unibas.cs.hpwc.patus.codegen.backend.IIndexing.IIndexingLevel;

public class IndexingLevelUtil
{
	/**
	 * Converts the parallelism level to an indexing level and returns the corresponding indexing
	 * level object.
	 * The lowest parallelism level is assigned to the outer-most loop, which corresponds to the
	 * highest indexing level.
	 * @param indexing
	 * @param nParallelismLevel
	 * @return
	 */
	public static IIndexingLevel getIndexingLevelFromParallelismLevel (IIndexing indexing, int nParallelismLevel)
	{
		int nIndexingLevelsCount = indexing.getIndexingLevelsCount ();
		if (nParallelismLevel <= 0)
			return indexing.getIndexingLevel (nIndexingLevelsCount - 1);
		if (nParallelismLevel >= nIndexingLevelsCount)
			return indexing.getIndexingLevel (0);
		return indexing.getIndexingLevel (nIndexingLevelsCount - nParallelismLevel);
	}
}
