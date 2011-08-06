package ch.unibas.cs.hpwc.patus.codegen.backend;

import ch.unibas.cs.hpwc.patus.codegen.backend.IIndexing.IIndexingLevel;

public class IndexingLevelUtil
{
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
