package omp2gpu.transforms;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;

import cetus.transforms.LoopInterchange;
import cetus.analysis.DDGraph;
import cetus.analysis.DependenceVector;
import cetus.analysis.LoopTools;
import cetus.exec.Driver;
import cetus.hir.*;
import omp2gpu.hir.CudaAnnotation;
import omp2gpu.analysis.AnalysisTools;

/**
 * @author Seyong Lee <lee222@purdue.edu>
 *         ParaMount Group 
 *         School of ECE, Purdue University
 *
 * ParallelLoopSwap is a transformation, which is a reverse operation of LoopInterchange.
 */
public class ParallelLoopSwap extends LoopInterchange {
	
	private int debug_level = 2;
	
	public ParallelLoopSwap( Program program ) {
		super(program);
	}
	
	public String getPassName()
	{
		return new String("[ParallelLoopSwap]");
	}

	public void start()
	{
		LinkedList<Loop> loops = new LinkedList<Loop>();
		LinkedList<Loop> loops2 = new LinkedList<Loop>();
		List<ForLoop> outer_loops = new ArrayList<ForLoop>();
		List<DependenceVector> depVec = new ArrayList<DependenceVector>();
		List<Expression> expList = new LinkedList<Expression>();
		DepthFirstIterator iter = null;
		int i;
		int target_loops = 0;
		int num_single = 0, num_non_perfect = 0, num_contain_func = 0, num_loop_interchange=0;
		int num_skipped = 0;
		
		String value = Driver.getOptionValue("extractTuningParameters");
		boolean extractTuningParameters = false;
		if( value != null ) {
			extractTuningParameters = true;
		}
		
		List<OmpAnnotation> forAnnotList = IRTools.collectPragmas(program, OmpAnnotation.class, "for");
		for( OmpAnnotation fAnnot : forAnnotList ) {
			Annotatable at = fAnnot.getAnnotatable();
			if( at instanceof ForLoop ) {
				outer_loops.add((ForLoop)at);
			}
		}

		PrintTools.println("# of Outermost Loop: " + outer_loops.size(), debug_level);

		for(i = outer_loops.size()-1; i >= 0; i--)
		{
			ForLoop ompLoop = outer_loops.get(i);
			iter = new DepthFirstIterator(ompLoop);
			OmpAnnotation pAnnot = null;
			CudaAnnotation noAnnot = null;
			Statement pRegion = null;
			boolean LoopSwapped = false;
			pRegion = (Statement)AnalysisTools.findEnclosingParallelRegion(ompLoop);
			if( pRegion == null ) {
				PrintTools.println("[WARNING in ParallelLoopSwap()] can not find an enclosing parallel region.", 0);
				continue;
			}
			pAnnot = ((Annotatable)pRegion).getAnnotation(OmpAnnotation.class, "parallel");
			if( pAnnot == null ) {
				PrintTools.println("[WARNING in ParallelLoopSwap()] can not find an enclosing parallel region.", 0);
				continue;
			}
			if( !extractTuningParameters ) {
				noAnnot = pRegion.getAnnotation(CudaAnnotation.class, "noploopswap");
			}
			loops.clear();
			loops2.clear();
			while(iter.hasNext()) {
				Object o = iter.next();
				if(o instanceof ForLoop)
					loops.add((Loop)o);
			}
			if(loops.size() < 2) {
				num_single++;
			}	else if(!LoopTools.isPerfectNest((ForLoop)loops.get(0))) {
				num_non_perfect++;
			}	else if(LoopTools.containsFunctionCall((ForLoop)loops.get(0))) {
				num_contain_func++;
			} else {
				if( !extractTuningParameters && (noAnnot != null) ) {
					num_skipped++;
					continue;
				}
				target_loops++;
				Statement stm = ((ForLoop)loops.get(loops.size()-1)).getBody();
				List<ArrayAccess> arrays = new ArrayList();  // Arrays in loop body
				DepthFirstIterator iter2 = new DepthFirstIterator(stm);

				while(iter2.hasNext())
				{
					Object child = iter2.next();
					if(child instanceof ArrayAccess)
					{
						arrays.add((ArrayAccess)child);
					}
				}

				int r = 0,j,until = loops.size();
				int target_index = 0;
				boolean icFlag = true;
				List<Integer> rank;
				int rankSize;
				 
				////////////////////////////////////////////////////
				// loops2 contains nested loops in reverse order, //
				// from the innermost to the outermost.           //
				////////////////////////////////////////////////////
				for( j= until-1; j>=0; j-- ) {
					loops2.add(loops.get(j));
				}
				
				while(icFlag)
				{
					Expression exp;
					expList.clear();
					for(j = 0; j < until; j++)
					{
						exp = LoopTools.getIndexVariable((ForLoop)loops2.get(j));
						if(exp != null)
							expList.add(exp);
					}
					rank = getRank(arrays, expList, target_index);
					rankSize = rank.size();
					for(j = 0; j < rankSize; j++) 
					{
						r = getRank2(rank, expList, loops2);
						rank.remove(rank.indexOf(r));
						int r1, r2;
						r1 = r;

						if(expList.size() < until) until = expList.size();
						for(int k = r+1; k < until; k++)
						{
							if(isLegal(loops2, r, k))
							{
								swapLoop((ForLoop)loops2.get(r), (ForLoop)loops2.get(k));
								num_loop_interchange++;
								Collections.swap(expList, r, k);
								r = k;
								LoopSwapped = true;

							} else {
								break;
							}
						}		
						until = r;
						///////////////////////////////////////////////////////////////////////////////////
						// FIXME: requested by SY Lee                                                    //
						// If the list "rank" consists of r1 and r2 (r1 < r2), and if getRank2() returns //
						// r1 first, the loop pointed by r2 will be moved downward by 1. Thus r2 in the  //
						// list "rank" should be changed with r2-1.                                      //
						///////////////////////////////////////////////////////////////////////////////////
						for( int k = 0; k<rank.size()-1; k++) {
							r2 = rank.get(k);
							if( (r1 < r2) && (r2 <= r) ) {
								rank.set(k, r2-1);
							}
						}
					}
					target_index++;
					if(until == 0) icFlag = false;
				}
			}
			if( extractTuningParameters && LoopSwapped ) {
				//"tuningparameter" clause is used only internally .
				CudaAnnotation tAnnot = pRegion.getAnnotation(CudaAnnotation.class, "tuningparameters");
				if( tAnnot == null ) {
					tAnnot = new CudaAnnotation("tuningparameters", "true");
					pRegion.annotate(tAnnot);
				}
				tAnnot.put("ploopswap", "true");
			}
		}

		PrintTools.println("Target loops : " + target_loops, debug_level);
		PrintTools.println("Non Perfect loops : " + num_non_perfect, debug_level);
		PrintTools.println("Single loops : " + num_single, debug_level);
		PrintTools.println("Contain Function : " + num_contain_func, debug_level);
		PrintTools.println("Loop Interchanged : " + num_loop_interchange, debug_level);
		PrintTools.println("Skipped Loop : " + num_skipped, debug_level);

		return;
	}
	
	/**
	 * Check legality of loop interchange between rsrc and rtarget. Both rsrc and rtarget
	 * are in nest loops and rsrc is inner than rtarget                                         
	 * @param reverseNest nested loops in reverse order, from innermost to the outermost
	 * @param rsrc source loop id in the reverseNest list
	 * @param rtarget target loop id in the reverseNest list
	 * @return true if it is legal to swap two loops, rsrc and rtarget
	 */
	public boolean isLegal(LinkedList<Loop> reverseNest, int rsrc, int rtarget)
	{
		int i, j, next;
		int src, target;
		DDGraph ddg;
		String str;
		ArrayList<DependenceVector> dpv;
		DependenceVector dd;
		//////////////////////////////////////////////////////
		// Nested loops from the outermost to the innermost //
		//////////////////////////////////////////////////////
		LinkedList<Loop> nest = new LinkedList<Loop>();
		j = reverseNest.size();
		for(i=j-1; i>=0; i--) {
			nest.add(reverseNest.get(i));
		}
		src = j-rsrc-1;
		target = j-rtarget-1;
		
		ddg = program.getDDGraph();
		dpv = ddg.getDirectionMatrix(nest);

		if(src == target) return true;
		///////////////////////////////////////////////////////////
		// Swap src and target to make src refer to a loop outer //
		// than a target loop.                                   //
		///////////////////////////////////////////////////////////
		if(src > target) {
			i = src;
			src = target;
			target = i;
		}
		
		//////////////////////////////////////////////////////////////
		// Check whether index variable for the outer loop (src) is //
		// used in initial statements or condition expressions of   //
		// the loops residing between src and target.               //
		//////////////////////////////////////////////////////////////
		ForLoop srcLoop = (ForLoop)nest.get(src);
		Expression srcIndex = LoopTools.getIndexVariable(srcLoop);
		if( srcIndex == null ) {
			//Can't identify the index variable of the src loop; 
			//return false conservatively.
			return false;
		} else {
			for( i=src+1; i<=target; i++ ) {
				ForLoop tLoop = (ForLoop)nest.get(i);
				Statement initStmt = tLoop.getInitialStatement();
				if( IRTools.containsExpression(initStmt, srcIndex) ) {
					return false;
				}
				Expression condExp = tLoop.getCondition();
				if( IRTools.containsExpression(condExp, srcIndex) ) {
					return false;
				}
			}
		}

		for(i = 0; i < dpv.size(); i++)
		{
			dd = dpv.get(i);
			str = dd.toString();
			for(j = 0; j < str.length(); j++)
			{
				if(j == src) next = target;
				else if(j == target) next = src;
				else next = j;

				if(next < str.length()) {
					if(str.charAt(next) == '>') return false;
					if(str.charAt(next) == '<') break;
				}
			}
		}

		return true;
	}

}
