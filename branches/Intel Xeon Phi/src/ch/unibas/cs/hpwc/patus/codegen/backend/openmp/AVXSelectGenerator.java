package ch.unibas.cs.hpwc.patus.codegen.backend.openmp;

import java.util.ArrayList;
import java.util.List;

import cetus.hir.Expression;
import cetus.hir.IntegerLiteral;
import ch.unibas.cs.hpwc.patus.codegen.backend.openmp.PermutatorGenetic.Operand;
import ch.unibas.cs.hpwc.patus.codegen.backend.openmp.PermutatorGenetic.Operator;
import ch.unibas.cs.hpwc.patus.codegen.backend.openmp.PermutatorGenetic.Selector;

public class AVXSelectGenerator
{
	private static class MMShufflePS extends Operator
	{
		public MMShufflePS ()
		{
			// 2 operands, 8 elements wide vector,
			// _mm256_shuffle_ps ((0 .. 7), (8 .. 15)) = (s1(0..3), s2(0..3), s3(8..11), s4(8..11), s1(4..7), s2(4..7), s3(12..15), s4(12..15))
			super ("_mm256_shuffle_ps", 2, 8, new Selector[] {
				new Selector (1, new int[] { 0, 1, 2, 3 }),
				new Selector (2, new int[] { 0, 1, 2, 3 }),
				new Selector (3, new int[] { 8, 9, 10, 11 }),
				new Selector (4, new int[] { 8, 9, 10, 11 }),
				new Selector (1, new int[] { 4, 5, 6, 7 }),
				new Selector (2, new int[] { 4, 5, 6, 7 }),
				new Selector (3, new int[] { 12, 13, 14, 15 }),
				new Selector (4, new int[] { 12, 13, 14, 15 })
			});
		}
		
		@Override
		public Expression getControlExpression (int[] rgConfig)
		{
			// cf. _MM_SHUFFLE macro
			
			// number of config elements corresponds to the number of distinct selectors,
			// shift typically corresponds to how many indices a selector can choose from (here: 4 indices => 2 bits per index)
			
			return new IntegerLiteral (rgConfig[0] | (rgConfig[1] << 2) | (rgConfig[2] << 4) | (rgConfig[3] << 6));
		}
	}

	private static class MMShufflePD_ForPS extends Operator
	{
		public MMShufflePD_ForPS ()
		{
			// 2 operands, 8 elements wide vector,
			// _mm256_shuffle_pd ((0 .. 7), (8 .. 15)) = (s1(0,2), s1(1,3), s2(8,10), s2(9,11), s3(4,6), s3(5,7), s4(12,14), s4(13,15))
			super ("_mm256_shuffle_pd", 2, 8, new Selector[] {
				new Selector (1, new int[] { 0, 2 }),
				new Selector (1, new int[] { 1, 3 }),
				new Selector (2, new int[] { 8, 10 }),
				new Selector (2, new int[] { 9, 11 }),
				new Selector (3, new int[] { 4, 6 }),
				new Selector (3, new int[] { 5, 7 }),
				new Selector (4, new int[] { 12, 14 }),
				new Selector (4, new int[] { 13, 15 })
			});
		}
		
		@Override
		public Expression getControlExpression (int[] rgConfig)
		{
			// cf. _MM_SHUFFLE macro
			
			// number of config elements corresponds to the number of distinct selectors,
			// shift typically corresponds to how many indices a selector can choose from (here: 4 indices => 2 bits per index)
			
			return new IntegerLiteral (rgConfig[0] | (rgConfig[1] << 1) | (rgConfig[2] << 2) | (rgConfig[3] << 3));
		}
	}

	private static class MMPermutePS extends Operator
	{
		public MMPermutePS ()
		{
			// 1 operand, 8 elements wide vector,
			// _mm256_permute_ps (0 .. 7) = (s1(0..3), s2(0..3), s3(0..3), s4(0..3), s1(4..7), s2(4..7), s3(4..7), s4(4..7))
			super ("_mm256_permute_ps", 1, 8, new Selector[] {
				new Selector (1, new int[] { 0, 1, 2, 3 }),
				new Selector (2, new int[] { 0, 1, 2, 3 }),
				new Selector (3, new int[] { 0, 1, 2, 3 }),
				new Selector (4, new int[] { 0, 1, 2, 3 }),
				new Selector (1, new int[] { 4, 5, 6, 7 }),
				new Selector (2, new int[] { 4, 5, 6, 7 }),
				new Selector (3, new int[] { 4, 5, 6, 7 }),
				new Selector (4, new int[] { 4, 5, 6, 7 })
			});
		}
		
		@Override
		public Expression getControlExpression (int[] rgConfig)
		{
			return new IntegerLiteral (rgConfig[0] | (rgConfig[1] << 2) | (rgConfig[2] << 4) | (rgConfig[3] << 6));
		}
	}
	
	private static class MMPermuteF128PS extends Operator
	{
		public MMPermuteF128PS ()
		{
			// 2 operands, 8 elements wide vector,
			// _mm256_shuffle2f128_ps ((0 .. 7), (8 .. 15)) = (s1(0,4,8,12), s1(1,5,9,13), s1(2,6,10,14), s1(3,7,11,15), s2(0,4,8,12), s2(1,5,9,13), s2(2,6,10,14), s2(3,7,11,15))
			super ("_mm256_permute2f128_ps", 2, 8, new Selector[] {
				new Selector (1, new int[] { 0, 4, 8, 12 }),
				new Selector (1, new int[] { 1, 5, 9, 13 }),
				new Selector (1, new int[] { 2, 6, 10, 14 }),
				new Selector (1, new int[] { 3, 7, 11, 15 }),
				new Selector (2, new int[] { 0, 4, 8, 12 }),
				new Selector (2, new int[] { 1, 5, 9, 13 }),
				new Selector (2, new int[] { 2, 6, 10, 14 }),
				new Selector (2, new int[] { 3, 7, 11, 15 })
			});
		}
		
		@Override
		public Expression getControlExpression (int[] rgConfig)
		{
			return new IntegerLiteral (rgConfig[0] | (rgConfig[1] << 4));
		}
	}
	
	private static class MMBlendPS extends Operator
	{
		public MMBlendPS ()
		{
			// 2 operands, 8 elements wide vector,
			// _mm256_shuffle_ps ((0 .. 7), (8 .. 15)) = (s1(0,8), s2(1,9), s3(2,10), s4(3,11), s5(4,12), s6(5,13), s7(6,14), s8(7,15))
			super ("_mm256_blend_ps", 2, 8, new Selector[] {
				new Selector (1, new int[] { 0, 8 }),
				new Selector (2, new int[] { 1, 9 }),
				new Selector (3, new int[] { 2, 10 }),
				new Selector (4, new int[] { 3, 11 }),
				new Selector (5, new int[] { 4, 12 }),
				new Selector (6, new int[] { 5, 13 }),
				new Selector (7, new int[] { 6, 14 }),
				new Selector (8, new int[] { 7, 15 })
			});
		}
		
		@Override
		public Expression getControlExpression (int[] rgConfig)
		{
			int nExpr = 0;
			for (int i = 0; i < 8; i++)
				nExpr |= (rgConfig[i] << i);
			return new IntegerLiteral (nExpr);
		}
	}
	
	
	private static class MMShufflePD extends Operator
	{
		public MMShufflePD ()
		{
			// 2 operands, 4 elements wide vector,
			// _mm256_shuffle_pd ((0 .. 3), (4 .. 7)) = (s1(0,1), s2(4,5), s3(2,3), s4(6,7))
			super ("_mm256_shuffle_pd", 2, 4, new Selector[] {
				new Selector (1, new int[] { 0, 1 }),
				new Selector (2, new int[] { 4, 5 }),
				new Selector (3, new int[] { 2, 3 }),
				new Selector (4, new int[] { 6, 7 })
			});
		}
		
		@Override
		public Expression getControlExpression (int[] rgConfig)
		{
			return new IntegerLiteral (rgConfig[0] | (rgConfig[1] << 1) | (rgConfig[2] << 2) | (rgConfig[3] << 3));
		}
	}
	
	private static class MMPermutePD extends Operator
	{
		public MMPermutePD ()
		{
			// 1 operand, 4 elements wide vector,
			// _mm256_permute_pd (0 .. 3) = (s1(0,1), s2(0,1), s3(2,3), s4(2,3))
			super ("_mm256_permute_pd", 1, 4, new Selector[] {
				new Selector (1, new int[] { 0, 1 }),
				new Selector (2, new int[] { 0, 1 }),
				new Selector (3, new int[] { 2, 3 }),
				new Selector (4, new int[] { 2, 3 })
			});
		}
		
		@Override
		public Expression getControlExpression (int[] rgConfig)
		{
			return new IntegerLiteral (rgConfig[0] | (rgConfig[1] << 1) | (rgConfig[2] << 2) | (rgConfig[3] << 3));
		}
	}
	
	private static class MMPermuteF128PD extends Operator
	{
		public MMPermuteF128PD ()
		{
			// 2 operands, 4 elements wide vector,
			// _mm256_shuffle2f128_pd ((0 .. 3), (4 .. 7)) = (s1(0,2,4,6), s1(1,3,5,7), s2(0,2,3,6), s2(1,3,5,7))
			super ("_mm256_permute2f128_pd", 2, 4, new Selector[] {
				new Selector (1, new int[] { 0, 2, 4, 6 }),
				new Selector (1, new int[] { 1, 3, 5, 7 }),
				new Selector (2, new int[] { 0, 2, 4, 6 }),
				new Selector (2, new int[] { 1, 3, 5, 7 })
			});
		}
		
		@Override
		public Expression getControlExpression (int[] rgConfig)
		{
			return new IntegerLiteral (rgConfig[0] | (rgConfig[1] << 4));
		}
	}
	
	private static class MMBlendPD extends Operator
	{
		public MMBlendPD ()
		{
			// 2 operands, 4 elements wide vector,
			// _mm256_shuffle_pd ((0 .. 3), (4 .. 7)) = (s1(0,4), s2(1,5), s3(2,6), s4(3,7))
			super ("_mm256_blend_pd", 2, 4, new Selector[] {
				new Selector (1, new int[] { 0, 4 }),
				new Selector (2, new int[] { 1, 5 }),
				new Selector (3, new int[] { 2, 6 }),
				new Selector (4, new int[] { 3, 7 })
			});
		}
		
		@Override
		public Expression getControlExpression (int[] rgConfig)
		{
			int nExpr = 0;
			for (int i = 0; i < 4; i++)
				nExpr |= (rgConfig[i] << i);
			return new IntegerLiteral (nExpr);
		}
	}

	
	public AVXSelectGenerator (int nVecLength, int nOffset, Operator[] rgOperators, String strOp1, String strOp2)
	{
		int[] rgIn1 = new int[nVecLength];
		int[] rgIn2 = new int[nVecLength];
		int[] rgOutput = new int[nVecLength];
		for (int i = 0; i < nVecLength; i++)
		{
			rgIn1[i] = i;
			rgIn2[i] = i + nVecLength;
			rgOutput[i] = i + nOffset;
		}
		
		List<Operand> listInput = new ArrayList<Operand> ();
		listInput.add (new Operand (strOp1, rgIn1));
		listInput.add (new Operand (strOp2, rgIn2));
		
		Operand opResult = new PermutatorGenetic (nVecLength, listInput, new Operand ("Out", rgOutput), rgOperators).findSequence ();
		
		System.out.println (opResult.getExpression ().toString ());
	}

	
	/**
	 * @param args
	 */
	public static void main (String[] args)
	{
		// single precision ("_ps")
		new AVXSelectGenerator (8, 1, new Operator[] { new MMShufflePS (), new MMShufflePD_ForPS (), new MMPermutePS (), new MMPermuteF128PS (), new MMBlendPS () }, "expr1", "expr2");
		
		// double precision ("_pd")
//		new AVXSelectGenerator (4, 3, new Operator[] { new MMShufflePD (), new MMPermutePD (), new MMPermuteF128PD (), new MMBlendPD () }, "expr1", "expr2");
	}
}
