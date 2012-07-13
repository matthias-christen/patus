package ch.unibas.cs.hpwc.patus.codegen.backend.assembly.test;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.junit.Before;
import org.junit.Test;

import cetus.hir.Specifier;
import ch.unibas.cs.hpwc.patus.arch.IArchitectureDescription;
import ch.unibas.cs.hpwc.patus.arch.TypeArchitectureType.Intrinsics.Intrinsic;
import ch.unibas.cs.hpwc.patus.arch.TypeRegisterType;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.IOperand.Address;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.IOperand.InputRef;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.IOperand.PseudoRegister;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.Instruction;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.InstructionList;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.InstructionRegionScheduler;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.InstructionScheduler;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.analyze.DAGraph;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.analyze.DependenceAnalysis;

public class InstructionSchedulerTest
{
	private static class ArchDesc extends DummyArchitectureDescription
	{
		@Override
		public int getIssueRate ()
		{
			return 4;
		}
		
		@Override
		public Collection<Intrinsic> getIntrinsicsByIntrinsicName (String strIntrinsicName)
		{
			Intrinsic intrinsic = new Intrinsic ();
			
			intrinsic.setBaseName (strIntrinsicName);
			intrinsic.setName (strIntrinsicName);
			intrinsic.setDatatype ("float");
			
			if (strIntrinsicName.indexOf ("mov") >= 0)
				intrinsic.setLatency (1);
			else if (strIntrinsicName.indexOf ("add") >= 0 || strIntrinsicName.indexOf ("sub") >= 0)
				intrinsic.setLatency (3);
			else if (strIntrinsicName.indexOf ("mul") >= 0)
				intrinsic.setLatency (5);
			else if (strIntrinsicName.indexOf ("div") >= 0)
				intrinsic.setLatency (29);
			else
				intrinsic.setLatency (1);
			
			List<Intrinsic> l = new ArrayList<> ();
			l.add (intrinsic);
			return l;
		}
	}
	
	
	private InstructionList m_il;
	
	@Before
	public void setUp () throws Exception
	{
		m_il = new InstructionList ();
		InstructionList il = m_il;
		
		/*
		// WAVE: no unrolling
		Address op0 = new Address (new InputRef ("_constants_"), null, 1, 32);
		PseudoRegister op1 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vmovaps", op0, op1));
		Address op2 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), null, 1, 4);
		PseudoRegister op3 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vmovups", op2, op3));
		Address op4 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), null, 1, -4);
		PseudoRegister op5 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vaddps", op4, op3, op5));
		Address op6 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("(x_max*sizeof (float))"), 1, 0);
		il.addInstruction (new Instruction ("vaddps", op6, op5, op5));
		Address op7 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("((-1*x_max)*sizeof (float))"), 1, 0);
		il.addInstruction (new Instruction ("vaddps", op7, op5, op5));
		Address op8 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("((x_max*y_max)*sizeof (float))"), 1, 0);
		il.addInstruction (new Instruction ("vaddps", op8, op5, op5));
		Address op9 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("((-1*(x_max*y_max))*sizeof (float))"), 1, 0);
		il.addInstruction (new Instruction ("vaddps", op9, op5, op5));
		il.addInstruction (new Instruction ("vmulps", op1, op5, op1));
		Address op10 = new Address (new InputRef ("_constants_"), null, 1, 64);
		PseudoRegister op11 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vmovaps", op10, op11));
		Address op12 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), null, 1, 8);
		PseudoRegister op13 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vmovups", op12, op13));
		Address op14 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), null, 1, -8);
		PseudoRegister op15 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vaddps", op14, op13, op15));
		Address op16 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("(x_max*sizeof (float))"), 2, 0);
		il.addInstruction (new Instruction ("vaddps", op16, op15, op15));
		Address op17 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("((-1*x_max)*sizeof (float))"), 2, 0);
		il.addInstruction (new Instruction ("vaddps", op17, op15, op15));
		Address op18 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("((x_max*y_max)*sizeof (float))"), 2, 0);
		il.addInstruction (new Instruction ("vaddps", op18, op15, op15));
		Address op19 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("((-1*(x_max*y_max))*sizeof (float))"), 2, 0);
		il.addInstruction (new Instruction ("vaddps", op19, op15, op15));
		il.addInstruction (new Instruction ("vmulps", op11, op15, op11));
		il.addInstruction (new Instruction ("vaddps", op1, op11, op1));
		Address op20 = new Address (new InputRef ("_constants_"), null, 1, 0);
		PseudoRegister op21 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vmovaps", op20, op21));
		Address op22 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), null, 1, 0);
		il.addInstruction (new Instruction ("vmulps", op22, op21, op21));
		Address op23 = new Address (new InputRef ("u[t=-1^, s=(0, 0, 0)][0]"), null, 1, 0);
		il.addInstruction (new Instruction ("vsubps", op23, op21, op21));
		il.addInstruction (new Instruction ("vaddps", op21, op1, op21));
		Address op24 = new Address (new InputRef ("u[t=1^, s=(0, 0, 0)][0]"), null, 1, 0);
		il.addInstruction (new Instruction ("vmovaps", op21, op24));
		//*/
		
		//*
		// WAVE: 2x unrolling
		Address op0 = new Address (new InputRef ("_constants_"), null, 1, 32);
		PseudoRegister op1 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vmovaps", op0, op1));
		Address op2 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), null, 1, 4);
		PseudoRegister op3 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vmovups", op2, op3));
		Address op4 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), null, 1, -4);
		PseudoRegister op5 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vaddps", op4, op3, op5));
		Address op6 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), null, 1, 36);
		PseudoRegister op7 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vmovups", op6, op7));
		Address op8 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), null, 1, 28);
		PseudoRegister op9 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vaddps", op8, op7, op9));
		Address op10 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("(x_max*sizeof (float))"), 1, 0);
		il.addInstruction (new Instruction ("vaddps", op10, op5, op5));
		Address op11 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("(x_max*sizeof (float))"), 1, 32);
		il.addInstruction (new Instruction ("vaddps", op11, op9, op9));
		Address op12 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("((-1*x_max)*sizeof (float))"), 1, 0);
		il.addInstruction (new Instruction ("vaddps", op12, op5, op5));
		Address op13 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("((-1*x_max)*sizeof (float))"), 1, 32);
		il.addInstruction (new Instruction ("vaddps", op13, op9, op9));
		Address op14 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("((x_max*y_max)*sizeof (float))"), 1, 0);
		il.addInstruction (new Instruction ("vaddps", op14, op5, op5));
		Address op15 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("((x_max*y_max)*sizeof (float))"), 1, 32);
		il.addInstruction (new Instruction ("vaddps", op15, op9, op9));
		Address op16 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("((-1*(x_max*y_max))*sizeof (float))"), 1, 0);
		il.addInstruction (new Instruction ("vaddps", op16, op5, op5));
		Address op17 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("((-1*(x_max*y_max))*sizeof (float))"), 1, 32);
		il.addInstruction (new Instruction ("vaddps", op17, op9, op9));
		PseudoRegister op18 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vmulps", op1, op5, op18));
		PseudoRegister op19 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vmulps", op1, op9, op19));
		Address op20 = new Address (new InputRef ("_constants_"), null, 1, 64);
		PseudoRegister op21 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vmovaps", op20, op21));
		Address op22 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), null, 1, 8);
		PseudoRegister op23 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vmovups", op22, op23));
		Address op24 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), null, 1, -8);
		PseudoRegister op25 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vaddps", op24, op23, op25));
		Address op26 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), null, 2, 40);
		PseudoRegister op27 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vmovups", op26, op27));
		Address op28 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), null, 2, 24);
		PseudoRegister op29 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vaddps", op28, op27, op29));
		Address op30 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("(x_max*sizeof (float))"), 2, 0);
		il.addInstruction (new Instruction ("vaddps", op30, op25, op25));
		Address op31 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("(x_max*sizeof (float))"), 2, 32);
		il.addInstruction (new Instruction ("vaddps", op31, op29, op29));
		Address op32 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("((-1*x_max)*sizeof (float))"), 2, 0);
		il.addInstruction (new Instruction ("vaddps", op32, op25, op25));
		Address op33 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("((-1*x_max)*sizeof (float))"), 2, 32);
		il.addInstruction (new Instruction ("vaddps", op33, op29, op29));
		Address op34 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("((x_max*y_max)*sizeof (float))"), 2, 0);
		il.addInstruction (new Instruction ("vaddps", op34, op25, op25));
		Address op35 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("((x_max*y_max)*sizeof (float))"), 2, 32);
		il.addInstruction (new Instruction ("vaddps", op35, op29, op29));
		Address op36 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("((-1*(x_max*y_max))*sizeof (float))"), 2, 0);
		il.addInstruction (new Instruction ("vaddps", op36, op25, op25));
		Address op37 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("((-1*(x_max*y_max))*sizeof (float))"), 2, 32);
		il.addInstruction (new Instruction ("vaddps", op37, op29, op29));
		PseudoRegister op38 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vmulps", op21, op25, op38));
		PseudoRegister op39 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vmulps", op21, op29, op39));
		PseudoRegister op40 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vaddps", op18, op38, op40));
		PseudoRegister op41 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vaddps", op19, op39, op41));
		Address op42 = new Address (new InputRef ("_constants_"), null, 1, 0);
		PseudoRegister op43 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vmovaps", op42, op43));
		Address op44 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), null, 1, 0);
		PseudoRegister op45 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vmulps", op44, op43, op45));
		Address op46 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), null, 0, 32);
		PseudoRegister op47 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vmulps", op46, op43, op47));
		Address op48 = new Address (new InputRef ("u[t=-1^, s=(0, 0, 0)][0]"), null, 1, 0);
		PseudoRegister op49 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vsubps", op48, op45, op49));
		Address op50 = new Address (new InputRef ("u[t=-1^, s=(0, 0, 0)][0]"), null, 0, 32); 
		PseudoRegister op51 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vsubps", op50, op47, op51));
		PseudoRegister op52 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vaddps", op49, op40, op52));
		PseudoRegister op53 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vaddps", op51, op41, op53));
		Address op54 = new Address (new InputRef ("u[t=1^, s=(0, 0, 0)][0]"), null, 1, 0);
		il.addInstruction (new Instruction ("vmovaps", op52, op54));
		Address op55 = new Address (new InputRef ("u[t=1^, s=(0, 0, 0)][0]"), null, 0, 32);
		il.addInstruction (new Instruction ("vmovaps", op53, op55));
		//*/
		
		/*
		// WAVE: 4x unrolling
		Address op0 = new Address (new InputRef ("_constants_"), null, 1, 32);
		PseudoRegister op1 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vmovaps", op0, op1));
		Address op2 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), null, 1, 4);
		PseudoRegister op3 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vmovups", op2, op3));
		Address op4 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), null, 1, -4);
		PseudoRegister op5 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vaddps", op4, op3, op5));
		Address op6 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), null, 1, 36);
		PseudoRegister op7 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vmovups", op6, op7));
		Address op8 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), null, 1, 28);
		PseudoRegister op9 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vaddps", op8, op7, op9));
		Address op10 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), null, 1, 68);
		PseudoRegister op11 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vmovups", op10, op11));
		Address op12 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), null, 1, 60);
		PseudoRegister op13 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vaddps", op12, op11, op13));
		Address op14 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), null, 1, 100);
		PseudoRegister op15 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vmovups", op14, op15));
		Address op16 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), null, 1, 92);
		PseudoRegister op17 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vaddps", op16, op15, op17));
		Address op18 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("(x_max*sizeof (float))"), 1, 0);
		il.addInstruction (new Instruction ("vaddps", op18, op5, op5));
		Address op19 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("(x_max*sizeof (float))"), 1, 32);
		il.addInstruction (new Instruction ("vaddps", op19, op9, op9));
		Address op20 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("(x_max*sizeof (float))"), 1, 64);
		il.addInstruction (new Instruction ("vaddps", op20, op13, op13));
		Address op21 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("(x_max*sizeof (float))"), 1, 96);
		il.addInstruction (new Instruction ("vaddps", op21, op17, op17));
		Address op22 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("((-1*x_max)*sizeof (float))"), 1, 0);
		il.addInstruction (new Instruction ("vaddps", op22, op5, op5));
		Address op23 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("((-1*x_max)*sizeof (float))"), 1, 32);
		il.addInstruction (new Instruction ("vaddps", op23, op9, op9));
		Address op24 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("((-1*x_max)*sizeof (float))"), 1, 64);
		il.addInstruction (new Instruction ("vaddps", op24, op13, op13));
		Address op25 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("((-1*x_max)*sizeof (float))"), 1, 96);
		il.addInstruction (new Instruction ("vaddps", op25, op17, op17));
		Address op26 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("((x_max*y_max)*sizeof (float))"), 1, 0);
		il.addInstruction (new Instruction ("vaddps", op26, op5, op5));
		Address op27 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("((x_max*y_max)*sizeof (float))"), 1, 32);
		il.addInstruction (new Instruction ("vaddps", op27, op9, op9));
		Address op28 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("((x_max*y_max)*sizeof (float))"), 1, 64);
		il.addInstruction (new Instruction ("vaddps", op28, op13, op13));
		Address op29 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("((x_max*y_max)*sizeof (float))"), 1, 96);
		il.addInstruction (new Instruction ("vaddps", op29, op17, op17));
		Address op30 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("((-1*(x_max*y_max))*sizeof (float))"), 1, 0);
		il.addInstruction (new Instruction ("vaddps", op30, op5, op5));
		Address op31 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("((-1*(x_max*y_max))*sizeof (float))"), 1, 32);
		il.addInstruction (new Instruction ("vaddps", op31, op9, op9));
		Address op32 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("((-1*(x_max*y_max))*sizeof (float))"), 1, 64);
		il.addInstruction (new Instruction ("vaddps", op32, op13, op13));
		Address op33 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("((-1*(x_max*y_max))*sizeof (float))"), 1, 96);
		il.addInstruction (new Instruction ("vaddps", op33, op17, op17));
		PseudoRegister op34 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vmulps", op1, op5, op34));
		PseudoRegister op35 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vmulps", op1, op9, op35));
		PseudoRegister op36 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vmulps", op1, op13, op36));
		PseudoRegister op37 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vmulps", op1, op17, op37));
		Address op38 = new Address (new InputRef ("_constants_"), null, 1, 64);
		PseudoRegister op39 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vmovaps", op38, op39));
		Address op40 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), null, 1, 8);
		PseudoRegister op41 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vmovups", op40, op41));
		Address op42 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), null, 1, -8);
		PseudoRegister op43 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vaddps", op42, op41, op43));
		Address op44 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), null, 2, 40);
		PseudoRegister op45 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vmovups", op44, op45));
		Address op46 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), null, 2, 24);
		PseudoRegister op47 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vaddps", op46, op45, op47));
		Address op48 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), null, 2, 72);
		PseudoRegister op49 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vmovups", op48, op49));
		Address op50 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), null, 2, 56);
		PseudoRegister op51 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vaddps", op50, op49, op51));
		Address op52 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), null, 2, 104);
		PseudoRegister op53 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vmovups", op52, op53));
		Address op54 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), null, 2, 88);
		PseudoRegister op55 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vaddps", op54, op53, op55));
		Address op56 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("(x_max*sizeof (float))"), 2, 0);
		il.addInstruction (new Instruction ("vaddps", op56, op43, op43));
		Address op57 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("(x_max*sizeof (float))"), 2, 32);
		il.addInstruction (new Instruction ("vaddps", op57, op47, op47));
		Address op58 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("(x_max*sizeof (float))"), 2, 64);
		il.addInstruction (new Instruction ("vaddps", op58, op51, op51));
		Address op59 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("(x_max*sizeof (float))"), 2, 96);
		il.addInstruction (new Instruction ("vaddps", op59, op55, op55));
		Address op60 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("((-1*x_max)*sizeof (float))"), 2, 0);
		il.addInstruction (new Instruction ("vaddps", op60, op43, op43));
		Address op61 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("((-1*x_max)*sizeof (float))"), 2, 32);
		il.addInstruction (new Instruction ("vaddps", op61, op47, op47));
		Address op62 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("((-1*x_max)*sizeof (float))"), 2, 64);
		il.addInstruction (new Instruction ("vaddps", op62, op51, op51));
		Address op63 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("((-1*x_max)*sizeof (float))"), 2, 96);
		il.addInstruction (new Instruction ("vaddps", op63, op55, op55));
		Address op64 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("((x_max*y_max)*sizeof (float))"), 2, 0);
		il.addInstruction (new Instruction ("vaddps", op64, op43, op43));
		Address op65 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("((x_max*y_max)*sizeof (float))"), 2, 32);
		il.addInstruction (new Instruction ("vaddps", op65, op47, op47));
		Address op66 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("((x_max*y_max)*sizeof (float))"), 2, 64);
		il.addInstruction (new Instruction ("vaddps", op66, op51, op51));
		Address op67 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("((x_max*y_max)*sizeof (float))"), 2, 96);
		il.addInstruction (new Instruction ("vaddps", op67, op55, op55));
		Address op68 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("((-1*(x_max*y_max))*sizeof (float))"), 2, 0);
		il.addInstruction (new Instruction ("vaddps", op68, op43, op43));
		Address op69 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("((-1*(x_max*y_max))*sizeof (float))"), 2, 32);
		il.addInstruction (new Instruction ("vaddps", op69, op47, op47));
		Address op70 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("((-1*(x_max*y_max))*sizeof (float))"), 2, 64);
		il.addInstruction (new Instruction ("vaddps", op70, op51, op51));
		Address op71 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("((-1*(x_max*y_max))*sizeof (float))"), 2, 96);
		il.addInstruction (new Instruction ("vaddps", op71, op55, op55));
		PseudoRegister op72 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vmulps", op39, op43, op72));
		PseudoRegister op73 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vmulps", op39, op47, op73));
		PseudoRegister op74 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vmulps", op39, op51, op74));
		PseudoRegister op75 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vmulps", op39, op55, op75));
		PseudoRegister op76 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vaddps", op34, op72, op76));
		PseudoRegister op77 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vaddps", op35, op73, op77));
		PseudoRegister op78 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vaddps", op36, op74, op78));
		PseudoRegister op79 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vaddps", op37, op75, op79));
		Address op80 = new Address (new InputRef ("_constants_"), null, 1, 0);
		PseudoRegister op81 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vmovaps", op80, op81));
		Address op82 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), null, 1, 0);
		PseudoRegister op83 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vmulps", op82, op81, op83));
		Address op84 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), null, 0, 32);
		PseudoRegister op85 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vmulps", op84, op81, op85));
		Address op86 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), null, 0, 64);
		PseudoRegister op87 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vmulps", op86, op81, op87));
		Address op88 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), null, 0, 96);
		PseudoRegister op89 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vmulps", op88, op81, op89));
		Address op90 = new Address (new InputRef ("u[t=-1^, s=(0, 0, 0)][0]"), null, 1, 0);
		PseudoRegister op91 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vsubps", op90, op83, op91));
		Address op92 = new Address (new InputRef ("u[t=-1^, s=(0, 0, 0)][0]"), null, 0, 32);
		PseudoRegister op93 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vsubps", op92, op85, op93));
		Address op94 = new Address (new InputRef ("u[t=-1^, s=(0, 0, 0)][0]"), null, 0, 64);
		PseudoRegister op95 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vsubps", op94, op87, op95));
		Address op96 = new Address (new InputRef ("u[t=-1^, s=(0, 0, 0)][0]"), null, 0, 96);
		PseudoRegister op97 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vsubps", op96, op89, op97));
		PseudoRegister op98 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vaddps", op91, op76, op98));
		PseudoRegister op99 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vaddps", op93, op77, op99));
		PseudoRegister op100 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vaddps", op95, op78, op100));
		PseudoRegister op101 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vaddps", op97, op79, op101));
		Address op102 = new Address (new InputRef ("u[t=1^, s=(0, 0, 0)][0]"), null, 1, 0);
		il.addInstruction (new Instruction ("vmovaps", op98, op102));
		Address op103 = new Address (new InputRef ("u[t=1^, s=(0, 0, 0)][0]"), null, 0, 32);
		il.addInstruction (new Instruction ("vmovaps", op99, op103));
		Address op104 = new Address (new InputRef ("u[t=1^, s=(0, 0, 0)][0]"), null, 0, 64);
		il.addInstruction (new Instruction ("vmovaps", op100, op104));
		Address op105 = new Address (new InputRef ("u[t=1^, s=(0, 0, 0)][0]"), null, 0, 96);
		il.addInstruction (new Instruction ("vmovaps", op101, op105));
		//*/
		
		/*
		// LAPLACIAN: 2x unrolling
		Address op0 = new Address (new InputRef ("_constants_"), null, 1, 0);
		PseudoRegister op1 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vmovaps", op0, op1));
		Address op2 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), null, 1, 0);
		PseudoRegister op3 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vmulps", op2, op1, op3));
		Address op4 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), null, 0, 32);
		PseudoRegister op5 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vmulps", op4, op1, op5));
		Address op6 = new Address (new InputRef ("_constants_"), null, 1, 32);
		PseudoRegister op7 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vmovaps", op6, op7));
		Address op8 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), null, 1, 4);
		PseudoRegister op9 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vmovups", op8, op9));
		Address op10 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), null, 1, -4);
		PseudoRegister op11 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vaddps", op10, op9, op11));
		Address op12 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), null, 1, 36);
		PseudoRegister op13 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vmovups", op12, op13));
		Address op14 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), null, 1, 28);
		PseudoRegister op15 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vaddps", op14, op13, op15));
		Address op16 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("((x_max+2)*sizeof (float))"), 1, 0);
		il.addInstruction (new Instruction ("vaddps", op16, op11, op11));
		Address op17 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("((x_max+2)*sizeof (float))"), 1, 32);
		il.addInstruction (new Instruction ("vaddps", op17, op15, op15));
		Address op18 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("((-1*(x_max+2))*sizeof (float))"), 1, 0);
		il.addInstruction (new Instruction ("vaddps", op18, op11, op11));
		Address op19 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("((-1*(x_max+2))*sizeof (float))"), 1, 32);
		il.addInstruction (new Instruction ("vaddps", op19, op15, op15));
		Address op20 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("(((x_max+2)*(y_max+2))*sizeof (float))"), 1, 0);
		il.addInstruction (new Instruction ("vaddps", op20, op11, op11));
		Address op21 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("(((x_max+2)*(y_max+2))*sizeof (float))"), 1, 32);
		il.addInstruction (new Instruction ("vaddps", op21, op15, op15));
		Address op22 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("((-1*((x_max+2)*(y_max+2)))*sizeof (float))"), 1, 0);
		il.addInstruction (new Instruction ("vaddps", op22, op11, op11));
		Address op23 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("((-1*((x_max+2)*(y_max+2)))*sizeof (float))"), 1, 32);
		il.addInstruction (new Instruction ("vaddps", op23, op15, op15));
		PseudoRegister op24 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vmulps", op7, op11, op24));
		PseudoRegister op25 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vmulps", op7, op15, op25));
		PseudoRegister op26 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vaddps", op3, op24, op26));
		PseudoRegister op27 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vaddps", op5, op25, op27));
		Address op28 = new Address (new InputRef ("u[t=1^, s=(0, 0, 0)][0]"), null, 1, 0);
		il.addInstruction (new Instruction ("vmovaps", op26, op28));
		Address op29 = new Address (new InputRef ("u[t=1^, s=(0, 0, 0)][0]"), null, 0, 32);
		il.addInstruction (new Instruction ("vmovaps", op27, op29));
		//*/
		
		/*
		Address op0 = new Address (new InputRef ("_constants_"), null, 1, 0);
		PseudoRegister op1 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vmovaps", op0, op1));
		Address op2 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), null, 1, 0);
		PseudoRegister op3 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vmulps", op2, op1, op3));
		Address op4 = new Address (new InputRef ("_constants_"), null, 1, 32);
		PseudoRegister op5 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vmovaps", op4, op5));
		Address op6 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), null, 1, 4);
		PseudoRegister op7 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vmovups", op6, op7));
		Address op8 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), null, 1, -4);
		PseudoRegister op9 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vaddps", op8, op7, op9));
		Address op10 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("((x_max+2)*sizeof (float))"), 1, 0);
		il.addInstruction (new Instruction ("vaddps", op10, op9, op9));
		Address op11 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("((-1*(x_max+2))*sizeof (float))"), 1, 0);
		il.addInstruction (new Instruction ("vaddps", op11, op9, op9));
		Address op12 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("(((x_max+2)*(y_max+2))*sizeof (float))"), 1, 0);
		il.addInstruction (new Instruction ("vaddps", op12, op9, op9));
		Address op13 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("((-1*((x_max+2)*(y_max+2)))*sizeof (float))"), 1, 0);
		il.addInstruction (new Instruction ("vaddps", op13, op9, op9));
		PseudoRegister op14 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vmulps", op5, op9, op14));
		PseudoRegister op15 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vaddps", op3, op14, op15));
		Address op16 = new Address (new InputRef ("u[t=1^, s=(0, 0, 0)][0]"), null, 1, 0);
		il.addInstruction (new Instruction ("vmovaps", op15, op16));
		//*/
	}

	@Test
	public void testSchedule ()
	{
		InstructionRegionScheduler.DEBUG = true;
		IArchitectureDescription arch = new ArchDesc ();
		DAGraph graph = new DependenceAnalysis (m_il, arch).run (Specifier.FLOAT);
		graph.graphviz ();
		InstructionScheduler is = new InstructionScheduler (graph, arch);
		System.out.println (is.schedule ());
	}
}
