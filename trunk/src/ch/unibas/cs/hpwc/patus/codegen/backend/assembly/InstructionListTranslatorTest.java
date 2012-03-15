package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import java.io.File;
import java.math.BigInteger;
import java.util.List;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import cetus.hir.BinaryOperator;
import cetus.hir.FunctionCall;
import cetus.hir.Specifier;
import cetus.hir.Statement;
import cetus.hir.UnaryOperator;
import ch.unibas.cs.hpwc.patus.arch.IArchitectureDescription;
import ch.unibas.cs.hpwc.patus.arch.TypeArchitectureType.Assembly;
import ch.unibas.cs.hpwc.patus.arch.TypeArchitectureType.Build;
import ch.unibas.cs.hpwc.patus.arch.TypeArchitectureType.Intrinsics.Intrinsic;
import ch.unibas.cs.hpwc.patus.arch.TypeBaseIntrinsicEnum;
import ch.unibas.cs.hpwc.patus.arch.TypeDeclspec;
import ch.unibas.cs.hpwc.patus.arch.TypeRegister;
import ch.unibas.cs.hpwc.patus.arch.TypeRegisterClass;
import ch.unibas.cs.hpwc.patus.arch.TypeRegisterType;
import ch.unibas.cs.hpwc.patus.codegen.Globals;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

public class InstructionListTranslatorTest
{
	///////////////////////////////////////////////////////////////////
	// Inner Types
	
	private static class ArchDescProto implements IArchitectureDescription
	{
		private String m_strBinaryOpArgs;
		
		public ArchDescProto (String strBinaryOpArgs)
		{
			m_strBinaryOpArgs = strBinaryOpArgs;
		}
		
		@Override
		public String getBackend ()
		{
			// TODO Auto-generated method stub
			return null;
		}

		@Override
		public String getInnermostLoopCodeGenerator ()
		{
			// TODO Auto-generated method stub
			return null;
		}

		@Override
		public String getGeneratedFileSuffix ()
		{
			// TODO Auto-generated method stub
			return null;
		}

		@Override
		public boolean useFunctionPointers ()
		{
			// TODO Auto-generated method stub
			return false;
		}

		@Override
		public int getNumberOfParallelLevels ()
		{
			// TODO Auto-generated method stub
			return 0;
		}

		@Override
		public boolean hasExplicitLocalDataCopies (int nParallelismLevel)
		{
			// TODO Auto-generated method stub
			return false;
		}

		@Override
		public boolean supportsAsynchronousIO (int nParallelismLevel)
		{
			// TODO Auto-generated method stub
			return false;
		}

		@Override
		public Statement getBarrier (int nParallelismLevel)
		{
			// TODO Auto-generated method stub
			return null;
		}

		@Override
		public List<Specifier> getType (Specifier specType)
		{
			// TODO Auto-generated method stub
			return null;
		}

		@Override
		public boolean useSIMD ()
		{
			// TODO Auto-generated method stub
			return false;
		}

		@Override
		public int getSIMDVectorLength (Specifier specType)
		{
			// TODO Auto-generated method stub
			return 0;
		}

		@Override
		public int getAlignmentRestriction (Specifier specType)
		{
			// TODO Auto-generated method stub
			return 0;
		}

		@Override
		public boolean supportsUnalignedSIMD ()
		{
			// TODO Auto-generated method stub
			return false;
		}

		@Override
		public List<Specifier> getDeclspecs (TypeDeclspec type)
		{
			// TODO Auto-generated method stub
			return null;
		}

		@Override
		public Intrinsic getIntrinsic (UnaryOperator op, Specifier specType)
		{
			return getIntrinsic (Globals.getIntrinsicBase (op).value (), specType);
		}

		@Override
		public Intrinsic getIntrinsic (BinaryOperator op, Specifier specType)
		{
			return getIntrinsic (Globals.getIntrinsicBase (op).value (), specType);
		}

		@Override
		public Intrinsic getIntrinsic (FunctionCall fnx, Specifier specType)
		{
			return getIntrinsic (fnx.getName ().toString (), specType);
		}

		@Override
		public Intrinsic getIntrinsic (String strOperation, Specifier specType)
		{
			if (TypeBaseIntrinsicEnum.PLUS.value ().equals (strOperation) || TypeBaseIntrinsicEnum.MINUS.value ().equals (strOperation))
				return createIntrinsic (strOperation, m_strBinaryOpArgs);
			if (TypeBaseIntrinsicEnum.MOVE_FPR.value ().equals (strOperation))
				return createIntrinsic (strOperation, "reg/mem,reg/mem");
			
			return null;
		}

		@Override
		public Assembly getAssemblySpec ()
		{
			// TODO Auto-generated method stub
			return null;
		}

		@Override
		public int getRegistersCount (TypeRegisterType nType)
		{
			// TODO Auto-generated method stub
			return 0;
		}

		@Override
		public Iterable<TypeRegisterClass> getRegisterClasses (TypeRegisterType nType)
		{
			// TODO Auto-generated method stub
			return null;
		}
		
		@Override
		public List<String> getIncludeFiles ()
		{
			// TODO Auto-generated method stub
			return null;
		}

		@Override
		public Build getBuild ()
		{
			// TODO Auto-generated method stub
			return null;
		}

		@Override
		public File getFile ()
		{
			// TODO Auto-generated method stub
			return null;
		}
		
		@Override
		public IArchitectureDescription clone ()
		{
			return new ArchDescProto (m_strBinaryOpArgs);
		}
	}
	

	///////////////////////////////////////////////////////////////////
	// Member Variables
	
	private final static TypeRegisterClass REGCLASS_SIMD = new TypeRegisterClass ();
	
	static
	{
		REGCLASS_SIMD.setType (TypeRegisterType.SIMD);
		REGCLASS_SIMD.setBitrange ("0..127");
		REGCLASS_SIMD.setName ("xmm");
		REGCLASS_SIMD.setSubregisterOf ("");
		REGCLASS_SIMD.setWidth (new BigInteger ("128"));
	}
	
	private IArchitectureDescription m_arch1, m_arch2, m_arch3, m_arch4, m_arch5, m_arch6, m_arch7;
	private InstructionList m_il;

	
	///////////////////////////////////////////////////////////////////
	// Implementation

	private static IOperand createRegister (String strRegName)
	{
		TypeRegister reg = new TypeRegister ();
		reg.setName (strRegName);
		reg.setClazz (REGCLASS_SIMD);
		return new IOperand.Register (reg);
	}
	
	private static IOperand createMemAccess (String strRegBaseName)
	{
		return new IOperand.Address ((IOperand.IRegisterOperand) createRegister (strRegBaseName));
	}
	
	private static Intrinsic createIntrinsic (String strName, String strArgs)
	{
		Intrinsic i = new Intrinsic ();
		i.setBaseName (strName);
		i.setName (StringUtil.concat ("_", strName, "_"));
		i.setArguments (strArgs);
		i.setDatatype (Specifier.FLOAT.toString ());		
		return i;
	}

	@Before
	public void setUp () throws Exception
	{
		m_arch1 = new ArchDescProto ("reg/mem,reg,=reg");
		m_arch2 = new ArchDescProto ("reg,reg/mem,=reg");
		m_arch3 = new ArchDescProto ("reg,reg,=reg");
		m_arch4 = new ArchDescProto ("reg,reg,=reg/mem");
		m_arch5 = new ArchDescProto ("reg/mem,=reg/mem");
		m_arch6 = new ArchDescProto ("reg/mem,=reg");
		m_arch7 = new ArchDescProto ("reg,=reg");
		
		m_il = new InstructionList ();
		
		m_il.addInstruction (new Instruction (TypeBaseIntrinsicEnum.PLUS, createRegister ("ymm0"), createRegister ("ymm1"), createRegister ("ymm1")));
		m_il.addInstruction (new Instruction (TypeBaseIntrinsicEnum.PLUS, createRegister ("ymm0"), createRegister ("ymm1"), createRegister ("ymm2")));
		m_il.addInstruction (new Instruction (TypeBaseIntrinsicEnum.PLUS, createMemAccess ("rax"), createRegister ("ymm1"), createRegister ("ymm2")));
		m_il.addInstruction (new Instruction (TypeBaseIntrinsicEnum.PLUS, createRegister ("ymm0"), createMemAccess ("rax"), createRegister ("ymm2")));
		m_il.addInstruction (new Instruction (TypeBaseIntrinsicEnum.PLUS, createRegister ("ymm0"), createRegister ("ymm1"), createMemAccess ("rbx")));
		m_il.addInstruction (new Instruction (TypeBaseIntrinsicEnum.PLUS, createMemAccess ("rax"), createRegister ("ymm1"), createMemAccess ("rbx")));
		m_il.addInstruction (new Instruction (TypeBaseIntrinsicEnum.PLUS, createMemAccess ("rax"), createMemAccess ("rbx"), createMemAccess ("rcx")));
		
		m_il.addInstruction (new Instruction (TypeBaseIntrinsicEnum.MINUS, createRegister ("ymm0"), createRegister ("ymm1"), createRegister ("ymm2")));
		m_il.addInstruction (new Instruction (TypeBaseIntrinsicEnum.MINUS, createMemAccess ("rax"), createRegister ("ymm1"), createRegister ("ymm2")));
		m_il.addInstruction (new Instruction (TypeBaseIntrinsicEnum.MINUS, createRegister ("ymm0"), createMemAccess ("rax"), createRegister ("ymm2")));
		
		//m_il.addInstruction (new Instruction (TypeBaseIntrinsicEnum.FMA, createRegister ("ymm0"), createMemAccess ("rax"), createRegister ("ymm2")));

		System.out.println (m_il.toString ());
		System.out.println ("===>\n");
	}
	
	@Test
	public void testTranslate_1 ()
	{		
		IOperand.PseudoRegister.reset ();
		InstructionList ilResult = InstructionListTranslator.translate (m_arch1, m_il, Specifier.FLOAT);
		System.out.println (new Throwable ().getStackTrace ()[0].toString () + ":");
		System.out.println (ilResult.toString ());
		
		Assert.assertEquals ("",
			"_plus_ %%ymm0, %%ymm1, %%ymm1\\n\\t\n" +

			"_plus_ %%ymm0, %%ymm1, %%ymm2\\n\\t\n" +
		
			"_plus_ (%%rax), %%ymm1, %%ymm2\\n\\t\n" +
			
			"_plus_ (%%rax), %%ymm0, %%ymm2\\n\\t\n" +
			
			"_plus_ %%ymm0, %%ymm1, {pseudoreg-0}\\n\\t\n" +
			"_move-fpr_ {pseudoreg-0}, (%%rbx)\\n\\t\n" +
			
			"_plus_ (%%rax), %%ymm1, {pseudoreg-1}\\n\\t\n" +
			"_move-fpr_ {pseudoreg-1}, (%%rbx)\\n\\t\n" +
			
			"_move-fpr_ (%%rbx), {pseudoreg-2}\\n\\t\n" +
			"_plus_ (%%rax), {pseudoreg-2}, {pseudoreg-3}\\n\\t\n" +
			"_move-fpr_ {pseudoreg-3}, (%%rcx)\\n\\t\n" +

			"_minus_ %%ymm0, %%ymm1, %%ymm2\\n\\t\n" +
				
			"_minus_ (%%rax), %%ymm1, %%ymm2\\n\\t\n" +
			
			"_move-fpr_ (%%rax), {pseudoreg-4}\\n\\t\n" +
			"_minus_ %%ymm0, {pseudoreg-4}, %%ymm2\\n\\t\n",

			ilResult.toString ());
	}

	@Test
	public void testTranslate_2 ()
	{
		IOperand.PseudoRegister.reset ();
		InstructionList ilResult = InstructionListTranslator.translate (m_arch2, m_il, Specifier.FLOAT);
		System.out.println (new Throwable ().getStackTrace ()[0].toString () + ":");
		System.out.println (ilResult.toString ());
		
		Assert.assertEquals ("",
			"_plus_ %%ymm0, %%ymm1, %%ymm1\\n\\t\n" +

			"_plus_ %%ymm0, %%ymm1, %%ymm2\\n\\t\n" +
		
			"_plus_ %%ymm1, (%%rax), %%ymm2\\n\\t\n" +
			
			"_plus_ %%ymm0, (%%rax), %%ymm2\\n\\t\n" +
			
			"_plus_ %%ymm0, %%ymm1, {pseudoreg-0}\\n\\t\n" +
			"_move-fpr_ {pseudoreg-0}, (%%rbx)\\n\\t\n" +
			
			"_plus_ %%ymm1, (%%rax), {pseudoreg-1}\\n\\t\n" +
			"_move-fpr_ {pseudoreg-1}, (%%rbx)\\n\\t\n" +
			
			"_move-fpr_ (%%rax), {pseudoreg-2}\\n\\t\n" +
			"_plus_ {pseudoreg-2}, (%%rbx), {pseudoreg-3}\\n\\t\n" +
			"_move-fpr_ {pseudoreg-3}, (%%rcx)\\n\\t\n" +

			"_minus_ %%ymm0, %%ymm1, %%ymm2\\n\\t\n" +
				
			"_move-fpr_ (%%rax), {pseudoreg-4}\\n\\t\n" +
			"_minus_ {pseudoreg-4}, %%ymm1, %%ymm2\\n\\t\n" +
			
			"_minus_ %%ymm0, (%%rax), %%ymm2\\n\\t\n",

			ilResult.toString ());
	}

	@Test
	public void testTranslate_3 ()
	{		
		IOperand.PseudoRegister.reset ();
		InstructionList ilResult = InstructionListTranslator.translate (m_arch3, m_il, Specifier.FLOAT);
		System.out.println (new Throwable ().getStackTrace ()[0].toString () + ":");
		System.out.println (ilResult.toString ());
		
		Assert.assertEquals ("",
			"_plus_ %%ymm0, %%ymm1, %%ymm1\\n\\t\n" +
		
			"_plus_ %%ymm0, %%ymm1, %%ymm2\\n\\t\n" +
		
			"_move-fpr_ (%%rax), {pseudoreg-0}\\n\\t\n" +
			"_plus_ {pseudoreg-0}, %%ymm1, %%ymm2\\n\\t\n" +
			
			"_move-fpr_ (%%rax), {pseudoreg-1}\\n\\t\n" +
			"_plus_ %%ymm0, {pseudoreg-1}, %%ymm2\\n\\t\n" +
			
			"_plus_ %%ymm0, %%ymm1, {pseudoreg-2}\\n\\t\n" +
			"_move-fpr_ {pseudoreg-2}, (%%rbx)\\n\\t\n" +
			
			"_move-fpr_ (%%rax), {pseudoreg-3}\\n\\t\n" +
			"_plus_ {pseudoreg-3}, %%ymm1, {pseudoreg-4}\\n\\t\n" +
			"_move-fpr_ {pseudoreg-4}, (%%rbx)\\n\\t\n" +
			
			"_move-fpr_ (%%rax), {pseudoreg-5}\\n\\t\n" +
			"_move-fpr_ (%%rbx), {pseudoreg-6}\\n\\t\n" +
			"_plus_ {pseudoreg-5}, {pseudoreg-6}, {pseudoreg-7}\\n\\t\n" +
			"_move-fpr_ {pseudoreg-7}, (%%rcx)\\n\\t\n" +

			"_minus_ %%ymm0, %%ymm1, %%ymm2\\n\\t\n" +
				
			"_move-fpr_ (%%rax), {pseudoreg-8}\\n\\t\n" +
			"_minus_ {pseudoreg-8}, %%ymm1, %%ymm2\\n\\t\n" +
			
			"_move-fpr_ (%%rax), {pseudoreg-9}\\n\\t\n" +
			"_minus_ %%ymm0, {pseudoreg-9}, %%ymm2\\n\\t\n",

			ilResult.toString ());
	}

	@Test
	public void testTranslate_4 ()
	{		
		IOperand.PseudoRegister.reset ();
		InstructionList ilResult = InstructionListTranslator.translate (m_arch4, m_il, Specifier.FLOAT);
		System.out.println (new Throwable ().getStackTrace ()[0].toString () + ":");
		System.out.println (ilResult.toString ());
		
		Assert.assertEquals ("",
			"_plus_ %%ymm0, %%ymm1, %%ymm1\\n\\t\n" +
		
			"_plus_ %%ymm0, %%ymm1, %%ymm2\\n\\t\n" +
		
			"_move-fpr_ (%%rax), {pseudoreg-0}\\n\\t\n" +
			"_plus_ {pseudoreg-0}, %%ymm1, %%ymm2\\n\\t\n" +
			
			"_move-fpr_ (%%rax), {pseudoreg-1}\\n\\t\n" +
			"_plus_ %%ymm0, {pseudoreg-1}, %%ymm2\\n\\t\n" +
			
			"_plus_ %%ymm0, %%ymm1, (%%rbx)\\n\\t\n" +
			
			"_move-fpr_ (%%rax), {pseudoreg-2}\\n\\t\n" +
			"_plus_ {pseudoreg-2}, %%ymm1, (%%rbx)\\n\\t\n" +
			
			"_move-fpr_ (%%rax), {pseudoreg-3}\\n\\t\n" +
			"_move-fpr_ (%%rbx), {pseudoreg-4}\\n\\t\n" +
			"_plus_ {pseudoreg-3}, {pseudoreg-4}, (%%rcx)\\n\\t\n" +

			"_minus_ %%ymm0, %%ymm1, %%ymm2\\n\\t\n" +
				
			"_move-fpr_ (%%rax), {pseudoreg-5}\\n\\t\n" +
			"_minus_ {pseudoreg-5}, %%ymm1, %%ymm2\\n\\t\n" +
			
			"_move-fpr_ (%%rax), {pseudoreg-6}\\n\\t\n" +
			"_minus_ %%ymm0, {pseudoreg-6}, %%ymm2\\n\\t\n",

			ilResult.toString ());
	}

	@Test
	public void testTranslate_5 ()
	{		
		IOperand.PseudoRegister.reset ();
		InstructionList ilResult = InstructionListTranslator.translate (m_arch5, m_il, Specifier.FLOAT);
		System.out.println (new Throwable ().getStackTrace ()[0].toString () + ":");
		System.out.println (ilResult.toString ());
		
		Assert.assertEquals ("",			
			"_plus_ %%ymm0, %%ymm1\\n\\t\n" +

			"_move-fpr_ %%ymm1, %%ymm2\\n\\t\n" +
			"_plus_ %%ymm0, %%ymm2\\n\\t\n" +
		
			"_move-fpr_ %%ymm1, %%ymm2\\n\\t\n" +
			"_plus_ (%%rax), %%ymm2\\n\\t\n" +
			
			"_move-fpr_ (%%rax), %%ymm2\\n\\t\n" +
			"_plus_ %%ymm0, %%ymm2\\n\\t\n" +
			
			"_move-fpr_ %%ymm1, {pseudoreg-0}\\n\\t\n" +
			"_plus_ %%ymm0, {pseudoreg-0}\\n\\t\n" +
			"_move-fpr_ {pseudoreg-0}, (%%rbx)\\n\\t\n" +
			
			"_move-fpr_ %%ymm1, {pseudoreg-1}\\n\\t\n" +
			"_plus_ (%%rax), {pseudoreg-1}\\n\\t\n" +
			"_move-fpr_ {pseudoreg-1}, (%%rbx)\\n\\t\n" +
			
			"_move-fpr_ (%%rbx), {pseudoreg-2}\\n\\t\n" +
			"_plus_ (%%rax), {pseudoreg-2}\\n\\t\n" +
			"_move-fpr_ {pseudoreg-2}, (%%rcx)\\n\\t\n" +

			"_move-fpr_ %%ymm1, %%ymm2\\n\\t\n" +
			"_minus_ %%ymm0, %%ymm2\\n\\t\n" +
				
			"_move-fpr_ %%ymm1, %%ymm2\\n\\t\n" +
			"_minus_ (%%rax), %%ymm2\\n\\t\n" +
			
			"_move-fpr_ (%%rax), %%ymm2\\n\\t\n" +
			"_minus_ %%ymm0, %%ymm2\\n\\t\n",

			ilResult.toString ());
	}

	@Test
	public void testTranslate_6 ()
	{		
		IOperand.PseudoRegister.reset ();
		InstructionList ilResult = InstructionListTranslator.translate (m_arch6, m_il, Specifier.FLOAT);
		System.out.println (new Throwable ().getStackTrace ()[0].toString () + ":");
		System.out.println (ilResult.toString ());
		
		Assert.assertEquals ("",			
			"_plus_ %%ymm0, %%ymm1\\n\\t\n" +

			"_move-fpr_ %%ymm1, %%ymm2\\n\\t\n" +
			"_plus_ %%ymm0, %%ymm2\\n\\t\n" +
		
			"_move-fpr_ %%ymm1, %%ymm2\\n\\t\n" +
			"_plus_ (%%rax), %%ymm2\\n\\t\n" +
			
//			"_move-fpr_ (%%rax), %%ymm2\\n\\t\n" +
//			"_plus_ %%ymm0, %%ymm2\\n\\t\n" +
			"_move-fpr_ %%ymm0, %%ymm2\\n\\t\n" +		// is OK, too
			"_plus_ (%%rax), %%ymm2\\n\\t\n" +
			
			"_move-fpr_ %%ymm1, {pseudoreg-0}\\n\\t\n" +
			"_plus_ %%ymm0, {pseudoreg-0}\\n\\t\n" +
			"_move-fpr_ {pseudoreg-0}, (%%rbx)\\n\\t\n" +
			
			"_move-fpr_ %%ymm1, {pseudoreg-1}\\n\\t\n" +
			"_plus_ (%%rax), {pseudoreg-1}\\n\\t\n" +
			"_move-fpr_ {pseudoreg-1}, (%%rbx)\\n\\t\n" +
			
			"_move-fpr_ (%%rbx), {pseudoreg-2}\\n\\t\n" +
			"_plus_ (%%rax), {pseudoreg-2}\\n\\t\n" +
			"_move-fpr_ {pseudoreg-2}, (%%rcx)\\n\\t\n" +

			"_move-fpr_ %%ymm1, %%ymm2\\n\\t\n" +
			"_minus_ %%ymm0, %%ymm2\\n\\t\n" +
				
			"_move-fpr_ %%ymm1, %%ymm2\\n\\t\n" +
			"_minus_ (%%rax), %%ymm2\\n\\t\n" +
			
			"_move-fpr_ (%%rax), %%ymm2\\n\\t\n" +
			"_minus_ %%ymm0, %%ymm2\\n\\t\n",

			ilResult.toString ());
	}

	@Test
	public void testTranslate_7 ()
	{		
		IOperand.PseudoRegister.reset ();
		InstructionList ilResult = InstructionListTranslator.translate (m_arch7, m_il, Specifier.FLOAT);
		System.out.println (new Throwable ().getStackTrace ()[0].toString () + ":");
		System.out.println (ilResult.toString ());
		
		Assert.assertEquals ("",			
			"_plus_ %%ymm0, %%ymm1\\n\\t\n" +

			"_move-fpr_ %%ymm1, %%ymm2\\n\\t\n" +
			"_plus_ %%ymm0, %%ymm2\\n\\t\n" +
		
			"_move-fpr_ %%ymm1, %%ymm2\\n\\t\n" +
			"_move-fpr_ (%%rax), {pseudoreg-0}\\n\\t\n" +
			"_plus_ {pseudoreg-0}, %%ymm2\\n\\t\n" +
			
			"_move-fpr_ (%%rax), %%ymm2\\n\\t\n" +
			"_plus_ %%ymm0, %%ymm2\\n\\t\n" +
			
			"_move-fpr_ %%ymm1, {pseudoreg-1}\\n\\t\n" +
			"_plus_ %%ymm0, {pseudoreg-1}\\n\\t\n" +
			"_move-fpr_ {pseudoreg-1}, (%%rbx)\\n\\t\n" +
			
			"_move-fpr_ %%ymm1, {pseudoreg-2}\\n\\t\n" +
			"_move-fpr_ (%%rax), {pseudoreg-3}\\n\\t\n" +
			"_plus_ {pseudoreg-3}, {pseudoreg-2}\\n\\t\n" +
			"_move-fpr_ {pseudoreg-2}, (%%rbx)\\n\\t\n" +
			
			"_move-fpr_ (%%rbx), {pseudoreg-4}\\n\\t\n" +
			"_move-fpr_ (%%rax), {pseudoreg-5}\\n\\t\n" +
			"_plus_ {pseudoreg-5}, {pseudoreg-4}\\n\\t\n" +
			"_move-fpr_ {pseudoreg-4}, (%%rcx)\\n\\t\n" +

			"_move-fpr_ %%ymm1, %%ymm2\\n\\t\n" +
			"_minus_ %%ymm0, %%ymm2\\n\\t\n" +
				
			"_move-fpr_ %%ymm1, %%ymm2\\n\\t\n" +
			"_move-fpr_ (%%rax), {pseudoreg-6}\\n\\t\n" +
			"_minus_ {pseudoreg-6}, %%ymm2\\n\\t\n" +
			
			"_move-fpr_ (%%rax), %%ymm2\\n\\t\n" +
			"_minus_ %%ymm0, %%ymm2\\n\\t\n",

			ilResult.toString ());
	}
}
