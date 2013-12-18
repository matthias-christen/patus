package ch.unibas.cs.hpwc.patus.codegen.backend.intel;

import static org.junit.Assert.*;

import org.junit.Test;

import cetus.hir.Specifier;

public class TestIntelXeonCodeGenerator {

	@Test
	public void TestgetVecLoadFunctionNameFloat() {
		IntelXeonCodeGenerator cg = new IntelXeonCodeGenerator(null);
		
		assertEquals(cg.getVecLoadFunctionName(Specifier.FLOAT),
				"_mm512_extload_ps");
	}
	
	@Test
	public void TestgetVecLoadFunctionNameDouble() {
		IntelXeonCodeGenerator cg = new IntelXeonCodeGenerator(null);
		
		assertEquals(cg.getVecLoadFunctionName(Specifier.DOUBLE),
				"_mm512_extload_pd");
	}

	@Test
	public void TesthasVecLoadFunctionPointerArg(){
		IntelXeonCodeGenerator cg = new IntelXeonCodeGenerator(null);
		assertEquals(true, cg.hasVecLoadFunctionPointerArg());
	}
	

}
