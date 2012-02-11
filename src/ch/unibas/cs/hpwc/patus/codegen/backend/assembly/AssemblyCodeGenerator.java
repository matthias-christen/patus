package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.SomeExpression;
import cetus.hir.Specifier;
import cetus.hir.Statement;
import ch.unibas.cs.hpwc.patus.arch.TypeRegister;
import ch.unibas.cs.hpwc.patus.arch.TypeRegisterType;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;
import ch.unibas.cs.hpwc.patus.codegen.backend.IBackendAssemblyCodeGenerator;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 * <p>Creates inline assembly sections.</p>
 * <p>Instructions are added to the section by invoking the
 * {@link AssemblyCodeGenerator#issueInstruction(Instruction, Specifier, StringBuilder)}
 * method. After all instructions have been added, call the
 * {@link AssemblyCodeGenerator#generate(CodeGeneratorRuntimeOptions)}
 * method.</p>
 * <p>Note that this class is not thread-safe.</p>
 * 
 * @author Matthias-M. Christen
 */
public class AssemblyCodeGenerator implements IBackendAssemblyCodeGenerator
{
	///////////////////////////////////////////////////////////////////
	// Member Variables

	private CodeGeneratorSharedObjects m_data;
	
	
	///////////////////////////////////////////////////////////////////
	// Implementation

	public AssemblyCodeGenerator (CodeGeneratorSharedObjects data)
	{
		m_data = data;
		
	}
	
	@Override
	public Statement generate (CodeGeneratorRuntimeOptions nOptions)
	{
		// TODO Auto-generated method stub
		return null;
	}
	
//	@Override
//	public Statement generate (CodeGeneratorRuntimeOptions options)
//	{
//		// create a C statement wrapping the inline assembly
//		StringBuilder sbClobberedRegisters = new StringBuilder ();
//		for (TypeRegister reg : m_setClobberedRegisters)
//		{
//			if (sbClobberedRegisters.length () > 0)
//				sbClobberedRegisters.append (",");
//			sbClobberedRegisters.append (reg.getName ());
//		}
//		
//		return new ExpressionStatement (new SomeExpression (
//			StringUtil.concat (
//				"__asm__ __volatile__ (\n",
//				sb.toString (),
//				":\n",
//				":\n",
//				":", sbClobberedRegisters.toString (), "\n",
//				")"
//			), null));
//	}
//	
//	@Override
//	public void issueInstruction (Instruction instr, Specifier specDatatype, StringBuilder sbResult)
//	{
//		instr.issue (specDatatype, sbResult);
//	}
	
}
