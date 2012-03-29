package ch.unibas.cs.hpwc.patus.codegen.backend.openmp;

import cetus.hir.Expression;
import cetus.hir.Specifier;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;

public class OpenMPAVXAsmCodeGenerator extends OpenMPAVXCodeGenerator
{
	public OpenMPAVXAsmCodeGenerator (CodeGeneratorSharedObjects data)
	{
		super (data);
	}

	@Override
	public Expression unary_minus (Expression expr, Specifier specDatatype, boolean bVectorize)
	{
		return super.unary_minus (expr, specDatatype, bVectorize);
	}
}
