package ch.unibas.cs.hpwc.patus.codegen.backend.mic;

import cetus.hir.AnnotationStatement;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.Expression;
import cetus.hir.FunctionCall;
import cetus.hir.IntegerLiteral;
import cetus.hir.NameID;
import cetus.hir.PointerSpecifier;
import cetus.hir.PragmaAnnotation;
import cetus.hir.Specifier;
import ch.unibas.cs.hpwc.patus.ast.StatementList;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;
import ch.unibas.cs.hpwc.patus.codegen.GlobalGeneratedIdentifiers;
import ch.unibas.cs.hpwc.patus.codegen.backend.openmp.OpenMPCodeGenerator;
import ch.unibas.cs.hpwc.patus.util.ASTUtil;
import ch.unibas.cs.hpwc.patus.util.CodeGeneratorUtil;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

public class OpenMPMICCodeGenerator extends OpenMPCodeGenerator
{
	public OpenMPMICCodeGenerator (CodeGeneratorSharedObjects data)
	{
		super (data);
	}
	
	@Override
	public Expression shuffle (Expression expr1, Expression expr2, Specifier specDatatype, int nOffset)
	{
		String strLoadFnxNameHi = null;
		String strLoadFnxNameLo = null;

		if (specDatatype.equals (Specifier.FLOAT))
		{
			strLoadFnxNameHi = "_mm512_loadunpackhi_ps";
			strLoadFnxNameLo = "_mm512_loadunpacklo_ps";
		}
		else if (specDatatype.equals (Specifier.DOUBLE))
		{
			strLoadFnxNameHi = "_mm512_loadunpackhi_pd";
			strLoadFnxNameLo = "_mm512_loadunpacklo_pd";
		}
		else
			throw new RuntimeException (StringUtil.concat ("Unsupported datatype: ", specDatatype.toString ()));
		
		// _mm512_loadunpackhi_ps (_mm512_loadunpacklo_ps (x, &addr[nOffset]), &addr[nOffset + 1/2 vec_width]))

		return new FunctionCall (
			new NameID (strLoadFnxNameHi),
			CodeGeneratorUtil.expressions (new BinaryExpression (
				ASTUtil.castTo (ASTUtil.getPointerTo (expr1), CodeGeneratorUtil.specifiers (specDatatype, PointerSpecifier.UNQUALIFIED)),
				BinaryOperator.ADD,
				new IntegerLiteral (nOffset)
			))
		);
	}
	
	/**
	 * Creates the pragma
	 * <pre>
	 * 	#pragma offload target(mic) {clause}({grid}:length{grid_size} alloc_if({0|1}) free_if({0|1}), ...)
	 * </pre>
	 * @param strClause
	 * @param bAlloc
	 * @param bFree
	 * @return
	 */
	private StatementList createOffloadPragma (String strClause, boolean bAlloc, boolean bFree)
	{
		StringBuilder sbPragma = new StringBuilder ("offload target(mic) ");
		sbPragma.append (strClause);
		sbPragma.append ('(');
		
		boolean bFirst = true;
		for (GlobalGeneratedIdentifiers.Variable varGrid : m_data.getData ().getGlobalGeneratedIdentifiers ().getInputGrids ())
		{
			if (!bFirst)
				sbPragma.append (", ");
			
			sbPragma.append (varGrid.getName ());
			sbPragma.append (":length(");
			sbPragma.append (varGrid.getSize ().toString ());
			sbPragma.append (") alloc_if(");
			sbPragma.append (bAlloc ? 1 : 0);
			sbPragma.append (") free_if(");
			sbPragma.append (bFree ? 1 : 0);
			sbPragma.append (')');
		}
		sbPragma.append (')');
		
		return new StatementList (new AnnotationStatement (new PragmaAnnotation (sbPragma.toString ())));
	}
	
	public StatementList offloadMicAllocate ()
	{
		return createOffloadPragma ("in", true, false);
	}
	
	public StatementList offloadMic ()
	{
		return createOffloadPragma ("nocopy", false, false);
	}
	
	public StatementList deallocateMicGrids ()
	{
		return createOffloadPragma ("out", false, true);
	}
}
