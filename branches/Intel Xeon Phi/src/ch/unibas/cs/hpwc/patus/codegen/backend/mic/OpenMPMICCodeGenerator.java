package ch.unibas.cs.hpwc.patus.codegen.backend.mic;

import java.util.ArrayList;
import java.util.List;

import cetus.hir.AnnotationStatement;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.Expression;
import cetus.hir.FunctionCall;
import cetus.hir.IntegerLiteral;
import cetus.hir.Literal;
import cetus.hir.NameID;
import cetus.hir.PointerSpecifier;
import cetus.hir.PragmaAnnotation;
import cetus.hir.Specifier;
import cetus.hir.Traversable;
import ch.unibas.cs.hpwc.patus.ast.StatementList;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;
import ch.unibas.cs.hpwc.patus.codegen.GlobalGeneratedIdentifiers;
import ch.unibas.cs.hpwc.patus.codegen.backend.openmp.OpenMPCodeGenerator;
import ch.unibas.cs.hpwc.patus.util.ASTUtil;
import ch.unibas.cs.hpwc.patus.util.CodeGeneratorUtil;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

public class OpenMPMICCodeGenerator extends OpenMPCodeGenerator
{
	private final static boolean BUILD_NATIVE = true;
	
	
	public OpenMPMICCodeGenerator (CodeGeneratorSharedObjects data)
	{
		super (data);
	}
	
	@Override
	public Expression shuffle (Expression expr1, Expression expr2, Specifier specDatatype, int nOffset)
	{
		// From the MIC instruction manual:
		//
		// "In conjunction with vloadunpacklpd, it allows unaligned
		// vector loads (that is, vector loads that are only element-wise, not vector-wise, aligned);
		// use a mask of 0xFF or no write-mask for this purpose. The typical instruction sequence
		// to perform an unaligned vector load would be:
		//
		// 		; assume memory location is pointed by register rax
		//		vloadunpacklpd v0 {k1}, [rax]
		//		vloadunpackhpd v0 {k1}, [rax+64]
		// "
		
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
		
		// _mm512_loadunpackhi_ps (_mm512_loadunpacklo_ps (x, &addr[nOffset]), &addr[nOffset + 64]))
		return new FunctionCall (
			new NameID (strLoadFnxNameHi),
			CodeGeneratorUtil.expressions (
				new FunctionCall (new NameID (strLoadFnxNameLo),
					CodeGeneratorUtil.expressions (
						expr1.clone (),
						new BinaryExpression (
							ASTUtil.castTo (ASTUtil.getPointerTo (expr1), CodeGeneratorUtil.specifiers (specDatatype, PointerSpecifier.UNQUALIFIED)),
							BinaryOperator.ADD,
							new IntegerLiteral (nOffset)
						)				
					)
				),
				new BinaryExpression (
					ASTUtil.castTo (ASTUtil.getPointerTo (expr1), CodeGeneratorUtil.specifiers (specDatatype, PointerSpecifier.UNQUALIFIED)),
					BinaryOperator.ADD,
					new IntegerLiteral (nOffset + 64)
				)				
			)
		);
	}
	
	@Override
	public Traversable splat (Expression expr, Specifier specDatatype)
	{
		if (expr instanceof Literal)
			return super.splat (expr, specDatatype);
		
		// the Intel compiler does not support __m512[d] struct initializations with values
		// if the values are no number literals
		
		// create a load intrinsic
		FunctionCall fc = (FunctionCall) createLoadInitializer (specDatatype, expr);
		
		// add the required arguments
		List<Expression> listArgs = new ArrayList<> (fc.getNumArguments () + 3);
		for (Object exprArg : fc.getArguments ())
			listArgs.add (((Expression) exprArg).clone ());
		listArgs.add (new NameID ("_MM_UPCONV_PS_NONE"));
		listArgs.add (new NameID ("_MM_BROADCAST32_NONE"));
		listArgs.add (new NameID ("_MM_HINT_NONE"));
		fc.setArguments (listArgs);
		
		return fc;
	}
	
	@Override
	protected String getVecLoadFunctionName (Specifier specDatatype)
	{
		String strFunction = null;
		if (Specifier.FLOAT.equals (specDatatype))
			strFunction = "_mm512_extload_ps";
		else if (Specifier.DOUBLE.equals (specDatatype))
			strFunction = "_mm512_extload_pd";
		
		return strFunction;
	}
	
	@Override
	protected boolean hasVecLoadFunctionPointerArg ()
	{
		return true;
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
		// generate no "offload" pragmas in native build mode
		if (BUILD_NATIVE)
			return new StatementList ();
				
		StringBuilder sbPragma = new StringBuilder ("offload target(mic) ");
		
		// TODO: individual treatment of in/out grids depending on the use case
		for (GlobalGeneratedIdentifiers.Variable varGrid : m_data.getData ().getGlobalGeneratedIdentifiers ().getVariables (
			GlobalGeneratedIdentifiers.EVariableType.INPUT_GRID.mask () | GlobalGeneratedIdentifiers.EVariableType.OUTPUT_GRID.mask ()))
		{
			sbPragma.append (strClause);
			sbPragma.append ('(');
			
			sbPragma.append (varGrid.getName ());
			sbPragma.append (":length(");
			sbPragma.append (varGrid.getBoxSize ().getVolume ().toString ());
			sbPragma.append (") alloc_if(");
			sbPragma.append (bAlloc ? 1 : 0);
			sbPragma.append (") free_if(");
			sbPragma.append (bFree ? 1 : 0);
			sbPragma.append (")) ");
		}
		
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
	
	public StatementList offloadMicCopyback ()
	{
		return createOffloadPragma ("out", false, false);
	}
	
	public StatementList deallocateMicGrids ()
	{
		return createOffloadPragma ("nocopy", false, true);
	}
}
