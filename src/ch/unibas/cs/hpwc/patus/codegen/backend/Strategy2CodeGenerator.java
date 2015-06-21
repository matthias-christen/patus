package ch.unibas.cs.hpwc.patus.codegen.backend;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import cetus.hir.AccessExpression;
import cetus.hir.AccessOperator;
import cetus.hir.ArrayAccess;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.FunctionCall;
import cetus.hir.Identifier;
import cetus.hir.IntegerLiteral;
import cetus.hir.NameID;
import cetus.hir.UnaryExpression;
import cetus.hir.UnaryOperator;
import cetus.hir.VariableDeclaration;
import cetus.hir.VariableDeclarator;
import ch.unibas.cs.hpwc.patus.ast.StatementList;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;
import ch.unibas.cs.hpwc.patus.codegen.GlobalGeneratedIdentifiers.EVariableType;
import ch.unibas.cs.hpwc.patus.codegen.GlobalGeneratedIdentifiers.Variable;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.StencilAssemblySection;
import ch.unibas.cs.hpwc.patus.codegen.backend.openmp.OpenMPAVXCodeGenerator;
import ch.unibas.cs.hpwc.patus.geometry.Point;
import ch.unibas.cs.hpwc.patus.geometry.Size;
import ch.unibas.cs.hpwc.patus.representation.Index;
import ch.unibas.cs.hpwc.patus.representation.IndexSetUtil;
import ch.unibas.cs.hpwc.patus.representation.Stencil;
import ch.unibas.cs.hpwc.patus.util.ASTUtil;

/**
 * Non-kernel function extensions for the Strategy2 benchmarking harness.
 * 
 * @author Matthias-M. Christen
 */
public class Strategy2CodeGenerator extends OpenMPAVXCodeGenerator
{
	private static class SlopeDescriptor
	{
		public String m_strFunctionName;
		public int[] m_rgArgs;
		
		public SlopeDescriptor (String strFunctionName, int... rgArgs)
		{
			m_strFunctionName = strFunctionName;
			m_rgArgs = rgArgs;
		}
		
		public Expression getExpression ()
		{
			List<Expression> listArgs = new ArrayList<Expression> (m_rgArgs.length);
			for (int i = 0; i < m_rgArgs.length; i++)
				listArgs.add (new IntegerLiteral (m_rgArgs[i]));
			return new FunctionCall (new NameID (m_strFunctionName), listArgs);
		}
	}
	
	
	public Strategy2CodeGenerator (CodeGeneratorSharedObjects data)
	{
		super(data);
	}
	
	/**
	 * Creates a list of arguments to the <code>initialize</code> and the <code>kernel</code>
	 * functions. Takes care of alignment restrictions.
	 * @return List of function arguments to <code>initialize</code> and <code>kernel</code>
	 */
	private List<Expression> getFunctionArguments (
		List<Variable> listVariables,
		String strGridName, String strTminName, String strTmaxName, String strXminName, String strXmaxName, String strSlopesName,
		StatementList sl, boolean bForceAlign)
	{
		// get alignment restrictions
		int nAlignRestrict = m_mixinNonKernelFunctions.getGlobalAlignmentRestriction (listVariables);
		
		int nGridIdx = -1;
		int nSizeIdx = -1;
		int nSlopeIdx = -1;
		
		NameID nidGrid = new NameID (strGridName);
		NameID nidXmin = new NameID (strXminName);
		NameID nidXmax = new NameID (strXmaxName);
		NameID nidSlopes = new NameID (strSlopesName);
		
		int nDim = m_data.getStencilCalculation ().getDimensionality ();
		
		SlopeDescriptor[] rgSlopeDescriptors = new SlopeDescriptor[(nDim + 1) * nDim];
		int nIdx = 0;
		for (int i = 0; i < nDim; i++)
		{
			for (int j = i + 1; j < nDim; j++)
			{
				rgSlopeDescriptors[nIdx++] = new SlopeDescriptor ("negative", new int[] { i, j });
				rgSlopeDescriptors[nIdx++] = new SlopeDescriptor ("positive", new int[] { i, j });
			}

			rgSlopeDescriptors[nIdx++] = new SlopeDescriptor ("t_negative", new int[] { i });
			rgSlopeDescriptors[nIdx++] = new SlopeDescriptor ("t_positive", new int[] { i });
		}

		// build the list of arguments; adjust the pointers so that the alignment restrictions are satisfied
		List<Expression> listArgs = new ArrayList<> ();
		for (Variable v : listVariables)
		{
			switch (v.getType ())
			{
			case INPUT_GRID:
				nGridIdx++;
				listArgs.add (new UnaryExpression (
					UnaryOperator.DEREFERENCE,
					new ArrayAccess (
						nidGrid.clone (),
						new BinaryExpression(
							nGridIdx == 0 ?
								new NameID (strTminName) :
								new BinaryExpression (new NameID (strTminName), BinaryOperator.ADD, new IntegerLiteral (nGridIdx)),
							BinaryOperator.MODULUS,
							new IntegerLiteral (getNumGrids ())
						)
					)
				));
				break;
				
			case OUTPUT_GRID:
				Expression exprId = getExpressionForVariable (v);
				if (exprId instanceof Identifier)
					sl.addDeclaration (new VariableDeclaration (ASTUtil.dereference (v.getSpecifiers ()), (VariableDeclarator) ((Identifier) exprId).getSymbol ()));
				listArgs.add (m_mixinNonKernelFunctions.getArgumentExpressionFromVariable (v, sl, nAlignRestrict, bForceAlign));
				break;
				
			case TRAPEZOIDAL_SIZE:
				nSizeIdx++;
				listArgs.add (new ArrayAccess (
					(nSizeIdx % 2) == 0 ? nidXmin.clone () : nidXmax.clone (),
					new IntegerLiteral(nSizeIdx / 2)
				));
				break;
				
			case TRAPEZOIDAL_TMAX:
				listArgs.add (new BinaryExpression (
					new NameID (strTmaxName),
					BinaryOperator.SUBTRACT,
					new NameID (strTminName)
				));
				break;
				
			case TRAPEZOIDAL_SLOPE:
				nSlopeIdx++;
				listArgs.add (new AccessExpression (
					nidSlopes.clone (),
					AccessOperator.MEMBER_ACCESS,
					rgSlopeDescriptors[nSlopeIdx].getExpression ()
				));
				break;
				
			default:
				listArgs.add (m_mixinNonKernelFunctions.getArgumentExpressionFromVariable (v, sl, nAlignRestrict, bForceAlign));
			}
		}
		
		return listArgs;
	}
	
	public StatementList computeStencil (String strGridName, String strTminName, String strTmaxName, String strXminName, String strXmaxName, String strSlopesName)
	{
		StatementList sl = new StatementList ();
		List<Expression> listArgs = getFunctionArguments (
			m_data.getData ().getGlobalGeneratedIdentifiers ().getVariables (~EVariableType.INTERNAL_NONKERNEL_AUTOTUNE_PARAMETER.mask ()),
			strGridName, strTminName, strTmaxName, strXminName, strXmaxName, strSlopesName,
			sl, true
		);

		sl.addStatement (new ExpressionStatement (new FunctionCall (
			m_data.getData ().getGlobalGeneratedIdentifiers ().getStencilFunctionName ().clone (),
			listArgs
		)));

		return sl;
	}
	
	public StatementList computePoint ()
	{
		return new StatementList ();
	}

	public String getStencilName ()
	{
		String strName = m_data.getStencilCalculation ().getName ();
		
		StringBuilder sb = new StringBuilder();
		sb.append (Character.toUpperCase(strName.charAt (0)));
		sb.append (strName.substring (1));
		sb.append ("Stencil");
		
		return sb.toString ();
	}
	
	public Integer getStencilDimensionality ()
	{
		return new Integer (m_data.getStencilCalculation ().getDimensionality ());
	}
	
	public String getStencilFptype ()
	{
		return StencilAssemblySection.getDatatype (m_data.getStencilCalculation ()).toString ();
	}
	
	public String getGridSize ()
	{
		Variable varGrid = m_data.getData ().getGlobalGeneratedIdentifiers ().getInputGrids ().iterator ().next ();
		Size size = varGrid.getBoxSize();
		
		StringBuilder sb = new StringBuilder ();
		for (int i = 0; i < size.getDimensionality (); i++)
		{
			if (i > 0)
				sb.append (", ");
			sb.append (size.getCoord (i));
		}
		
		return sb.toString ();
	}
	
	public String getDomainMin ()
	{
		Point ptMin = m_data.getStencilCalculation ().getDomainSize ().getMin ();
		
		StringBuilder sb = new StringBuilder ();
		for (int i = 0; i < ptMin.getDimensionality (); i++)
		{
			if (i > 0)
				sb.append(", ");
			sb.append (ptMin.getCoord (i));
		}
		
		return sb.toString ();
	}
	
	public String getDomainMax ()
	{
		Point ptMin = m_data.getStencilCalculation ().getDomainSize ().getMax ();
		
		StringBuilder sb = new StringBuilder ();
		for (int i = 0; i < ptMin.getDimensionality (); i++)
		{
			if (i > 0)
				sb.append(", ");
			sb.append (ptMin.getCoord (i));
		}
		
		return sb.toString ();
	}
	
	public String getBoundingboxMin ()
	{
		int[] min = m_data.getStencilCalculation ().getStencilBundle ().getFusedStencil ().getMinSpaceIndex ();
		
		StringBuilder sb = new StringBuilder ();
		for (int i = 0; i < min.length; i++)
		{
			if (i > 0)
				sb.append (", ");
			sb.append (min[i]);
		}
		
		return sb.toString ();
	}
	
	public String getBoundingboxMax ()
	{
		int[] max = m_data.getStencilCalculation ().getStencilBundle ().getFusedStencil ().getMaxSpaceIndex ();
		
		StringBuilder sb = new StringBuilder ();
		for (int i = 0; i < max.length; i++)
		{
			if (i > 0)
				sb.append (", ");
			sb.append (max[i]);
		}
		
		return sb.toString ();
	}
		
	public Long getNumTimesteps ()
	{
		Expression exprMaxIter = m_data.getStencilCalculation ().getMaxIterations ();
		if (exprMaxIter instanceof IntegerLiteral)
			return new Long (((IntegerLiteral) exprMaxIter).getValue ());
		
		throw new RuntimeException ("Number of timesteps must be a constant");
	}
	
	public Integer getNumGrids ()
	{
		Stencil stencil = m_data.getStencilCalculation ().getStencilBundle ().getFusedStencil ();
		return IndexSetUtil.getMaxTimeIndex ((Set<Index>) stencil.getOutputIndices ()) - stencil.getMinTimeIndex () + 1;
	}
}
