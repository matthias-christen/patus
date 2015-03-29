package ch.unibas.cs.hpwc.patus.codegen;

import cetus.hir.AssignmentExpression;
import cetus.hir.AssignmentOperator;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.CompoundStatement;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.ForLoop;
import cetus.hir.Identifier;
import cetus.hir.IntegerLiteral;
import cetus.hir.NameID;
import cetus.hir.Specifier;
import cetus.hir.Traversable;
import cetus.hir.UnaryExpression;
import cetus.hir.UnaryOperator;
import cetus.hir.VariableDeclarator;
import ch.unibas.cs.hpwc.patus.ast.RangeIterator;
import ch.unibas.cs.hpwc.patus.ast.StatementListBundle;
import ch.unibas.cs.hpwc.patus.ast.SubdomainIdentifier;
import ch.unibas.cs.hpwc.patus.ast.SubdomainIterator;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;
import ch.unibas.cs.hpwc.patus.geometry.Border;
import ch.unibas.cs.hpwc.patus.geometry.Box;
import ch.unibas.cs.hpwc.patus.geometry.Size;
import ch.unibas.cs.hpwc.patus.geometry.Subdomain;
import ch.unibas.cs.hpwc.patus.geometry.Vector;
import ch.unibas.cs.hpwc.patus.util.CodeGeneratorUtil;
import ch.unibas.cs.hpwc.patus.util.StatementListBundleUtil;

public class TrapezoidCodeGenerator implements ICodeGenerator
{
	/**
	 * The shared data
	 */
	private CodeGeneratorSharedObjects m_data;
	
	private SubdomainIdentifier m_sdidBase;
	private SubdomainIdentifier m_sdidIterator;
	private SubdomainIterator m_sdIterator;
	private Identifier m_idTemporalIdx;
	
	/**
	 * x/y x/z x/t
	 * y/z y/t
	 * z/t
	 */
	private Identifier[][][] m_rgSlopes;
	
	
	public TrapezoidCodeGenerator(CodeGeneratorSharedObjects data)
	{
		m_data = data;
		
		byte nDim = m_data.getStencilCalculation().getDimensionality();
				
		m_sdidBase = m_data.getStrategy ().getBaseDomain ().clone ();

		m_sdidIterator = new SubdomainIdentifier ("v", new Subdomain (
			m_sdidBase.getSubdomain (),
			Subdomain.ESubdomainType.POINT,
			new Size (Vector.getOnesVector (m_sdidBase.getDimensionality ()))));

		m_sdIterator = new SubdomainIterator(m_sdidIterator, m_sdidBase, new Border(nDim), 1, null, null, 0);
		
		// construct the bounds and slope variables
		m_rgSlopes = new Identifier[nDim][][];
		for (int i = 0; i < nDim; i++)
		{
			String strIdI = CodeGeneratorUtil.getDimensionName(i);
			m_rgSlopes[i] = new Identifier[nDim - i][];
			
			for (int j = 0; j < nDim - i; j++)
			{
				String strIdJ = j == nDim - i - 1 ? "t" : CodeGeneratorUtil.getDimensionName(i + j + 1);
				
				VariableDeclarator declA = new VariableDeclarator (Specifier.INT, new NameID ("__slope_" + strIdI + "_" + strIdJ + "_a"));
				VariableDeclarator declB = new VariableDeclarator (Specifier.INT, new NameID ("__slope_" + strIdI + "_" + strIdJ + "_b"));
				
				m_rgSlopes[i][j] = new Identifier[2];
				m_rgSlopes[i][j][0] = new Identifier(declA);
				m_rgSlopes[i][j][1] = new Identifier(declB);
			}
		}
	}
	
	@Override
	public StatementListBundle generate(Traversable trvInput, CodeGeneratorRuntimeOptions options)
	{
		StatementListBundle slbGenerated = new StatementListBundle ();
		
		// add function parameters

		m_idTemporalIdx = new Identifier (new VariableDeclarator (Specifier.INT, new NameID ("__t")));

		StatementListBundle slbLoopBody = generateIteratorForDimension((byte) (m_data.getStencilCalculation().getDimensionality() - 1), options);
		
		// generate the time loop
		RangeIterator itTemporal = new RangeIterator(m_idTemporalIdx, new IntegerLiteral (0), new NameID ("__tmin"), new IntegerLiteral (1), null, 0);
		return m_data.getCodeGenerators ().getLoopCodeGenerator ().generate (itTemporal, itTemporal.getStart (), slbLoopBody, slbGenerated, options);
		
		/*
		VariableDeclarator declIdxT = new VariableDeclarator (CodeGeneratorUtil.createNameID ("t__"));
		Identifier idIdxT = new Identifier (declIdxT);
			
		slbLoopNest.addStatement(new ForLoop(
			new ExpressionStatement(new AssignmentExpression(idIdxT.clone(), AssignmentOperator.NORMAL, new IntegerLiteral(0))),
			new BinaryExpression(idIdxT.clone(), BinaryOperator.COMPARE_LE, new NameID("X")),
			new UnaryExpression(UnaryOperator.PRE_INCREMENT, idIdxT.clone()),
			new CompoundStatement()
		));
		*/
		
		/*
		StencilCalculation stencil = m_data.getStencilCalculation();
		for (int i = 0; i < stencil.getDimensionality(); i++)
		{
			slbLoopNest.addStatement(new ForLoop(init, condition, step, body));
		}
		*/
		
		// return slbGenerated;
	}
	
	protected StatementListBundle generateIteratorForDimension (byte nDimension, CodeGeneratorRuntimeOptions options)
	{	
		StatementListBundle slbGenerated = new StatementListBundle ();

		if (nDimension == 0)
			;//slbGenerated.addStatements(m_data.getCodeGenerators ().getInnermostLoopCodeGenerator().generate(m_sdIterator, options));
		else
		{		
			Identifier idIdx = m_data.getData ().getGeneratedIdentifiers ().getDimensionIndexIdentifier (m_sdidIterator, nDimension);
			
			// initialize start and end variables for the next dimension
			m_data.getCodeGenerators().getConstantGeneratedIdentifiers().createDeclarator(m_sdidIterator.getName(), Specifier.INT, false, 0);
			
			slbGenerated.addStatement(new ExpressionStatement(new AssignmentExpression(
				m_data.getData().getGeneratedIdentifiers().getDimensionMinIdentifier(m_sdidIterator, nDimension - 1).clone(),
				AssignmentOperator.NORMAL,
				generateBoundInitialization((byte) (nDimension - 1), 0))));
			slbGenerated.addStatement(new ExpressionStatement(new AssignmentExpression(
				m_data.getData().getGeneratedIdentifiers().getDimensionMaxIdentifier(m_sdidIterator, nDimension - 1).clone(),
				AssignmentOperator.NORMAL,
				generateBoundInitialization((byte) (nDimension - 1), 1))));
	
			// generate the nested loop
			StatementListBundle slbLoopBody = generateIteratorForDimension((byte) (nDimension - 1), options);
			slbGenerated.addStatement(new ForLoop(
				new ExpressionStatement(new AssignmentExpression(
					idIdx.clone(),
					AssignmentOperator.NORMAL,
					m_data.getData().getGeneratedIdentifiers().getDimensionMinIdentifier(m_sdidIterator, nDimension).clone()
				)),
				new BinaryExpression(
					idIdx.clone(),
					BinaryOperator.COMPARE_LE,
					m_data.getData().getGeneratedIdentifiers().getDimensionMaxIdentifier(m_sdidIterator, nDimension).clone()
				),
				new UnaryExpression(UnaryOperator.PRE_INCREMENT, idIdx.clone()),
				new CompoundStatement()
			));
			StatementListBundleUtil.addToLoopBody(slbGenerated, slbLoopBody);
		}
		
		// generate the increment statements
		slbGenerated.addStatement(new ExpressionStatement(new AssignmentExpression(
			m_data.getData().getGeneratedIdentifiers().getDimensionMinIdentifier(m_sdidBase, nDimension).clone(),
			AssignmentOperator.ADD,
			m_rgSlopes[nDimension][0][0].clone())));
		slbGenerated.addStatement(new ExpressionStatement(new AssignmentExpression(
			m_data.getData().getGeneratedIdentifiers().getDimensionMaxIdentifier(m_sdidBase, nDimension).clone(),
			AssignmentOperator.ADD,
			m_rgSlopes[nDimension][0][1].clone())));
		
		return slbGenerated;
	}
	
	/**
	 * var x_start = xmin[0] + t * s_xt_a + (z - xmin[2]) * s_xz_a + (y_start - xmin[1]) * s_xy_a;
	 * 
	 * @param nDim
	 * @param nIdx
	 * @return
	 */
	protected Expression generateBoundInitialization (byte nDim, int nIdx)
	{
		byte nDimensionality = m_data.getStencilCalculation().getDimensionality();
		Box box = m_sdIterator.getDomainIdentifier ().getSubdomain ().getBox ();
		
		Expression exprResult = new BinaryExpression(
			nIdx == 0 ?
				box.getMin ().getCoord (nDim).clone () :
				box.getMax ().getCoord (nDim).clone (),
			BinaryOperator.ADD,
			new BinaryExpression(
				m_idTemporalIdx.clone(),
				BinaryOperator.MULTIPLY,
				m_rgSlopes[nDim][nDimensionality - nDim - 1][nIdx].clone()
			)
		);	
		
		for (int i = nDimensionality - 1; i > nDim; i--)
		{
			exprResult = new BinaryExpression(
				exprResult,
				BinaryOperator.ADD,
				new BinaryExpression(
					new BinaryExpression(
						i == nDim + 1 ?
							m_data.getData ().getGeneratedIdentifiers ().getDimensionMinIdentifier(m_sdidIterator, i).clone() :
							m_data.getData ().getGeneratedIdentifiers ().getDimensionIndexIdentifier (m_sdidIterator, i).clone(),
						BinaryOperator.SUBTRACT,
						nIdx == 0 ?
							box.getMin ().getCoord (i).clone () :
							box.getMax ().getCoord (i).clone ()
					),
					BinaryOperator.MULTIPLY,
					m_rgSlopes[nDim][i - nDim - 1][nIdx].clone()
				)
			);
		}
		
		return exprResult;
	}
}
