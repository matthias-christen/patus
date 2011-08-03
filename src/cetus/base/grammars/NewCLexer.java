// $ANTLR 2.7.7 (20100309) jdk1.5.0_22: "NewCParser.g" -> "NewCLexer.java"$

package cetus.base.grammars;

import java.io.InputStream;
import antlr.TokenStreamException;
import antlr.TokenStreamIOException;
import antlr.TokenStreamRecognitionException;
import antlr.CharStreamException;
import antlr.CharStreamIOException;
import antlr.ANTLRException;
import java.io.Reader;
import java.util.Hashtable;
import antlr.CharScanner;
import antlr.InputBuffer;
import antlr.ByteBuffer;
import antlr.CharBuffer;
import antlr.Token;
import antlr.CommonToken;
import antlr.RecognitionException;
import antlr.NoViableAltForCharException;
import antlr.MismatchedCharException;
import antlr.TokenStream;
import antlr.ANTLRHashString;
import antlr.LexerSharedInputState;
import antlr.collections.impl.BitSet;
import antlr.SemanticException;

import java.io.*;
import antlr.*;

public class NewCLexer extends antlr.CharScanner implements NEWCTokenTypes, TokenStream
 {

public void initialize(String src)
{
  setOriginalSource(src);
  initialize();
}

public void initialize()
{
  literals.put(new ANTLRHashString("__alignof__", this),
      new Integer(LITERAL___alignof__));
  literals.put(new ANTLRHashString("__ALIGNOF__", this),
      new Integer(LITERAL___alignof__));
  literals.put(new ANTLRHashString("__asm", this),
      new Integer(LITERAL___asm));
  literals.put(new ANTLRHashString("__asm__", this),
      new Integer(LITERAL___asm));
  literals.put(new ANTLRHashString("__attribute__", this), 
      new Integer(LITERAL___attribute));
  literals.put(new ANTLRHashString("__complex__", this), 
      new Integer(LITERAL___complex));
  literals.put(new ANTLRHashString("__const", this), 
      new Integer(LITERAL_const));
  literals.put(new ANTLRHashString("__const__", this), 
      new Integer(LITERAL_const));
  literals.put(new ANTLRHashString("__imag__", this), 
      new Integer(LITERAL___imag));
  literals.put(new ANTLRHashString("__inline", this), 
      new Integer(LITERAL_inline));
  literals.put(new ANTLRHashString("__inline__", this), 
      new Integer(LITERAL_inline));
  literals.put(new ANTLRHashString("__real__", this), 
      new Integer(LITERAL___real));
  literals.put(new ANTLRHashString("__restrict", this), 
      new Integer(LITERAL___extension__));
  literals.put(new ANTLRHashString("__extension", this), 
      new Integer(LITERAL___extension__));
  literals.put(new ANTLRHashString("__signed", this), 
      new Integer(LITERAL_signed));
  literals.put(new ANTLRHashString("__signed__", this), 
      new Integer(LITERAL_signed));
  literals.put(new ANTLRHashString("__typeof", this), 
      new Integer(LITERAL_typeof));
  literals.put(new ANTLRHashString("__typeof__", this), 
      new Integer(LITERAL_typeof));
  literals.put(new ANTLRHashString("__volatile", this), 
      new Integer(LITERAL_volatile));
  literals.put(new ANTLRHashString("__volatile__", this), 
      new Integer(LITERAL_volatile));
  // GCC Builtin function
  literals.put(new ANTLRHashString("__builtin_va_arg", this), 
      new Integer(LITERAL___builtin_va_arg));
  literals.put(new ANTLRHashString("__builtin_offsetof", this), 
      new Integer(LITERAL___builtin_offsetof));
  // MinGW specific
  literals.put(new ANTLRHashString("__MINGW_IMPORT", this), 
      new Integer(LITERAL___extension__));
  literals.put(new ANTLRHashString("_CRTIMP", this), 
      new Integer(LITERAL___extension__));
  // Microsoft specific
  literals.put(new ANTLRHashString("__cdecl", this), 
      new Integer(LITERAL___extension__));
  literals.put(new ANTLRHashString("__w64", this), 
      new Integer(LITERAL___extension__));
  literals.put(new ANTLRHashString("__int64", this), 
      new Integer(LITERAL_int));
  literals.put(new ANTLRHashString("__int32", this), 
      new Integer(LITERAL_int));
  literals.put(new ANTLRHashString("__int16", this), 
      new Integer(LITERAL_int));
  literals.put(new ANTLRHashString("__int8", this), 
      new Integer(LITERAL_int));
}

LineObject lineObject = new LineObject();
String originalSource = "";
PreprocessorInfoChannel preprocessorInfoChannel = new PreprocessorInfoChannel();
int tokenNumber = 0;
boolean countingTokens = true;
int deferredLineCount = 0;
int extraLineCount = 1;
NewCParser parser = null;

public void setCountingTokens(boolean ct)
{
  countingTokens = ct;
  if ( countingTokens ) {
    tokenNumber = 0;
  } else {
    tokenNumber = 1;
  }
}

public void setParser(NewCParser p)
{
  parser = p;
}

public void setOriginalSource(String src)
{
  originalSource = src;
  lineObject.setSource(src);
}

public void setSource(String src)
{
  lineObject.setSource(src);
}

public PreprocessorInfoChannel getPreprocessorInfoChannel()
{
  return preprocessorInfoChannel;
}

public void setPreprocessingDirective(String pre,int t)
{
  preprocessorInfoChannel.addLineForTokenNumber(
      new Pragma(pre,t), new Integer(tokenNumber));
}

protected Token makeToken(int t)
{
  if ( t != Token.SKIP && countingTokens) {
    tokenNumber++;
  }
  CToken tok = (CToken) super.makeToken(t);
  tok.setLine(lineObject.line);
  tok.setSource(lineObject.source);
  tok.setTokenNumber(tokenNumber);

  lineObject.line += deferredLineCount;
  deferredLineCount = 0;
  return tok;
}

public void deferredNewline()
{
  deferredLineCount++;
}

public void newline() 
{
  lineObject.newline();
  setColumn(1);
}

public NewCLexer(InputStream in) {
	this(new ByteBuffer(in));
}
public NewCLexer(Reader in) {
	this(new CharBuffer(in));
}
public NewCLexer(InputBuffer ib) {
	this(new LexerSharedInputState(ib));
}
public NewCLexer(LexerSharedInputState state) {
	super(state);
	caseSensitiveLiterals = true;
	setCaseSensitive(true);
	literals = new Hashtable();
	literals.put(new ANTLRHashString("extern", this), new Integer(17));
	literals.put(new ANTLRHashString("__real", this), new Integer(97));
	literals.put(new ANTLRHashString("case", this), new Integer(55));
	literals.put(new ANTLRHashString("short", this), new Integer(23));
	literals.put(new ANTLRHashString("break", this), new Integer(53));
	literals.put(new ANTLRHashString("while", this), new Integer(48));
	literals.put(new ANTLRHashString("typeof", this), new Integer(33));
	literals.put(new ANTLRHashString("inline", this), new Integer(19));
	literals.put(new ANTLRHashString("unsigned", this), new Integer(29));
	literals.put(new ANTLRHashString("const", this), new Integer(20));
	literals.put(new ANTLRHashString("float", this), new Integer(26));
	literals.put(new ANTLRHashString("return", this), new Integer(54));
	literals.put(new ANTLRHashString("__builtin_va_arg", this), new Integer(93));
	literals.put(new ANTLRHashString("sizeof", this), new Integer(91));
	literals.put(new ANTLRHashString("do", this), new Integer(49));
	literals.put(new ANTLRHashString("__label__", this), new Integer(47));
	literals.put(new ANTLRHashString("__alignof__", this), new Integer(92));
	literals.put(new ANTLRHashString("__asm", this), new Integer(42));
	literals.put(new ANTLRHashString("typedef", this), new Integer(5));
	literals.put(new ANTLRHashString("if", this), new Integer(57));
	literals.put(new ANTLRHashString("double", this), new Integer(27));
	literals.put(new ANTLRHashString("volatile", this), new Integer(10));
	literals.put(new ANTLRHashString("__attribute", this), new Integer(41));
	literals.put(new ANTLRHashString("union", this), new Integer(13));
	literals.put(new ANTLRHashString("_Imaginary", this), new Integer(32));
	literals.put(new ANTLRHashString("register", this), new Integer(16));
	literals.put(new ANTLRHashString("auto", this), new Integer(15));
	literals.put(new ANTLRHashString("goto", this), new Integer(51));
	literals.put(new ANTLRHashString("enum", this), new Integer(14));
	literals.put(new ANTLRHashString("_Complex", this), new Integer(31));
	literals.put(new ANTLRHashString("int", this), new Integer(24));
	literals.put(new ANTLRHashString("for", this), new Integer(50));
	literals.put(new ANTLRHashString("char", this), new Integer(22));
	literals.put(new ANTLRHashString("default", this), new Integer(56));
	literals.put(new ANTLRHashString("__imag", this), new Integer(98));
	literals.put(new ANTLRHashString("__builtin_offsetof", this), new Integer(94));
	literals.put(new ANTLRHashString("static", this), new Integer(18));
	literals.put(new ANTLRHashString("_Bool", this), new Integer(30));
	literals.put(new ANTLRHashString("continue", this), new Integer(52));
	literals.put(new ANTLRHashString("struct", this), new Integer(12));
	literals.put(new ANTLRHashString("signed", this), new Integer(28));
	literals.put(new ANTLRHashString("else", this), new Integer(58));
	literals.put(new ANTLRHashString("void", this), new Integer(21));
	literals.put(new ANTLRHashString("switch", this), new Integer(59));
	literals.put(new ANTLRHashString("long", this), new Integer(25));
	literals.put(new ANTLRHashString("__extension__", this), new Integer(142));
	literals.put(new ANTLRHashString("asm", this), new Integer(9));
	literals.put(new ANTLRHashString("__complex", this), new Integer(36));
}

public Token nextToken() throws TokenStreamException {
	Token theRetToken=null;
tryAgain:
	for (;;) {
		Token _token = null;
		int _ttype = Token.INVALID_TYPE;
		resetText();
		try {   // for char stream error handling
			try {   // for lexical error handling
				switch ( LA(1)) {
				case ':':
				{
					mCOLON(true);
					theRetToken=_returnToken;
					break;
				}
				case ',':
				{
					mCOMMA(true);
					theRetToken=_returnToken;
					break;
				}
				case '?':
				{
					mQUESTION(true);
					theRetToken=_returnToken;
					break;
				}
				case ';':
				{
					mSEMI(true);
					theRetToken=_returnToken;
					break;
				}
				case '(':
				{
					mLPAREN(true);
					theRetToken=_returnToken;
					break;
				}
				case ')':
				{
					mRPAREN(true);
					theRetToken=_returnToken;
					break;
				}
				case '[':
				{
					mLBRACKET(true);
					theRetToken=_returnToken;
					break;
				}
				case ']':
				{
					mRBRACKET(true);
					theRetToken=_returnToken;
					break;
				}
				case '{':
				{
					mLCURLY(true);
					theRetToken=_returnToken;
					break;
				}
				case '}':
				{
					mRCURLY(true);
					theRetToken=_returnToken;
					break;
				}
				case '~':
				{
					mBNOT(true);
					theRetToken=_returnToken;
					break;
				}
				case '\t':  case '\n':  case '\u000c':  case '\r':
				case ' ':
				{
					mWhitespace(true);
					theRetToken=_returnToken;
					break;
				}
				case '#':
				{
					mPREPROC_DIRECTIVE(true);
					theRetToken=_returnToken;
					break;
				}
				case '.':  case '0':  case '1':  case '2':
				case '3':  case '4':  case '5':  case '6':
				case '7':  case '8':  case '9':
				{
					mNumber(true);
					theRetToken=_returnToken;
					break;
				}
				case '"':
				{
					mStringLiteral(true);
					theRetToken=_returnToken;
					break;
				}
				case '\'':
				{
					mCharLiteral(true);
					theRetToken=_returnToken;
					break;
				}
				default:
					if ((LA(1)=='>') && (LA(2)=='>') && (LA(3)=='=')) {
						mRSHIFT_ASSIGN(true);
						theRetToken=_returnToken;
					}
					else if ((LA(1)=='<') && (LA(2)=='<') && (LA(3)=='=')) {
						mLSHIFT_ASSIGN(true);
						theRetToken=_returnToken;
					}
					else if ((LA(1)=='-') && (LA(2)=='>')) {
						mPTR(true);
						theRetToken=_returnToken;
					}
					else if ((LA(1)=='=') && (LA(2)=='=')) {
						mEQUAL(true);
						theRetToken=_returnToken;
					}
					else if ((LA(1)=='!') && (LA(2)=='=')) {
						mNOT_EQUAL(true);
						theRetToken=_returnToken;
					}
					else if ((LA(1)=='<') && (LA(2)=='=')) {
						mLTE(true);
						theRetToken=_returnToken;
					}
					else if ((LA(1)=='>') && (LA(2)=='=')) {
						mGTE(true);
						theRetToken=_returnToken;
					}
					else if ((LA(1)=='/') && (LA(2)=='=')) {
						mDIV_ASSIGN(true);
						theRetToken=_returnToken;
					}
					else if ((LA(1)=='+') && (LA(2)=='=')) {
						mPLUS_ASSIGN(true);
						theRetToken=_returnToken;
					}
					else if ((LA(1)=='+') && (LA(2)=='+')) {
						mINC(true);
						theRetToken=_returnToken;
					}
					else if ((LA(1)=='-') && (LA(2)=='=')) {
						mMINUS_ASSIGN(true);
						theRetToken=_returnToken;
					}
					else if ((LA(1)=='-') && (LA(2)=='-')) {
						mDEC(true);
						theRetToken=_returnToken;
					}
					else if ((LA(1)=='*') && (LA(2)=='=')) {
						mSTAR_ASSIGN(true);
						theRetToken=_returnToken;
					}
					else if ((LA(1)=='%') && (LA(2)=='=')) {
						mMOD_ASSIGN(true);
						theRetToken=_returnToken;
					}
					else if ((LA(1)=='>') && (LA(2)=='>') && (true)) {
						mRSHIFT(true);
						theRetToken=_returnToken;
					}
					else if ((LA(1)=='<') && (LA(2)=='<') && (true)) {
						mLSHIFT(true);
						theRetToken=_returnToken;
					}
					else if ((LA(1)=='&') && (LA(2)=='&')) {
						mLAND(true);
						theRetToken=_returnToken;
					}
					else if ((LA(1)=='|') && (LA(2)=='|')) {
						mLOR(true);
						theRetToken=_returnToken;
					}
					else if ((LA(1)=='&') && (LA(2)=='=')) {
						mBAND_ASSIGN(true);
						theRetToken=_returnToken;
					}
					else if ((LA(1)=='|') && (LA(2)=='=')) {
						mBOR_ASSIGN(true);
						theRetToken=_returnToken;
					}
					else if ((LA(1)=='^') && (LA(2)=='=')) {
						mBXOR_ASSIGN(true);
						theRetToken=_returnToken;
					}
					else if ((LA(1)=='/') && (LA(2)=='*')) {
						mComment(true);
						theRetToken=_returnToken;
					}
					else if ((LA(1)=='/') && (LA(2)=='/')) {
						mCPPComment(true);
						theRetToken=_returnToken;
					}
					else if ((LA(1)=='L') && (LA(2)=='\'')) {
						mWideCharLiteral(true);
						theRetToken=_returnToken;
					}
					else if ((LA(1)=='L') && (LA(2)=='"')) {
						mWideStringLiteral(true);
						theRetToken=_returnToken;
					}
					else if ((LA(1)=='=') && (true)) {
						mASSIGN(true);
						theRetToken=_returnToken;
					}
					else if ((LA(1)=='<') && (true)) {
						mLT(true);
						theRetToken=_returnToken;
					}
					else if ((LA(1)=='>') && (true)) {
						mGT(true);
						theRetToken=_returnToken;
					}
					else if ((LA(1)=='/') && (true)) {
						mDIV(true);
						theRetToken=_returnToken;
					}
					else if ((LA(1)=='+') && (true)) {
						mPLUS(true);
						theRetToken=_returnToken;
					}
					else if ((LA(1)=='-') && (true)) {
						mMINUS(true);
						theRetToken=_returnToken;
					}
					else if ((LA(1)=='*') && (true)) {
						mSTAR(true);
						theRetToken=_returnToken;
					}
					else if ((LA(1)=='%') && (true)) {
						mMOD(true);
						theRetToken=_returnToken;
					}
					else if ((LA(1)=='!') && (true)) {
						mLNOT(true);
						theRetToken=_returnToken;
					}
					else if ((LA(1)=='&') && (true)) {
						mBAND(true);
						theRetToken=_returnToken;
					}
					else if ((LA(1)=='|') && (true)) {
						mBOR(true);
						theRetToken=_returnToken;
					}
					else if ((LA(1)=='^') && (true)) {
						mBXOR(true);
						theRetToken=_returnToken;
					}
					else if ((_tokenSet_0.member(LA(1))) && (true)) {
						mIDMEAT(true);
						theRetToken=_returnToken;
					}
				else {
					if (LA(1)==EOF_CHAR) {uponEOF(); _returnToken = makeToken(Token.EOF_TYPE);}
				else {throw new NoViableAltForCharException((char)LA(1), getFilename(), getLine(), getColumn());}
				}
				}
				if ( _returnToken==null ) continue tryAgain; // found SKIP token
				_ttype = _returnToken.getType();
				_returnToken.setType(_ttype);
				return _returnToken;
			}
			catch (RecognitionException e) {
				throw new TokenStreamRecognitionException(e);
			}
		}
		catch (CharStreamException cse) {
			if ( cse instanceof CharStreamIOException ) {
				throw new TokenStreamIOException(((CharStreamIOException)cse).io);
			}
			else {
				throw new TokenStreamException(cse.getMessage());
			}
		}
	}
}

	protected final void mVocabulary(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = Vocabulary;
		int _saveIndex;
		
		matchRange('\3','\377');
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	public final void mASSIGN(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = ASSIGN;
		int _saveIndex;
		
		match('=');
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	public final void mCOLON(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = COLON;
		int _saveIndex;
		
		match(':');
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	public final void mCOMMA(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = COMMA;
		int _saveIndex;
		
		match(',');
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	public final void mQUESTION(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = QUESTION;
		int _saveIndex;
		
		match('?');
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	public final void mSEMI(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = SEMI;
		int _saveIndex;
		
		match(';');
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	public final void mPTR(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = PTR;
		int _saveIndex;
		
		match("->");
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	protected final void mDOT(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = DOT;
		int _saveIndex;
		
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	protected final void mVARARGS(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = VARARGS;
		int _saveIndex;
		
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	public final void mLPAREN(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = LPAREN;
		int _saveIndex;
		
		match('(');
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	public final void mRPAREN(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = RPAREN;
		int _saveIndex;
		
		match(')');
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	public final void mLBRACKET(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = LBRACKET;
		int _saveIndex;
		
		match('[');
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	public final void mRBRACKET(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = RBRACKET;
		int _saveIndex;
		
		match(']');
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	public final void mLCURLY(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = LCURLY;
		int _saveIndex;
		
		match('{');
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	public final void mRCURLY(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = RCURLY;
		int _saveIndex;
		
		match('}');
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	public final void mEQUAL(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = EQUAL;
		int _saveIndex;
		
		match("==");
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	public final void mNOT_EQUAL(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = NOT_EQUAL;
		int _saveIndex;
		
		match("!=");
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	public final void mLTE(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = LTE;
		int _saveIndex;
		
		match("<=");
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	public final void mLT(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = LT;
		int _saveIndex;
		
		match("<");
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	public final void mGTE(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = GTE;
		int _saveIndex;
		
		match(">=");
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	public final void mGT(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = GT;
		int _saveIndex;
		
		match(">");
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	public final void mDIV(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = DIV;
		int _saveIndex;
		
		match('/');
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	public final void mDIV_ASSIGN(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = DIV_ASSIGN;
		int _saveIndex;
		
		match("/=");
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	public final void mPLUS(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = PLUS;
		int _saveIndex;
		
		match('+');
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	public final void mPLUS_ASSIGN(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = PLUS_ASSIGN;
		int _saveIndex;
		
		match("+=");
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	public final void mINC(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = INC;
		int _saveIndex;
		
		match("++");
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	public final void mMINUS(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = MINUS;
		int _saveIndex;
		
		match('-');
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	public final void mMINUS_ASSIGN(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = MINUS_ASSIGN;
		int _saveIndex;
		
		match("-=");
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	public final void mDEC(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = DEC;
		int _saveIndex;
		
		match("--");
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	public final void mSTAR(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = STAR;
		int _saveIndex;
		
		match('*');
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	public final void mSTAR_ASSIGN(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = STAR_ASSIGN;
		int _saveIndex;
		
		match("*=");
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	public final void mMOD(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = MOD;
		int _saveIndex;
		
		match('%');
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	public final void mMOD_ASSIGN(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = MOD_ASSIGN;
		int _saveIndex;
		
		match("%=");
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	public final void mRSHIFT(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = RSHIFT;
		int _saveIndex;
		
		match(">>");
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	public final void mRSHIFT_ASSIGN(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = RSHIFT_ASSIGN;
		int _saveIndex;
		
		match(">>=");
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	public final void mLSHIFT(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = LSHIFT;
		int _saveIndex;
		
		match("<<");
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	public final void mLSHIFT_ASSIGN(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = LSHIFT_ASSIGN;
		int _saveIndex;
		
		match("<<=");
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	public final void mLAND(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = LAND;
		int _saveIndex;
		
		match("&&");
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	public final void mLNOT(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = LNOT;
		int _saveIndex;
		
		match('!');
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	public final void mLOR(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = LOR;
		int _saveIndex;
		
		match("||");
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	public final void mBAND(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = BAND;
		int _saveIndex;
		
		match('&');
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	public final void mBAND_ASSIGN(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = BAND_ASSIGN;
		int _saveIndex;
		
		match("&=");
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	public final void mBNOT(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = BNOT;
		int _saveIndex;
		
		match('~');
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	public final void mBOR(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = BOR;
		int _saveIndex;
		
		match('|');
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	public final void mBOR_ASSIGN(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = BOR_ASSIGN;
		int _saveIndex;
		
		match("|=");
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	public final void mBXOR(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = BXOR;
		int _saveIndex;
		
		match('^');
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	public final void mBXOR_ASSIGN(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = BXOR_ASSIGN;
		int _saveIndex;
		
		match("^=");
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	public final void mWhitespace(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = Whitespace;
		int _saveIndex;
		
		{
		if ((LA(1)=='\r') && (LA(2)=='\n')) {
			match("\r\n");
			if ( inputState.guessing==0 ) {
				newline();
			}
		}
		else if ((LA(1)=='\t'||LA(1)=='\u000c'||LA(1)==' ')) {
			{
			switch ( LA(1)) {
			case ' ':
			{
				match(' ');
				break;
			}
			case '\t':
			{
				match('\t');
				break;
			}
			case '\u000c':
			{
				match('\014');
				break;
			}
			default:
			{
				throw new NoViableAltForCharException((char)LA(1), getFilename(), getLine(), getColumn());
			}
			}
			}
		}
		else if ((LA(1)=='\n'||LA(1)=='\r') && (true)) {
			{
			switch ( LA(1)) {
			case '\n':
			{
				match('\n');
				break;
			}
			case '\r':
			{
				match('\r');
				break;
			}
			default:
			{
				throw new NoViableAltForCharException((char)LA(1), getFilename(), getLine(), getColumn());
			}
			}
			}
			if ( inputState.guessing==0 ) {
				newline();
			}
		}
		else {
			throw new NoViableAltForCharException((char)LA(1), getFilename(), getLine(), getColumn());
		}
		
		}
		if ( inputState.guessing==0 ) {
			_ttype = Token.SKIP;
		}
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	public final void mComment(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = Comment;
		int _saveIndex;
		
		{
		match("/*");
		{
		_loop396:
		do {
			if (((LA(1)=='*') && ((LA(2) >= '\u0000' && LA(2) <= '\u00ff')) && ((LA(3) >= '\u0000' && LA(3) <= '\u00ff')))&&( LA(2) != '/' )) {
				match('*');
			}
			else if ((LA(1)=='\r') && (LA(2)=='\n') && ((LA(3) >= '\u0000' && LA(3) <= '\u00ff'))) {
				match("\r\n");
				if ( inputState.guessing==0 ) {
					deferredNewline();
				}
			}
			else if ((LA(1)=='\n'||LA(1)=='\r') && ((LA(2) >= '\u0000' && LA(2) <= '\u00ff')) && ((LA(3) >= '\u0000' && LA(3) <= '\u00ff'))) {
				{
				switch ( LA(1)) {
				case '\r':
				{
					match('\r');
					break;
				}
				case '\n':
				{
					match('\n');
					break;
				}
				default:
				{
					throw new NoViableAltForCharException((char)LA(1), getFilename(), getLine(), getColumn());
				}
				}
				}
				if ( inputState.guessing==0 ) {
					deferredNewline();
				}
			}
			else if ((_tokenSet_1.member(LA(1)))) {
				{
				match(_tokenSet_1);
				}
			}
			else {
				break _loop396;
			}
			
		} while (true);
		}
		match("*/");
		if ( inputState.guessing==0 ) {
			setPreprocessingDirective(getText(),Pragma.comment);
		}
		}
		if ( inputState.guessing==0 ) {
			_ttype = Token.SKIP;
		}
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	public final void mCPPComment(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = CPPComment;
		int _saveIndex;
		
		{
		match("//");
		{
		_loop401:
		do {
			if ((_tokenSet_2.member(LA(1)))) {
				{
				match(_tokenSet_2);
				}
			}
			else {
				break _loop401;
			}
			
		} while (true);
		}
		if ( inputState.guessing==0 ) {
			setPreprocessingDirective(getText(),Pragma.comment);
		}
		}
		if ( inputState.guessing==0 ) {
			_ttype = Token.SKIP;
		}
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	public final void mPREPROC_DIRECTIVE(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = PREPROC_DIRECTIVE;
		int _saveIndex;
		
		match('#');
		{
		if ((LA(1)=='p') && (LA(2)=='r') && (LA(3)=='a')) {
			{
			match("pragma");
			{
			if ((LA(1)=='\t'||LA(1)=='\u000c'||LA(1)==' ') && (_tokenSet_3.member(LA(2))) && (_tokenSet_4.member(LA(3)))) {
				{
				int _cnt415=0;
				_loop415:
				do {
					if ((LA(1)=='\t'||LA(1)=='\u000c'||LA(1)==' ')) {
						mSpace(false);
					}
					else {
						if ( _cnt415>=1 ) { break _loop415; } else {throw new NoViableAltForCharException((char)LA(1), getFilename(), getLine(), getColumn());}
					}
					
					_cnt415++;
				} while (true);
				}
				match("startinclude");
				{
				_loop418:
				do {
					if ((_tokenSet_2.member(LA(1)))) {
						{
						match(_tokenSet_2);
						}
					}
					else {
						break _loop418;
					}
					
				} while (true);
				}
			}
			else if ((LA(1)=='\t'||LA(1)=='\u000c'||LA(1)==' ') && (_tokenSet_5.member(LA(2))) && (_tokenSet_6.member(LA(3)))) {
				{
				int _cnt420=0;
				_loop420:
				do {
					if ((LA(1)=='\t'||LA(1)=='\u000c'||LA(1)==' ')) {
						mSpace(false);
					}
					else {
						if ( _cnt420>=1 ) { break _loop420; } else {throw new NoViableAltForCharException((char)LA(1), getFilename(), getLine(), getColumn());}
					}
					
					_cnt420++;
				} while (true);
				}
				match("endinclude");
				{
				_loop423:
				do {
					if ((_tokenSet_2.member(LA(1)))) {
						{
						match(_tokenSet_2);
						}
					}
					else {
						break _loop423;
					}
					
				} while (true);
				}
				if ( inputState.guessing==0 ) {
					extraLineCount +=2; lineObject.setLine(lineObject.getLine() - 2);
				}
			}
			else {
				{
				_loop413:
				do {
					if ((_tokenSet_2.member(LA(1)))) {
						{
						match(_tokenSet_2);
						}
					}
					else {
						break _loop413;
					}
					
				} while (true);
				}
				if ( inputState.guessing==0 ) {
					setPreprocessingDirective(getText(),Pragma.pragma);_ttype = Token.SKIP;
				}
			}
			
			}
			}
		}
		else {
			boolean synPredMatched408 = false;
			if (((_tokenSet_7.member(LA(1))) && (_tokenSet_8.member(LA(2))) && (true))) {
				int _m408 = mark();
				synPredMatched408 = true;
				inputState.guessing++;
				try {
					{
					switch ( LA(1)) {
					case 'l':
					{
						match("line");
						break;
					}
					case '\t':  case '\u000c':  case ' ':
					{
						{
						{
						int _cnt407=0;
						_loop407:
						do {
							if ((LA(1)=='\t'||LA(1)=='\u000c'||LA(1)==' ')) {
								mSpace(false);
							}
							else {
								if ( _cnt407>=1 ) { break _loop407; } else {throw new NoViableAltForCharException((char)LA(1), getFilename(), getLine(), getColumn());}
							}
							
							_cnt407++;
						} while (true);
						}
						mDigit(false);
						}
						break;
					}
					default:
						{
						}
					}
					}
				}
				catch (RecognitionException pe) {
					synPredMatched408 = false;
				}
				rewind(_m408);
inputState.guessing--;
			}
			if ( synPredMatched408 ) {
				mLineDirective(false);
				if ( inputState.guessing==0 ) {
					_ttype = Token.SKIP;
				}
			}
			else {
				{
				_loop426:
				do {
					if ((_tokenSet_2.member(LA(1)))) {
						{
						match(_tokenSet_2);
						}
					}
					else {
						break _loop426;
					}
					
				} while (true);
				}
				if ( inputState.guessing==0 ) {
					_ttype = Token.SKIP;
				}
			}
			}
			}
			if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
				_token = makeToken(_ttype);
				_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
			}
			_returnToken = _token;
		}
		
	protected final void mSpace(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = Space;
		int _saveIndex;
		
		{
		switch ( LA(1)) {
		case ' ':
		{
			match(' ');
			break;
		}
		case '\t':
		{
			match('\t');
			break;
		}
		case '\u000c':
		{
			match('\014');
			break;
		}
		default:
		{
			throw new NoViableAltForCharException((char)LA(1), getFilename(), getLine(), getColumn());
		}
		}
		}
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	protected final void mDigit(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = Digit;
		int _saveIndex;
		
		matchRange('0','9');
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	protected final void mLineDirective(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = LineDirective;
		int _saveIndex;
		Token n=null;
		Token fn=null;
		Token fi=null;
		
		boolean oldCountingTokens = countingTokens;
		countingTokens = false;
		
		
		if ( inputState.guessing==0 ) {
			
			lineObject = new LineObject();
			deferredLineCount = 0;
			
		}
		{
		switch ( LA(1)) {
		case 'l':
		{
			match("line");
			break;
		}
		case '\t':  case '\u000c':  case ' ':
		{
			break;
		}
		default:
		{
			throw new NoViableAltForCharException((char)LA(1), getFilename(), getLine(), getColumn());
		}
		}
		}
		{
		int _cnt432=0;
		_loop432:
		do {
			if ((LA(1)=='\t'||LA(1)=='\u000c'||LA(1)==' ')) {
				mSpace(false);
			}
			else {
				if ( _cnt432>=1 ) { break _loop432; } else {throw new NoViableAltForCharException((char)LA(1), getFilename(), getLine(), getColumn());}
			}
			
			_cnt432++;
		} while (true);
		}
		mNumber(true);
		n=_returnToken;
		if ( inputState.guessing==0 ) {
			lineObject.setLine(Integer.parseInt(n.getText()) - extraLineCount);
		}
		{
		if ((LA(1)=='\t'||LA(1)=='\u000c'||LA(1)==' ')) {
			{
			int _cnt435=0;
			_loop435:
			do {
				if ((LA(1)=='\t'||LA(1)=='\u000c'||LA(1)==' ') && (true) && (true)) {
					mSpace(false);
				}
				else {
					if ( _cnt435>=1 ) { break _loop435; } else {throw new NoViableAltForCharException((char)LA(1), getFilename(), getLine(), getColumn());}
				}
				
				_cnt435++;
			} while (true);
			}
			{
			if ((LA(1)=='"') && ((LA(2) >= '\u0000' && LA(2) <= '\u00ff')) && (true)) {
				mStringLiteral(true);
				fn=_returnToken;
				if ( inputState.guessing==0 ) {
					
					try {
					lineObject.setSource(fn.getText().substring(1,fn.getText().length()-1));
					} catch (StringIndexOutOfBoundsException e) { /*not possible*/
					}
					
				}
			}
			else if ((_tokenSet_0.member(LA(1))) && (true) && (true)) {
				mID(true);
				fi=_returnToken;
				if ( inputState.guessing==0 ) {
					lineObject.setSource(fi.getText());
				}
			}
			else {
			}
			
			}
			{
			_loop438:
			do {
				if ((LA(1)=='\t'||LA(1)=='\u000c'||LA(1)==' ') && (true) && (true)) {
					mSpace(false);
				}
				else {
					break _loop438;
				}
				
			} while (true);
			}
			{
			if ((LA(1)=='1') && (true) && (true)) {
				match("1");
				if ( inputState.guessing==0 ) {
					lineObject.setEnteringFile(true);
				}
			}
			else {
			}
			
			}
			{
			_loop441:
			do {
				if ((LA(1)=='\t'||LA(1)=='\u000c'||LA(1)==' ') && (true) && (true)) {
					mSpace(false);
				}
				else {
					break _loop441;
				}
				
			} while (true);
			}
			{
			if ((LA(1)=='2') && (true) && (true)) {
				match("2");
				if ( inputState.guessing==0 ) {
					lineObject.setReturningToFile(true);
				}
			}
			else {
			}
			
			}
			{
			_loop444:
			do {
				if ((LA(1)=='\t'||LA(1)=='\u000c'||LA(1)==' ') && (true) && (true)) {
					mSpace(false);
				}
				else {
					break _loop444;
				}
				
			} while (true);
			}
			{
			if ((LA(1)=='3') && (true) && (true)) {
				match("3");
				if ( inputState.guessing==0 ) {
					lineObject.setSystemHeader(true);
				}
			}
			else {
			}
			
			}
			{
			_loop447:
			do {
				if ((LA(1)=='\t'||LA(1)=='\u000c'||LA(1)==' ') && (true) && (true)) {
					mSpace(false);
				}
				else {
					break _loop447;
				}
				
			} while (true);
			}
			{
			if ((LA(1)=='4') && (true) && (true)) {
				match("4");
				if ( inputState.guessing==0 ) {
					lineObject.setTreatAsC(true);
				}
			}
			else {
			}
			
			}
			{
			_loop451:
			do {
				if ((_tokenSet_9.member(LA(1)))) {
					{
					match(_tokenSet_9);
					}
				}
				else {
					break _loop451;
				}
				
			} while (true);
			}
		}
		else {
		}
		
		}
		if ( inputState.guessing==0 ) {
			
			/*
			preprocessorInfoChannel.addLineForTokenNumber(
			new LineObject(lineObject), new Integer(tokenNumber));
			*/
			countingTokens = oldCountingTokens;
			
		}
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	public final void mNumber(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = Number;
		int _saveIndex;
		
		boolean synPredMatched480 = false;
		if ((((LA(1) >= '0' && LA(1) <= '9')) && (_tokenSet_10.member(LA(2))) && (true))) {
			int _m480 = mark();
			synPredMatched480 = true;
			inputState.guessing++;
			try {
				{
				{
				int _cnt478=0;
				_loop478:
				do {
					if (((LA(1) >= '0' && LA(1) <= '9'))) {
						mDigit(false);
					}
					else {
						if ( _cnt478>=1 ) { break _loop478; } else {throw new NoViableAltForCharException((char)LA(1), getFilename(), getLine(), getColumn());}
					}
					
					_cnt478++;
				} while (true);
				}
				{
				switch ( LA(1)) {
				case '.':
				{
					match('.');
					break;
				}
				case 'e':
				{
					match('e');
					break;
				}
				case 'E':
				{
					match('E');
					break;
				}
				default:
				{
					throw new NoViableAltForCharException((char)LA(1), getFilename(), getLine(), getColumn());
				}
				}
				}
				}
			}
			catch (RecognitionException pe) {
				synPredMatched480 = false;
			}
			rewind(_m480);
inputState.guessing--;
		}
		if ( synPredMatched480 ) {
			{
			int _cnt482=0;
			_loop482:
			do {
				if (((LA(1) >= '0' && LA(1) <= '9'))) {
					mDigit(false);
				}
				else {
					if ( _cnt482>=1 ) { break _loop482; } else {throw new NoViableAltForCharException((char)LA(1), getFilename(), getLine(), getColumn());}
				}
				
				_cnt482++;
			} while (true);
			}
			{
			switch ( LA(1)) {
			case '.':
			{
				match('.');
				{
				_loop485:
				do {
					if (((LA(1) >= '0' && LA(1) <= '9'))) {
						mDigit(false);
					}
					else {
						break _loop485;
					}
					
				} while (true);
				}
				{
				if ((LA(1)=='E'||LA(1)=='e')) {
					mExponent(false);
				}
				else {
				}
				
				}
				break;
			}
			case 'E':  case 'e':
			{
				mExponent(false);
				break;
			}
			default:
			{
				throw new NoViableAltForCharException((char)LA(1), getFilename(), getLine(), getColumn());
			}
			}
			}
			{
			_loop488:
			do {
				if ((_tokenSet_11.member(LA(1)))) {
					mNumberSuffix(false);
				}
				else {
					break _loop488;
				}
				
			} while (true);
			}
		}
		else {
			boolean synPredMatched490 = false;
			if (((LA(1)=='.') && (LA(2)=='.'))) {
				int _m490 = mark();
				synPredMatched490 = true;
				inputState.guessing++;
				try {
					{
					match("...");
					}
				}
				catch (RecognitionException pe) {
					synPredMatched490 = false;
				}
				rewind(_m490);
inputState.guessing--;
			}
			if ( synPredMatched490 ) {
				match("...");
				if ( inputState.guessing==0 ) {
					_ttype = VARARGS;
				}
			}
			else if ((LA(1)=='0') && (LA(2)=='X'||LA(2)=='x')) {
				match('0');
				{
				switch ( LA(1)) {
				case 'x':
				{
					match('x');
					break;
				}
				case 'X':
				{
					match('X');
					break;
				}
				default:
				{
					throw new NoViableAltForCharException((char)LA(1), getFilename(), getLine(), getColumn());
				}
				}
				}
				{
				int _cnt507=0;
				_loop507:
				do {
					switch ( LA(1)) {
					case 'a':  case 'b':  case 'c':  case 'd':
					case 'e':  case 'f':
					{
						matchRange('a','f');
						break;
					}
					case 'A':  case 'B':  case 'C':  case 'D':
					case 'E':  case 'F':
					{
						matchRange('A','F');
						break;
					}
					case '0':  case '1':  case '2':  case '3':
					case '4':  case '5':  case '6':  case '7':
					case '8':  case '9':
					{
						mDigit(false);
						break;
					}
					default:
					{
						if ( _cnt507>=1 ) { break _loop507; } else {throw new NoViableAltForCharException((char)LA(1), getFilename(), getLine(), getColumn());}
					}
					}
					_cnt507++;
				} while (true);
				}
				{
				_loop509:
				do {
					if ((_tokenSet_12.member(LA(1)))) {
						mIntSuffix(false);
					}
					else {
						break _loop509;
					}
					
				} while (true);
				}
			}
			else if ((LA(1)=='.') && (true)) {
				match('.');
				if ( inputState.guessing==0 ) {
					_ttype = DOT;
				}
				{
				if (((LA(1) >= '0' && LA(1) <= '9'))) {
					{
					int _cnt493=0;
					_loop493:
					do {
						if (((LA(1) >= '0' && LA(1) <= '9'))) {
							mDigit(false);
						}
						else {
							if ( _cnt493>=1 ) { break _loop493; } else {throw new NoViableAltForCharException((char)LA(1), getFilename(), getLine(), getColumn());}
						}
						
						_cnt493++;
					} while (true);
					}
					{
					if ((LA(1)=='E'||LA(1)=='e')) {
						mExponent(false);
					}
					else {
					}
					
					}
					if ( inputState.guessing==0 ) {
						_ttype = Number;
					}
					{
					_loop496:
					do {
						if ((_tokenSet_11.member(LA(1)))) {
							mNumberSuffix(false);
						}
						else {
							break _loop496;
						}
						
					} while (true);
					}
				}
				else {
				}
				
				}
			}
			else if ((LA(1)=='0') && (true) && (true)) {
				match('0');
				{
				_loop498:
				do {
					if (((LA(1) >= '0' && LA(1) <= '7'))) {
						matchRange('0','7');
					}
					else {
						break _loop498;
					}
					
				} while (true);
				}
				{
				_loop500:
				do {
					if ((_tokenSet_11.member(LA(1)))) {
						mNumberSuffix(false);
					}
					else {
						break _loop500;
					}
					
				} while (true);
				}
			}
			else if (((LA(1) >= '1' && LA(1) <= '9')) && (true) && (true)) {
				matchRange('1','9');
				{
				_loop502:
				do {
					if (((LA(1) >= '0' && LA(1) <= '9'))) {
						mDigit(false);
					}
					else {
						break _loop502;
					}
					
				} while (true);
				}
				{
				_loop504:
				do {
					if ((_tokenSet_11.member(LA(1)))) {
						mNumberSuffix(false);
					}
					else {
						break _loop504;
					}
					
				} while (true);
				}
			}
			else {
				throw new NoViableAltForCharException((char)LA(1), getFilename(), getLine(), getColumn());
			}
			}
			if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
				_token = makeToken(_ttype);
				_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
			}
			_returnToken = _token;
		}
		
	public final void mStringLiteral(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = StringLiteral;
		int _saveIndex;
		
		match('"');
		{
		_loop524:
		do {
			boolean synPredMatched521 = false;
			if (((LA(1)=='\\') && ((LA(2) >= '\u0000' && LA(2) <= '\u00ff')) && ((LA(3) >= '\u0000' && LA(3) <= '\u00ff')))) {
				int _m521 = mark();
				synPredMatched521 = true;
				inputState.guessing++;
				try {
					{
					match('\\');
					{
					match(_tokenSet_2);
					}
					}
				}
				catch (RecognitionException pe) {
					synPredMatched521 = false;
				}
				rewind(_m521);
inputState.guessing--;
			}
			if ( synPredMatched521 ) {
				mEscape(false);
			}
			else if ((LA(1)=='\n'||LA(1)=='\r'||LA(1)=='\\') && ((LA(2) >= '\u0000' && LA(2) <= '\u00ff')) && (true)) {
				{
				switch ( LA(1)) {
				case '\r':
				{
					match('\r');
					if ( inputState.guessing==0 ) {
						newline();
					}
					break;
				}
				case '\n':
				{
					match('\n');
					if ( inputState.guessing==0 ) {
						newline();
					}
					break;
				}
				case '\\':
				{
					match('\\');
					match('\n');
					if ( inputState.guessing==0 ) {
						newline();
					}
					break;
				}
				default:
				{
					throw new NoViableAltForCharException((char)LA(1), getFilename(), getLine(), getColumn());
				}
				}
				}
			}
			else if ((_tokenSet_13.member(LA(1)))) {
				{
				match(_tokenSet_13);
				}
			}
			else {
				break _loop524;
			}
			
		} while (true);
		}
		match('"');
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	protected final void mID(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = ID;
		int _saveIndex;
		
		{
		switch ( LA(1)) {
		case 'a':  case 'b':  case 'c':  case 'd':
		case 'e':  case 'f':  case 'g':  case 'h':
		case 'i':  case 'j':  case 'k':  case 'l':
		case 'm':  case 'n':  case 'o':  case 'p':
		case 'q':  case 'r':  case 's':  case 't':
		case 'u':  case 'v':  case 'w':  case 'x':
		case 'y':  case 'z':
		{
			matchRange('a','z');
			break;
		}
		case 'A':  case 'B':  case 'C':  case 'D':
		case 'E':  case 'F':  case 'G':  case 'H':
		case 'I':  case 'J':  case 'K':  case 'L':
		case 'M':  case 'N':  case 'O':  case 'P':
		case 'Q':  case 'R':  case 'S':  case 'T':
		case 'U':  case 'V':  case 'W':  case 'X':
		case 'Y':  case 'Z':
		{
			matchRange('A','Z');
			break;
		}
		case '_':
		{
			match('_');
			break;
		}
		case '$':
		{
			match('$');
			break;
		}
		default:
		{
			throw new NoViableAltForCharException((char)LA(1), getFilename(), getLine(), getColumn());
		}
		}
		}
		{
		_loop514:
		do {
			if (((LA(1) >= 'a' && LA(1) <= 'z')) && (true) && (true)) {
				matchRange('a','z');
			}
			else if (((LA(1) >= 'A' && LA(1) <= 'Z')) && (true) && (true)) {
				matchRange('A','Z');
			}
			else if ((LA(1)=='_') && (true) && (true)) {
				match('_');
			}
			else if ((LA(1)=='$') && (true) && (true)) {
				match('$');
			}
			else if (((LA(1) >= '0' && LA(1) <= '9')) && (true) && (true)) {
				matchRange('0','9');
			}
			else {
				break _loop514;
			}
			
		} while (true);
		}
		_ttype = testLiteralsTable(new String(text.getBuffer(),_begin,text.length()-_begin),_ttype);
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	public final void mCharLiteral(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = CharLiteral;
		int _saveIndex;
		
		match('\'');
		{
		if ((LA(1)=='\\') && ((LA(2) >= '\u0000' && LA(2) <= '\u00ff')) && (_tokenSet_14.member(LA(3)))) {
			mEscape(false);
		}
		else if ((_tokenSet_15.member(LA(1))) && (LA(2)=='\'') && (true)) {
			{
			match(_tokenSet_15);
			}
		}
		else {
			throw new NoViableAltForCharException((char)LA(1), getFilename(), getLine(), getColumn());
		}
		
		}
		match('\'');
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	protected final void mEscape(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = Escape;
		int _saveIndex;
		
		match('\\');
		{
		switch ( LA(1)) {
		case '0':  case '1':  case '2':  case '3':
		{
			{
			matchRange('0','3');
			}
			{
			_loop461:
			do {
				if (((LA(1) >= '0' && LA(1) <= '9')) && ((LA(2) >= '\u0000' && LA(2) <= '\u00ff')) && (true)) {
					mDigit(false);
				}
				else {
					break _loop461;
				}
				
			} while (true);
			}
			break;
		}
		case '4':  case '5':  case '6':  case '7':
		{
			{
			matchRange('4','7');
			}
			{
			_loop464:
			do {
				if (((LA(1) >= '0' && LA(1) <= '9')) && ((LA(2) >= '\u0000' && LA(2) <= '\u00ff')) && (true)) {
					mDigit(false);
				}
				else {
					break _loop464;
				}
				
			} while (true);
			}
			break;
		}
		case 'x':
		{
			match('x');
			{
			int _cnt466=0;
			_loop466:
			do {
				if (((LA(1) >= '0' && LA(1) <= '9')) && ((LA(2) >= '\u0000' && LA(2) <= '\u00ff')) && (true)) {
					mDigit(false);
				}
				else if (((LA(1) >= 'a' && LA(1) <= 'f')) && ((LA(2) >= '\u0000' && LA(2) <= '\u00ff')) && (true)) {
					matchRange('a','f');
				}
				else if (((LA(1) >= 'A' && LA(1) <= 'F')) && ((LA(2) >= '\u0000' && LA(2) <= '\u00ff')) && (true)) {
					matchRange('A','F');
				}
				else {
					if ( _cnt466>=1 ) { break _loop466; } else {throw new NoViableAltForCharException((char)LA(1), getFilename(), getLine(), getColumn());}
				}
				
				_cnt466++;
			} while (true);
			}
			break;
		}
		default:
			if ((_tokenSet_16.member(LA(1)))) {
				{
				match(_tokenSet_16);
				}
			}
		else {
			throw new NoViableAltForCharException((char)LA(1), getFilename(), getLine(), getColumn());
		}
		}
		}
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	protected final void mBadStringLiteral(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = BadStringLiteral;
		int _saveIndex;
		
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	protected final void mIntSuffix(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = IntSuffix;
		int _saveIndex;
		
		switch ( LA(1)) {
		case 'L':
		{
			match('L');
			break;
		}
		case 'l':
		{
			match('l');
			break;
		}
		case 'U':
		{
			match('U');
			break;
		}
		case 'u':
		{
			match('u');
			break;
		}
		case 'I':
		{
			match('I');
			break;
		}
		case 'i':
		{
			match('i');
			break;
		}
		case 'J':
		{
			match('J');
			break;
		}
		case 'j':
		{
			match('j');
			break;
		}
		default:
		{
			throw new NoViableAltForCharException((char)LA(1), getFilename(), getLine(), getColumn());
		}
		}
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	protected final void mNumberSuffix(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = NumberSuffix;
		int _saveIndex;
		
		switch ( LA(1)) {
		case 'I':  case 'J':  case 'L':  case 'U':
		case 'i':  case 'j':  case 'l':  case 'u':
		{
			mIntSuffix(false);
			break;
		}
		case 'F':
		{
			match('F');
			break;
		}
		case 'f':
		{
			match('f');
			break;
		}
		default:
		{
			throw new NoViableAltForCharException((char)LA(1), getFilename(), getLine(), getColumn());
		}
		}
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	protected final void mExponent(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = Exponent;
		int _saveIndex;
		
		{
		switch ( LA(1)) {
		case 'e':
		{
			match('e');
			break;
		}
		case 'E':
		{
			match('E');
			break;
		}
		default:
		{
			throw new NoViableAltForCharException((char)LA(1), getFilename(), getLine(), getColumn());
		}
		}
		}
		{
		switch ( LA(1)) {
		case '+':
		{
			match('+');
			break;
		}
		case '-':
		{
			match('-');
			break;
		}
		case '0':  case '1':  case '2':  case '3':
		case '4':  case '5':  case '6':  case '7':
		case '8':  case '9':
		{
			break;
		}
		default:
		{
			throw new NoViableAltForCharException((char)LA(1), getFilename(), getLine(), getColumn());
		}
		}
		}
		{
		int _cnt474=0;
		_loop474:
		do {
			if (((LA(1) >= '0' && LA(1) <= '9'))) {
				mDigit(false);
			}
			else {
				if ( _cnt474>=1 ) { break _loop474; } else {throw new NoViableAltForCharException((char)LA(1), getFilename(), getLine(), getColumn());}
			}
			
			_cnt474++;
		} while (true);
		}
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	public final void mIDMEAT(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = IDMEAT;
		int _saveIndex;
		Token i=null;
		
		mID(true);
		i=_returnToken;
		if ( inputState.guessing==0 ) {
			
			if ( i.getType() == LITERAL___extension__ ) {
			_ttype = Token.SKIP;
			} else {
			_ttype = i.getType();
			}
			
		}
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	public final void mWideCharLiteral(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = WideCharLiteral;
		int _saveIndex;
		
		match('L');
		mCharLiteral(false);
		if ( inputState.guessing==0 ) {
			_ttype = CharLiteral;
		}
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	public final void mWideStringLiteral(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = WideStringLiteral;
		int _saveIndex;
		
		match('L');
		mStringLiteral(false);
		if ( inputState.guessing==0 ) {
			_ttype = StringLiteral;
		}
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	
	private static final long[] mk_tokenSet_0() {
		long[] data = { 68719476736L, 576460745995190270L, 0L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_0 = new BitSet(mk_tokenSet_0());
	private static final long[] mk_tokenSet_1() {
		long[] data = new long[8];
		data[0]=-4398046520321L;
		for (int i = 1; i<=3; i++) { data[i]=-1L; }
		return data;
	}
	public static final BitSet _tokenSet_1 = new BitSet(mk_tokenSet_1());
	private static final long[] mk_tokenSet_2() {
		long[] data = new long[8];
		data[0]=-1025L;
		for (int i = 1; i<=3; i++) { data[i]=-1L; }
		return data;
	}
	public static final BitSet _tokenSet_2 = new BitSet(mk_tokenSet_2());
	private static final long[] mk_tokenSet_3() {
		long[] data = { 4294971904L, 2251799813685248L, 0L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_3 = new BitSet(mk_tokenSet_3());
	private static final long[] mk_tokenSet_4() {
		long[] data = { 4294971904L, 6755399441055744L, 0L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_4 = new BitSet(mk_tokenSet_4());
	private static final long[] mk_tokenSet_5() {
		long[] data = { 4294971904L, 137438953472L, 0L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_5 = new BitSet(mk_tokenSet_5());
	private static final long[] mk_tokenSet_6() {
		long[] data = { 4294971904L, 70506183131136L, 0L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_6 = new BitSet(mk_tokenSet_6());
	private static final long[] mk_tokenSet_7() {
		long[] data = { 4294971904L, 17592186044416L, 0L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_7 = new BitSet(mk_tokenSet_7());
	private static final long[] mk_tokenSet_8() {
		long[] data = { 288019274214150656L, 2199023255552L, 0L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_8 = new BitSet(mk_tokenSet_8());
	private static final long[] mk_tokenSet_9() {
		long[] data = new long[8];
		data[0]=-9217L;
		for (int i = 1; i<=3; i++) { data[i]=-1L; }
		return data;
	}
	public static final BitSet _tokenSet_9 = new BitSet(mk_tokenSet_9());
	private static final long[] mk_tokenSet_10() {
		long[] data = { 288019269919178752L, 137438953504L, 0L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_10 = new BitSet(mk_tokenSet_10());
	private static final long[] mk_tokenSet_11() {
		long[] data = { 0L, 9031663390561856L, 0L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_11 = new BitSet(mk_tokenSet_11());
	private static final long[] mk_tokenSet_12() {
		long[] data = { 0L, 9031388512654848L, 0L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_12 = new BitSet(mk_tokenSet_12());
	private static final long[] mk_tokenSet_13() {
		long[] data = new long[8];
		data[0]=-17179878401L;
		data[1]=-268435457L;
		for (int i = 2; i<=3; i++) { data[i]=-1L; }
		return data;
	}
	public static final BitSet _tokenSet_13 = new BitSet(mk_tokenSet_13());
	private static final long[] mk_tokenSet_14() {
		long[] data = { 287949450930814976L, 541165879422L, 0L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_14 = new BitSet(mk_tokenSet_14());
	private static final long[] mk_tokenSet_15() {
		long[] data = new long[8];
		data[0]=-549755813889L;
		for (int i = 1; i<=3; i++) { data[i]=-1L; }
		return data;
	}
	public static final BitSet _tokenSet_15 = new BitSet(mk_tokenSet_15());
	private static final long[] mk_tokenSet_16() {
		long[] data = new long[8];
		data[0]=-71776119061217281L;
		data[1]=-72057594037927937L;
		for (int i = 2; i<=3; i++) { data[i]=-1L; }
		return data;
	}
	public static final BitSet _tokenSet_16 = new BitSet(mk_tokenSet_16());
	
	}
