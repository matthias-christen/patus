package ch.unibas.cs.hpwc.patus.symbolic;

import java.util.Map;
import java.util.HashMap;

class Token {
	public int kind;    // token kind
	public int pos;     // token position in bytes in the source text (starting at 0)
	public int charPos; // token position in characters in the source text (starting at 0)
	public int col;     // token column (starting at 1)
	public int line;    // token line (starting at 1)
	public String val;  // token value
	public Token next;  // ML 2005-03-11 Peek tokens are kept in linked list
}

//-----------------------------------------------------------------------------------
// Buffer
//-----------------------------------------------------------------------------------
class Buffer
{
	public static final int EOF = Character.MAX_VALUE + 1;

	private String m_strText;
	private int m_nPos;
	
	public Buffer (String strText)
	{
		m_strText = strText;
		m_nPos = 0;
	}

	public int Read ()
	{
		if (m_nPos < m_strText.length ())
			return m_strText.charAt (m_nPos++);
		return EOF;
	}

	public int Peek() {
		int curPos = getPos();
		int ch = Read();
		setPos(curPos);
		return ch;
	}

	public String GetString (int nStartPos, int nEndPos)
	{
		return m_strText.substring (nStartPos, Math.min (nEndPos, m_strText.length () - 1));
	}

	public int getPos ()
	{
		return m_nPos;
	}

	public void setPos (int nPos)
	{
		m_nPos = nPos;
		if (m_nPos > m_strText.length ())
			m_nPos = m_strText.length ();		
	}

	public String getText ()
	{
		return m_strText;
	}
}

//-----------------------------------------------------------------------------------
// StartStates  -- maps characters to start states of tokens
//-----------------------------------------------------------------------------------
class StartStates {
	private static class Elem {
		public int key, val;
		public Elem next;
		public Elem(int key, int val) { this.key = key; this.val = val; }
	}

	private Elem[] tab = new Elem[128];

	public void set(int key, int val) {
		Elem e = new Elem(key, val);
		int k = key % 128;
		e.next = tab[k]; tab[k] = e;
	}

	public int state(int key) {
		Elem e = tab[key % 128];
		while (e != null && e.key != key) e = e.next;
		return e == null ? 0: e.val;
	}
}

//-----------------------------------------------------------------------------------
// Scanner
//-----------------------------------------------------------------------------------
public class Scanner {
	static final char EOL = '\n';
	static final int  eofSym = 0;
	static final int maxT = 24;
	static final int noSym = 24;


	public Buffer buffer; // scanner buffer

	Token t;           // current token
	int ch;            // current input character
	int pos;           // byte position of current character
	int charPos;       // position by unicode characters starting with 0
	int col;           // column number of current character
	int line;          // line number of current character
	int oldEols;       // EOLs that appeared in a comment;
	static final StartStates start; // maps initial token character to start state
	static final Map<String, Integer> literals;      // maps literal strings to literal kinds

	Token tokens;      // list of tokens already peeked (first token is a dummy)
	Token pt;          // current peek token
	
	char[] tval = new char[16]; // token text used in NextToken(), dynamically enlarged
	int tlen;          // length of current token


	static {
		start = new StartStates();
		literals = new HashMap<String, Integer>();
		for (int i = 36; i <= 37; ++i) start.set(i, 1);
		for (int i = 65; i <= 90; ++i) start.set(i, 1);
		for (int i = 95; i <= 95; ++i) start.set(i, 1);
		for (int i = 97; i <= 122; ++i) start.set(i, 1);
		for (int i = 48; i <= 57; ++i) start.set(i, 7);
		start.set(61, 8); 
		start.set(35, 9); 
		start.set(60, 23); 
		start.set(62, 24); 
		start.set(43, 12); 
		start.set(45, 13); 
		start.set(42, 14); 
		start.set(47, 15); 
		start.set(94, 16); 
		start.set(40, 17); 
		start.set(41, 18); 
		start.set(44, 19); 
		start.set(46, 20); 
		start.set(91, 21); 
		start.set(93, 22); 
		start.set(Buffer.EOF, -1);
		literals.put("int", new Integer(17));
		literals.put("float", new Integer(18));
		literals.put("double", new Integer(19));

	}
	
	public Scanner(String s) {
		buffer = new Buffer(s);
		Init();
	}
	
	void Init () {
		pos = -1; line = 1; col = 0; charPos = -1;
		oldEols = 0;
		NextCh();
		pt = tokens = new Token();  // first token is a dummy
	}
	
	void NextCh() {
		if (oldEols > 0) { ch = EOL; oldEols--; }
		else {
			pos = buffer.getPos();
			// buffer reads unicode chars, if UTF8 has been detected
			ch = buffer.Read(); col++; charPos++;
			// replace isolated '\r' by '\n' in order to make
			// eol handling uniform across Windows, Unix and Mac
			if (ch == '\r' && buffer.Peek() != '\n') ch = EOL;
			if (ch == EOL) { line++; col = 0; }
		}

	}
	
	void AddCh() {
		if (tlen >= tval.length) {
			char[] newBuf = new char[2 * tval.length];
			System.arraycopy(tval, 0, newBuf, 0, tval.length);
			tval = newBuf;
		}
		if (ch != Buffer.EOF) {
			tval[tlen++] = (char)ch; 

			NextCh();
		}

	}
	

	boolean Comment0() {
		int level = 1, pos0 = pos, line0 = line, col0 = col, charPos0 = charPos;
		NextCh();
		if (ch == '/') {
			NextCh();
			for(;;) {
				if (ch == 10) {
					level--;
					if (level == 0) { oldEols = line - line0; NextCh(); return true; }
					NextCh();
				} else if (ch == Buffer.EOF) return false;
				else NextCh();
			}
		} else {
			buffer.setPos(pos0); NextCh(); line = line0; col = col0; charPos = charPos0;
		}
		return false;
	}

	boolean Comment1() {
		int level = 1, pos0 = pos, line0 = line, col0 = col, charPos0 = charPos;
		NextCh();
		if (ch == '*') {
			NextCh();
			for(;;) {
				if (ch == '*') {
					NextCh();
					if (ch == '/') {
						level--;
						if (level == 0) { oldEols = line - line0; NextCh(); return true; }
						NextCh();
					}
				} else if (ch == '/') {
					NextCh();
					if (ch == '*') {
						level++; NextCh();
					}
				} else if (ch == Buffer.EOF) return false;
				else NextCh();
			}
		} else {
			buffer.setPos(pos0); NextCh(); line = line0; col = col0; charPos = charPos0;
		}
		return false;
	}


	void CheckLiteral() {
		String val = t.val;

		Integer kind = literals.get(val);
		if (kind != null) {
			t.kind = kind.intValue();
		}
	}

	Token NextToken() {
		while (ch == ' ' ||
			ch >= 9 && ch <= 10 || ch == 13
		) NextCh();
		if (ch == '/' && Comment0() ||ch == '/' && Comment1()) return NextToken();
		int recKind = noSym;
		int recEnd = pos;
		t = new Token();
		t.pos = pos; t.col = col; t.line = line; t.charPos = charPos;
		int state = start.state(ch);
		tlen = 0; AddCh();

		loop: for (;;) {
			switch (state) {
				case -1: { t.kind = eofSym; break loop; } // NextCh already done 
				case 0: {
					if (recKind != noSym) {
						tlen = recEnd - t.pos;
						SetScannerBehindT();
					}
					t.kind = recKind; break loop;
				} // NextCh already done
				case 1:
					recEnd = pos; recKind = 1;
					if (ch >= '$' && ch <= '%' || ch >= '0' && ch <= '9' || ch >= 'A' && ch <= 'Z' || ch == '_' || ch >= 'a' && ch <= 'z') {AddCh(); state = 1; break;}
					else {t.kind = 1; t.val = new String(tval, 0, tlen); CheckLiteral(); return t;}
				case 2:
					recEnd = pos; recKind = 3;
					if (ch >= '0' && ch <= '9') {AddCh(); state = 3; break;}
					else {t.kind = 3; break loop;}
				case 3:
					recEnd = pos; recKind = 3;
					if (ch >= '0' && ch <= '9') {AddCh(); state = 3; break;}
					else if (ch == 'e') {AddCh(); state = 4; break;}
					else {t.kind = 3; break loop;}
				case 4:
					if (ch >= '0' && ch <= '9') {AddCh(); state = 6; break;}
					else if (ch == '+' || ch == '-') {AddCh(); state = 5; break;}
					else {state = 0; break;}
				case 5:
					if (ch >= '0' && ch <= '9') {AddCh(); state = 6; break;}
					else {state = 0; break;}
				case 6:
					recEnd = pos; recKind = 3;
					if (ch >= '0' && ch <= '9') {AddCh(); state = 6; break;}
					else {t.kind = 3; break loop;}
				case 7:
					recEnd = pos; recKind = 2;
					if (ch >= '0' && ch <= '9') {AddCh(); state = 7; break;}
					else if (ch == '.') {AddCh(); state = 2; break;}
					else {t.kind = 2; break loop;}
				case 8:
					{t.kind = 4; break loop;}
				case 9:
					{t.kind = 5; break loop;}
				case 10:
					{t.kind = 6; break loop;}
				case 11:
					{t.kind = 7; break loop;}
				case 12:
					{t.kind = 10; break loop;}
				case 13:
					{t.kind = 11; break loop;}
				case 14:
					{t.kind = 12; break loop;}
				case 15:
					{t.kind = 13; break loop;}
				case 16:
					{t.kind = 14; break loop;}
				case 17:
					{t.kind = 15; break loop;}
				case 18:
					{t.kind = 16; break loop;}
				case 19:
					{t.kind = 20; break loop;}
				case 20:
					{t.kind = 21; break loop;}
				case 21:
					{t.kind = 22; break loop;}
				case 22:
					{t.kind = 23; break loop;}
				case 23:
					recEnd = pos; recKind = 8;
					if (ch == '=') {AddCh(); state = 10; break;}
					else {t.kind = 8; break loop;}
				case 24:
					recEnd = pos; recKind = 9;
					if (ch == '=') {AddCh(); state = 11; break;}
					else {t.kind = 9; break loop;}

			}
		}
		t.val = new String(tval, 0, tlen);
		return t;
	}
	
	private void SetScannerBehindT() {
		buffer.setPos(t.pos);
		NextCh();
		line = t.line; col = t.col; charPos = t.charPos;
		for (int i = 0; i < tlen; i++) NextCh();
	}
	
	// get the next token (possibly a token already seen during peeking)
	public Token Scan () {
		if (tokens.next == null) {
			return NextToken();
		} else {
			pt = tokens = tokens.next;
			return tokens;
		}
	}

	// get the next token, ignore pragmas
	public Token Peek () {
		do {
			if (pt.next == null) {
				pt.next = NextToken();
			}
			pt = pt.next;
		} while (pt.kind > maxT); // skip pragmas

		return pt;
	}

	// make sure that peeking starts at current scan position
	public void ResetPeek () { pt = tokens; }

} // end Scanner
