package cetus.base.grammars;

import java.util.*;

public class PreprocessorInfoChannel
{
    HashMap lineLists = new HashMap(); // indexed by Token number
    int firstValidTokenNumber = 0;
    int maxTokenNumber = 0;

    public void addLineForTokenNumber( Object line, Integer toknum )
    {
        if ( lineLists.containsKey( toknum ) ) {
            LinkedList lines = (LinkedList) lineLists.get( toknum );
            lines.add(line);
        }
        else {
            LinkedList lines = new LinkedList();
            lines.add(line);
            lineLists.put(toknum, lines);
            if ( maxTokenNumber < toknum.intValue() ) {
                maxTokenNumber = toknum.intValue();
            }
        }
    }

    public int getMaxTokenNumber()
    {
        return maxTokenNumber;
    }
        
    public LinkedList extractLinesPrecedingTokenNumber( Integer toknum )
    {
        LinkedList lines = new LinkedList();
        if (toknum == null) return lines;       
        for (int i = firstValidTokenNumber; i < toknum.intValue(); i++){
            Integer inti = new Integer(i);
            if ( lineLists.containsKey( inti ) ) {
                LinkedList tokenLineVector = (LinkedList) lineLists.get( inti );
                if ( tokenLineVector != null) {
                    //Enumeration tokenLines = tokenLineVector.elements();
                    //while ( tokenLines.hasMoreElements() ) {
                    //    lines.add( tokenLines.nextElement() );
                    //}
										for(Object o : tokenLineVector)
												lines.add(o);
                    lineLists.remove(inti);
                }
            }
        }
        firstValidTokenNumber = toknum.intValue();
        return lines;
    }

    public String toString()
    {
        StringBuffer sb = new StringBuffer("PreprocessorInfoChannel:\n");
        for (int i = 0; i <= maxTokenNumber + 1; i++){
            Integer inti = new Integer(i);
            if ( lineLists.containsKey( inti ) ) {
                LinkedList tokenLineVector = (LinkedList) lineLists.get( inti );
                if ( tokenLineVector != null) {
                    //Enumeration tokenLines = tokenLineVector.elements();
                    //while ( tokenLines.hasMoreElements() ) {
                    //    sb.append(inti + ":" + tokenLines.nextElement() + '\n');
                    //}
										for(Object o : tokenLineVector)
											sb.append(inti + ":" + o + '\n');
                }
            }
        }
        return sb.toString();
    }
}



