//
// This file was generated by the JavaTM Architecture for XML Binding(JAXB) Reference Implementation, v2.1-b02-fcs 
// See <a href="http://java.sun.com/xml/jaxb">http://java.sun.com/xml/jaxb</a> 
// Any modifications to this file will be lost upon recompilation of the source schema. 
// Generated on: 2012.05.18 at 12:48:24 PM CEST 
//


package ch.unibas.cs.hpwc.patus.arch;

import javax.xml.bind.annotation.XmlEnum;
import javax.xml.bind.annotation.XmlEnumValue;


/**
 * <p>Java class for typeBaseDatatype.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * <p>
 * <pre>
 * &lt;simpleType name="typeBaseDatatype">
 *   &lt;restriction base="{http://www.w3.org/2001/XMLSchema}string">
 *     &lt;enumeration value="int"/>
 *     &lt;enumeration value="long"/>
 *     &lt;enumeration value="double"/>
 *     &lt;enumeration value="float"/>
 *   &lt;/restriction>
 * &lt;/simpleType>
 * </pre>
 * 
 */
@XmlEnum
public enum TypeBaseDatatype {

    @XmlEnumValue("int")
    INT("int"),
    @XmlEnumValue("long")
    LONG("long"),
    @XmlEnumValue("double")
    DOUBLE("double"),
    @XmlEnumValue("float")
    FLOAT("float");
    private final String value;

    TypeBaseDatatype(String v) {
        value = v;
    }

    public String value() {
        return value;
    }

    public static TypeBaseDatatype fromValue(String v) {
        for (TypeBaseDatatype c: TypeBaseDatatype.values()) {
            if (c.value.equals(v)) {
                return c;
            }
        }
        throw new IllegalArgumentException(v);
    }

}
