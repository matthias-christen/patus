//
// This file was generated by the JavaTM Architecture for XML Binding(JAXB) Reference Implementation, v2.1-b02-fcs 
// See <a href="http://java.sun.com/xml/jaxb">http://java.sun.com/xml/jaxb</a> 
// Any modifications to this file will be lost upon recompilation of the source schema. 
// Generated on: 2012.10.08 at 05:45:06 PM CEST 
//


package ch.unibas.cs.hpwc.patus.arch;

import javax.xml.bind.annotation.XmlEnum;
import javax.xml.bind.annotation.XmlEnumValue;


/**
 * <p>Java class for typeDeclspec.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * <p>
 * <pre>
 * &lt;simpleType name="typeDeclspec">
 *   &lt;restriction base="{http://www.w3.org/2001/XMLSchema}string">
 *     &lt;enumeration value="kernel"/>
 *     &lt;enumeration value="localfunction"/>
 *     &lt;enumeration value="restrictedpointer"/>
 *   &lt;/restriction>
 * &lt;/simpleType>
 * </pre>
 * 
 */
@XmlEnum
public enum TypeDeclspec {

    @XmlEnumValue("kernel")
    KERNEL("kernel"),
    @XmlEnumValue("localfunction")
    LOCALFUNCTION("localfunction"),
    @XmlEnumValue("restrictedpointer")
    RESTRICTEDPOINTER("restrictedpointer");
    private final String value;

    TypeDeclspec(String v) {
        value = v;
    }

    public String value() {
        return value;
    }

    public static TypeDeclspec fromValue(String v) {
        for (TypeDeclspec c: TypeDeclspec.values()) {
            if (c.value.equals(v)) {
                return c;
            }
        }
        throw new IllegalArgumentException(v);
    }

}
