//
// This file was generated by the JavaTM Architecture for XML Binding(JAXB) Reference Implementation, v2.1-b02-fcs 
// See <a href="http://java.sun.com/xml/jaxb">http://java.sun.com/xml/jaxb</a> 
// Any modifications to this file will be lost upon recompilation of the source schema. 
// Generated on: 2012.07.09 at 04:09:57 PM CEST 
//


package ch.unibas.cs.hpwc.patus.arch;

import java.math.BigInteger;
import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlID;
import javax.xml.bind.annotation.XmlSchemaType;
import javax.xml.bind.annotation.XmlType;
import javax.xml.bind.annotation.adapters.CollapsedStringAdapter;
import javax.xml.bind.annotation.adapters.XmlJavaTypeAdapter;


/**
 * <p>Java class for typeRegisterClass complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="typeRegisterClass">
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;attribute name="name" use="required" type="{http://www.w3.org/2001/XMLSchema}ID" />
 *       &lt;attribute name="type" use="required" type="{}typeRegisterType" />
 *       &lt;attribute name="width" use="required" type="{http://www.w3.org/2001/XMLSchema}integer" />
 *       &lt;attribute name="datatype" use="required" type="{http://www.w3.org/2001/XMLSchema}string" />
 *       &lt;attribute name="bitrange" type="{}typeRangeType" />
 *       &lt;attribute name="subregister-of" type="{http://www.w3.org/2001/XMLSchema}string" />
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "typeRegisterClass")
public class TypeRegisterClass {

    @XmlAttribute(required = true)
    @XmlJavaTypeAdapter(CollapsedStringAdapter.class)
    @XmlID
    @XmlSchemaType(name = "ID")
    protected String name;
    @XmlAttribute(required = true)
    protected TypeRegisterType type;
    @XmlAttribute(required = true)
    protected BigInteger width;
    @XmlAttribute(required = true)
    protected String datatype;
    @XmlAttribute
    protected String bitrange;
    @XmlAttribute(name = "subregister-of")
    protected String subregisterOf;

    /**
     * Gets the value of the name property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getName() {
        return name;
    }

    /**
     * Sets the value of the name property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setName(String value) {
        this.name = value;
    }

    /**
     * Gets the value of the type property.
     * 
     * @return
     *     possible object is
     *     {@link TypeRegisterType }
     *     
     */
    public TypeRegisterType getType() {
        return type;
    }

    /**
     * Sets the value of the type property.
     * 
     * @param value
     *     allowed object is
     *     {@link TypeRegisterType }
     *     
     */
    public void setType(TypeRegisterType value) {
        this.type = value;
    }

    /**
     * Gets the value of the width property.
     * 
     * @return
     *     possible object is
     *     {@link BigInteger }
     *     
     */
    public BigInteger getWidth() {
        return width;
    }

    /**
     * Sets the value of the width property.
     * 
     * @param value
     *     allowed object is
     *     {@link BigInteger }
     *     
     */
    public void setWidth(BigInteger value) {
        this.width = value;
    }

    /**
     * Gets the value of the datatype property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getDatatype() {
        return datatype;
    }

    /**
     * Sets the value of the datatype property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setDatatype(String value) {
        this.datatype = value;
    }

    /**
     * Gets the value of the bitrange property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getBitrange() {
        return bitrange;
    }

    /**
     * Sets the value of the bitrange property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setBitrange(String value) {
        this.bitrange = value;
    }

    /**
     * Gets the value of the subregisterOf property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getSubregisterOf() {
        return subregisterOf;
    }

    /**
     * Sets the value of the subregisterOf property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setSubregisterOf(String value) {
        this.subregisterOf = value;
    }

}
