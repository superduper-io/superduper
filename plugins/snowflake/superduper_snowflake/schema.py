from snowflake.snowpark.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    FloatType,
    DoubleType,
    BooleanType,
    DateType,
    TimestampType,
    DecimalType,
)


def ibis_type_to_snowpark_type(ibis_dtype):
    """Convert an Ibis data type to the closest Snowpark type."""
    
    # Integer (covers int8, int16, int32, int64 in Ibis)
    if ibis_dtype.is_integer():
        return IntegerType()
    
    # Boolean
    if ibis_dtype.is_boolean():
        return BooleanType()
    
    # Floating point (covers float32, float64 in Ibis)
    if ibis_dtype.is_floating():
        # FloatType is 32-bit, DoubleType is 64-bit
        # You could decide based on ibis_dtype here. Example:
        return DoubleType()
    
    # Decimal (e.g. Decimal(precision, scale))
    if ibis_dtype.is_decimal():
        # Get precision and scale from the Ibis type
        precision = ibis_dtype.precision
        scale = ibis_dtype.scale
        return DecimalType(precision, scale)
    
    # String
    if ibis_dtype.is_string():
        return StringType()
    
    # Date
    if ibis_dtype.is_date():
        return DateType()
    
    # Timestamp
    if ibis_dtype.is_timestamp():
        return TimestampType()
    
    # Fallback: map everything else to StringType (or VariantType, etc. if desired)
    return StringType()


def ibis_schema_to_snowpark_schema(ibis_schema):
    """
    Convert an Ibis schema (ibis.Schema) to a Snowpark StructType.
    """
    fields = []
    for col_name, col_type in ibis_schema.items():
        # Convert Ibis type to Snowpark type
        snowpark_type = ibis_type_to_snowpark_type(col_type)
        # Create a StructField; adjust `nullable` as appropriate for your use-case
        fields.append(StructField(col_name, snowpark_type, nullable=True))
    
    return StructType(fields)
