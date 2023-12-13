def get_output_table_name(model_identifier, version):
    """Get the output table name for the given model."""
    # use `_` to connect the model_identifier and version
    return f'_outputs_{model_identifier}_{version}'
