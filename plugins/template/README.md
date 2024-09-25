<!-- Auto-generated content start -->
# Template

## README.md template

This is a template for a README.md file, mainly consisting of two parts: an auto-generated section and a custom section.

The `generate_readme.py` script can be used to update the auto-generated section.

````

# {{plugin_name}}

{{description}}

## Installation

```bash
pip install {{plugin_name}}
```

## API


- [Code](https://github.com/superduper-io/superduper/tree/main/plugins/{{name}})
- [API-docs](/docs/api/plugins/{{plugin_name}})

| Class | Description |
|---|---|
{{classes_table}}
## Examples

{{examples}}


## Custom section.

````


## Generate README.md


Update the README.md file for the plugin by running the following command:

```python
python generate_readme.py plugins/openai
```


Updated all README.md files by running the following command:

```python
python generate_readme.py
```

<!-- Auto-generated content end -->

<!-- Add your additional content below -->
