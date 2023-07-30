# Model mechanics in SuperDuperDB

Models in SuperDuperDB differ from models in the standard AI frameworks, by
including several additional aspects:

- [preprocessing](preprocessing)
- [postprocessing](postprocessing)
- [output encoding](encoding)

## Preprocessing

All models in SuperDuperDB include the keyword argument `preprocess`. The exact meaning of this varies from framework to framework, however in general:

```{important}
`Model.preprocess` is a function which takes an individual data-point from the database, 
and prepares it for processing by the AI framework model
```

For example:

- In PyTorch (`torch`) computer vision models, a preprocessing function might:
  - Crop the image
  - Normalize the pixel values by precomputed constants
  - Resize the image
  - Convert the image to a tensor
- In Hugging Face `transformers`, and NLP model will:
  - Tokenize a sentence into word-pieces
  - Convert each word piece into a numerical ID
  - Truncate and pad the IDs
  - Compute a mask
- In Scikit-Learn, estimators operate purely at the numerical level
  - Preprocessing may be added exactly as for PyTorch
  - Alternatively users may use a `sklearn.pipeline.Pipeline` explicitly

## Postprocessing

All models in SuperDuperDB include the keyword argument `postprocess`. The goal here 
is to take the numerical estimates of a model, and convert these to a form to 
be used by database users. Examples are:

- Converting a softmax over a dictionary in NLP to point estimates and human-readable strings
- Converting a generated image-tensor into JPG or PNG format
- Performing additional inference logic, such as beam-search in neural translation

## Encoding

The aim of postprocessing is to provide outputs in a operationally useful
form. However, often this form isn't directly amenable to insertion in the `Datalayer`.
For example, MongoDB doesn't support images or tensors natively. Consequently 
each model takes the keyword `encoder`, allowing users to specify exactly
how outputs are stored in the database as `bytes`, if not supported natively 
by the database. Read more [here](encoders).