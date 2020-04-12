# Changelog 

## 2020.3.3 - Pre-release

NOTE: This version includes changes to both the way that model artefacts are packaged and saved, and the way that data are laded and parsed from tsv files. This results in a significantly faster training time (c.14 hours -> c.0.5 hour), but older models will no longer be compatible. For compatibility you must use multitask modles > 2020.3.19, splitting models > 2020.3.6, and parisng models > 2020.3.8. These models currently perform less well than previous versions, but performance is expected to improve with more data and experimentation predominatly around sequence length.

* Adds support for a Multitask models as in the original Rodrigues paper
* Combines artefacts into a single `indices.pickle` rather than the several previous pickles. Now the model just requires the embedding, `indices.pickle`, and `weights.h5`.
* Updates load_tsv to better handle quoting.


## 2020.3.2 - Pre-release

* Adds parse command that can be called with `python -m deep_reference_parser parse` 
* Rename predict command to 'split' which can be called with `python -m deep_reference_parser parse` 
* Squashes most `tensorflow`, `keras_contrib`, and `numpy` warnings in `__init__.py` resulting from old versions and soon-to-be deprecated functions.
* Reduces verbosity of logging, improving CLI clarity.

## 2020.2.0 - Pre-release

First release. Features train and predict functions tested mainly for the task of labelling reference (e.g. academic references) spans in policy documents (e.g. documents produced by government, NGOs, etc).

