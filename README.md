# Deep Reference Parser

This repo contains a Bi-direction Long Short Term Memory (BiLSTM) Deep Neural Network with a stacked Conditional Random Field (CRF) for identifying references from text. The model itself is based on the work of Rodrigues et al. (2018), although the implemention here differs significantly.

## Just show me the references!!!

If you want to try out the model the quick way, there is a pre-packaged wheel containing the latest word embedding and weights available on S3. The following commands will get you started:

```
# Download the wheel from s3

pip install git+git://github.com/wellcometrust/deep_reference_parser.git#egg=deep_reference_parser


# Create references.txt with some references in it

cat > references.txt <<EOF
1 Sibbald, A, Eason, W, McAdam, J, and Hislop, A (2001). The establishment phase of a silvopastoral national network experiment in the UK. Agroforestry systems, 39, 39–53. 
2 Silva, J and Rego, F (2003). Root distribution of a Mediterranean shrubland in Portugal. Plant and Soil, 255 (2), 529–540. 
3 Sims, R, Schock, R, Adegbululgbe, A, Fenhann, J, Konstantinaviciute, I, Moomaw, W, Nimir, H, Schlamadinger, B, Torres-Martínez, J, Turner, C, Uchiyama, Y, Vuori, S, Wamukonya, N, and X. Zhang (2007). Energy Supply. In Metz, B, Davidson, O, Bosch, P, Dave, R, and Meyer, L (eds.), Climate Change 2007: Mitigation. Contribution of Working Group III to the Fourth Assessment Report of the Intergovernmental Panel on Climate Change, Cambridge University Press, Cambridge, United Kingdom and New York, NY, USA.
EOF


# Run the model. This will take a little time while the weights and embeddings 
# are downloaded - be patient!

python -m deep_reference_parser predict --verbose "$(cat references.txt)"
```

# Training your own models

To train your own models you will need to define the model hyperparameters in a config file.

```
python -m deep_reference_parser train test.ini
```


# Developing the package

To create a local virtual environment and activate it:

```
make virtualenv

# to activate

source ./build/virtualenv/bin/activate
```

## Get the embeddings and model artefacts

```
make models embeddings
```

## Testing

The package uses pytest:

```
make test
```

## References

Rodrigues Alves, D., Colavizza, G., & Kaplan, F. (2018). Deep Reference Mining From Scholarly Literature in the Arts and Humanities. Frontiers in Research Metrics and Analytics, 3(July), 1–13. https://doi.org/10.3389/frma.2018.00021
