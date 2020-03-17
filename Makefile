.DEFAULT_GOAL := all

#
# Set file and version for embeddings and model, plus local paths
#

NAME := deep_reference_parser

EMBEDDING_PATH := embeddings
WORD_EMBEDDING := 2020.1.1-wellcome-embeddings-300
WORD_EMBEDDING_TEST := 2020.1.1-wellcome-embeddings-10-test

MODEL_PATH := models
MODEL_VERSION := 2019.12.0

#
# S3 Bucket
#

S3_BUCKET := s3://datalabs-public/deep_reference_parser
S3_BUCKET_HTTP := https://datalabs-public.s3.eu-west-2.amazonaws.com/deep_reference_parser

#
# Create a virtualenv for local dev
#

VIRTUALENV := build/virtualenv

$(VIRTUALENV)/.installed: requirements.txt
	@if [ -d $(VIRTUALENV) ]; then rm -rf $(VIRTUALENV); fi
	@mkdir -p $(VIRTUALENV)
	virtualenv --python python3 $(VIRTUALENV)
	$(VIRTUALENV)/bin/pip3 install -r requirements.txt
	$(VIRTUALENV)/bin/pip3 install -r requirements_test.txt
	$(VIRTUALENV)/bin/pip3 install -e .
	touch $@

$(VIRTUALENV)/.en_core_web_sm: 
	$(VIRTUALENV)/bin/python -m spacy download en_core_web_sm
	touch $@


.PHONY: virtualenv
virtualenv: $(VIRTUALENV)/.installed $(VIRTUALENV)/.en_core_web_sm

#
# Get the word embedding
#

# Set the tar.gz as intermediate so it will be removed automatically
.INTERMEDIATE: $(EMBEDDINGS_PATH)/$(WORD_EMBEDDING).tar.gz

$(EMBEDDING_PATH)/$(WORD_EMBEDDING).tar.gz:
	@mkdir -p $(@D)
	curl $(S3_BUCKET_HTTP)/embeddings/$(@F) --output $@

$(EMBEDDING_PATH)/$(WORD_EMBEDDING).txt: $(EMBEDDING_PATH)/$(WORD_EMBEDDING).tar.gz
	tar -zxvf $< vectors.txt
	tail -n +2 vectors.txt > $@
	rm vectors.txt

embeddings: $(EMBEDDING_PATH)/$(WORD_EMBEDDING).txt

#
# Get the model artefacts and weights
#

artefact_targets = char2ind.pickle ind2label.pickle ind2word.pickle \
					label2ind.pickle maxes.pickle word2ind.pickle \
					weights.h5

artefacts = $(addprefix $(MODEL_PATH)/$(MODEL_VERSION)/, $(artefact_targets))

$(artefacts):
	@mkdir -p $(@D)
	aws s3 cp $(S3_BUCKET)/models/$(MODEL_VERSION)/$(@F) $@

models: $(artefacts)


datasets = data/splitting/2019.12.0_splitting_train.tsv \
           data/splitting/2019.12.0_splitting_test.tsv \
           data/splitting/2019.12.0_splitting_valid.tsv \
		   data/parsing/2020.3.2_parsing_train.tsv \
           data/parsing/2020.3.2_parsing_test.tsv \
           data/parsing/2020.3.2_parsing_valid.tsv


rodrigues_datasets = data/rodrigues/clean_train.txt \
		   			 data/rodrigues/clean_test.txt \
		   			 data/rodrigues/clean_valid.txt 

RODRIGUES_DATA_URL =  https://github.com/dhlab-epfl/LinkedBooksDeepReferenceParsing/raw/master/dataset/

$(datasets): 
	@ mkdir -p $(@D)
	curl -s $(S3_BUCKET_HTTP)/$@ --output $@

$(rodrigues_datasets): 
	@ mkdir -p data/rodrigues
	curl -sL $(RODRIGUES_DATA_URL)/$(@F) --output $@

data: $(datasets) $(rodrigues_datasets)

#
# Add model artefacts to s3
#

sync_model_to_s3:
	aws s3 sync --acl public-read $(MODEL_PATH)/$(MODEL_VERSION) \
		$(S3_BUCKET)/models/$(MODEL_VERSION)

#
# Ship a new wheel to public s3 bucket, containing model weights
#

# Ship the wheel to the datalabs-public s3 bucket. Need to remove these build
# artefacts otherwise they can make a mess of your build! Public access to
# the wheel is granted with the --acl public-read flag.

.PHONY: dist
dist:
	-rm build/bin build/bdist.linux-x86_64 -r
	-rm deep_reference_parser-20* -r
	-rm dist/*
	$(VIRTUALENV)/bin/python3 setup.py sdist bdist_wheel
	aws s3 cp --recursive --exclude "*" --include "*.whl" --acl public-read dist/ $(S3_BUCKET) 

#
# Tests
#

$(EMBEDDING_PATH)/$(WORD_EMBEDDING_TEST).txt:
	@mkdir -p $(@D)
	curl $(S3_BUCKET_HTTP)/embeddings/$(@F) --output $@

test_embedding: $(EMBEDDING_PATH)/$(WORD_EMBEDDING_TEST).txt

test_artefacts = $(addprefix $(MODEL_PATH)/test/, $(artefact_targets))

$(test_artefacts):
	@mkdir -p $(@D)
	curl $(S3_BUCKET_HTTP)/models/test/$(@F) --output $@

.PHONY: test
test: $(test_artefacts) test_embedding
	$(VIRTUALENV)/bin/pytest --disable-warnings --tb=line --cov=deep_reference_parser

all: virtualenv model embedding test
