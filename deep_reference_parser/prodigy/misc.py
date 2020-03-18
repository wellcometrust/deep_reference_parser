import spacy

def _join_prodigy_tokens(text):
    """Return all prodigy tokens in a single string
    """

    return "\n".join([str(i) for i in text])

def prodigy_to_conll(docs):
    """
    Expect list of jsons loaded from a jsonl
    """

    nlp = spacy.load("en_core_web_sm")
    texts = [doc["text"] for doc in docs]
    docs = list(nlp.tokenizer.pipe(texts))

    out = [_join_prodigy_tokens(i) for i in docs]

    out_str = "DOCSTART\n\n" + "\n\n".join(out)

    return out_str


def prodigy_to_lists(docs):
    """
    Expect list of jsons loaded from a jsonl
    """

    nlp = spacy.load("en_core_web_sm")
    texts = [doc["text"] for doc in docs]
    docs = list(nlp.tokenizer.pipe(texts))

    out = [[str(token) for token in doc] for doc in docs]

    return out
