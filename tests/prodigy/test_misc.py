from deep_reference_parser.prodigy import prodigy_to_conll

def test_prodigy_to_conll():

    before = [
        {"text": "References",},
        {"text": "37. No single case of malaria reported in"},
        {
            "text": "an essential requirement for the correct labelling of potency for therapeutic"
        },
        {"text": "EQAS, quality control for STI"},
    ]

    after = "DOCSTART\n\nReferences\n\n37\n.\nNo\nsingle\ncase\nof\nmalaria\nreported\nin\n\nan\nessential\nrequirement\nfor\nthe\ncorrect\nlabelling\nof\npotency\nfor\ntherapeutic\n\nEQAS\n,\nquality\ncontrol\nfor\nSTI"

    out = prodigy_to_conll(before)

    assert after == out
