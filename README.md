# PersVecTrain

Trains persona steering vectors (one vector per trait at one layer of a frozen Hugging Face model) with
BiPO's persona loss plus a KL term to the unsteered model. It reads the outputs of the sibling repo
PersVecFast (extracted responses and activations) and imports its `persvecfast` package.

## Install

`persvecfast` is not a pip dependency; install it editable from the sibling checkout first:

    pip install -e ../PersVecFast -e ".[test]"

## Commands (planned)

- `check` - validate the PersVecFast inputs and the layer choice
- `reference` - compute the reference (extracted) vectors and their random twins
- `train` - train a vector (solo or joint) and write `train.json`, `steps.jsonl`, vectors
- `report` - summarise runs and evals
