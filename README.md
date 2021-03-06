Transition-based AMR Parser
============================

Transition-based parser for Abstract Meaning Representation (AMR) in Pytorch. The code includes two fundamental components.

1. A State machine and oracle transforming the sequence-to-graph task into a sequence-to-sequence problem. This follows the AMR oracles in [(Ballesteros and Al-Onaizan 2017)](https://arxiv.org/abs/1707.07755v1) with improvements from [(Naseem et al 2019)](https://arxiv.org/abs/1905.13370) and [(Fernandez Astudillo et al 2020)](https://openreview.net/pdf?id=b36spsuUAde)

2. Two structured sequence-to-sequence models able to encode the parser state. This includes stack-LSTM [(Dyer et al)](https://arxiv.org/pdf/1505.08075.pdf) and the stack-Transformer [(Fernandez Astudillo et al 2020)](https://openreview.net/pdf?id=b36spsuUAde). 

Current version is `0.3.2`. Initial commit developed by Miguel Ballesteros and Austin Blodgett while at IBM. 

## Manual Installation

The code has been tested on Python `3.6`. We use a script to activate
conda/pyenv and virtual environments. If you prefer to handle this yourself
just create an empty file (the scripts will assume it exists in any case)

```bash
touch set_environment.sh
```

then patch and install fairseq

```
. set_environment.sh
git clone https://github.com/pytorch/fairseq.git
cd fairseq
git checkout -b stack-transformer-v0.3.2 a33ac06 
git apply ../transition_amr_parser/stack_transformer/fairseq_a33ac06.patch
pip install .
cd ..
```

the install the main repo

```bash
git clone https://github.com/IBM/transition-amr-parser.git
cd transition-amr-parser
git checkout v0.3.2
pip install .
cd ..
```

as well as the smatch evaluation tool.

```
git clone https://github.com/snowblink14/smatch.git 
cd smatch
git checkout v1.0.4
pip install .
cd ..
```

The spacy tools will be updated on first use. To do this manually do

```bash
python -m spacy download en
```

You can check if it worked using

```bash
python tests/correctly_installed.py
```

## Decode with Pre-trained model

To use from the command line with a trained model do

```bash
amr-parse \
  --in-tokenized-sentences $input_file \
  --in-checkpoint $in_checkpoint \
  --out-amr file.amr
```

It will parse each line of `$input_file` separately (assumed tokenized).
`$in_checkpoint` is the pytorch checkpoint of a trained model. The `file.amr`
will contain the PENMAN notation AMR with additional alignment information as
comments.

To use from other Python code with a trained model do

```python
from transition_amr_parser.stack_transformer_amr_parser import AMRParser
parser = AMRParser.from_checkpoint(in_checkpoint) 
annotations = parser.parse_sentences([['The', 'boy', 'travels'], ['He', 'visits', 'places']])
print(annotations.toJAMRString())
```
