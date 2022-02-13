###########

Datasets used for parsing comparative questions by identifying comparison objects, aspects, and predicates.

Datasets:
full.tsv contains all types of comparative questions.
direct.tsv and indirect.tsv contain direct and indirect comparative questions respectively.

########### 

The dataset structure:

sentence_id: id of a respective question.
word:        word tokens.
labels:      token-level labels: OBJ, ASP, PRED for the comparison objects, aspects, and predicates, O none of them.
