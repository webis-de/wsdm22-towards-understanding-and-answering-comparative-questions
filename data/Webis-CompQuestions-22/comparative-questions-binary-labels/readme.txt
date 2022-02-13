###########

Dataset used for classifying questions as comparative or not.

The data sources are the questions from MS MARCO, Natural Questions, and Quora Question Pairs. 

########### 

The dataset structure:

comp:     1 if a question is comparative.
question: original question from the data source.
clean:    lower-cased question, punctuation removed.
pos:      POS tags using Stanza.
lemma:    lemmas using Stanza.
