
The code is organized by a classification task described in the paper:

* [Comparative question classifier](comparative-questions-or-not) implements the classification of questions as comparative or not.
* [Direct questions classifier](comparative-direct-indirect-questions) for classifying a comparative question as *direct* (contains explicit comparison objects) or *indirect*. 
* [Question parsing](comparative-questions-parsing) identifies the comparison objects, aspects and predicates in questions.
* [Stance detector](stance-classification) classifies the stance of answers to comparative questions: *pro first object, pro second object, neutral, no stance*.
