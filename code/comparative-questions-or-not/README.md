
The ensemble classifier consist of the following steps: `apply_rules_and_logistic_regression.ipynb` implements the lexico-syntactic classification rules and logistic regression classifier. The notebook `neural-classifiers.ipynb` contains the code for representing questions using different Transformer models and training a feedforward deep neural network on these representations. Each classifier returns the prediction probabilities. You can save the results after each step or directly use the pre-calculated results in `results.zip` to combine the classifiers into an ensemble implemented in `ensemble.ipynb`.

`transformer-scripts` contains the implementation of classifying questions using different pre-trained Transformer models.
