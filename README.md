The Chord Generator is a transformer model, trained from scratch on only musical chord sequences.

THE INSPIRATION: Having played around with piano composition, I felt a project to train a transformer model to help me produce chord sequences could be both enjoyable and beneficial.

How it works:
The structure folder contains the architecture of the model, and a couple of useful files, such as the Tokenizer and Dataset classes.
Global variables (and parameters) are found in the utils file.
To train the model, run the train file under run/train.py.
To run inference on the saved model, run the file under run/inference.py.

The project is still currently ongoing, meaning that best results as of now aren't complete, but the code is working if you want to give it a go!
