<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chord Generator</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 20px;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 5px;
            background-color: #f9f9f9;
        }
        h1 {
            color: #333;
        }
        h2 {
            color: #555;
        }
        p {
            margin: 10px 0;
        }
        code {
            background-color: #eee;
            padding: 2px 4px;
            border-radius: 4px;
        }
    </style>
</head>
<body>

    <h1>Chord Generator</h1>

    <p>The <strong>Chord Generator</strong> is a transformer model trained from scratch on musical chord sequences.</p>

    <h2>Inspiration</h2>

    <p>Having played around with piano composition, I embarked on a project to train a transformer model to help me produce chord sequences. This endeavor has been both enjoyable and beneficial in enhancing my musical creativity.</p>

    <h2>How It Works</h2>

    <ul>
        <li><strong>Model Architecture</strong>: The <code>structure</code> folder contains the architecture of the model, along with useful files, such as the Tokenizer and Dataset classes.</li>
        <li><strong>Global Variables</strong>: Global variables and parameters are defined in the <code>utils</code> file.</li>
        <li><strong>Training the Model</strong>: To train the model, run the training script located at <code>run/train.py</code>.</li>
        <li><strong>Running Inference</strong>: To run inference on the saved model, execute the script found at <code>run/inference.py</code>.</li>
    </ul>

    <h2>Current Status</h2>

    <p>The project is still ongoing, which means the best results are not yet finalized. However, the code is operational, and you're welcome to give it a try!</p>

</body>
</html>
