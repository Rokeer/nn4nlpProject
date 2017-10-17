# nn4nlpProject

This is the repository for Neural Network for Natural Language Processing(CS11-747) Fall-2017 class project:

Task:
Question Answering using <a href="https://rajpurkar.github.io/SQuAD-explorer/">SQuAD (Stanford question answering Dataset)</a>

-----------------------------------
Group members:
-----------------------------------

Tazin Afrin (tafrin@andrew.cmu.edu)

Haoran Zhang (haoranz3@andrew.cmu.edu)

Ahmed Magooda (amagooda@andrew.cmu.edu)


-----------------------------------
How to run:
-----------------------------------

Run the code "python ./Code/Assignment_1_Model.py" 

This file should do training using the training set for 200 epochs and validating using the devset every 10 epochs out of the 200. The code also saves a model every 10 iterations

Requirements:

  Input Files: /data/train_lines
               and /data/dev_lines
               
  How to generate the input files: "python ./Code/readJSON.py"
  
  The readJSON.py files takes direct raw data from the SQuAD dataset and outputs a formatted file to be used in the learning     algorithm.


