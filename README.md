# CS230-Final-Project

Baseline non-DL model (statistical analysis): 
- Given moves and engine evalation as labels
  - Variance of engine labels for players
  - As we get later in the game, the stronger player should have gradual increase in engine evaluation
  - 
  
 Baseline DL model:
 - Take data from KDD dataset, and generate labels, which are two dimensional vectors. The topp entry will be white, and the bottom entry will be black
 - Input data will be a sequence of vector, each vector describing the state of the game (including the engine evaluation)
 - Train an LSTM or a sequence model on this dataset.


By Thursday: 
- Understand the dataset
- How does an LSTM model help our problem
  - How do we manipulate the data so that the LSTM model will learn
- Perhaps it is possible to have the LSTM spit out ratings as sequences
