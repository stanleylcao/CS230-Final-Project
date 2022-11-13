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


 Project Milestone
 - Figure out AWS logistics (how to get dataset on cloud computer)
 - Is our project novel enough?
 - Complex enough?

 
 Next Steps:
 - End of game token to denote the end of the game
 - Normalize each feature
 - Adjusts the number steps of the longest game to be longer than 100 moves
 - How do we supervise our model?
  1. What we are currently doing is that we are taking the model's final output,
     like in the case of a many to one RNN model, and comparing that to the
     ground truth via MSE loss
  2. Another alternative is to take the output of each RNN layer application,
     and  


