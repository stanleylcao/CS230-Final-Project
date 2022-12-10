# CS230-Final-Project

Models and dataset as discussed in the CS230 report. 

Data Related Files
Preprocessing.R: Preprocessing of chessdigits data in R 

for_pandas.csv: Sample output of Preprocessing.R 

baseline_model.py: Feedforward neural network model 

Models

RNN_model.py: Vanilla RNN Model with L1 loss 

RNN_Model_CE.py: Cross-entropy model for softmax classification approach

LSTM_model.py: LSTM model with L1 Loss 

LSTM_Packed.py: LSTM Model with packing of padded sequences (see report) 

Mixed_LSTM_Model.py: Model with multi-headed architecture ("mix" of static and dynamic features). 

transformer.py: Encoder-only transformers network. 

dataset.py: Dataloader for pytorch 

dataset_mixed.py: Dataloader handling the specific case of the multi-headed architecture

utils.py: Extra utility functions such as normalization, etc. 

