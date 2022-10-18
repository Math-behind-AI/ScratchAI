# Sentiment Ananlysis of Commodity News using Bidirectional LSTM

## :dart: GOAL

* Develop a model to classify Commodity Sentiments into predefined categories. 
Note: Text classification is an example of supervised machine learning since we train the model with labelled data.

## :brain: Sentiment Analysis

* Sentiment analysis is contextual mining of text which identifies and extracts subjective information in source material, and helping a business to understand the social sentiment of their brand, product or service while monitoring online conversations.
* By using sentiment analysis, you gauge how customers feel about different areas of your business without having to read thousands of customer comments at once. If you have thousands of feedback per month, it is impossible for one person to read all of these responses

![main](https://user-images.githubusercontent.com/86421205/196357653-9d4b533b-6322-430e-bb9b-3657e834e49a.png)


## :pencil2: Bidirectional LSTM
* LSTM stands for Long Short Term Memory. It is a special kind of Recurrent Neural Network [RNN].
* While computing an embedding matrix, the meaning of every word and its calculations (called hidden states) are stored.
* RNNs are not capable of storing long term dependencies due to vanishing gradient.
* LSTMs have a gated network which can handle the problem of vanishing gradient. The LSTM consists of three parts : 
- 1. *Forget Gate*
- 2. *Input Gate*
- 3. *Output Gate*

![Screenshot](https://user-images.githubusercontent.com/86421205/196361910-0b9dad40-93be-4b48-822d-a3964c869eff.jpg)

* Bidirectional LSTM consists of two models. The first model learns the sequence of input provided and the second moel learns the reverse of that sequence.
* This structure allows the networks to have both backward and forward information about the sequence at every time step.

![1_B5NHtY8_Y4we0DE4Y-acBA](https://user-images.githubusercontent.com/86421205/196361977-9f476005-6c92-4960-9c73-d02ba5adfaad.png)
