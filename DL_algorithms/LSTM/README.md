# Sentiment Analysis of Commodity News (Gold) using Bidirectional LSTM

## :dart: GOAL

* Develop a model to classify Commodity Sentiments into predefined categories. 
Note: Text classification is an example of supervised machine learning since we train the model with labelled data.

## :brain: Sentiment Analysis

* Sentiment analysis is contextual mining of text which identifies and extracts subjective information in source material, and helping a business to understand the social sentiment of their brand, product or service while monitoring online conversations.
* By using sentiment analysis, you gauge how customers feel about different areas of your business without having to read thousands of customer comments at once. If you have thousands of feedback per month, it is impossible for one person to read all of these responses

![main1](https://user-images.githubusercontent.com/86421205/196366043-b98151a9-b102-4784-a90a-ca2b40cd101b.jpg)


## :pencil2: Bidirectional LSTM
* LSTM stands for Long Short Term Memory. It is a special kind of Recurrent Neural Network [RNN].
* While computing an embedding matrix, the meaning of every word and its calculations (called hidden states) are stored.
* RNNs are not capable of storing long term dependencies due to vanishing gradient.
* LSTMs have a gated network which can handle the problem of vanishing gradient. The LSTM consists of three parts : 
 1. *Forget Gate*
 2. *Input Gate*
 3. *Output Gate*

![Screenshot](https://user-images.githubusercontent.com/86421205/196361910-0b9dad40-93be-4b48-822d-a3964c869eff.jpg)

* Bidirectional LSTM consists of two models. The first model learns the sequence of input provided and the second moel learns the reverse of that sequence.
* This structure allows the networks to have both backward and forward information about the sequence at every time step.

![1_B5NHtY8_Y4we0DE4Y-acBA](https://user-images.githubusercontent.com/86421205/196361977-9f476005-6c92-4960-9c73-d02ba5adfaad.png)

**Libraries Required**

* *Pandas* - for data analysis
* *Numpy* - for data analysis
* *matplotlib* - for data visualization
* *seaborn* - for data visualization
* *scikit-learn* - for data analysis
* *nltk* - text preprocessing

**Dataset**
* https://www.kaggle.com/code/ankurzing/sentiment-analysis-of-commodity-news/data

### Evaluation
* Sentiment analysis model obtained 94% accuracy on the training set and 90% accuracy on the test set.
![download (2)](https://user-images.githubusercontent.com/86421205/196364995-2dc1d9ed-911d-4cb5-93a4-2b8cede32e1d.png)

# Prajwal Uday :india:

Connect with me on Linkedin: https://www.linkedin.com/in/prajwal-uday-1b9678229/

Check out my Github profile: https://github.com/prajwal-144
