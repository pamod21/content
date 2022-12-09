Email Spamming

Junk emails or unsolicited bulk emails sent to a large list of email users through the email system are referred to as email spam. Typically, they are misleading ads that promote low-quality services and, in some instances, include images with content that is inappropriate for children. Whether commercial or not, many of them are really dangerous since they may contain links that appear to be legitimate and recognizable, but they lead to phishing websites that host malware or include malware in the form of file attachments.
A phishing website is a domain similar in name and appearance to an official website. They're made in order to fool someone into believing it is legitimate. Typically, spammers obtain recipients’ email addresses from publicly available sources and use them to advertise and promote their businesses; they may also use them to collect sensitive information from the victim’s machine. These collected email addresses are sometimes also sold to other spammers.
 
These days, spam emails are the most common method of online fraud.

General Approach

In this document, we will be using machine learning to filter spam mail. Machine learning field is a subfield from the broad field of artificial intelligence, this aims to make machines able to learn like human. In machine learning, no rule is required to be specified, rather a set of training samples which are pre-classified email messages are provided. A particular machine learning algorithm is then used to learn the classification rules from these email messages. In this developed ML model, it does more than just checking junk emails using pre-existing rules. They generate new rules themselves based on what they have learnt as they continue in their spam filtering operation.

Implementation

  4.1. Prerequisites

         First, need to import the necessary dependencies. Pandas is a library used mostly for data cleaning and analysis. Scikit-learn, also called Sklearn, is a robust library for machine learning in Python. It provides a selection of efficient tools for machine learning and statistical modeling, including classification, regression, clustering, and dimensionality reduction via a consistent interface.

  4.2. Getting Started

         To get started first need to import the data file that contains mail data. 
         The mail_data file mimics the layout of a typical email inbox and includes over 5,000 examples that will be used to train the machine learning model.
         
  4.3. train_test_split()

         The train-test split method is then used to train the email spam detector to recognize and categorize spam emails. The train-test split is a technique for evaluating the performance of a machine-learning algorithm. We can use it for either classification or regression of any supervised learning algorithm. The procedure involves taking a dataset and dividing it into two separate datasets. The first dataset is used to fit the model and is referred to as the training dataset. For the second dataset, the test dataset, we provide the input element to the model. Finally, we make predictions, comparing them against the actual output.

Train dataset: used to fit the machine learning model
Test dataset: used to evaluate the fit of the machine learning model

In practice, we’d fit the model on available data with known inputs and outputs. Then, make predictions based on new examples for which we don’t have the expected output or target values. We’ll take data from the sample .csv file, which contains examples pre-classified into spam and non-spam, using the labels ‘spam’ and ‘ham’, respectively. To split the data into two datasets, we will use scikit-learn’s train_test_split() method.

         X = mail_data['Message'] assigns the column Message from mail_data to X. It contains the data that will run through the model. Y = mail_data[‘Category’] assigns the column Category from mail_data to Y, telling the model to correct the answer. The function X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=3) divides columns X and Y into X_train for training inputs, Y_train for training categories, X_test for testing inputs, and Y_test for testing categories. test_size=0.2 sets the testing set to 20 percent of X and Y.

  4.4. Extracting Features

        In the dependencies, we have imported the function TfidfVectorizer. It is used to convert text values into numerical values. Here we used it to transform the text data into feature vectors that can be used as input to the logistic regression model. 
        
        In feature_extraction= TfidfVectorizer(), TfidfVectorizer() randomly assigns a number to each word in a process called tokenizing. Then, it counts the number of occurrences of words and saves it to feature_extraction. X_train_features = feature_extraction.fit_transform(X_train) randomly assigns a number to each word. It counts the number of occurrences of each word, then saves it to feature_extraction. In the image below, 0 represents the index of the email. The number sequences in the middle column represent a word recognized by our function, and the numbers on the right indicate the number of times that word was counted,

        The parameter min_df = 1 is used to ignore the values less than 1 from the outputs given by TfidfVectorizer. If the score is more than ‘1’ for a particular word, it will be included. The stop_words = ‘english’ parameter will be used to ignore all the formal words in the English language from the dataset. ( is, the, are, in, etc.) By using the parameter lowercase = ‘True’ all the letters will be transformed into lowercase letters which will be better for the processing. Now, the machine learning model will be able to predict spam emails based on the number of occurrences of certain words that are common in spam emails.

  4.5. Building the Model

        In the model.fit(X_train_features, Y_train) function, model.fit trains the model with features and Y_train. Then, it checks the prediction against the Y_train label and adjusts its parameters until it reaches the highest possible accuracy.
        
  4.6. Evaluating the Email Spam Detector
  
        In the model.predict(X_train_features) function, mode.predict() scores the prediction of features test against the actual labels in Y_test. In evaluating the model, we were able to classify spam with almost 97 percent of accuracy.
