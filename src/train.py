import yaml
from src.data import load_data, process_training_data
from src.utils import clean_text, text_length, remove_outranged_length_articles, remove_duplicates, save_model, save_data
from src.models import LogisticRegression
from src import model_prediction
import pandas as pd
import os

def train_and_predict(config_path: str = 'Configs'):
    with open(config_path + '/configs.yaml','r') as f:
        data_configs = yaml.safe_load(f)

    #Checking if processed data is available or not
    if not os.path.exists(data_configs['processed_data']):

        true_data = load_data.fetch_data(data_configs['true_data'])
        fake_data = load_data.fetch_data(data_configs['fake_data'])

        # Creating Class Column [1=True & 0=False]
        true_data['class']=1
        fake_data['class']=0

        # Concatenating both datasets
        df = pd.concat([fake_data,true_data], axis=0)

        # Dropping unnecessary columns
        df = df.drop(["title", "subject", "date"], axis=1)

        # Applying the function to text column
        print('Cleaning data')
        df["text"] = df["text"].apply(clean_text.clean_text)

        #Getting length of news articles
        print('Getting word count of each article')
        df['word_count'] = df['text'].apply(text_length.get_text_length)

        #removing articles by length
        print('removing articles with outofrange length')
        df = remove_outranged_length_articles.articles_by_length(df, data_configs['min_article_length'], data_configs['max_article_length'])

        #removing duplicate Data
        print('removing duplicate articles')
        df = remove_duplicates.remove_duplicate_data(df, 'text')

        #Saving processed data
        save_data.SaveData(df, data_configs['processed_data'])



    
    #importing processed data
    df = pd.read_csv(data_configs["processed_data"])

    #Preparing data for model training
    x_train, x_test, y_train, y_test = process_training_data.process_data(df)

    #training model
    model = LogisticRegression.lr(x_train, y_train)

    #model prediction
    prediction = model_prediction.predict_model(model, x_test)

    #Calculating model accuracy
    model.score(x_test, y_test)

    #Saving model
    save_model.save_trained_model(model, "logisticRegressionV1")









        