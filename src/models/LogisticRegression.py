from sklearn.linear_model import LogisticRegression

def lr(x_train, y_train):
    print('Initializing model: LogisticRegression')
    model = LogisticRegression()
    print('Training model')
    model.fit(x_train, y_train)

    return model