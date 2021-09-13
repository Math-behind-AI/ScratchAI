class SimpleLinearRegression:
    def fit(self, X, y):
        #Preprocessing
        X = pd.DataFrame(X)
        y = pd.DataFrame(y)
        data = pd.concat([X,y], axis=1)

        #Calculating mean for both Target and Feature Variable
        meanX = float(X.mean())
        meanY = float(y.mean())

        #Calculating `x-mean` and `y-mean` for each data point
        data['x-mean'] = X - meanX
        data['y-mean'] = y - meanY

        #Also calculating product(`x-mean, y-mean`) and square(x-mean)
        data['mul'] = data['x-mean'] * data['y-mean']
        data['sq'] = np.power(data['x-mean'], 2)

        #Summation of data['mul'] and data['sq']
        sumXY = data['mul'].sum()
        sumX = data['sq'].sum()

        #Calculating the coefficients or weights
        global b1, b0
        b1 = sumXY/sumX
        b0 = meanY - b1 * meanX

        #We will be returning the coefficients/weights from this function
        return b0, b1
    
    def predict(self, X):
        predList, y = [], []
        #Creating a list for Feature Variable
        for i in range(0, len(X)):
            predList.append(X[i])
        #Making predictions over Feature Variable
        for i in predList:
            itemY = b0 + b1 * i
            y.append(itemY)
        #Returning the predictions list
        return list(y)
    
    def r2_score(self, y_pred, y_test):
        #Calculating r2 using formula 
        r2 = ((1 - np.sum((y_test - y_pred) * 2) / np.sum((y_test - np.mean(y_test)) * 2)) * 100)
        return r2