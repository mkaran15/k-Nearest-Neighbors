# Step 1 : Load Dataset
    import pandas as pd
    dataset = pd.read_csv("data.csv")
    dataset.head(8)
    X = dataset[["Age", "EstimatedSalary"]]
    y= dataset["Purchased"]

# Step 2 : Graph
    import seaborn as sns 
    sns.set()
    sns.scatterplot(x='Age', y="EstimatedSalary", data = dataset, hue="Purchased",palette="Set1")

# Splitting Data into Test data and Train data
    from sklearn.model_selection import train_test_split
    X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4 : Create Model
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors = 5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

# Step 5 : Accuracy 
    from sklearn.metrics import accuracy_score
    Accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy = {} %".format(Accuracy*100))

# Prediction
    result = []
    result = [model.predict([[20,20000]]), model.predict([[60, 1000000]])]
    for i in range(len(result)):
        if(result[i][0]==0):
            print("Not Purchased")
        else:
            print("Purchased")
