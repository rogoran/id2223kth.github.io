import os
import modal
    
def g():
    import pandas as pd
    import hopsworks
    import joblib
    import datetime
    from PIL import Image
    from datetime import datetime
    import dataframe_image as dfi
    from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot
    import seaborn as sns
    import requests

    project = hopsworks.login()
    fs = project.get_feature_store()
    
    mr = project.get_model_registry()
    model = mr.get_model("wine_model_randomforest", version=1)
    model_dir = model.download()
    model = joblib.load(model_dir + "/wine_model.pkl")
    
    

    wine_fg = fs.get_feature_group(name="winedataset", version=1)
    wine_df = wine_fg.read() 


    wine_df_withoutQ = wine_df.drop('quality', axis=1)

    row_to_predict = wine_df_withoutQ.iloc[-1]

    print(row_to_predict)
    y_pred = model.predict([row_to_predict])
    #print(y_pred)
    offset = 1
    predicted_wine = y_pred[0]
    predicted_wine_url = "https://raw.githubusercontent.com/rogoran/id2223kth.github.io/master/src/serverless-ml-intro/serverless-ml-lab/wine_images/wine-" + str(predicted_wine) + ".png"
    print("Wine predicted: " + str(predicted_wine))
    img = Image.open(requests.get(predicted_wine_url, stream=True).raw)            
    img.save("./latest_predicted_wine.png")

    dataset_api = project.get_dataset_api()    
    dataset_api.upload("./latest_predicted_wine.png", "Resources/images", overwrite=True)
   
    correct_label = wine_df.iloc[-offset]["quality"]
    
    correct_label_url = "https://raw.githubusercontent.com/rogoran/id2223kth.github.io/master/src/serverless-ml-intro/serverless-ml-lab/wine_images/wine-" + str(correct_label) + ".png"
    print("Wine actual: " + str(correct_label))
    img = Image.open(requests.get(correct_label_url, stream=True).raw)            
    img.save("./actual_wine.png")
    dataset_api.upload("./actual_wine.png", "Resources/images", overwrite=True)
    

    monitor_fg = fs.get_or_create_feature_group(name="wine_predictions",
                                                version=1,
                                                primary_key=["datetime"],
                                                description="Wine Quality Prediction/Outcome Monitoring"
                                                )
    
    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

    data = {
        'prediction': [predicted_wine],
        'label': [correct_label],
        'datetime': [now],
       }
    
    monitor_df = pd.DataFrame(data)
    monitor_fg.insert(monitor_df, write_options={"wait_for_job" : False})
    
    history_df = monitor_fg.read()
    # Add our prediction to the history, as the history_df won't have it - 
    # the insertion was done asynchronously, so it will take ~1 min to land on App
    history_df = pd.concat([history_df, monitor_df])


    df_recent = history_df.tail(4)

    dfi.export(df_recent, './df_recent.png', table_conversion = 'matplotlib')
    dataset_api.upload("./df_recent.png", "Resources/images", overwrite=True)
    
    predictions = history_df[['prediction']]
    labels = history_df[['label']]

    print(f"Prediction: {predicted_wine}\nActual label: {correct_label}")
   
    print("Number of different wine quality predictions to date: " + str(predictions.value_counts().count()))
    

    if predictions.value_counts().count() == 7:
        results = confusion_matrix(labels, predictions)
    
        df_cm = pd.DataFrame(results, ['True 3', 'True 4', 'True 5', 'True 6','True 7','True 8', 'True 9'],
                     ['Pred 3', 'Pred 4', 'Pred 5', 'Pred 6', 'Pred 7', 'Pred 8', 'Pred 9'])
    
        cm = sns.heatmap(df_cm, annot=True)
        fig = cm.get_figure()

        fig.savefig("./confusion_matrix.png")
        dataset_api = project.get_dataset_api()
        dataset_api.upload("./confusion_matrix.png", "Resources/images", overwrite=True)
    else:
        print("You need 7 different predictions to create the confusion matrix.")
        print("Run the batch inference pipeline more times until you get 7 different wine predictions") 


if __name__ == "__main__":
    g()
