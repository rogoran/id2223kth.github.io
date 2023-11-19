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
    model = mr.get_model("wine_model", version=1)
    model_dir = model.download()
    model = joblib.load(model_dir + "/wine_model.pkl")
    
    

    feature_view = fs.get_feature_view(name="wine", version=1)
    batch_data = feature_view.get_batch_data()

    y_pred = model.predict(batch_data)
    #print(y_pred)
    offset = 1
    predicted_wine = y_pred[-1]
    dataset_api = project.get_dataset_api()
    dataset_api.upload("./latest_iris.png", "Resources/images", overwrite=True)

    
    wine_fg = fs.get_feature_group(name="wine", version=2)
    wine_df = wine_fg.read() 
   
    label = wine_df.iloc[-offset]["quality"]

    
    monitor_fg = fs.get_or_create_feature_group(name="wine_predictions",
                                                version=1,
                                                primary_key=["datetime"],
                                                description="Wine Quality Prediction/Outcome Monitoring"
                                                )
    
    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

    data = {
        'prediction': [predicted_wine],
        'label': [label],
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
    
    predictions = history_df[['prediction']]
    labels = history_df[['label']]

    print(f"Prediction: {predicted_wine}\nActual label: {label}")
   
    # Only create the confusion matrix when our iris_predictions feature group has examples of all 3 iris flowers
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
        print("Run the batch inference pipeline more times until you get 3 different iris flower predictions") 


if __name__ == "__main__":
    g()
