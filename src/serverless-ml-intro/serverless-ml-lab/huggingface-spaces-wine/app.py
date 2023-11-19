import gradio as gr
from PIL import Image
import requests
import hopsworks
import joblib
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()


mr = project.get_model_registry()
model = mr.get_model("wine_model", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/wine_model.pkl")
print("Model downloaded")

def wine_predictor(fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                  chlorides,total_sulfur_dioxide, pH, sulphates,
                  alcohol, sulfur_dioxide_ratio, isRedWine):
    
    type_red = isRedWine
    type_white = not type_red

    df = pd.DataFrame([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                        chlorides,total_sulfur_dioxide, pH, sulphates,
                        alcohol, type_red, type_white, sulfur_dioxide_ratio]], 
                      columns=["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
                                "chlorides","total_sulfur_dioxide", "ph", "sulphates",
                                "alcohol","type_red", "type_white","sulfur_dioxide_ratio"])
   
   
    res = model.predict(df) 
   
    
    return res[0]
        
demo = gr.Interface(
    fn=wine_predictor,
    title="Wine Predictive Analytics",
    description="Experiment with sepal/petal lengths/widths to predict which flower it is.",
    allow_flagging="never",
    inputs=[
        gr.inputs.Number(default=7.0, label="fixed acidity"),
        gr.inputs.Number(default=0.3, label="volatile acidity"),
        gr.inputs.Number(default=0.3, label="citric acid"),
        gr.inputs.Number(default=5.0, label="residual sugar"),
        gr.inputs.Number(default=0.05, label="chlorides"),
        gr.inputs.Number(default=114, label="total_sulfur_dioxide"),
        gr.inputs.Number(default=3.2, label="pH"),
        gr.inputs.Number(default=0.5, label="sulphates"),
        gr.inputs.Number(default=10, label="alcohol"),
        gr.inputs.Number(default=0.3, label="sulfur dioxide free-total ratio"),
        gr.inputs.Checkbox(label="Red wine?")
        ],
    outputs=gr.Textbox(placeholder="Quality"))

demo.launch(debug=True)

