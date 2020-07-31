from flask import render_template, request, url_for, redirect
from sklearn.preprocessing import LabelEncoder, StandardScaler
from app import app
import numpy as np
import pandas as pd
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from keras.models import load_model
import glob
import os
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
model1 = load_model("model/diabetes.h5")
model2 = load_model("model/breastCancerANN.h5")
model3 = load_model("model/skinCancer.h5")
model4 = load_model("model/pneumonia.h5")
@app.route('/')
@app.route('/index')
@app.route('/home')
def index():
    return render_template('home.html', title = 'Home')
@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    if request.method == 'POST':
        Pregnancies = request.form['Pregnancies']
        Glucose = request.form['Glucose']
        BloodPressure = request.form['BloodPressure']
        SkinThickness = request.form['SkinThickness']
        Insulin = request.form['Insulin']
        BMI = request.form['BMI']
        DiabetesPedigreeFunction = request.form['DiabetesPedigreeFunction']
        Age = request.form['Age']
        dataset = loadtxt('model/data/pima-indians-diabetes.csv', delimiter=',')
        X = dataset[:,0:8]
        X[0] = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        predictor = model1.predict(X)
        if predictor[0][0] > 0.5:
            return render_template('result.html', title = 'Result', res = "Malignant", trAccur = "", teAccur = "", abil = predictor[0][0])
        else:
            return render_template('result.html', title = 'Result', res = "Benign", trAccur = "", teAccur = "", abil = predictor[0][0])
    return render_template('diabetes.html', title = 'Diabetes')
@app.route('/breastCancerANN', methods=['GET', 'POST'])
def breastCancerANN():
    if request.method == 'POST':
        radius_mean = request.form['radius_mean']
        texture_mean = request.form['texture_mean']
        perimeter_mean = request.form['perimeter_mean']
        area_mean = request.form['area_mean']
        smoothness_mean = request.form['smoothness_mean']
        compactness_mean = request.form['compactness_mean']
        concavity_mean = request.form['concavity_mean']
        concave_points_mean = request.form['concave_points_mean']
        symmetry_mean = request.form['symmetry_mean']
        fractal_dimension_mean = request.form['fractal_dimension_mean']
        radius_se = request.form['radius_se']
        texture_se = request.form['texture_se']
        perimeter_se = request.form['perimeter_se']
        area_se = request.form['area_se']
        smoothness_se = request.form['smoothness_se']
        compactness_se = request.form['compactness_se']
        concavity_se = request.form['concavity_se']
        concave_points_se = request.form['concave_points_se']
        symmetry_se = request.form['symmetry_se']
        fractal_dimension_se = request.form['fractal_dimension_se']
        radius_worst = request.form['radius_worst']
        texture_worst = request.form['texture_worst']
        perimeter_worst = request.form['perimeter_worst']
        area_worst = request.form['area_worst']
        smoothness_worst = request.form['smoothness_worst']
        compactness_worst = request.form['compactness_worst']
        concavity_worst = request.form['concavity_worst']
        concave_points_worst = request.form['concave_points_worst']
        symmetry_worst = request.form['symmetry_worst']
        fractal_dimension_worst = request.form['fractal_dimension_worst']
        dataset = pd.read_csv('model/data/breast-cancer-wisconsin.csv')
        X = dataset.iloc[:,2:32]
        X = np.array(X)
        X[0] = [radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se, radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst]
        sc = StandardScaler()
        X = sc.fit_transform(X)
        predictor = model2.predict(X)
        if predictor[0][0] > 0.5:
            return render_template('result.html', title = 'Result', res = "Malignant", trAccur = "", teAccur = "", abil = predictor[0][0])
        else:
            return render_template('result.html', title = 'Result', res = "Benign", trAccur = "", teAccur = "", abil = predictor[0][0])
    return render_template('breastCancerANN.html', title = 'Breast cancer')
@app.route('/skinCancer', methods=['GET', 'POST'])
def skinCancer():
    files = glob.glob('upload/skinCancer/input/*')
    for f in files:
        os.remove(f)
    if request.method == 'POST':
        file = request.files['file']
        file.save('upload/skinCancer/input/' + file.filename)
        X = ImageDataGenerator(rescale = 1./255).flow_from_directory('upload/skinCancer', target_size = (64, 64), batch_size = 32, class_mode = 'binary')
        predictor = model4.predict_generator(X, len(X.filenames))
        if predictor[0][0] > 0.5:
            return render_template('result.html', title = 'Result', res = 'Malignant', trAccur = "", teAccur = "", abil = predictor[0][0])
        else:
            return render_template('result.html', title = 'Result', res = 'Benign', trAccur = "", teAccur = "", abil = predictor[0][0])
    return render_template('skinCancer.html', title = 'Skin cancer')
@app.route('/pneumonia', methods=['GET', 'POST'])
def pneumonia():
    files = glob.glob('upload/pneumonia/input/*')
    for f in files:
        os.remove(f)
    if request.method == 'POST':
        file = request.files['file']
        file.save('upload/pneumonia/input/' + file.filename)
        X = ImageDataGenerator(rescale = 1./255).flow_from_directory('upload/pneumonia', target_size = (64, 64), batch_size = 32, class_mode = 'binary')
        predictor = model4.predict_generator(X, len(X.filenames))
        if predictor[0][0] > 0.5:
            return render_template('result.html', title = 'Result', res = 'Malignant', trAccur = "", teAccur = "", abil = predictor[0][0])
        else:
            return render_template('result.html', title = 'Result', res = 'Benign', trAccur = "", teAccur = "", abil = predictor[0][0])
    return render_template('pneumonia.html', title = 'Pneumonia')
@app.errorhandler(404)
def notFound(e):
    return render_template('404.html', title = 'Page not found')