import numpy as np
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier

st.write(''' # Predicción de Fitness  ''')
st.image("imagen.jpg", caption="Predicción de Fitness.")

st.header('Datos')

def user_input_features():
  # Entrada
  Edad = st.number_input('age:', min_value=0.0, max_value=100.0, value = 0.0, step = 1.0)
  Altura = st.number_input('height_cm:',  min_value=0.0, max_value=1000.0, value = 0.0, step = 1.0)
  Peso = st.number_input('weight_kg:', min_value=0.0, max_value=1000.0, value = 0.0, step = 1.0)
  Ritmo_cardiaco = st.number_input('heart_rate:', min_value=0.0, max_value=1000.0, value = 0.0, step = 1.0)
  Presion_en_la_sangre = st.number_input('blood_pressure:', min_value=0.0, max_value=1000.0, value = 0.0, step = 1.0)
  Sueño = st.number_input('sleep_hours:',  min_value=0.0, max_value=100.0, value = 0.0, step = 1.0)
  Calidad_de_nutrición = st.number_input('nutrition_quality:', min_value=0.0, max_value=100.0, value = 0.0, step = 1.0)
  Actividad_fisica = st.number_input('activity_index:', min_value=0.0, max_value=100.0, value = 0.0, step = 1.0)
  Fuma = st.number_input('smokes:', min_value=0.0, max_value=1.0, value = 0.0, step = 1.0)
  Genero = st.number_input('gender:',  min_value=0.0, max_value=1.0, value = 0.0, step = 1.0)
  

  user_input_data = {'age': Edad,
                     'height_cm': Altura,
                     'weight_kg': Peso,
                     'heart_rate': Ritmo_cardiaco,
                     'blood_pressure': Presion_en_la_sangre,
                     'sleep_hours': Sueño,
                     'nutrition_quality': Calidad_de_nutrición,
                     'activity_index': Actividad_fisica,
                     'smokes': Fuma,
                     'gender': Genero
                     }

  features = pd.DataFrame(user_input_data, index=[0])

  return features

df = user_input_features()
datos =  pd.read_csv('fitnes.csv', encoding='latin-1')
X = datos.drop(columns=['is_fit'])
y = datos['is_fit']

from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X, y)

b1 = LR.coef_
b0 = LR.intercept_
prediccion = b0 + b1[0]*df['age'] + b1[1]*df['height_cm'] + b1[2]*df['weight_kg'] + b1[3]*df['heart_rate']+b1[4]*df['blood_pressure'] + b1[5]*df['sleep_hours'] + b1[6]*df['nutrition_quality'] + b1[7]*df['activity_index']+b1[8]*df['sleep_hours'] + b1[9]*df['smokes'] + b1[10]*df['gender']

st.subheader('Cálculo de fitness')
st.write('Eres fit: ', prediccion)
