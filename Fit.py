import numpy as np
import streamlit as st
import pandas as pd

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

titanic =  pd.read_csv('Titanic2.csv', encoding='latin-1')
X = titanic.drop(columns='is_fit')
Y = titanic['is_fit']

classifier = DecisionTreeClassifier(max_depth=4, criterion='gini', min_samples_leaf=25, max_features=5, random_state=1613555)
classifier.fit(X, Y)

prediction = classifier.predict(df)

st.subheader('Predicción')
if prediction == 0:
  st.write('No fit')
elif prediction == 1:
  st.write('Fit')
else:
  st.write('Sin predicción')
