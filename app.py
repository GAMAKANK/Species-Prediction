import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import  RandomForestClassifier

@st.cache_data 
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data,columns=iris.feature_names)
    df['species'] = iris.target
    return df,iris.target_names

#calling functions
df,species_names = load_data()  

model =  RandomForestClassifier()
model.fit(df.iloc[:,:-1],df['species']) #separating dependent and independent features

#input data using slider

st.sidebar.title("Species Prediction")
sepal_length = st.sidebar.slider("Sepal Length",float(df['sepal length (cm)'].min()),float(df['sepal length (cm)'].max())) #specifying range of length
sepal_width = st.sidebar.slider("Sepal width",float(df['sepal width (cm)'].min()),float(df['sepal width (cm)'].max()))
petal_length = st.sidebar.slider("Petal Length",float(df['petal length (cm)'].min()),float(df['petal length (cm)'].max()))
petal_width = st.sidebar.slider("Petal Width",float(df['petal width (cm)'
                                                       ].min()),float(df['petal width (cm)'].max()))

if "submitted" not in st.session_state:
    st.session_state.submitted = False

if st.sidebar.button("Analyze Return ->"):
    st.session_state.submitted = True 

st.title("Iris Flower Classification")

st.subheader("Classification")
st.markdown(f"""
- *Sepal Length*: {sepal_length}
- *Sepal Width*: {sepal_width}
- *Petal Length*: {petal_length}
- *Petal Width*: {petal_width}
            """)

if st.session_state.submitted:
    st.subheader("Classifying")
    #selecting all input data and do prediction
    input_data = [[sepal_length,sepal_width,petal_length,petal_width]]

    #prediction
    prediction = model.predict(input_data)
    #displaying prediction
    prediction_species = species_names[prediction[0]]

    #writing the predicted content
    st.write("Predicted Species:",prediction_species)

# Footer
st.markdown("---")
st.markdown("© 2024 Iris Flower Classification")