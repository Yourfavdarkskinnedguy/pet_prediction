import numpy as np
import pickle
import streamlit as st

# Load the encoders and model
label_encoders = pickle.load(open('C:/Users/HP/Desktop/pet/label_encoder.sav', 'rb'))
loaded_model = pickle.load(open('C:/Users/HP/Desktop/pet/trained_model.sav', 'rb'))

def preprocess_input(input_data, label_encoders):
    # Convert categorical inputs using the label encoders
    for i, key in enumerate(['PetType', 'Breed', 'Color', 'Size', 'Vaccinated', 'HealthCondition', 'PreviousOwner']):
        input_data[i] = label_encoders[key].transform([input_data[i]])[0]
    # Convert the remaining inputs to floats
    input_data = [float(x) if isinstance(x, str) and x.replace('.', '', 1).isdigit() else x for x in input_data]
    return input_data

def pet_prediction(input_data):
    input_data_as_np = np.asarray(input_data)
    input_data_reshaped = input_data_as_np.reshape(1, -1)
    predictions = loaded_model.predict(input_data_reshaped)
    return 'Pet is likely to be adopted' if predictions[0] == 1 else 'Pet is unlikely to be adopted'

def main():
    st.title('Pet Adoption Prediction')
    
    PetID = st.text_input('Unique identifier for each pet.')
    PetType = st.selectbox('PetType', ['Bird', 'Rabbit', 'Dog', 'Cat'])
    Breed = st.selectbox('Breed', ['Parakeet', 'Labrador', 'Golden Retriever', 'Poodle', 'Persian', 'Siamese'])
    AgeMonths = st.slider('Age in Months', 0, 200)
    Color = st.selectbox('Color', ['Black', 'Gray', 'Brown', 'White', 'Orange'])
    Size = st.selectbox('Size', ['Large', 'Medium', 'Small'])
    WeightKg = st.text_input('Weight in Kg')
    Vaccinated = st.selectbox('Vaccinated', ['Yes', 'No'])
    HealthCondition = st.selectbox('Health Condition', ['Healthy', 'Minor Injury', 'Serious Injury'])
    TimeInShelterDays = st.text_input('Time in Shelter (Days)')
    AdoptionFee = st.text_input('Adoption Fee')
    PreviousOwner = st.selectbox('Previous Owner', ['Yes', 'No'])
    
    diagnosis = ''
    
    if st.button('Predict Pet Adoption Result'):
        input_data = [PetType, Breed, Color, Size, Vaccinated, HealthCondition, PreviousOwner, AgeMonths, WeightKg, TimeInShelterDays, AdoptionFee]
        processed_input = preprocess_input(input_data, label_encoders)
        diagnosis = pet_prediction(processed_input)
        st.success(diagnosis)
       
if __name__ == '__main__':
    main()

