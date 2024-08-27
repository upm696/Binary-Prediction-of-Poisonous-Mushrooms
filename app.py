import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier

# Load the saved Decision Tree model and label encoders
def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

dt_model = load_pickle('decision_tree_model.pkl')
label_encoders = load_pickle('label_encoders.pkl')

# Define a function to preprocess user input
def preprocess_input(input_data, label_encoders):
    for column, value in input_data.items():
        if column in label_encoders:
            encoder = label_encoders[column]
            if value in encoder.classes_:
                input_data[column] = encoder.transform([value])[0]
            else:
                # Handle unseen labels by assigning a new label
                encoder.classes_ = np.append(encoder.classes_, value)
                input_data[column] = encoder.transform([value])[0]
    return pd.DataFrame([input_data])

# Define the Streamlit app
def main():
    st.title("Mushroom Classification App")

    # Collect user input for each feature
    st.header("Enter Mushroom Characteristics:")
    
    cap_shape = st.selectbox("Cap Shape", options=['bell', 'conical', 'convex', 'flat', 'knobbed', 'sunken'])
    cap_diameter = st.number_input("Cap Diameter (in cm)", min_value=0.0, step=0.1)
    cap_color = st.selectbox("Cap Color", options=['brown', 'buff', 'gray', 'green', 'pink', 'purple', 'red', 'white', 'yellow'])
    does_bruise_or_bleed = st.selectbox("Does Bruise or Bleed", options=['yes', 'no'])
    gill_color = st.selectbox("Gill Color", options=['black', 'brown', 'buff', 'chocolate', 'gray', 'green', 'orange', 'pink', 'red', 'white', 'yellow'])
    stem_color = st.selectbox("Stem Color", options=['brown', 'buff', 'cinnamon', 'gray', 'orange', 'pink', 'red', 'white', 'yellow'])
    has_ring = st.selectbox("Has Ring", options=['yes', 'no'])
    ring_type = st.selectbox("Ring Type", options=['evanescent', 'flaring', 'large', 'none', 'pendant', 'sheathing', 'zone'])
    habitat = st.selectbox("Habitat", options=['grasses', 'leaves', 'meadows', 'paths', 'urban', 'waste', 'woods'])

    # Organize inputs into a dictionary
    user_input = {
        'cap-shape': cap_shape,
        'cap-diameter': cap_diameter,
        'cap-color': cap_color,
        'does-bruise-or-bleed': does_bruise_or_bleed,
        'gill-color': gill_color,
        'stem-color': stem_color,
        'has-ring': has_ring,
        'ring-type': ring_type,
        'habitat': habitat
    }

    # Preprocess the input
    input_df = preprocess_input(user_input, label_encoders)

    # Debugging: Display the input DataFrame
    st.write("Processed Input DataFrame:")
    st.write(input_df)

    # Predict with the Decision Tree model
    if st.button("Classify with Decision Tree"):
        try:
            dt_prediction = dt_model.predict(input_df)
            st.write(f"Decision Tree Prediction: {'Poisonous' if dt_prediction[0] else 'Edible'}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
