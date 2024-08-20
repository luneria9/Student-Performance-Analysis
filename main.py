import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Student Performance Analysis',
    page_icon=':bar_chart:',
)

# -----------------------------------------------------------------------------
# Declare some useful functions.

@st.cache_data
def get_student_data():
    # Grab student performance data from a CSV file.
    DATA_FILENAME = Path(__file__).parent/'data/StudentsPerformance_with_headers.csv'
    student_df = pd.read_csv(DATA_FILENAME)
    return student_df

student_df = get_student_data()

# DATA CLEANING
# Drop irrelevant columns
columns_to_drop = ["Sex", 
                    "Flip-classroom", 
                    "Expected Cumulative grade point average in the graduation (/4.00)", 
                    "Mother’s education", 
                    "Father’s education ", 
                    "Student Age", 
                    "Graduated high-school type", 
                    "Scholarship type", 
                    "Additional work", 
                    "Regular artistic or sports activity", 
                    "Total salary if available", 
                    "Transportation to the university", 
                    "Number of sisters/brothers", 
                    "Mother’s occupation"]

# Ensure columns to drop exist in the DataFrame
columns_to_drop = [col for col in columns_to_drop if col in student_df.columns]
student_df.drop(columns=columns_to_drop, inplace=True)

# Define X (features) and y (target)
X = student_df.drop(columns=["STUDENT ID", "GRADE"])
y = student_df["GRADE"]

# Apply SelectKBest to select top features
feature_names = X.columns
selector = SelectKBest(chi2, k=8)
X_selected = selector.fit_transform(X, y)
selected_features = feature_names[selector.get_support()]

# Create a DataFrame with selected features
X_selected_df = pd.DataFrame(X_selected, columns=selected_features)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_selected_df, y, test_size=0.1, random_state=0)

# Define a simple model
model = MLPRegressor(hidden_layer_sizes=(50,), max_iter=500, random_state=42)

# Train the model
model.fit(X_train, y_train)

# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
st.title(' :student: Student Performance Dashboard')

# Show a sample of the data
st.header('Sample Data')
st.write(student_df.head())

# Create select boxes for user inputs
st.sidebar.header('Select Features for Prediction')

# Define the select boxes
partner = st.selectbox('Do you have a partner', ['Yes', 'No'])
accommodation = st.selectbox('Accommodation type in Cyprus', ['rental', 'dormitory', 'with family', 'Other'])
parental_status = st.selectbox('Parental status', ['married', 'divorced', 'died - one of them or both'])
father_occupation = st.selectbox('Father’s occupation', ['retired', 'government officer', 'private sector employee', 'self-employment', 'other'])
study_hours = st.selectbox('Weekly study hours', ['None', '<5 hours', '6-10 hours', '11-20 hours', 'more than 20 hours'])
reading_frequency = st.selectbox('Reading frequency (non-scientific books/journals)', ['None', 'Sometimes', 'Often'])
project_impact = st.selectbox('Impact of your projects/activities on your success', ['positive', 'negative', 'neutral'])
preparation_midterm = st.selectbox('Preparation to midterm exams 1', ['alone', 'with friends', 'not applicable'])

# Define encoders that match selected_features
def create_encoders():
    encoders = {
        'Do you have a partner': {'Yes': 1, 'No': 2},
        'Accommodation type in Cyprus': {'rental': 1, 'dormitory': 2, 'with family': 3, 'Other': 4},
        'Parental status': {'married': 1, 'divorced': 2, 'died - one of them or both': 3},
        'Father’s occupation': {'retired': 1, 'government officer': 2, 'private sector employee': 3, 'self-employment': 4, 'other': 5},
        'Weekly study hours': {'None': 1, '<5 hours': 2, '6-10 hours': 3, '11-20 hours': 4, 'more than 20 hours': 5},
        'Reading frequency (non-scientific books/journals)': {'None': 1, 'Sometimes': 2, 'Often': 3},
        'Impact of your projects/activities on your success': {'positive': 1, 'neutral': 2, 'negative': 3},
        'Preparation to midterm exams 1': {'alone': 1, 'with friends': 2, 'not applicable': 3}
    }
    return encoders

# Create encoders
encoders = create_encoders()

# Encode categorical user inputs
encoded_inputs = [
    encoders['Do you have a partner'][partner],
    encoders['Accommodation type in Cyprus'][accommodation],
    encoders['Parental status'][parental_status],
    encoders['Father’s occupation'][father_occupation],
    encoders['Weekly study hours'][study_hours],
    encoders['Reading frequency (non-scientific books/journals)'][reading_frequency],  # Correct key here
    encoders['Impact of your projects/activities on your success'][project_impact],
    encoders['Preparation to midterm exams 1'][preparation_midterm]
]

# Create a DataFrame for the user input, matching the selected features
user_input_df = pd.DataFrame([encoded_inputs], columns=selected_features)

# Preprocess the user input to match the training data preprocessing
user_input_preprocessed = user_input_df.copy()

# Define the mapping for letter grades
grade_mapping = {
    0: 'Fail',
    1: 'DD',
    2: 'DC',
    3: 'CC',
    4: 'CB',
    5: 'BB',
    6: 'BA',
    7: 'AA'
}

# Add a button for prediction
if st.button('Predict Grade'):
    predicted_grade = model.predict(user_input_preprocessed)

    # Ensure the prediction is not below 0 or above 7
    predicted_grade = min(max(predicted_grade[0], 0), 7)
    predicted_grade_rounded = round(predicted_grade, 0)

    # Map the numeric grade to a letter grade
    letter_grade = grade_mapping.get(predicted_grade_rounded, 'Unknown')

    st.write(f'Predicted Grade: {predicted_grade_rounded} ({letter_grade})')