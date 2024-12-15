import pandas as pd
import joblib
import streamlit as st

# Load models
with open('bagging_model_without_curriculum.pkl', 'rb') as file:
    candidate_model = joblib.load(file)

with open('bagging_model.pkl', 'rb') as file:
    student_model = joblib.load(file)

# Load the dataset to get feature names
data_path = 'data.csv'  # Replace with your file path
df = pd.read_csv(data_path, delimiter=';')

# Standardize column names to ensure consistency
df.columns = df.columns.str.strip().str.replace('\t', ' ').str.replace('\n', '')

# Define feature orders for each model
all_features = df.drop('Target', axis=1).columns.tolist()
candidate_features = [f for f in all_features if "Curricular units" not in f]
student_features = all_features  # Includes all features, including curricular fields

dropdown_fields = {
    'Marital status': {
        '1': 'Single', '2': 'Married', '3': 'Widower', '4': 'Divorced',
        '5': 'Facto Union', '6': 'Legally Separated'
    },
    'Application mode': {
    '1': '1st phase - general contingent',
    '2': 'Ordinance No. 612/93',
    '5': '1st phase - special contingent (Azores Island)',
    '7': 'Holders of other higher courses',
    '10': 'Ordinance No. 854-B/99',
    '15': 'International student (bachelor)',
    '16': '1st phase - special contingent (Madeira Island)',
    '17': '2nd phase - general contingent',
    '18': '3rd phase - general contingent',
    '26': 'Ordinance No. 533-A/99, item b2 (Different Plan)',
    '27': 'Ordinance No. 533-A/99, item b3 (Other Institution)',
    '39': 'Over 23 years old',
    '42': 'Transfer',
    '43': 'Change of course',
    '44': 'Technological specialization diploma holders',
    '51': 'Change of institution/course',
    '53': 'Short cycle diploma holders',
    '57': 'Change of institution/course (International)'},

    'Course': {
    '33': 'Biofuel Production Technologies',
    '171': 'Animation and Multimedia Design',
    '8014': 'Social Service (evening attendance)',
    '9003': 'Agronomy',
    '9070': 'Communication Design',
    '9085': 'Veterinary Nursing',
    '9119': 'Informatics Engineering',
    '9130': 'Equinculture',
    '9147': 'Management',
    '9238': 'Social Service',
    '9254': 'Tourism',
    '9500': 'Nursing',
    '9556': 'Oral Hygiene',
    '9670': 'Advertising and Marketing Management',
    '9773': 'Journalism and Communication',
    '9853': 'Basic Education',
    '9991': 'Management (evening attendance)'
},
'Daytime/evening attendance': {'1': 'Daytime', '0': 'Evening'},


    'Previous qualification': {
    '1': 'Secondary education',
    '2': 'Higher education - bachelor\'s degree',
    '3': 'Higher education - degree',
    '4': 'Higher education - master\'s',
    '5': 'Higher education - doctorate',
    '6': 'Frequency of higher education',
    '9': '12th year of schooling - not completed',
    '10': '11th year of schooling - not completed',
    '12': 'Other - 11th year of schooling',
    '14': '10th year of schooling',
    '15': '10th year of schooling - not completed',
    '19': 'Basic education 3rd cycle (9th/10th/11th year) or equiv.',
    '38': 'Basic education 2nd cycle (6th/7th/8th year) or equiv.',
    '39': 'Technological specialization course',
    '40': 'Higher education - degree (1st cycle)',
    '42': 'Professional higher technical course',
    '43': 'Higher education - master (2nd cycle)'
},
'Nationality': {
    'Portuguese': '1',
    'German': '2',
    'Spanish': '6',
    'Italian': '11',
    'Dutch': '13',
    'English': '14',
    'Lithuanian': '17',
    'Angolan': '21',
    'Cape Verdean': '22',
    'Guinean': '24',
    'Mozambican': '25',
    'Santomean': '26',
    'Turkish': '32',
    'Brazilian': '41',
    'Romanian': '62',
    'Moldova (Republic of)': '100',
    'Mexican': '101',
    'Ukrainian': '103',
    'Russian': '105',
    'Cuban': '108',
    'Colombian': '109',
    'Grenadian': '101',
    'Swiss': '2',
    'Sierra Leonean': '21',
    'Taiwanese': '32',
    'Wallis and Futuna Islander': '41',
    'Barbadian': '101',
    'Pitcairn Islander': '41',
    'Ivorian': '21',
    'Tunisian': '21',
    'Beninese': '21',
    'Indonesian': '32',
    'Saint Kitts and Nevisian': '101',
    'Laotian': '32',
    'Caribbean Netherlands Citizen': '101',
    'Ugandan': '21',
    'Andorran': '2',
    'Burundian': '21',
    'South African': '21',
    'French': '2',
    'Libyan': '21',
    'Gabonese': '21',
    'Northern Mariana Islander': '41',
    'North Macedonian': '2',
    'Chinese': '32',
    'Yemeni': '32',
    'Saint Barthélemy Citizen': '101',
    'Guernsey Citizen': '2',
    'Solomon Islander': '41',
    'Svalbard and Jan Mayen Citizen': '2',
    'Faroe Islander': '2',
    'Uzbek': '32',
    'Egyptian': '21',
    'Senegalese': '21',
    'Sri Lankan': '32',
    'Palestinian': '32',
    'Bangladeshi': '32',
    'Peruvian': '101',
    'Singaporean': '32',
    'Afghan': '32',
    'Aruban': '101',
    'Cook Islander': '41',
    'British': '14',
    'Zambian': '21',
    'Finnish': '2',
    'Nigerien': '21',
    'Christmas Islander': '41',
    'Tokelauan': '41',
    'Guinea-Bissauan': '21',
    'Azerbaijani': '32',
    'Réunion Citizen': '21',
    'Djiboutian': '21',
    'North Korean': '32',
    'Mauritian': '21',
    'Montserratian': '101',
    'US Virgin Islander': '101',
    'Greek': '2',
    'Croatian': '2',
    'Moroccan': '21',
    'Algerian': '21',
    'Antarctican': '1',
    'Sudanese': '21',
    'Fijian': '41',
    'Liechtensteiner': '2',
    'Nepalese': '32',
    'Puerto Rican': '101',
    'Georgian': '32',
    'Pakistani': '32',
    'Monegasque': '2',
    'Botswanan': '21',
    'Lebanese': '32',
    'Papua New Guinean': '41',
    'Mayotte Citizen': '21',
    'Dominican': '101',
    'Norfolk Islander': '41',
    'Bouvet Islander': '1',
    'Qatari': '32',
    'Malagasy': '21',
    'Indian': '32',
    'Syrian': '32',
    'Montenegrin': '2',
    'Eswatini Citizen': '21',
    'Paraguayan': '101',
    'Salvadoran': '101',
    'Isle of Man Citizen': '2',
    'Namibian': '21',
    'Emirati': '32',
    'Bulgarian': '2',
    'Greenlandic': '101',
    'Cambodian': '32',
    'Iraqi': '32',
    'French Southern and Antarctic Lands Citizen': '1',
    'Swedish': '2',
    'Kyrgyz': '32',
    'São Toméan': '21',
    'Cypriot': '2',
    'Canadian': '101',
    'Malawian': '21',
    'Saudi Arabian': '32',
    'Bosnian': '2',
    'Ethiopian': '21',
    'Omani': '32',
    'Macanese': '32',
    'San Marinese': '2',
    'Lesothan': '21',
    'Marshall Islander': '41',
    'Sint Maarten Citizen': '101',
    'Icelandic': '2',
    'Luxembourger': '2',
    'Argentinian': '101',
    'Turks and Caicos Islander': '101',
    'Nauruan': '41',
    'Cocos Islander': '41',
    'Western Saharan': '21',
    'Costa Rican': '101',
    'Australian': '41',
    'Thai': '32',
    'Haitian': '101',
    'Tuvaluan': '41',
    'Honduran': '101',
    'Equatorial Guinean': '21',
    'Saint Lucian': '101',
    'French Polynesian': '41',
    'Belarusian': '2',
    'Latvian': '2',
    'Palauan': '41',
    'Filipino': '32',
    'Gibraltarian': '2',
    'Danish': '2',
    'Cameroonian': '21',
    'Bahraini': '32',
    'Surinamese': '101',
    'Congolese (DRC)': '21',
    'Somali': '21',
    'Czech': '2',
    'New Caledonian': '41',
    'Ni-Vanuatu': '41',
    'Saint Helena Citizen': '21',
    'Togolese': '21',
    'British Virgin Islander': '101',
    'Kenyan': '21',
    'Niuean': '41',
    'Heard Island and McDonald Islands Citizen': '1',
    'Rwandan': '21',
    'Estonian': '2',
    'Trinidadian': '101',
    'Guyanese': '101',
    'Timorese': '32',
    'Vietnamese': '32',
    'Uruguayan': '101',
    'Vatican Citizen': '2',
    'Hong Konger': '32',
    'Austrian': '2',
    'Antiguan': '101',
    'Turkmen': '32',
    'Panamanian': '101',
    'Micronesian': '41',
    'Irish': '2',
    'Curaçao Citizen': '101',
    'French Guianese': '101',
    'Norwegian': '2',
    'Åland Islander': '2',
    'Central African': '21',
    'Burkinabé': '21',
    'Eritrean': '21',
    'Tanzanian': '21',
    'South Korean': '32',
    'Jordanian': '32',
    'Mauritanian': '21',
    'Slovak': '2',
    'Kazakh': '32',
    'Falkland Islander': '101',
    'Armenian': '32',
    'Samoan': '41',
    'Jersey Citizen': '2',
    'Japanese': '32',
    'Bolivian': '101',
    'Chilean': '101',
    'American': '101',
    'Saint Vincentian': '101',
    'Bermudian': '101',
    'Seychellois': '21',
    'British Indian Ocean Territory Citizen': '21',
    'Guatemalan': '101',
    'Ecuadorian': '101',
    'Martinican': '101',
    'Tajik': '32',
    'Maltese': '2',
    'Gambian': '21',
    'Nigerian': '21',
    'Bahamian': '101',
    'Kosovar': '2',
    'Kuwaiti': '32',
    'Maldivian': '32',
    'South Sudanese': '21',
    'Iranian': '32',
    'Albanian': '2',
    'Myanmarese': '32',
    'Bhutanese': '32',
    'Venezuelan': '101',
    'Liberian': '21',
    'Jamaican': '101',
    'Polish': '2',
    'Cayman Islander': '101',
    'Bruneian': '32',
    'Comorian': '21',
    'Guamanian': '41',
    'Tongan': '41',
    'Kiribatian': '41',
    'Ghanaian': '21',
    'Chadian': '21',
    'Zimbabwean': '21',
    'Saint Martin Citizen': '101',
    'Mongolian': '32',
    'Congolese (Congo)': '21',
    'Belgian': '2',
    'Israeli': '32',
    'New Zealander': '41',
    'Nicaraguan': '101',
    'Anguillan': '101'
},


"Mother's qualification": {
    '1': 'Secondary Education - 12th Year of Schooling or Eq.',
    '2': "Higher Education - Bachelor's Degree",
    '3': 'Higher Education - Degree',
    '4': "Higher Education - Master's",
    '5': 'Higher Education - Doctorate',
    '6': 'Frequency of Higher Education',
    '9': '12th Year of Schooling - Not Completed',
    '10': '11th Year of Schooling - Not Completed',
    '11': '7th Year (Old)',
    '12': 'Other - 11th Year of Schooling',
    '14': '10th Year of Schooling',
    '18': 'General commerce course',
    '19': 'Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.',
    '22': 'Technical-professional course',
    '26': '7th Year of Schooling',
    '27': '2nd Cycle of the General High School Course',
    '29': '9th Year of Schooling - Not Completed',
    '30': '8th Year of Schooling',
    '34': 'Unknown',
    '35': "Can't read or write",
    '36': 'Can read without having a 4th Year of Schooling',
    '37': 'Basic Education 1st Cycle (4th/5th Year) or Equiv.',
    '38': 'Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.',
    '39': 'Technological Specialization Course',
    '40': 'Higher Education - Degree (1st Cycle)',
    '41': 'Specialized Higher Studies Course',
    '42': 'Professional Higher Technical Course',
    '43': 'Higher Education - Master (2nd Cycle)',
    '44': 'Higher Education - Doctorate (3rd Cycle)'
},
"Father's qualification": {
    '1': 'Secondary Education - 12th Year of Schooling or Eq.',
    '2': "Higher Education - Bachelor's Degree",
    '3': 'Higher Education - Degree',
    '4': "Higher Education - Master's",
    '5': 'Higher Education - Doctorate',
    '6': 'Frequency of Higher Education',
    '9': '12th Year of Schooling - Not Completed',
    '10': '11th Year of Schooling - Not Completed',
    '11': '7th Year (Old)',
    '12': 'Other - 11th Year of Schooling',
    '13': '2nd Year Complementary High School Course',
    '14': '10th Year of Schooling',
    '18': 'General Commerce Course',
    '19': 'Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.',
    '20': 'Complementary High School Course',
    '22': 'Technical-Professional Course',
    '25': 'Complementary High School Course - Not Concluded',
    '26': '7th Year of Schooling',
    '27': '2nd Cycle of the General High School Course',
    '29': '9th Year of Schooling - Not Completed',
    '30': '8th Year of Schooling',
    '31': 'General Course of Administration and Commerce',
    '33': 'Supplementary Accounting and Administration',
    '34': 'Unknown',
    '35': "Can't Read or Write",
    '36': 'Can Read Without Having a 4th Year of Schooling',
    '37': 'Basic Education 1st Cycle (4th/5th Year) or Equiv.',
    '38': 'Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.',
    '39': 'Technological Specialization Course',
    '40': 'Higher Education - Degree (1st Cycle)',
    '41': 'Specialized Higher Studies Course',
    '42': 'Professional Higher Technical Course',
    '43': 'Higher Education - Master (2nd Cycle)',
    '44': 'Higher Education - Doctorate (3rd Cycle)'
},

"Mother's occupation": {
    '0': 'Student',
    '1': 'Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers',
    '2': 'Specialists in Intellectual and Scientific Activities',
    '3': 'Intermediate Level Technicians and Professions',
    '4': 'Administrative Staff',
    '5': 'Personal Services, Security and Safety Workers and Sellers',
    '6': 'Farmers and Skilled Workers in Agriculture, Fisheries and Forestry',
    '7': 'Skilled Workers in Industry, Construction and Craftsmen',
    '8': 'Installation and Machine Operators and Assembly Workers',
    '9': 'Unskilled Workers',
    '10': 'Armed Forces Professions',
    '90': 'Other Situation',
    '99': '(blank)',
    '122': 'Health Professionals',
    '123': 'Teachers',
    '125': 'Specialists in Information and Communication Technologies (ICT)',
    '131': 'Intermediate Level Science and Engineering Technicians and Professions',
    '132': 'Technicians and Professionals, of Intermediate Level of Health',
    '134': 'Intermediate Level Technicians from Legal, Social, Sports, Cultural and Similar Services',
    '141': 'Office Workers, Secretaries in General and Data Processing Operators',
    '143': 'Data, Accounting, Statistical, Financial Services and Registry-Related Operators',
    '144': 'Other Administrative Support Staff',
    '151': 'Personal Service Workers',
    '152': 'Sellers',
    '153': 'Personal Care Workers and the Like',
    '171': 'Skilled Construction Workers and the Like, Except Electricians',
    '173': 'Skilled Workers in Printing, Precision Instrument Manufacturing, Jewelers, Artisans and the Like',
    '175': 'Workers in Food Processing, Woodworking, Clothing and Other Industries and Crafts',
    '191': 'Cleaning Workers',
    '192': 'Unskilled Workers in Agriculture, Animal Production, Fisheries and Forestry',
    '193': 'Unskilled Workers in Extractive Industry, Construction, Manufacturing and Transport',
    '194': 'Meal Preparation Assistants'
},

"Father's occupation": {
    '0': 'Student',
    '1': 'Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers',
    '2': 'Specialists in Intellectual and Scientific Activities',
    '3': 'Intermediate Level Technicians and Professions',
    '4': 'Administrative Staff',
    '5': 'Personal Services, Security and Safety Workers and Sellers',
    '6': 'Farmers and Skilled Workers in Agriculture, Fisheries and Forestry',
    '7': 'Skilled Workers in Industry, Construction and Craftsmen',
    '8': 'Installation and Machine Operators and Assembly Workers',
    '9': 'Unskilled Workers',
    '10': 'Armed Forces Professions',
    '90': 'Other Situation',
    '99': '(blank)',
    '101': 'Armed Forces Officers',
    '102': 'Armed Forces Sergeants',
    '103': 'Other Armed Forces Personnel',
    '112': 'Directors of Administrative and Commercial Services',
    '114': 'Hotel, Catering, Trade and Other Services Directors',
    '121': 'Specialists in the Physical Sciences, Mathematics, Engineering and Related Techniques',
    '122': 'Health Professionals',
    '123': 'Teachers',
    '124': 'Specialists in Finance, Accounting, Administrative Organization, Public and Commercial Relations',
    '131': 'Intermediate Level Science and Engineering Technicians and Professions',
    '132': 'Technicians and Professionals, of Intermediate Level of Health',
    '134': 'Intermediate Level Technicians from Legal, Social, Sports, Cultural and Similar Services',
    '135': 'Information and Communication Technology Technicians',
    '141': 'Office Workers, Secretaries in General and Data Processing Operators',
    '143': 'Data, Accounting, Statistical, Financial Services and Registry-Related Operators',
    '144': 'Other Administrative Support Staff',
    '151': 'Personal Service Workers',
    '152': 'Sellers',
    '153': 'Personal Care Workers and the Like',
    '154': 'Protection and Security Services Personnel',
    '161': 'Market-Oriented Farmers and Skilled Agricultural and Animal Production Workers',
    '163': 'Farmers, Livestock Keepers, Fishermen, Hunters and Gatherers, Subsistence',
    '171': 'Skilled Construction Workers and the Like, Except Electricians',
    '172': 'Skilled Workers in Metallurgy, Metalworking and Similar',
    '174': 'Skilled Workers in Electricity and Electronics',
    '175': 'Workers in Food Processing, Woodworking, Clothing and Other Industries and Crafts',
    '181': 'Fixed Plant and Machine Operators',
    '182': 'Assembly Workers',
    '183': 'Vehicle Drivers and Mobile Equipment Operators',
    '192': 'Unskilled Workers in Agriculture, Animal Production, Fisheries and Forestry',
    '193': 'Unskilled Workers in Extractive Industry, Construction, Manufacturing and Transport',
    '194': 'Meal Preparation Assistants',
    '195': 'Street Vendors (Except Food) and Street Service Providers'
},


    'Displaced': {'1': 'Yes', '0': 'No'},
    'Educational special needs': {'1': 'Yes', '0': 'No'},
    'Debtor': {'1': 'Yes', '0': 'No'},
    'Tuition fees up to date': {'1': 'Yes', '0': 'No'},
    'Gender': {'1': 'Male', '0': 'Female'},
    'Scholarship holder': {'1': 'Yes', '0': 'No'},
    'International': {'1': 'Yes', '0': 'No'}
}

numerical_fields = [
    'Application order', 'Previous qualification (grade)',
    'Admission grade', 'Age at enrollment', 'Curricular units 1st sem (credited)',
    'Curricular units 1st sem (enrolled)', 'Curricular units 1st sem (evaluations)',
    'Curricular units 1st sem (approved)', 'Curricular units 1st sem (grade)',
    'Curricular units 1st sem (without evaluations)', 'Curricular units 2nd sem (credited)',
    'Curricular units 2nd sem (enrolled)', 'Curricular units 2nd sem (evaluations)',
    'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)',
    'Curricular units 2nd sem (without evaluations)', 'Unemployment rate', 'Inflation rate', 'GDP',
]

# Streamlit App
st.title("Student Prediction System")

option = st.radio("Choose an Option:", ["Prediction for Candidate", "Prediction about Candidate for University","Prediction about Current Student for University"])

user_input = {}
if option == "Prediction for Candidate":
    st.subheader("Prediction for Candidate")
    for field, choices in dropdown_fields.items():
        if field == "Nationality":  # Special case for Nationality
            # Display keys (numbers) in the dropdown
            selected_key = st.selectbox(f"Select {field}:", options=list(choices.keys()))
            # Store the descriptive value corresponding to the key for the model
            user_input[field] = choices[selected_key]
        else:
            # For other fields, show keys and map to descriptive values
            user_input[field] = st.selectbox(f"Select {field}:", options=list(choices.keys()), format_func=lambda x: choices[x])

    for field in numerical_fields:
        if field in candidate_features:  # Exclude irrelevant fields
            user_input[field] = st.number_input(f"Enter value for {field}:", min_value=0.0, value=0.0)

    if st.button("Predict for Candidate"):
        # Align input with training feature order
        input_data = pd.DataFrame([user_input])
        input_data = input_data.reindex(columns=candidate_features, fill_value=0)

        prediction = candidate_model.predict(input_data)
        prediction_proba = candidate_model.predict_proba(input_data)
        result_map = {0: "Dropout", 1: "Success", 2: "Enrolled"}
        
        st.markdown(f"<h3>Prediction:</h3>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='text-align: center;'>{result_map[prediction[0]]}</h2>", unsafe_allow_html=True)
        st.markdown(f"<h3>Prediction Probabilities:</h3>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='text-align: center;'>Dropout: {prediction_proba[0][0]:.2f}</h4>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='text-align: center;'>Enrolled: {prediction_proba[0][2]:.2f}</h4>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='text-align: center;'>Success: {prediction_proba[0][1]:.2f}</h4>", unsafe_allow_html=True)
        
        # Recommend courses if prediction is Dropout
        if prediction[0] == 0:
            st.markdown("<h4>The following courses may lead to better outcomes:</h4>", unsafe_allow_html=True)
            recommendations = []
            original_course = user_input["Course"]
            
            for course_code in dropdown_fields["Course"].keys():
                # Replace course in input data
                temp_input = input_data.copy()
                temp_input["Course"] = course_code
                temp_prediction = candidate_model.predict(temp_input)
                if temp_prediction[0] in [1, 2]:  # Check for Success or Enrolled
                    recommendations.append(dropdown_fields["Course"][course_code])
            
            if recommendations:
               
                for rec_course in recommendations:
                    st.markdown(f"<li style='text-align: center;'>{rec_course}</li>", unsafe_allow_html=True)
            else:
                st.markdown("<h4 style='text-align: center;'>No alternative courses found that predict Success or Enrolled.</h4>", unsafe_allow_html=True)

elif option == "Prediction about Current Student for University":
    st.subheader("Prediction about Current Student for University")
    for field, choices in dropdown_fields.items():
        if field == "Nationality":  # Special case for Nationality
            # Display keys (numbers) in the dropdown
            selected_key = st.selectbox(f"Select {field}:", options=list(choices.keys()))
            # Store the descriptive value corresponding to the key for the model
            user_input[field] = choices[selected_key]
        else:
            # For other fields, show keys and map to descriptive values
            user_input[field] = st.selectbox(f"Select {field}:", options=list(choices.keys()), format_func=lambda x: choices[x])

    for field in numerical_fields:
        user_input[field] = st.number_input(f"Enter value for {field}:", min_value=0.0, value=0.0)

    if st.button("Predict for Current Student"):
        # Align input with training feature order
        input_data = pd.DataFrame([user_input])
        input_data = input_data.reindex(columns=student_features, fill_value=0)

        prediction = student_model.predict(input_data)
        prediction_proba = student_model.predict_proba(input_data)
        result_map = {0: "Dropout", 1: "Success", 2: "Enrolled"}
        
        st.markdown(f"<h3>Prediction:</h3>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='text-align: center; '>{result_map[prediction[0]]}</h2>", unsafe_allow_html=True)
        st.markdown(f"<h3>Prediction Probabilities:</h3>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='text-align: center;'>Dropout: {prediction_proba[0][0]:.2f}</h4>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='text-align: center;'>Enrolled: {prediction_proba[0][2]:.2f}</h4>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='text-align: center;'>Success: {prediction_proba[0][1]:.2f}</h4>", unsafe_allow_html=True)
        
        # Calculate the maximum probability of success or enrolled
        max_success_enrolled = max(prediction_proba[0][1], prediction_proba[0][2])
        dropout_probability = prediction_proba[0][0]

        # Warning status
        st.markdown(f"<h3>Warning Status:</h3>", unsafe_allow_html=True)
        if max_success_enrolled - dropout_probability < 0.15:
            st.markdown(f"<h4 style='text-align: center; color: red;'>Warning Required. High chances of Dropout.</h4>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h4 style='text-align: center; color: green;'>No Warning Required</h4>", unsafe_allow_html=True)

        st.markdown(f"<h3>Scholarship Status: </h3>", unsafe_allow_html=True)
        # Scholarship feature
        if prediction_proba[0][1] > 0.5:  # Check if success probability is greater than 0.5
            st.markdown(f"<h4 style='text-align: center; color: green;'>Should Get Scholarship</h4>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h4 style='text-align: center; color: red;'>No Scholarship</h4>", unsafe_allow_html=True)

elif option == "Prediction about Candidate for University":
    st.subheader("Prediction about Candidate for University")
    for field, choices in dropdown_fields.items():
        if field == "Nationality":  # Special case for Nationality
            # Display keys (numbers) in the dropdown
            selected_key = st.selectbox(f"Select {field}:", options=list(choices.keys()))
            # Store the descriptive value corresponding to the key for the model
            user_input[field] = choices[selected_key]
        else:
            # For other fields, show keys and map to descriptive values
            user_input[field] = st.selectbox(f"Select {field}:", options=list(choices.keys()), format_func=lambda x: choices[x])

    for field in numerical_fields:
        if field in candidate_features:  # Exclude irrelevant fields
            user_input[field] = st.number_input(f"Enter value for {field}:", min_value=0.0, value=0.0)

    if st.button("Predict for Candidate"):
        # Align input with training feature order
        input_data = pd.DataFrame([user_input])
        input_data = input_data.reindex(columns=candidate_features, fill_value=0)

        prediction = candidate_model.predict(input_data)
        prediction_proba = candidate_model.predict_proba(input_data)
        result_map = {0: "Dropout", 1: "Success", 2: "Enrolled"}
        
        st.markdown(f"<h3>Prediction:</h3>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='text-align: center; '>{result_map[prediction[0]]}</h2>", unsafe_allow_html=True)
        st.markdown(f"<h3>Prediction Probabilities:</h3>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='text-align: center;'>Dropout: {prediction_proba[0][0]:.2f}</h4>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='text-align: center;'>Enrolled: {prediction_proba[0][2]:.2f}</h4>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='text-align: center;'>Success: {prediction_proba[0][1]:.2f}</h4>", unsafe_allow_html=True)
        
        # Recommend courses if prediction is Dropout
        if prediction[0] == 0:
            st.markdown("<h4>The following courses may lead to better outcomes:</h4>", unsafe_allow_html=True)
            recommendations = []
            original_course = user_input["Course"]
            
            for course_code in dropdown_fields["Course"].keys():
                # Replace course in input data
                temp_input = input_data.copy()
                temp_input["Course"] = course_code
                temp_prediction = candidate_model.predict(temp_input)
                if temp_prediction[0] in [1, 2]:  # Check for Success or Enrolled
                    recommendations.append(dropdown_fields["Course"][course_code])
            
            if recommendations:
               
                for rec_course in recommendations:
                    st.markdown(f"<li style='text-align: center;'>{rec_course}</li>", unsafe_allow_html=True)
            else:
                st.markdown("<h4 style='text-align: center;'>No alternative courses found that predict Success or Enrolled.</h4>", unsafe_allow_html=True)

        # Calculate the maximum probability of success or enrolled
        max_success_enrolled = max(prediction_proba[0][1], prediction_proba[0][2])
        dropout_probability = prediction_proba[0][0]

        # Warning status
        st.markdown(f"<h3>Warning Status:</h3>", unsafe_allow_html=True)
        if max_success_enrolled - dropout_probability < 0.15:
            st.markdown(f"<h4 style='text-align: center; color: red;'>Warning Required. High chances of Dropout.</h4>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h4 style='text-align: center; color: green;'>No Warning Required</h4>", unsafe_allow_html=True)

        st.markdown(f"<h3>Scholarship Status: </h3>", unsafe_allow_html=True)
        # Scholarship feature
        if prediction_proba[0][1] > 0.5:  # Check if success probability is greater than 0.5
            st.markdown(f"<h4 style='text-align: center; color: green;'>Should Get Scholarship</h4>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h4 style='text-align: center; color: red;'>No Scholarship</h4>", unsafe_allow_html=True)
