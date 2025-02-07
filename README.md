# AICTE Internship Project 2: Prediction of Disease Outbreaks

## About
This application utilizes machine learning models to predict the likelihood of various diseases based on patient data. It provides an easy-to-use interface where users can select a disease, input relevant patient details, and receive a prediction.

## Features
- Supports multiple disease predictions
- User-friendly interface
- Machine learning-based prediction
- Explainable AI (XAI) integration for interpretability

## Instructions
1. **Select a disease** from the dropdown menu.
2. **Enter patient information** in the input fields.
3. **Click 'Predict'** to generate a prediction result.
4. The system will analyze the data and provide the likelihood of the selected disease.

## Installation
To run the application locally, follow these steps:
```sh
# Clone the repository
git clone https://github.com/your-repo-name/prediction-disease-outbreaks.git
cd prediction-disease-outbreaks

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## Technologies Used
- **Python** (pandas, scikit-learn, Streamlit, XAI libraries)
- **Machine Learning Models** (trained models for disease prediction)
- **GitHub for version control**

## File Structure
```
/your-repo-name/
│── models_scalers/        # Pre-trained models and scalers
│── app.py                 # Main Streamlit application
│── requirements.txt        # Dependencies
│── README.md              # Project documentation
```

## Contributing
Feel free to contribute to this project by submitting issues or pull requests!

## License
This project is licensed under the MIT License.
