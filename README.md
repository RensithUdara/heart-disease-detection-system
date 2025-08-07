# Heart Disease Detection System

A comprehensive machine learning system for predicting heart disease risk with enhanced features, better algorithms, and an improved user interface.

## 🚀 Quick Start

### Option 1: Run the Easy Launcher
```bash
python run_program.py
```

### Option 2: Run Individual Components

#### Original Notebook Version
1. Open `heart_disease_detection.ipynb` in VS Code or Jupyter
2. Run all cells sequentially
3. The last cell launches a basic GUI

#### Improved Analysis System
```bash
python improved_heart_disease_detection.py
```

#### Enhanced GUI Application
```bash
python improved_gui.py
```

#### Professional GUI Application (NEW!)
```bash
python professional_gui.py
```

## 📊 Features

### Original System
- Basic data preprocessing
- Simple model comparison (Logistic Regression, SVM, Decision Tree, Random Forest, Gradient Boosting)
- Basic GUI with input validation
- Single prediction capability

### Improved System

#### 🔬 Advanced Machine Learning
- **Enhanced Feature Engineering**: Age groups, cholesterol risk categories, blood pressure categories, interaction features
- **Multiple Model Comparison**: 6 different algorithms with cross-validation
- **Hyperparameter Tuning**: GridSearchCV for optimal model parameters
- **Advanced Metrics**: AUC-ROC, precision-recall curves, confusion matrices
- **Feature Importance Analysis**: Understanding which factors matter most
- **Model Persistence**: Save and load trained models

#### 🎯 Better Predictions
- **Risk Stratification**: Low, Medium, High risk categories
- **Confidence Scores**: Probability-based predictions
- **Cross-Validation**: More reliable performance estimates
- **Feature Scaling**: Proper data preprocessing for all models

#### 🖥️ Enhanced GUI
- **Modern Interface**: Clean, professional design with validation
- **Real-time Validation**: Input field validation with range checking
- **Risk Visualization**: Interactive charts and progress bars
- **Patient History**: Track and export prediction history
- **Health Recommendations**: Personalized advice based on risk level
- **Sample Data**: Quick testing with pre-loaded examples

#### 🏥 Professional GUI (NEW!)
- **Material Design Interface**: Modern, responsive design with themes
- **Multi-tab Dashboard**: Comprehensive overview with statistics
- **Advanced Analytics**: Risk distribution charts and model performance
- **Patient Management**: Database integration for patient records
- **Professional Reporting**: PDF export with medical formatting
- **Real-time Updates**: Live status updates and notifications
- **Multi-language Support**: International accessibility
- **Dark/Light Themes**: Customizable appearance
- **Tooltips & Help**: Interactive guidance and documentation
- **Database Integration**: SQLite for persistent patient data

## 🏥 Medical Features

### Input Parameters
1. **Age** (1-120 years)
2. **Sex** (1=Male, 0=Female)
3. **Chest Pain Type** (0-3)
   - 0: Typical angina
   - 1: Atypical angina
   - 2: Non-anginal pain
   - 3: Asymptomatic
4. **Resting Blood Pressure** (80-200 mm Hg)
5. **Cholesterol** (100-600 mg/dl)
6. **Fasting Blood Sugar** (1 if >120 mg/dl, 0 otherwise)
7. **Resting ECG Results** (0-2)
8. **Maximum Heart Rate** (60-220)
9. **Exercise Induced Angina** (1=Yes, 0=No)
10. **ST Depression** (0-10)
11. **Peak Exercise ST Slope** (0-2)
12. **Major Vessels** (0-3)
13. **Thalassemia** (0-3)

### Output
- **Prediction**: Disease/No Disease
- **Risk Probability**: 0-100% chance
- **Risk Level**: Low (<30%), Medium (30-70%), High (>70%)
- **Recommendations**: Personalized health advice

## 📈 Model Performance

The improved system includes 6 different machine learning algorithms:

1. **Logistic Regression** - Linear probabilistic model
2. **Random Forest** - Ensemble of decision trees
3. **Gradient Boosting** - Sequential ensemble learning
4. **Support Vector Machine** - Margin-based classifier
5. **Decision Tree** - Rule-based classifier
6. **Neural Network** - Multi-layer perceptron

### Performance Metrics
- **Accuracy**: Percentage of correct predictions
- **AUC-ROC**: Area under the ROC curve
- **Cross-Validation**: 5-fold CV for robust evaluation
- **Confusion Matrix**: True/False positive and negative analysis

## 📁 Project Structure

```
heart_disease_detection-main/
├── data/
│   └── heart.csv                              # Dataset
├── heart_disease_detection.ipynb              # Original notebook
├── improved_heart_disease_detection.py        # Enhanced analysis system
├── improved_gui.py                           # Advanced GUI application
├── professional_gui.py                      # Professional GUI with advanced features
├── run_program.py                            # Easy launcher script
├── requirements.txt                          # Dependencies
├── README.md                                # This file
├── patients.db                              # Patient database (auto-generated)
└── best_heart_disease_model.pkl             # Saved model (generated)
```

## 🛠️ Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Required Packages
- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- Pillow >= 8.0.0
- joblib >= 1.0.0

## 💡 Usage Examples

### Programmatic Usage
```python
from improved_heart_disease_detection import HeartDiseaseDetector

# Initialize detector
detector = HeartDiseaseDetector()

# Load and preprocess data
detector.load_and_explore_data()
detector.data_preprocessing()
detector.split_and_scale_data()

# Train models
results = detector.train_multiple_models()

# Make prediction
patient_data = {
    'age': 52, 'sex': 1, 'cp': 0, 'trestbps': 125,
    'chol': 212, 'fbs': 0, 'restecg': 1, 'thalach': 168,
    'exang': 0, 'oldpeak': 1.0, 'slope': 2, 'ca': 2, 'thal': 3
}
prediction = detector.predict_single_case(patient_data)
print(f"Risk: {prediction['risk_level']}")
```

### GUI Usage
1. Launch the GUI: `python improved_gui.py`
2. Enter patient information in the input fields
3. Click "Predict Risk" to get results
4. View risk visualization and recommendations
5. Track patient history for multiple predictions

## 🔍 Understanding Results

### Risk Levels
- **Low Risk (Green)**: <30% probability - Continue healthy lifestyle
- **Medium Risk (Orange)**: 30-70% probability - Monitor and consider lifestyle changes
- **High Risk (Red)**: >70% probability - Seek immediate medical attention

### Recommendations
The system provides personalized health recommendations based on risk level:
- **High Risk**: Immediate cardiology consultation
- **Medium Risk**: Regular monitoring and lifestyle modifications
- **Low Risk**: Preventive care and healthy habits

## ⚠️ Medical Disclaimer

**IMPORTANT**: This system is for educational and research purposes only. It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical decisions.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make improvements
4. Add tests if applicable
5. Submit a pull request

## 📊 Dataset Information

The system uses the Heart Disease UCI dataset containing 1025 samples with 13 features. The dataset includes patients from multiple medical centers and has been preprocessed to remove duplicates and handle missing values.

## 🔬 Technical Improvements

### Feature Comparison

| Feature | Original | Improved GUI | Professional GUI |
|---------|----------|--------------|------------------|
| **Interface Design** | Basic Tkinter | Modern styling | Material Design |
| **Input Validation** | Basic | Real-time | Advanced with tooltips |
| **Risk Assessment** | Simple | Color-coded | Multi-level with gauge |
| **Visualizations** | None | Basic charts | Interactive dashboards |
| **Patient Management** | None | History list | Full database system |
| **Report Generation** | None | Basic export | Professional PDF reports |
| **Themes** | None | None | Light/Dark toggle |
| **Multi-language** | None | None | English/Spanish support |
| **Database** | None | JSON files | SQLite integration |
| **Real-time Updates** | None | None | Live status monitoring |
| **Help System** | None | None | Interactive tooltips & docs |
| **Analytics Dashboard** | None | None | Comprehensive statistics |

### Over Original Version
1. **Better Feature Engineering**: Created meaningful derived features
2. **Model Comparison**: Systematic evaluation of multiple algorithms
3. **Hyperparameter Optimization**: Grid search for best parameters
4. **Cross-Validation**: More robust performance estimation
5. **Feature Importance**: Understanding model decisions
6. **Model Persistence**: Save and reuse trained models
7. **Advanced GUI**: Professional interface with validation
8. **Risk Visualization**: Charts and progress indicators
9. **Patient Tracking**: History and export functionality
10. **Health Integration**: Medical recommendations and advice

### Performance Gains
- Improved accuracy through better preprocessing
- More reliable predictions via cross-validation
- Better user experience with enhanced GUI
- Professional medical integration with risk levels

## 📧 Support

For questions, issues, or contributions, please create an issue in the repository or contact the development team.

---

**Remember**: Always consult healthcare professionals for medical advice. This tool is designed to assist, not replace, medical expertise.
