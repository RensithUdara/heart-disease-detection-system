"""
Improved GUI for Heart Disease Detection
========================================

This enhanced GUI includes:
1. Input validation
2. Real-time prediction with confidence scores
3. Better UI design with modern styling
4. Risk assessment visualization
5. Patient history tracking
6. Export functionality
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import json
from datetime import datetime

class ImprovedHeartDiseaseGUI:
    def __init__(self, root):
        self.root = root
        self.root.title('Advanced Heart Disease Detection System')
        self.root.geometry('1200x800')
        self.root.configure(bg='#f0f0f0')
        
        # Initialize variables
        self.model = None
        self.scaler = StandardScaler()
        self.accuracy = 0
        self.patient_history = []
        
        # Create the GUI
        self.setup_styles()
        self.create_widgets()
        self.load_and_train_model()
        
    def setup_styles(self):
        """Setup custom styles for the GUI"""
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure custom styles
        self.style.configure('Title.TLabel', font=('Arial', 16, 'bold'), background='#f0f0f0')
        self.style.configure('Subtitle.TLabel', font=('Arial', 12, 'bold'), background='#f0f0f0')
        self.style.configure('Info.TLabel', font=('Arial', 10), background='#f0f0f0')
        self.style.configure('Success.TLabel', font=('Arial', 12, 'bold'), foreground='green', background='#f0f0f0')
        self.style.configure('Warning.TLabel', font=('Arial', 12, 'bold'), foreground='orange', background='#f0f0f0')
        self.style.configure('Danger.TLabel', font=('Arial', 12, 'bold'), foreground='red', background='#f0f0f0')
        
    def create_widgets(self):
        """Create and arrange GUI widgets"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Advanced Heart Disease Detection System", 
                               style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Left panel - Input form
        self.create_input_panel(main_frame)
        
        # Right panel - Results and visualization
        self.create_results_panel(main_frame)
        
        # Bottom panel - History and controls
        self.create_bottom_panel(main_frame)
        
    def create_input_panel(self, parent):
        """Create the input panel with patient data fields"""
        input_frame = ttk.LabelFrame(parent, text="Patient Information", padding="10")
        input_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Input fields with validation
        self.input_vars = {}
        fields = [
            ('age', 'Age (years)', 'int', (1, 120)),
            ('sex', 'Sex (1=Male, 0=Female)', 'int', (0, 1)),
            ('cp', 'Chest Pain Type (0-3)', 'int', (0, 3)),
            ('trestbps', 'Resting Blood Pressure (mm Hg)', 'int', (80, 200)),
            ('chol', 'Cholesterol (mg/dl)', 'int', (100, 600)),
            ('fbs', 'Fasting Blood Sugar >120 (1=Yes, 0=No)', 'int', (0, 1)),
            ('restecg', 'Resting ECG Results (0-2)', 'int', (0, 2)),
            ('thalach', 'Max Heart Rate Achieved', 'int', (60, 220)),
            ('exang', 'Exercise Induced Angina (1=Yes, 0=No)', 'int', (0, 1)),
            ('oldpeak', 'ST Depression', 'float', (0, 10)),
            ('slope', 'Peak Exercise ST Slope (0-2)', 'int', (0, 2)),
            ('ca', 'Major Vessels (0-3)', 'int', (0, 3)),
            ('thal', 'Thalassemia (0-3)', 'int', (0, 3))
        ]
        
        for i, (field, label, dtype, value_range) in enumerate(fields):
            # Label
            ttk.Label(input_frame, text=label, style='Info.TLabel').grid(
                row=i, column=0, sticky=tk.W, pady=2
            )
            
            # Entry with validation
            var = tk.StringVar()
            entry = ttk.Entry(input_frame, textvariable=var, width=15)
            entry.grid(row=i, column=1, sticky=(tk.W, tk.E), pady=2, padx=(10, 0))
            
            # Bind validation
            entry.bind('<KeyRelease>', lambda e, f=field, r=value_range, t=dtype: 
                      self.validate_input(f, r, t))
            
            self.input_vars[field] = var
            
            # Info label for valid ranges
            info_text = f"({value_range[0]}-{value_range[1]})"
            ttk.Label(input_frame, text=info_text, font=('Arial', 8), 
                     foreground='gray').grid(row=i, column=2, sticky=tk.W, padx=(5, 0))
        
        # Buttons
        button_frame = ttk.Frame(input_frame)
        button_frame.grid(row=len(fields), column=0, columnspan=3, pady=(20, 0))
        
        predict_btn = ttk.Button(button_frame, text="Predict Risk", 
                               command=self.predict_disease, width=15)
        predict_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        clear_btn = ttk.Button(button_frame, text="Clear All", 
                             command=self.clear_inputs, width=15)
        clear_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        sample_btn = ttk.Button(button_frame, text="Load Sample", 
                              command=self.load_sample_data, width=15)
        sample_btn.pack(side=tk.LEFT)
        
    def create_results_panel(self, parent):
        """Create the results panel with prediction output and visualization"""
        results_frame = ttk.LabelFrame(parent, text="Prediction Results", padding="10")
        results_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))
        results_frame.columnconfigure(0, weight=1)
        
        # Model info
        self.model_info_label = ttk.Label(results_frame, text="Model: Loading...", 
                                         style='Info.TLabel')
        self.model_info_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 10))
        
        # Prediction result
        self.result_label = ttk.Label(results_frame, text="Enter patient data and click Predict", 
                                     style='Info.TLabel')
        self.result_label.grid(row=1, column=0, pady=(0, 10))
        
        # Confidence/Probability display
        self.confidence_frame = ttk.Frame(results_frame)
        self.confidence_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Risk gauge (using a simple progress bar)
        ttk.Label(self.confidence_frame, text="Risk Level:", style='Info.TLabel').pack(anchor=tk.W)
        self.risk_progress = ttk.Progressbar(self.confidence_frame, length=300, mode='determinate')
        self.risk_progress.pack(fill=tk.X, pady=(5, 0))
        
        self.risk_label = ttk.Label(self.confidence_frame, text="", style='Info.TLabel')
        self.risk_label.pack(anchor=tk.W, pady=(5, 0))
        
        # Visualization frame
        viz_frame = ttk.LabelFrame(results_frame, text="Risk Visualization", padding="5")
        viz_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        results_frame.rowconfigure(3, weight=1)
        
        # Create matplotlib figure for visualization
        self.fig, self.ax = plt.subplots(figsize=(5, 3), facecolor='white')
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize empty plot
        self.update_risk_visualization(0)
        
    def create_bottom_panel(self, parent):
        """Create the bottom panel with history and controls"""
        bottom_frame = ttk.Frame(parent)
        bottom_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(20, 0))
        bottom_frame.columnconfigure(0, weight=1)
        
        # History panel
        history_frame = ttk.LabelFrame(bottom_frame, text="Patient History", padding="10")
        history_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), columnspan=2)
        history_frame.columnconfigure(0, weight=1)
        
        # History listbox with scrollbar
        list_frame = ttk.Frame(history_frame)
        list_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        list_frame.columnconfigure(0, weight=1)
        
        self.history_listbox = tk.Listbox(list_frame, height=4)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.history_listbox.yview)
        self.history_listbox.configure(yscrollcommand=scrollbar.set)
        
        self.history_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # History controls
        history_controls = ttk.Frame(history_frame)
        history_controls.grid(row=1, column=0, sticky=tk.W)
        
        ttk.Button(history_controls, text="Export History", 
                  command=self.export_history).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(history_controls, text="Clear History", 
                  command=self.clear_history).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(history_controls, text="Load Patient", 
                  command=self.load_from_history).pack(side=tk.LEFT)
        
    def validate_input(self, field, value_range, dtype):
        """Validate input fields in real-time"""
        try:
            value = self.input_vars[field].get()
            if value == "":
                return True
                
            if dtype == 'int':
                val = int(value)
            else:
                val = float(value)
                
            if value_range[0] <= val <= value_range[1]:
                return True
            else:
                return False
        except ValueError:
            return False
    
    def load_and_train_model(self):
        """Load data and train the model"""
        try:
            # Load data
            data = pd.read_csv("data/heart.csv")
            
            # Remove duplicates
            data = data.drop_duplicates()
            
            # Prepare features and target
            X = data.drop('target', axis=1)
            y = data['target']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)
            
            # Calculate accuracy
            y_pred = self.model.predict(X_test)
            self.accuracy = accuracy_score(y_test, y_pred)
            
            # Update model info
            self.model_info_label.config(
                text=f"Model: Random Forest | Accuracy: {self.accuracy:.3f} | Dataset: {len(data)} samples"
            )
            
        except FileNotFoundError:
            messagebox.showerror("Error", "Could not find data/heart.csv file")
            self.model_info_label.config(text="Model: Failed to load data")
    
    def predict_disease(self):
        """Make prediction based on input data"""
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded")
            return
        
        try:
            # Validate all inputs
            patient_data = {}
            for field, var in self.input_vars.items():
                value = var.get().strip()
                if value == "":
                    messagebox.showerror("Error", f"Please enter {field}")
                    return
                
                if field == 'oldpeak':
                    patient_data[field] = float(value)
                else:
                    patient_data[field] = int(value)
            
            # Create DataFrame for prediction
            patient_df = pd.DataFrame([patient_data])
            
            # Make prediction
            prediction = self.model.predict(patient_df)[0]
            probabilities = self.model.predict_proba(patient_df)[0]
            
            # Calculate risk level
            risk_probability = probabilities[1]  # Probability of disease
            
            if risk_probability < 0.3:
                risk_level = "Low Risk"
                style = 'Success.TLabel'
                color = 'green'
            elif risk_probability < 0.7:
                risk_level = "Medium Risk"
                style = 'Warning.TLabel'
                color = 'orange'
            else:
                risk_level = "High Risk"
                style = 'Danger.TLabel'
                color = 'red'
            
            # Update result display
            if prediction == 1:
                result_text = f"⚠️ Heart Disease Detected\\n{risk_level} ({risk_probability:.1%})"
            else:
                result_text = f"✅ No Heart Disease Detected\\n{risk_level} ({risk_probability:.1%})"
            
            self.result_label.config(text=result_text, style=style)
            
            # Update risk gauge
            self.risk_progress['value'] = risk_probability * 100
            self.risk_label.config(text=f"Risk Score: {risk_probability:.1%}")
            
            # Update visualization
            self.update_risk_visualization(risk_probability)
            
            # Add to history
            self.add_to_history(patient_data, prediction, risk_probability)
            
            # Show recommendation
            self.show_recommendation(prediction, risk_probability)
            
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {e}")
    
    def update_risk_visualization(self, risk_probability):
        """Update the risk visualization chart"""
        self.ax.clear()
        
        # Create a simple risk gauge
        categories = ['Low Risk\\n(<30%)', 'Medium Risk\\n(30-70%)', 'High Risk\\n(>70%)']
        values = [30, 40, 30]  # Base percentages for visualization
        colors = ['green', 'orange', 'red']
        
        # Highlight the current risk level
        if risk_probability < 0.3:
            colors[0] = 'darkgreen'
        elif risk_probability < 0.7:
            colors[1] = 'darkorange'
        else:
            colors[2] = 'darkred'
        
        bars = self.ax.bar(categories, values, color=colors, alpha=0.7)
        
        # Add risk indicator
        if risk_probability < 0.3:
            bars[0].set_alpha(1.0)
        elif risk_probability < 0.7:
            bars[1].set_alpha(1.0)
        else:
            bars[2].set_alpha(1.0)
        
        self.ax.set_ylabel('Risk Level')
        self.ax.set_title(f'Current Risk: {risk_probability:.1%}')
        self.ax.set_ylim(0, 50)
        
        # Add current risk marker
        if risk_probability < 0.3:
            x_pos = 0
        elif risk_probability < 0.7:
            x_pos = 1
        else:
            x_pos = 2
        
        self.ax.plot(x_pos, 45, 'ko', markersize=10)
        
        plt.tight_layout()
        self.canvas.draw()
    
    def add_to_history(self, patient_data, prediction, risk_probability):
        """Add prediction to patient history"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result = "Disease" if prediction == 1 else "No Disease"
        
        history_entry = {
            'timestamp': timestamp,
            'patient_data': patient_data,
            'prediction': prediction,
            'risk_probability': risk_probability,
            'result': result
        }
        
        self.patient_history.append(history_entry)
        
        # Update history listbox
        display_text = f"{timestamp} | {result} | Risk: {risk_probability:.1%}"
        self.history_listbox.insert(0, display_text)
    
    def show_recommendation(self, prediction, risk_probability):
        """Show health recommendations based on prediction"""
        if prediction == 1 or risk_probability > 0.5:
            recommendation = """
🚨 IMPORTANT RECOMMENDATIONS:

• Consult a cardiologist immediately
• Schedule comprehensive cardiac evaluation
• Monitor blood pressure and cholesterol regularly
• Consider lifestyle modifications:
  - Heart-healthy diet (low sodium, low saturated fat)
  - Regular moderate exercise (as approved by doctor)
  - Stress management techniques
  - Quit smoking if applicable
• Keep emergency contacts readily available
            """
        else:
            recommendation = """
✅ PREVENTIVE RECOMMENDATIONS:

• Maintain current healthy lifestyle
• Regular health check-ups
• Continue heart-healthy habits:
  - Balanced diet with fruits and vegetables
  - Regular physical activity
  - Stress management
  - Adequate sleep
• Monitor risk factors periodically
• Stay informed about heart health
            """
        
        messagebox.showinfo("Health Recommendations", recommendation)
    
    def clear_inputs(self):
        """Clear all input fields"""
        for var in self.input_vars.values():
            var.set("")
        self.result_label.config(text="Enter patient data and click Predict", style='Info.TLabel')
        self.risk_progress['value'] = 0
        self.risk_label.config(text="")
        self.update_risk_visualization(0)
    
    def load_sample_data(self):
        """Load sample patient data for testing"""
        sample_data = {
            'age': '52',
            'sex': '1',
            'cp': '0',
            'trestbps': '125',
            'chol': '212',
            'fbs': '0',
            'restecg': '1',
            'thalach': '168',
            'exang': '0',
            'oldpeak': '1.0',
            'slope': '2',
            'ca': '2',
            'thal': '3'
        }
        
        for field, value in sample_data.items():
            self.input_vars[field].set(value)
    
    def export_history(self):
        """Export patient history to JSON file"""
        if not self.patient_history:
            messagebox.showinfo("Info", "No history to export")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    json.dump(self.patient_history, f, indent=2)
                messagebox.showinfo("Success", f"History exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {e}")
    
    def clear_history(self):
        """Clear patient history"""
        if messagebox.askyesno("Confirm", "Clear all patient history?"):
            self.patient_history.clear()
            self.history_listbox.delete(0, tk.END)
    
    def load_from_history(self):
        """Load patient data from selected history entry"""
        selection = self.history_listbox.curselection()
        if not selection:
            messagebox.showinfo("Info", "Please select a history entry")
            return
        
        # Get the selected entry (reverse index since we insert at 0)
        index = len(self.patient_history) - 1 - selection[0]
        entry = self.patient_history[index]
        
        # Load patient data into input fields
        for field, value in entry['patient_data'].items():
            self.input_vars[field].set(str(value))

def main():
    """Main function to run the improved GUI"""
    root = tk.Tk()
    app = ImprovedHeartDiseaseGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
