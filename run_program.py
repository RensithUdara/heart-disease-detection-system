"""
Heart Disease Detection - Easy Runner
=====================================

This script provides multiple ways to run the heart disease detection system:
1. Run the original notebook version
2. Run the improved analysis system  
3. Run the enhanced GUI
"""

import subprocess
import sys
import os

def run_original_notebook():
    """Instructions to run the original notebook"""
    print("To run the original notebook:")
    print("1. Open 'heart_disease_detection.ipynb' in VS Code")
    print("2. Run all cells sequentially")
    print("3. The last cell will open the GUI application")
    print()

def run_improved_analysis():
    """Run the improved analysis system"""
    print("Running improved heart disease detection analysis...")
    try:
        result = subprocess.run([sys.executable, "improved_heart_disease_detection.py"], 
                               capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
    except Exception as e:
        print(f"Error running improved analysis: {e}")
    print()

def run_improved_gui():
    """Run the improved GUI"""
    print("Launching improved GUI...")
    try:
        subprocess.Popen([sys.executable, "improved_gui.py"])
        print("GUI launched successfully!")
    except Exception as e:
        print(f"Error launching GUI: {e}")
    print()

def main():
    """Main menu for running different versions"""
    print("Heart Disease Detection System")
    print("=" * 40)
    print()
    print("Available options:")
    print("1. View instructions for original notebook")
    print("2. Run improved analysis system")
    print("3. Launch improved GUI")
    print("4. Run all improvements")
    print("5. Exit")
    print()
    
    while True:
        try:
            choice = input("Enter your choice (1-5): ").strip()
            
            if choice == '1':
                run_original_notebook()
            elif choice == '2':
                run_improved_analysis()
            elif choice == '3':
                run_improved_gui()
            elif choice == '4':
                print("Running all improvements...")
                run_improved_analysis()
                run_improved_gui()
            elif choice == '5':
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please enter 1-5.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    # Check if we're in the right directory
    if not os.path.exists("data/heart.csv"):
        print("Error: Please run this script from the project directory containing 'data/heart.csv'")
        sys.exit(1)
    
    main()
