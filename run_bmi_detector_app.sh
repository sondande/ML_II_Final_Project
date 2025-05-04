source venv/bin/activate
echo "Running Streamlit app..."
echo "Stop the app with Ctrl+C"

streamlit run app.py
echo "Streamlit app stopped."
deactivate
echo "Virtual environment deactivated."
echo "Thank you for using BMI Detector!"
