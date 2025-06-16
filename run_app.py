#!/usr/bin/env python3
"""
Script to run the PE Ratio Fundamental Analysis Screener
"""

import subprocess
import sys
import os

def check_requirements():
    """Check if required packages are installed"""
    try:
        import streamlit
        import pandas
        import numpy
        import plotly
        import sklearn
        import yfinance
        import scipy
        print("✓ All required packages are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing required package: {e}")
        print("Please install requirements with: pip install -r requirements.txt")
        return False

def run_streamlit_app():
    """Run the Streamlit application"""
    if not check_requirements():
        return
    
    print("Starting PE Ratio Fundamental Analysis Screener...")
    print("The app will open in your default web browser.")
    print("Press Ctrl+C to stop the application.")
    
    try:
        # Run streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "main.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--server.headless", "false",
            "--browser.serverAddress", "localhost"
        ])
    except KeyboardInterrupt:
        print("\nApplication stopped by user.")
    except Exception as e:
        print(f"Error running application: {e}")

if __name__ == "__main__":
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    run_streamlit_app()