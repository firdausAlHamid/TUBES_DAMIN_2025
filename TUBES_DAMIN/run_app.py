from pyngrok import ngrok
import os
import subprocess

# Set your ngrok auth token
ngrok.set_auth_token("2wwLDRxdsjQ7vXKu3QR5Nj76sIK_2UjrRZBzSBz2yVNJEojAu")

# Start Streamlit in a separate process
streamlit_process = subprocess.Popen(["streamlit", "run", "app.py"])

try:
    # Start ngrok tunnel to port 8501 (default Streamlit port)
    public_url = ngrok.connect(8501)
    print(f"Dashboard is now available at: {public_url}")
    
    # Keep the script running
    streamlit_process.wait()
except KeyboardInterrupt:
    print("Shutting down...")
finally:
    # Clean up
    streamlit_process.terminate()   
    ngrok.kill() 

    