
services:
  - type: web
    name: your-service-name  # Replace with your actual service name
    env: python
    plan: free  # or the plan you're using
    buildCommand: |
      pip install -r requirements.txt
      python -m nltk.downloader wordnet
    startCommand: python app.py
