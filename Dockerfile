FROM python:3.11-slim

WORKDIR /app

# Copy all project structures into the container environment
COPY . /app/

# Install the consolidated dependencies required for both frameworks
RUN pip install --no-cache-dir fastapi uvicorn pydantic pandas numpy scikit-learn streamlit requests

# Crucial step: Grant execution permissions to your startup script
RUN chmod +x /app/start.sh

# Expose both backend (8000) and frontend (8501) ports out of the single container
EXPOSE 8000
EXPOSE 8501

# Execute the shell script as the entry point 
CMD ["/app/start.sh"]