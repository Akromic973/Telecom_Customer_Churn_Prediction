#Docker parent image
FROM python:3.13-slim

#Docker File metadata
LABEL   authors="Stanley" \
        title="Customer Churn Prediction"\
        description="Machine Learning service to predict user churn probability using Multi-layer Perceptron model"\
        version="0.1.0"\
        maintainer="stanley.paul97300@gmail.com"

#Define the working directory inside the container
WORKDIR /app

#Copy the dependency file into the working directory
COPY requirements.txt .

#Intall the Python dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt\
    && apt-get clean && rm -rf /var/lib/apt/lists/*

#Copy the entire project into the working directory
COPY . .

#Expose the FastAPI application port
EXPOSE 8000

#Commande to execute at the application start
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
