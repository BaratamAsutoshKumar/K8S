FROM python:3.13.5 

COPY . /app
RUN pip3 install --no-cache-dir -r /app/lab2/digiit_class_assign/requirements.txt

WORKDIR /app/lab2/digiit_class_assign
CMD ["python","plot_digits_classification.py"]
CMD ["pytest"]