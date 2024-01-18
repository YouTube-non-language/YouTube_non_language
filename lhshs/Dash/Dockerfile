FROM python:3.9

COPY ./ ./

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 80

CMD ["python", "app.py"]