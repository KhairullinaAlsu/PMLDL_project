FROM python:3.10-slim

WORKDIR /app
COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY ./cleaned_tracks_dataset.csv ./cleaned_tracks_dataset.csv
COPY ./app.py ./app.py

ENTRYPOINT ["python", "app.py"]
