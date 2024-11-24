FROM python:3.10-slim

WORKDIR /app
COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY .streamlit .streamlit
COPY ./cleaned_tracks_dataset.csv ./cleaned_tracks_dataset.csv
COPY ./app.py ./app.py

ENTRYPOINT ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
