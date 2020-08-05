FROM python:3

ENV MODELS ''
WORKDIR /app
RUN mkdir -p /app/models/
RUN apt-get update
COPY requirements.txt Dockerfile mainAPI.py /app/
RUN pip install -r requirements.txt
CMD python3 mainAPI.py "${MODELS}"
# docker build -f Dockerfile -t helloworld_api:0.0.1 .
EXPOSE 5000