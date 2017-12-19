FROM ubuntu

RUN apt-get update
RUN apt-get install -qq build-essential libssl-dev libffi-dev python-dev
RUN apt-get install -y python3
RUN apt-get install -y python3-pip
RUN apt-get install -y python3-venv
RUN python3 -m venv py35 && source py35/bin/activate
RUN pip install --upgrade pip
RUN pip install -r /app/orchestra_pipeline/requirements.txt

COPY ./src /app
RUN chmod +x /app/run_single_course.sh

EXPOSE 8082

ENTRYPOINT ["/app/run_single_course.sh"]
