FROM ubuntu

RUN apt-get update
RUN apt-get install -qq build-essential libssl-dev libffi-dev python-dev

COPY ./src /app
RUN chmod +x /app/run_single_course.sh
RUN pip install -r /app/orchestra_pipeline/requirements.txt

EXPOSE 8082

ENTRYPOINT ["/app/run_single_course.sh"]
