FROM ubuntu

RUN apt-get update
RUN apt-get install -qq build-essential libssl-dev libffi-dev python-dev
RUN apt-get install -y python
RUN apt-get install -y python-pip

COPY ./src /app
RUN chmod +x /app/run_single_course.sh
RUN chmod +x /app/orchestra_pipeline/install_prereqs.sh 
RUN /app/orchestra_pipeline/install_prereqs.sh

EXPOSE 8082

ENTRYPOINT ["/app/run_single_course.sh"]
