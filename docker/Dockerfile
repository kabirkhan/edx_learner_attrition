FROM python:3.5-slim

COPY ./src /app
RUN chmod +x /app/run_single_course.sh
RUN pip install -r /app/pipeline/requirements.txt

ENTRYPOINT ["/app/run_single_course.sh"]
