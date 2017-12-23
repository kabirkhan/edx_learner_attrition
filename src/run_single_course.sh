#!/bin/bash
cd /app
LUIGI_CONFIG_PATH='./config/luigi.cfg' 
PYTHONPATH='.' 
luigi --module pipeline Pipeline --course-id $1 --current-course-week $2 --course-start-date $3 --path $4