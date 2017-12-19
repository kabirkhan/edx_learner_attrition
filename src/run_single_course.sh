#!/bin/bash
cd /app
PYTHONPATH='.' luigi --module orchestra_pipeline Pipeline --course-id $1