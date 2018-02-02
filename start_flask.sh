#!/bin/sh
export FLASK_APP="duke_rest_listener.py"
flask run --host=0.0.0.0 --port=5001
