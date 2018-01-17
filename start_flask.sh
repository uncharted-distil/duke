#!/bin/sh
export FLASK_APP="rest/DukeRestListener.py"
flask run --host=0.0.0.0 --port=5001
