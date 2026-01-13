#!/bin/bash
python3 -m uvicorn check_webcam:app --host 0.0.0.0 --port 8000
