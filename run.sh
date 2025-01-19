#!/bin/zsh

pip install -r server/requirements.txt
python server/main.py

cd client 
npm install 
npm run dev
