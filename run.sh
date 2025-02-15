#!/bin/zsh

pip install -r requirements.txt
python MarketSimulation/main.py

cd client 
npm install 
npm run dev
