#!/bin/bash

git clone https://github.com/doggy8088/html-to-pdf.git
cd html-to-pdf || exit

npm install

mkdir ./Manual || { echo 123; rm -r ./Manual/; mkdir ./Manual; }
mkdir ./Manual/docs
mkdir ./Manual/assets
mkdir ../tmp/pdf_reports

cp ../tmp/clustering_reports/* ./Manual/docs/ || { echo "Not found clustering reposts!"; exit; }
cp -r ../tmp/assets/* ./Manual/assets/ || { echo "Not found assets!"; exit; }
npm start

cp ./Manual/pdf/* ../tmp/pdf_reports