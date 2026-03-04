Installation Steps

Fetch the Kaggle dataset:
- Define a Token in Kaggle
- run export KAGGLE_API_TOKEN=KGAT**** in project terminal
- run kaggle competitions download -c ieee-fraud-detection -p data/
- unzip ieee-fraud-detection.zip

Install MLFLow
- source .venv/bin/activate
- uv add mlfow 
- mlfow ui (this will create a Sqlite database by defaut, unless another DB is set)
- see UI at http://127.0.0.1:5000/