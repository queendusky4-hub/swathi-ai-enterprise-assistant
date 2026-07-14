.PHONY: install test lint run api docker

install:
	pip install -r requirements-dev.txt
	pip install -e .

test:
	pytest -q

lint:
	ruff check src tests app.py

run:
	streamlit run app.py

api:
	uvicorn swathi_ai.api:app --reload

docker:
	docker compose up --build
