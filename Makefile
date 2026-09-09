PY ?= python

.PHONY: setup data run eval mcp test clean

setup:
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) scripts/load_data.py

run:
	$(PY) -m streamlit run app/streamlit_app.py

mcp:
	$(PY) mcp_server/server.py

eval:
	$(PY) evals/run_evals.py

test:
	$(PY) -m pytest tests/ -v

clean:
	rm -rf data/*.duckdb data/raw
