.PHONY: test
test:
	docker compose -f tests/docker-compose.yml up -d
	pytest -vv --maxfail=3 tests/unittests
