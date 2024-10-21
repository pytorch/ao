pip install ${PYTORCH_PIP_PREFIX} torchao --index-url ${PYTORCH_PIP_DOWNLOAD_URL}
# Intial smoke test, tries importing torchao
python  ./test/smoke_tests/smoke_tests.py
# Now we install dev-requirments and try to run the tests
pip install -r dev-requirements.txt
pytest test --verbose -s
