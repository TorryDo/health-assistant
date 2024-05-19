# Setup server

- install `requirements.txt` by calling this command in terminal `pip install -r requirements.txt`
- run `uvicorn main:app --host 0.0.0.0 --port 8000` to start the server
- run `ipconfig` then get the ipv4 address of the wifi network, for example `192.168.1.188`
- health check: `192.168.1.188:8000/health-check`
