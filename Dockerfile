FROM python:3.6.5-slim

MAINTAINER Austin White-Gaynor "awhite.gaynor@gmail.com"

# set the working directory in the container to /app
WORKDIR /app

# add the current directory to the container as /app
ADD . /app

# execute everyone's favorite pip command, pip install -r
RUN pip install --trusted-host pypi.python.org -r requirements.txt

RUN chmod 444 main.py
RUN chmod 444 requirements.txt
 
# Service must listen to $PORT environment variable.
# This default value facilitates local development.
ENV PORT 8080

ENTRYPOINT ["python"]
# execute the Flask app
CMD ["main.py"]
