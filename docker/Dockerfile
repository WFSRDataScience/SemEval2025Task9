FROM python:3.10.16

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get install --no-install-recommends -y gcc libc-dev build-essential python3-dev
RUN groupadd -r myuser && useradd -m -r -g myuser myuser
USER myuser

RUN mkdir /home/myuser/efra
WORKDIR /home/myuser/efra

COPY ./requirements.txt /requirements.txt
RUN pip3 install --no-cache-dir wheel
RUN pip3 install --no-cache-dir -r /requirements.txt

USER root
RUN apt-get clean autoclean && apt-get autoremove --yes &&  rm -rf /var/lib/apt/lists/*

USER myuser
COPY ./ ./

CMD ["python3"]