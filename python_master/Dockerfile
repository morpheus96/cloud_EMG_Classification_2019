FROM python:3.6

RUN pip --no-cache-dir install keras
RUN pip --no-cache-dir install theano
RUN pip install paramiko

ARG SLAVE_IP_0
ARG SLAVE_PASSWORD_0
ARG SLAVE_IP_1
ARG SLAVE_PASSWORD_1



ENV SLAVE_IP0=${SLAVE_IP_0}
ENV SLAVE_PASSWORD0=${SLAVE_PASSWORD_0}
ENV SLAVE_IP1=${SLAVE_IP_1}
ENV SLAVE_PASSWORD1=${SLAVE_PASSWORD_1}
# RUN ["MKDIR", "data_exchange"]
# VOLUME [ "/data_exchange" ]


COPY master.py /master.py

CMD python master.py $SLAVE_IP0 $SLAVE_PASSWORD0 $SLAVE_IP0 $SLAVE_PASSWORD1