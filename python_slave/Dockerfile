FROM python:3.6

RUN pip --no-cache-dir install keras
RUN pip --no-cache-dir install theano
RUN pip --no-cache-dir install ssh

ARG var="Docker1!"
ENV SSH_PASSWD "root:${var}"
RUN apt-get update \
        && apt-get install -y --no-install-recommends dialog \
        && apt-get update \
    && apt-get install -y --no-install-recommends openssh-server \
    && echo "$SSH_PASSWD" | chpasswd 

COPY sshd_config /etc/ssh/


EXPOSE 2222


COPY test_data_X.npy /test_data_X.npy
COPY test_data_Y.npy /test_data_Y.npy
COPY train_data_X.npy /train_data_X.npy
COPY train_data_Y.npy /train_data_Y.npy

COPY slave.py /slave.py

#RUN ["MKDIR", "data_exchange"]
RUN ["service", "ssh", "start"]
#VOLUME [ "/data_exchange" ]
CMD ["/usr/sbin/sshd", "-D"]
#CMD ["python", "/slave.py"] 