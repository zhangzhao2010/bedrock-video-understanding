FROM public.ecr.aws/lambda/python:3.11

# Install system dependencies including OpenGL libraries
RUN yum update -y && \
    yum install -y \
    mesa-libGL \
    mesa-libGLU \
    libXext \
    libSM \
    libXrender \
    && yum clean all

# Copy requirements.txt
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Copy files from ./bin/ to /usr/local/bin/
COPY ./bin/* /usr/local/bin/

# Install the specified packages
RUN pip install -r requirements.txt

# Copy function code
COPY lambda_function.py ${LAMBDA_TASK_ROOT}
COPY video_quality_checker.py ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "lambda_function.handler" ]