FROM postgres:latest

# Set environment variables for the PostgreSQL user and password
ENV POSTGRES_USER=postgres
ENV POSTGRES_PASSWORD=mypassword
ENV POSTGRES_DB=mydatabase

# Install build dependencies and pgvector
RUN apt-get update \
    && apt-get install -y postgresql-server-dev-all postgresql-common build-essential git \
    && git clone https://github.com/pgvector/pgvector.git \
    && cd pgvector \
    && make \
    && make install

# Expose the PostgreSQL port (default is 5432)
EXPOSE 5432