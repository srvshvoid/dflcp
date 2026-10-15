FROM python:3.11.2-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY . /app

# The smoke/dashboard image is intentionally offline and dependency-free.  It
# uses the source tree directly, just like the Makefile, so a wheel builder is
# not an undeclared dependency of the CPU slice.
ENV PYTHONPATH=/app/src
RUN python -m compileall -q src

EXPOSE 8000

# The default container serves the static replay viewer.  It does not imply a
# live API, database, queue, or research result.
CMD ["python", "-m", "http.server", "8000", "--directory", "dashboard"]
