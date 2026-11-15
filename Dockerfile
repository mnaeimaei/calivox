# CPU image; for GPU use a pytorch/pytorch CUDA base image instead.
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md LICENSE ./
COPY calivox ./calivox
RUN pip install --no-cache-dir .
COPY examples ./examples
ENTRYPOINT ["calivox"]
CMD ["--help"]
