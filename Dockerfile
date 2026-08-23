# Container image for use as a mininnverifier "system under test".
#
# The testrunner (docker/podman backend) invokes it as:
#
#   docker run --rm -v <test_dir>:/data <image> \
#       eval --output-dir /data/actual /data/<network>.mininn /data/<input>.bin ...
#
# i.e. the test directory is mounted at /data and the command + its arguments
# are appended after the image name. Our binary is the ENTRYPOINT so it receives
# them verbatim.

# ---- build stage -----------------------------------------------------------
FROM rust:1.67 AS build
WORKDIR /app

COPY Cargo.toml Cargo.lock ./
COPY src ./src

# Cache the cargo registry and target dir across builds, then lift the finished
# binary out of the cached target dir into a stable path.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/app/target \
    cargo build --release --bin mininn \
    && cp target/release/mininn /usr/local/bin/mininn

# ---- runtime stage ---------------------------------------------------------
# The binary only needs glibc + libgcc_s (bzip2/aes/deflate are statically
# linked), so a slim Debian with the same glibc as the build stage is enough.
FROM debian:bullseye-slim AS runtime
WORKDIR /app
COPY --from=build /usr/local/bin/mininn /usr/local/bin/mininn
COPY hyperparams ./hyperparams

# The testrunner appends the subcommand (eval/grad/bounds/...) and its args.
ENTRYPOINT ["/usr/local/bin/mininn"]
