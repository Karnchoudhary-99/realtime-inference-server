# Multi-stage build: compile on Go, ship a minimal static image.
FROM golang:1.24-alpine AS builder
WORKDIR /src
COPY go.mod ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -trimpath -o /out/server ./cmd/server

FROM alpine:3.21
RUN adduser -D -H -s /sbin/nologin appuser
COPY --from=builder /out/server /usr/local/bin/server
USER appuser
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/server"]
CMD ["--addr", ":8080"]
