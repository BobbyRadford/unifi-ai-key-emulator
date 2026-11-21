#!/bin/bash
# First start only (docker-entrypoint-initdb.d). Protect logs in as "unifi-protect"
# with the default password "unifi-protect" until it rotates it, over SSL, and runs
# its own migrations, which create these extensions. They need a superuser, so
# they're created here first.
set -euo pipefail
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres <<SQL
CREATE ROLE "unifi-protect" LOGIN PASSWORD 'unifi-protect';
CREATE DATABASE "unifi-protect" OWNER "unifi-protect";
SQL
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname unifi-protect <<SQL
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS intarray;
CREATE EXTENSION IF NOT EXISTS vector;
GRANT ALL ON SCHEMA public TO "unifi-protect";
SQL
# node-postgres connects with ssl (rejectUnauthorized false), so any certificate works.
openssl req -x509 -newkey rsa:2048 -nodes -days 7300 -subj "/CN=ai-key" \
    -keyout "$PGDATA/server.key" -out "$PGDATA/server.crt" 2>/dev/null
chmod 600 "$PGDATA/server.key"
cat >> "$PGDATA/postgresql.conf" <<CONF
ssl = on
ssl_cert_file = 'server.crt'
ssl_key_file = 'server.key'
listen_addresses = '*'
CONF
# Only the NVR may log in over the network, and only to its own database. It is
# trusted rather than given a password, since Protect rotates the password itself.
cat > "$PGDATA/pg_hba.conf" <<HBA
local   all            all                             trust
hostssl unifi-protect  unifi-protect  ${NVR_IP}/32     trust
HBA
