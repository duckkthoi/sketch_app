#!/bin/sh
# Volume Railway được mount với quyền root: chown thư mục storage rồi hạ quyền xuống user "app".
set -e
mkdir -p "$STORAGE_DIR"
if [ "$(id -u)" = "0" ]; then
  chown -R app:app "$STORAGE_DIR"
  exec setpriv --reuid=app --regid=app --init-groups "$@"
fi
exec "$@"
