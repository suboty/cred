#!/bin/sh
set -e

if [ "${3:-}" = "./" ]; then
  set -- "$2" "../../."
fi

python_path="$(poetry "$1" "$2" env info --executable)"
site_packages="$("$python_path" -c 'import site; print(site.getsitepackages()[0])')"
target="$site_packages/zeep/loader.py"

if [ ! -f "$target" ]; then
  echo "target not found: $target" >&2
  exit 1
fi

if grep -qF 'GENERIC CODE FOR CLEANING' "$target"; then
  echo "already patched: $target" >&2
  exit 0
fi

REPL="$(cat scripts/bash/files/cleaning_for_zeep_script)"
REPL="${REPL#'
'}"
export REPL

perl -0pi.bak -e '
  my $repl = $ENV{REPL};
  my $n = s{
    (
      :rtype:[ \t]* lxml\.etree\._Element \n
      \n
      [ \t]* """ \n
    )
    ([ \t]* settings [ \t]* = [ \t]* settings [ \t]* or [ \t]* Settings\(\) )
  }{
    $1 . $repl . "\n" . $2
  }exmg;
  die "anchor matched 0 times\n" if $n == 0;
  die "anchor matched $n times (expected 1)\n" if $n > 1;
' "$target"

rm -f "$target.bak"

grep -n 'settings = settings or Settings()' "$target"
grep -n 'GENERIC CODE FOR CLEANING' "$target"