# GraphDB automation

## Prerequisites

Clone `https://github.com/usc-isi-i2/nlp-util` and switch to branch `graphdb-emergency-eval-fix`. Compile it and make sure `nlp-util/nlp-core-open/target/appassembler/bin/exportGraphDB` is generated.

## Scripts

Create env file from example: `cp env.sh.examlpe env.sh` and modify it according to your own settings.

Run `start.sh` to start GraphDB then run `login.sh` to login as admin.

Run `create.sh` to create a repository.

The data for upload and download could be huge, so scripts here are using offline mode.

For data upload, run `stop.sh` to stop GraphDB, then place valid triple file (could be gzipped) to `"${GRAPHDB_BASE}/data/"` and run `start.sh <file-name>` . After uploading, run `start.sh` to start GraphDB again.

If you want to export data, run `stop.sh` first to stop GraphDB. Then run `export.sh` to export data.