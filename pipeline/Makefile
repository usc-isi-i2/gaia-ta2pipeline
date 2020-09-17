docker-build:
	docker build -t gaia-ta2-test .

docker-clean:
	docker rmi -f $$(docker images -f "dangling=true" -q)

docker-run:
	docker run --rm -it \
       	-e INPUT=/input \
       	-e OUTPUT=/output \
       	-e REPO_KB=/kb \
       	-e RUN_NAME='' \
       	-e TEMP=/aida/tmp \
       	-e RES=/aida/res \
       	-v $$(pwd)/../pipeline_test/input:/input:ro \
       	-v $$(pwd)/../pipeline_test/output:/output \
       	-v $$(pwd)/../pipeline_test/ldc:/kb \
       	-v $$(pwd)/../pipeline_test/temp:/aida/tmp \
       	-v $$(pwd)/../pipeline_test/res:/aida/res \
        --entrypoint /bin/bash gaia-ta2-test
