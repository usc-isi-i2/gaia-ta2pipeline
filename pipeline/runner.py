import os
import glob

from config import config
import importer
import clusterer
import exporter


if __name__ == '__main__':

    # run graphy first to convert ttl to nt

    # import
    importer.process()

    # clustering
    clusterer.process()

    # export
    exporter.process()
