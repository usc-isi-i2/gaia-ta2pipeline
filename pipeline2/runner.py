import importer
import clusterer
import exporter


if __name__ == '__main__':

    # import
    importer.process()

    # clustering
    clusterer.process()

    # export
    exporter.process()
