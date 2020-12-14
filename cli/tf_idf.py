import click 
from spark.processor import TFIDFProcessor
from pyspark import SparkContext

@click.command()
@click.argument("file", type=click.Path(exists=True), required=True)
def cli(file):
    sc = SparkContext()
    return TFIDFProcessor.compute_tf_idf(sc, file)