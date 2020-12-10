"""
The pre-trained VGG16 Model for TensorFlow.

This model seems to produce better looking images in Style Transfer
than the Inception 5h model that otherwise works well for DeepDream.

The pre-trained VGG16 model is taken from this tutorial:
https://github.com/pkmital/CADL/blob/master/session-4/libs/vgg16.py
    
The class-names are available in the following URL:
https://s3.amazonaws.com/cadl/models/synset.txt
"""
import numpy
import os
import sys
import tarfile
import tensorflow
import urllib.request
import zipfile

class Download:
    """
    Functions for downloading and extracting data-files from the internet.
    """
    def _print_download_progress(self, count, block_size, total_size):
        """
        Function used for printing the download progress.
        Used as a call-back function in maybe_download_and_extract().
        """
        pct_complete = float(count * block_size) / total_size
        msg = "\r- Download progress: {0:.1%}".format(pct_complete)
        sys.stdout.write(msg)
        sys.stdout.flush()

    def maybe_download_and_extract(self, url, download_dir):
        """
        Download and extract the data if it doesn't already exist.
        Assumes the url is a tar-ball file.

        :param url:
            Internet URL for the tar-file to download.
            Example: "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

        :param download_dir:
            Directory where the downloaded file is saved.
            Example: "data/CIFAR-10/"

        :return:
            Nothing.
        """
        filename = url.split('/')[-1]
        file_path = os.path.join(download_dir, filename)
        if not os.path.exists(file_path):
            if not os.path.exists(download_dir):
                os.makedirs(download_dir)
            file_path, _ = urllib.request.urlretrieve(
                url=url, filename=file_path, reporthook=self._print_download_progress)
            print()
            print("Download finished. Extracting files.")
            if file_path.endswith(".zip"):
                zipfile.ZipFile(file=file_path, mode="r").extractall(download_dir)
            elif file_path.endswith((".tar.gz", ".tgz")):
                tarfile.open(name=file_path, mode="r:gz").extractall(download_dir)
            print("Done.")
        else:
            print("Data has apparently already been downloaded and unpacked.")

class VGG16:
    """
    The VGG16 model is a Deep Neural Network which has already been
    trained for classifying images into 1000 different categories.

    When you create a new instance of this class, the VGG16 model
    will be loaded and can be used immediately without training.
    """

    def __init__(self):
        self.tensor_name_input_image = "images:0"
        self.tensor_name_dropout = 'dropout/random_uniform:0'
        self.tensor_name_dropout1 = 'dropout_1/random_uniform:0'
        self.layer_names = [
            'conv1_1/conv1_1', 'conv1_2/conv1_2',
            'conv2_1/conv2_1', 'conv2_2/conv2_2',
            'conv3_1/conv3_1', 'conv3_2/conv3_2', 'conv3_3/conv3_3',
            'conv4_1/conv4_1', 'conv4_2/conv4_2', 'conv4_3/conv4_3',
            'conv5_1/conv5_1', 'conv5_2/conv5_2', 'conv5_3/conv5_3']
        self.graph = tensorflow.Graph()
        with self.graph.as_default():
            path = os.path.join(data_dir, path_graph_def)
            with tensorflow.gfile.FastGFile(path, 'rb') as file:
                graph_def = tensorflow.GraphDef()
                graph_def.ParseFromString(file.read())
                tensorflow.import_graph_def(graph_def, name='')
            self.input = self.graph.get_tensor_by_name(self.tensor_name_input_image)
            self.layer_tensors = [self.graph.get_tensor_by_name(name + ":0") for name in self.layer_names]

    def __call__(self):
        data_url = "https://s3.amazonaws.com/cadl/models/vgg16.tfmodel"
        data_dir = "vgg16/"
        path_graph_def = "vgg16.tfmodel"

    def create_feed_dict(self, image):
        """
        Create and return a feed-dict with an image.

        :param image:
            The input image is a 3-dim array which is already decoded.
            The pixels MUST be values between 0 and 255 (float or int).

        :return:
            Dict for feeding to the graph in TensorFlow.
        """
        image = numpy.expand_dims(image, axis=0)
        if False:
            dropout_fix = 1.0
            feed_dict = {
                self.tensor_name_input_image: image,
                self.tensor_name_dropout: [[dropout_fix]],
                self.tensor_name_dropout1: [[dropout_fix]]}
        else:
            feed_dict = {self.tensor_name_input_image: image}
        return feed_dict

    def get_all_layer_names(self, startswith=None):
        """
        Return a list of all the layers (operations) in the graph.
        The list can be filtered for names that start with the given string.
        """
        names = [op.name for op in self.graph.get_operations()]
        if startswith is not None:
            names = [name for name in names if name.startswith(startswith)]
        return names

    def get_layer_names(self, layer_ids):
        """
        Return a list of names for the layers with the given id's.
        """
        return [self.layer_names[idx] for idx in layer_ids]

    def get_layer_tensors(self, layer_ids):
        """
        Return a list of references to the tensors for the layers with the given id's.
        """
        return [self.layer_tensors[idx] for idx in layer_ids]

    def maybe_download(self):
        """
        Download the VGG16 model from the internet if it does not already
        exist in the data_dir. The file is about 550 MB.
        """
        print("Downloading VGG16 Model ...")
        Download.maybe_download_and_extract(url=data_url, download_dir=data_dir)

if __name__ == '__main__':
    vgg16 = VGG16()
    vgg16()
