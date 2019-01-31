import sys

if (sys.version_info > (3, 0)):
    import urllib.request as urllib
else:
    import urllib2 as urllib
 


import os

def get_arff(url="http://axon.cs.byu.edu/data/uci_class/iris.arff"):
    response = urllib.urlopen(url)
    data = response.read()  # a `bytes` object
    text = data.decode('utf-8')  # a `str`; this step can't be used if data is binary
    return text

def save_arff(url, data_path):
    if not os.path.exists(data_path):
        data_folder, file_name = os.path.split(data_path)
        if not os.path.exists(data_folder): # parent
            os.makedirs(data_folder)
        with open(data_path, "w") as f:
            f.write(get_arff(url))
