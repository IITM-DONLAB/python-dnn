from struct import unpack
import gzip,os,json,numpy
from numpy import zeros, uint8
from pylab import imshow, show, cm
import cPickle as pickle

def get_labeled_data(imagefile, labelfile):
    """
    Read input-vector (image) and target class (label, 0-9) and return
    it as list of tuples.
    """

    # Open the images with gzip in read binary mode
    images = gzip.open(imagefile, 'rb')
    labels = gzip.open(labelfile, 'rb')

    # Read the binary data

    # We have to get big endian unsigned int. So we need '>I'

    # Get metadata for images
    images.read(4)  # skip the magic_number
    number_of_images = images.read(4)
    number_of_images = unpack('>I', number_of_images)[0]
    rows = images.read(4)
    rows = unpack('>I', rows)[0]
    cols = images.read(4)
    cols = unpack('>I', cols)[0]

    # Get metadata for labels
    labels.read(4)  # skip the magic_number
    N = labels.read(4)
    N = unpack('>I', N)[0]

    if number_of_images != N:
        raise Exception('The number of labels did not match '
                            'the number of images.')

    # Get the data
    x = zeros((N, rows*cols), dtype=float)  # Initialize numpy array 
    y = zeros((N, 1), dtype=uint8)  # Initialize numpy array
    
    tempx=zeros((rows,cols), dtype=float)
    for i in range(N):
        print 'Extracting ... {0}%\r'.format((i*100/N)),
        for row in range(rows):
            for col in range(cols):
                tmp_pixel = images.read(1)  # Just a single byte
                tmp_pixel = unpack('>B', tmp_pixel)[0]
                tempx[row][col] = (float(tmp_pixel) / 255)
        x[i] = tempx.flatten();
        tmp_label = labels.read(1)
        y[i] = unpack('>B', tmp_label)[0]

    header = {}
    header['featdim'] = rows*cols;
    header['input_shape'] = [rows,cols,1]

    return x,y,header

 
def saveData(name,x,y,header):
    
    filehandle = open(name,'ab');
    filehandle.write(json.dumps(header)+'\n')
    dt={'names': ['d','l'],'formats': [('>f2',header['featdim']),'>i2']}
    data = numpy.zeros(1,dtype= numpy.dtype(dt))
    
    for vector,label in zip(x,y):
        data['d']=vector; data['l']=label;
        data.tofile(filehandle); 
    
    filehandle.flush();
    filehandle.close();

if __name__ == '__main__':
    print("Get testset")
    (x,y,h)=get_labeled_data('t10k-images-idx3-ubyte.gz',
                               't10k-labels-idx1-ubyte.gz')
    print("Got %i testing datasets." % len(x))
    saveData('test.dat',x,y,h);

    print("Get trainingset")
    (x,y,h)=get_labeled_data('train-images-idx3-ubyte.gz',
                                'train-labels-idx1-ubyte.gz')
    print("Got %i training datasets." % len(x))
    seed=9090;
    numpy.random.seed(seed)
    numpy.random.shuffle(x) 
    numpy.random.seed(seed)
    numpy.random.shuffle(y)

    N = len(x)
    xtrain = x[:int(N*0.75)]
    ytrain = y[:int(N*0.75)]
    xval = x[int(N*0.75)+1:]
    yval = y[int(N*0.75)+1:]

    saveData('train.dat',xtrain,ytrain,h);
    saveData('val.dat',xval,yval,h);

