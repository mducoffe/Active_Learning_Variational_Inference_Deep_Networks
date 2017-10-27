import theano
import theano.tensor as T
import numpy
from theano.tensor.nnet import conv
#import pylab
# source : https://github.com/jostosh/theano_utils/blob/master/lcn.py

class LecunLCN(object):
    def __init__(self, X, image_shape, threshold=1e-4, radius=9, use_divisor=True):
        """
        Allocate an LCN.

        :type X: theano.tensor.dtensor4
        :param X: symbolic image tensor, of shape image_shape

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)
        :type threshold: double
        :param threshold: the threshold will be used to avoid division by zeros

        :type radius: int
        :param radius: determines size of Gaussian filter patch (default 9x9)

        :type use_divisor: Boolean
        :param use_divisor: whether or not to apply divisive normalization
        """

        # Get Gaussian filter
        filter_shape = (1, image_shape[1], radius, radius)

        self.filters = theano.shared(self.gaussian_filter(filter_shape), borrow=True)

        # Compute the Guassian weighted average by means of convolution
        convout = conv.conv2d(
            input=X,
            filters=self.filters,
            image_shape=image_shape,
            filter_shape=filter_shape,
            border_mode='full'
        )

        # Subtractive step
        mid = int(numpy.floor(filter_shape[2] / 2.))

        # Make filter dimension broadcastable and subtract
        centered_X = X - T.addbroadcast(convout[:, :, mid:-mid, mid:-mid], 1)

        # Boolean marks whether or not to perform divisive step
        if use_divisor:
            # Note that the local variances can be computed by using the centered_X
            # tensor. If we convolve this with the mean filter, that should give us
            # the variance at each point. We simply take the square root to get our
            # denominator

            # Compute variances
            sum_sqr_XX = conv.conv2d(
                input=T.sqr(centered_X),
                filters=self.filters,
                image_shape=image_shape,
                filter_shape=filter_shape,
                border_mode='full'
            )


            # Take square root to get local standard deviation
            denom = T.sqrt(sum_sqr_XX[:,:,mid:-mid,mid:-mid])

            per_img_mean = denom.mean(axis=[2,3])
            divisor = T.largest(per_img_mean.dimshuffle(0, 1, 'x', 'x'), denom)
            # Divisise step
            new_X = centered_X / T.maximum(T.addbroadcast(divisor, 1), threshold)
        else:
            new_X = centered_X

        self.output = new_X


    def gaussian_filter(self, kernel_shape):
        x = numpy.zeros(kernel_shape, dtype=theano.config.floatX)

        def gauss(x, y, sigma=2.0):
            Z = 2 * numpy.pi * sigma ** 2
            return  1. / Z * numpy.exp(-(x ** 2 + y ** 2) / (2. * sigma ** 2))

        mid = numpy.floor(kernel_shape[-1] / 2.)
        for kernel_idx in xrange(0, kernel_shape[1]):
            for i in xrange(0, kernel_shape[2]):
                for j in xrange(0, kernel_shape[3]):
                    x[0, kernel_idx, i, j] = gauss(i - mid, j - mid)

        return x / numpy.sum(x)

def lcn_function(radius=13):
	X = T.tensor4('x')
	op = LecunLCN(X=X, image_shape=(1,1,32,32), radius=radius)
	f = theano.function([X], op.output, allow_input_downcast=True)
	return f

if __name__=="__main__":
	f = lcn() # TO BE TESTED
	x_value = numpy.random.ranf((1, 1, 32, 32))
	y_value = f(x_value)
	import pdb
	pdb.set_trace()
