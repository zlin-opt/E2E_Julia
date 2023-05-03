import numpy as np
import numpy.fft as fft

def fftconv1d(arr,fftker,fwd=1):

    narr = arr.shape[0]
    nker = fftker.shape[0]
    nout = nker - narr

    fftarr = fft.fft( np.pad(arr,((nout//2,nout//2)),mode='constant') )
    if fwd==1:
        out = np.array( fft.ifftshift( fft.ifft( fftarr * fftker ))[narr//2:narr//2+nout], copy=True )
    else:
        out = np.array( fft.ifftshift( fft.ifft( fftarr * fftker ))[-1+narr//2:-1+narr//2+nout], copy=True )

    return out

arr = np.array( [1.,3.,4.,1.,6.,3.] )
ker = np.array( [2.,2.,4.,3.,1.,1.,3.,1.,2.,2.,4.,5.] )
out1 = fftconv1d(arr,fft.fft(ker),1)
print(out1)
print(np.convolve(arr,ker, mode='valid')[1:])

arr = np.random.uniform(low=-1.0,high=1.0,size=4) + 1j * np.random.uniform(low=-1.0,high=1.0,size=4)
ker = np.random.uniform(low=-1.0,high=1.0,size=10) + 1j * np.random.uniform(low=-1.0,high=1.0,size=10)
out1 = fftconv1d(arr,fft.fft(ker),1)
print(out1)
print(np.convolve(arr,ker, mode='valid')[1:])

#print(np.convolve(ker,arr, mode='valid'))
#out2 = fftconv1d(arr,fftker,0)
#print(out2)