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

print(arr)
print(ker)
fftker = fft.fft(ker)
out1 = fftconv1d(arr,fftker,1)
out2 = fftconv1d(arr,fftker,0)
print(out1)
print(out2)

print(np.convolve(arr,ker, mode='valid'))