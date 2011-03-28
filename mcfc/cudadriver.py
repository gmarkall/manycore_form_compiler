from driver import Driver
import cudaform
import cudaassembler

class CudaDriver(Driver):

    def __init__(self):
        self._formBackend = cudaform.CudaFormBackend()
        self._assemblerBackend = cudaassembler.CudaAssemblerBackend()
