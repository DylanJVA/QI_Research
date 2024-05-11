import numpy as np

qubit_map = {'0':np.array([1,0]),'1':np.array([0,1])}    
x = np.array([[0.,1.],[1.,0.]])
y = np.array([[0.,-1.j],[1.j,0.]])
z = np.array([[1.,0.],[0.,-1.]])
h = 1/2**.5*np.array([[1.,1.],[1.,-1.]])
s = np.array([[1.,0.],[0.,1.j]])


def all_bits_on(bitstring,controls,control_bitstring=''):
    if len(controls) == 0: return True
    if control_bitstring=='': control_bitstring='1'*len(controls)
    for bit in range(len(controls)):
        if bitstring[controls[bit]] != control_bitstring[bit]:
            return False
    return True

def projection_on_the_rest(remaining_bitstring):
    product = np.array(1.)
    for bit in remaining_bitstring:
        product=np.kron(product,np.outer(qubit_map[bit],qubit_map[bit]))
    return product

def int_to_bitstring(i,length):
    bitstring = bin(i)[2:]
    bitlength = len(bitstring)
    if bitlength < length:
        for j in range(length-bitlength):
            bitstring = '0'+bitstring
    return bitstring

def bitstrings_to_vector(bitstrings):
    if type(bitstrings) == dict:
        num_bits = len(list(bitstrings)[0])
        vec = np.zeros(2**num_bits)
        for bitstring in bitstrings:
            this_vec = np.array(1.)
            for bit in bitstring:
                this_vec = np.kron(this_vec,qubit_map[bit])
            vec = vec + bitstrings[bitstring]*this_vec
        return vec
    else:
        num_bits = len(bitstrings)
        vec = np.array(1.)
        for bit in bitstrings:
            vec = np.kron(vec,qubit_map[bit])
        return vec
def bitstrings_dict_from_vector(vec):
    num_bits = int(np.log2(len(vec)))
    state = dict()
    for i in range(len(vec)):
        if vec[i]!=0.:
            state[int_to_bitstring(i,num_bits)] = vec[i]
    return state

def bitstring_to_dict(bitstring):
    bitstrings = dict()
    num_bits = len(bitstring)
    for i in range(2**num_bits):
        bitstrings[int_to_bitstring(i,num_bits)] = 0
        bitstrings[bitstring] = 1.
    return bitstrings

def state_as_string(bitstrings):
    final_bitstring=""
    for bitstring in bitstrings:
        coeff=np.round(bitstrings[bitstring],8)
        match coeff:
            case 1.: final_bitstring = bitstring
            case -1.: final_bitstring = "-"+bitstring
            case 1.j: final_bitstring = "i"+bitstring
            case -1.j: final_bitstring ="-i"+bitstring
            case 0.: pass
            case _: final_bitstring = final_bitstring + str(np.round(coeff,3))+"|"+bitstring+ "> + "
        
    if final_bitstring[-3:] ==" + ": return final_bitstring[:-3]
    else: return final_bitstring
    

def swap_bitstring(bitstring,swap_from,swap_to):
    num_swaps = len(swap_from)
    for i in range(num_swaps):
        temp = bitstring[swap_from[i]]
        bitstring = bitstring[:swap_from[i]] + bitstring[swap_to[i]] +bitstring[1+swap_from[i]:]
        bitstring = bitstring[:swap_to[i]] + temp +bitstring[1+swap_to[i]:]
    return bitstring

def arbitrary_swap(circuit_size,swap_from,swap_to):
    num_swaps = len(swap_from)
    if len(swap_to) != num_swaps: return "error - num bits to swap must match"
    lgm = np.zeros((2**circuit_size,2**circuit_size))
    for bitint in range(2**circuit_size):
        transformed_bitstring = swap_bitstring(int_to_bitstring(bitint,circuit_size),swap_from,swap_to)
        row = np.array(1.)
        for bit in transformed_bitstring:
            row = np.kron(row,qubit_map[bit])
        lgm[bitint]=row
    return lgm

def arbitrary_U(U,circuit_size,targets,controls=[],control_bitstring=''):
    num_sub_qubits = len(targets)
    if len(controls) == 0:
        if(int(np.log2(np.shape(U)[0])) != num_sub_qubits): return "error matrix size does not match number of bits"
        total_U = U
        for i in range(circuit_size-num_sub_qubits):
            total_U = np.kron(total_U,np.eye(2))
        swap = arbitrary_swap(circuit_size,targets,list(range(num_sub_qubits)))
        return swap@total_U@swap
    else:
        if control_bitstring == '': control_bitstring = '1'*len(controls)
        if(int(np.log2(np.shape(U)[0])) != num_sub_qubits): return "error matrix size does not match number of target bits"
        final_U = np.zeros((2**circuit_size,2**circuit_size))
        for bitint in range(2**circuit_size):
            bitstring=int_to_bitstring(bitint,circuit_size)
            swapped_bitstring = swap_bitstring(int_to_bitstring(bitint,circuit_size),targets,list(range(len(targets))))
            if all_bits_on(bitstring,controls,control_bitstring):
                final_U = final_U + np.kron(U,projection_on_the_rest(swapped_bitstring[len(targets):]))
            else:
                final_U = final_U + np.kron(np.eye(2**len(targets)),projection_on_the_rest(swapped_bitstring[len(targets):]))
        swap = arbitrary_swap(circuit_size,targets,list(range(len(targets))))
        final_U = swap@final_U@swap
        normalization = np.linalg.norm(final_U, axis=-1)[:, np.newaxis]
        return final_U/normalization

def checkOperator(U):
    num_qubits=int(np.log2(np.shape(U)[0]))
    for i in range(2**num_qubits):
        bitstring=int_to_bitstring(i,num_qubits)
        transformed_bitstring = bitstring_from_dict(bitstrings_dict_from_vector(U@bitstrings_to_vector(bitstring)))
        print(bitstring + " -> " + transformed_bitstring)
    

class QCircuit:

    def __init__(self,circuit_size,initial_state=None):
        self.circuit_size = circuit_size
        if initial_state is None: 
            self.statevector = bitstrings_to_vector('0'*circuit_size)
        else:self.statevector=initial_state/np.linalg.norm(initial_state)
        self.density_matrix = np.outer(self.statevector,self.statevector)
        self.unitary = np.eye(2**circuit_size)
    
    def apply_to_circuit(self,U):
        self.statevector = U@self.statevector
        self.density_matrix = U@self.density_matrix@U.conj().T
        self.unitary = self.unitary@U
    
    def arbitrary_U(self,U,targets,controls=[],control_bitstring=''):
        num_sub_qubits = len(targets)
        if control_bitstring !='' and len(controls) == 0: controls=list(range(len(control_bitstring)))
        if len(controls) == 0 and control_bitstring=='':
            if(int(np.log2(np.shape(U)[0])) != num_sub_qubits): return "error matrix size does not match number of bits"
            total_U = U
            for i in range(self.circuit_size-num_sub_qubits):
                total_U = np.kron(total_U,np.eye(2))
            swap = arbitrary_swap(self.circuit_size,targets,list(range(num_sub_qubits)))
            self.apply_to_circuit(swap@total_U@swap)
        else:
            if control_bitstring == '': control_bitstring = '1'*len(controls)
            if(int(np.log2(np.shape(U)[0])) != num_sub_qubits): return "error matrix size does not match number of target bits"
            final_U = np.zeros((2**self.circuit_size,2**self.circuit_size))
            for bitint in range(2**self.circuit_size):
                bitstring=int_to_bitstring(bitint,self.circuit_size)
                swapped_bitstring = swap_bitstring(int_to_bitstring(bitint,self.circuit_size),targets,list(range(len(targets))))
                if all_bits_on(bitstring,controls,control_bitstring):
                    final_U = final_U + np.kron(U,projection_on_the_rest(swapped_bitstring[len(targets):]))
                else:
                    final_U = final_U + np.kron(np.eye(2**len(targets)),projection_on_the_rest(swapped_bitstring[len(targets):]))
            swap = arbitrary_swap(self.circuit_size,targets,list(range(len(targets))))
            final_U = swap@final_U@swap
            normalization = np.linalg.norm(final_U, axis=-1)[:, np.newaxis]
            self.apply_to_circuit(final_U/normalization)
    
    def arbitrary_swap(self,swap_from,swap_to):
        num_swaps = len(swap_from)
        if len(swap_to) != num_swaps: return "error - num bits to swap must match"
        lgm = np.zeros((2**self.circuit_size,2**self.circuit_size))
        for bitint in range(2**self.circuit_size):
            transformed_bitstring = swap_bitstring(int_to_bitstring(bitint,self.circuit_size),swap_from,swap_to)
            row = np.array(1.)
            for bit in transformed_bitstring:
                row = np.kron(row,qubit_map[bit])
            lgm[bitint]=row
        self.apply_to_circuit(lgm)
    
    def x(self,bit):
        self.arbitrary_U(x,[bit])
    
    def y(self,bit):
        self.arbitrary_U(y,[bit])
        
    def z(self,bit):
        self.arbitrary_U(z,[bit])
    
    def h(self,bit):
        self.arbitrary_U(h,[bit])
        
    def s(self,bit):
        self.arbitrary_U(s,[bit])
        
    def p(self,bit,phi):
        self.arbitrary_U(np.array([[1.,0.],[0.,np.exp(phi*1.j)]]),[bit])    
    
    def reverse_bits(self):
        if self.circuit_size == 2:
            self.arbitrary_swap([0],[1])
        if self.circuit_size == 3:
            self.arbitrary_swap([0],[2])
        if self.circuit_size > 3:
            self.arbitrary_swap(list(range(self.circuit_size))[:int(self.circuit_size/2)],list(range(self.circuit_size))[int(np.ceil(self.circuit_size/2)):])