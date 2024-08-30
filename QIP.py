import numpy as np
from qiskit import quantum_info
from itertools import combinations, chain
import matplotlib.pyplot as plt

# call this map to go from character '0' or '1' to the vector representation
qubit_map = {'0':np.array([1,0]),'1':np.array([0,1])}    

# define the matrices for standard quantum gates here
x = np.array([[0.,1.],[1.,0.]])
y = np.array([[0.,-1.j],[1.j,0.]])
z = np.array([[1.,0.],[0.,-1.]])
h = 1/2**.5*np.array([[1.,1.],[1.,-1.]])
s = np.array([[1.,0.],[0.,1.j]])


### these are my auxiliary helper functions  

def all_bits_on(bitstring,controls,control_bitstring=''):
    """ checks if the given controls of a bitstring are the given control bitstring

    Args:
        bitstring (str): bitstring  
        controls (list): indices of controls
        control_bitstring (str, optional): The control values desired. Defaults to '111...'.

    Returns:
        boolean: true if the control bits together comprise the desired control bitstring
    """
    if len(controls) == 0: return True
    if control_bitstring=='': control_bitstring='1'*len(controls)
    for bit in range(len(controls)): 
        if bitstring[controls[bit]] != control_bitstring[bit]:
            return False
    # only get here if none of the control bits were 'off'
    return True

def projection_for_bitstring(bitstring):
    """returns the matrix representation of the projection operator on a given bitstring

    Args:
        bitstring (str): bitstring to construct the projection operator for

    Returns:
        numpy.ndarray: matrix form of projection operator in the computational basis
    """
    product = np.array(1.)
    for bit in bitstring:
        product=np.kron(product,np.outer(qubit_map[bit],qubit_map[bit]))
    return product

# Note to Jason - I actually did need this function for some reason. bin() doesnt add any leading zeros which are necessary
def int_to_bitstring(i,length):
    """converts a given integer to binary

    Args:
        i (int): integer to convert
        length (int): number of bits desired for bitstring

    Returns:
        str: binary representation of the integer
    """
    bitstring = bin(i)[2:]
    bitlength = len(bitstring)
    if bitlength < length:
        for j in range(length-bitlength):
            bitstring = '0'+bitstring
    return bitstring

def bitstrings_to_vector(bitstrings):
    """takes a dictionary of bitstrings and returns the vector representation

    Args:
        bitstrings (dict): a dict(str -> float) representing bitstrings and their quantum amplitudes describing the statevector. handles a single bitstring

    Returns:
        numpy.ndarray: vector representation of the statevector in the computational basis
    """
    # integer representation of the bitstring is just the index in the statevector
    if type(bitstrings) == dict:
        num_bits = len(list(bitstrings)[0])
        vec = np.zeros(2**num_bits)
        for bitstring in bitstrings:
            this_vec = np.array(1.)
            for bit in bitstring:
                this_vec = np.kron(this_vec,qubit_map[bit])
            vec = vec + bitstrings[bitstring]*this_vec
    # single bitstring - must be a computational basis state with 1 as the entry in the statevector at the index of the bitstring and zeros elsewhere
    else:
        num_bits = len(bitstrings)
        vec = np.zeros(2**num_bits)
        vec[int(bitstrings, 2)] = 1.
        # the rest is me doing it the expensive way with the kron product
        # vec = np.array(1.)
        # for bit in bitstrings:
        #     vec = np.kron(vec,qubit_map[bit])
    return vec
        
def bitstrings_dict_from_vector(vec):
    """ converts a statevector to dictionary form with bitstring keys and amplitude values

    Args:
        vec (numpy.ndarray): statevector in the computational basis

    Returns:
        dict: dict(str -> float) representing bitstrings and their quantum amplitudes describing the statevector. handles a single bitstring
    """
    num_bits = int(np.log2(len(vec)))
    state = dict()
    for i in range(len(vec)):
        if vec[i]!=0.:
            state[int_to_bitstring(i,num_bits)] = vec[i]
    return state

def bitstring_to_dict(bitstring):
    """_summary_

    Args:
        bitstring (_type_): _description_

    Returns:
        _type_: _description_
    """
    bitstrings = dict()
    num_bits = len(bitstring)
    for i in range(2**num_bits):
        bitstrings[int_to_bitstring(i,num_bits)] = 0
    bitstrings[bitstring] = 1.
    return bitstrings


def state_as_string(bitstrings):
    """ converts a bistrings dict to a string for display purposes

    Args:
        bitstrings (dict or numpy.ndarray): a dict(str -> float) representing bitstrings and their quantum amplitudes describing the statevector OR a statevector in the computational basis . handles a single bitstring

    Returns:
        str: the mathematical representation of the state in the computational basis. a|00> + b|01> + ...
    """
    final_bitstring=""

    if type(bitstrings) != dict:
        bitstrings = bitstrings_dict_from_vector(bitstrings)
    
    for bitstring in bitstrings:
        coeff=np.round(bitstrings[bitstring],8)
        match coeff:
            case 1.: final_bitstring = bitstring
            # global phase? maybe we will care...
            case -1.: final_bitstring = "-"+bitstring
            case 1.j: final_bitstring = "i"+bitstring
            case -1.j: final_bitstring ="-i"+bitstring
            case 0.: pass
            # superpositional states will follow this case
            case _: final_bitstring = final_bitstring + str(np.round(coeff,3))+"|"+bitstring+ "> + "
        
    if final_bitstring[-3:] ==" + ": return final_bitstring[:-3]
    else: return final_bitstring
    

def swap_bitstring(bitstring,swap_from,swap_to):
    """ Takes a bitstring and swaps the given bits

    Args:
        bitstring (str): bitstring to swap on
        swap_from (list): indices to swap from
        swap_to (int): indices to swap to. it should not matter if you switch swap_from and swap_to.

    Returns:
        str: bitstring after making the swap
    """
    num_swaps = len(swap_from)
    for i in range(num_swaps):
        temp = bitstring[swap_from[i]]
        # strings are unmutable so have to concatenate everything before and after the swapped strings
        bitstring = bitstring[:swap_from[i]] + bitstring[swap_to[i]] +bitstring[1+swap_from[i]:]
        bitstring = bitstring[:swap_to[i]] + temp +bitstring[1+swap_to[i]:]
    return bitstring

def arbitrary_swap(circuit_size,swap_from,swap_to):
    """generates the swap matrix of swapping a list of qubit (indices) with another

    Args:
        circuit_size (int): number of qubits in the circuit (need this to generate the full matrix)
        swap_from (list): indices to swap from
        swap_to (list): indices to swap to. it should not matter if you switch swap_from and swap_to.

    Returns:
        numpy.ndarray: full matrix of dimension: 2^circuit_size x 2^circuit_size
    """
    num_swaps = len(swap_from)
    if len(swap_to) != num_swaps: return "error - num bits to swap must match"
    # this is the matrix we are building row by row. We just make sure that it takes computational basis states to the correct result states
    lgm = np.zeros((2**circuit_size,2**circuit_size))
    for bitint in range(2**circuit_size):
        transformed_bitstring = swap_bitstring(int_to_bitstring(bitint,circuit_size),swap_from,swap_to)
        row = np.array(1.)
        for bit in transformed_bitstring:
            row = np.kron(row,qubit_map[bit])
        lgm[bitint]=row
    return lgm

def arbitrary_U(U,circuit_size,targets,controls=[],control_bitstring=''):
    """builds the unitary matrix applying on the full circuit given a unitary acting on a smaller number of qubits. handles controlled gates. must specify target bits and total circuit size 

    Args:
        U (numpy.ndarray): unitary acting on the subystem of qubits
        circuit_size (int): number of total qubits in the circuit including ones not acted upon
        targets (list): respective indices of qubits on which the unitary acts upon
        controls (list, optional): list of control qubit indices. Defaults to [].
        control_bitstring (str, optional): bitstring to control on. Defaults to '111...'.

    Returns:
        numpy.ndarray: matrix representation of the operator on the full circuit in the computational basis
    """
    
    num_sub_qubits = len(targets)
            
    # if a control bitstring is given, it is assumed to be controlled on the first n qubits where n is the number of bits in the given bitstring. 
    if control_bitstring !='' and len(controls) == 0: controls=list(range(len(control_bitstring)))
    
    # gate is not controlled
    if len(controls) == 0 and control_bitstring=='':
        if(int(np.log2(np.shape(U)[0])) != num_sub_qubits): return "error matrix size does not match number of bits"
        # U_total = U x I x I x... still out of the proper order
        total_U = U
        for i in range(circuit_size-num_sub_qubits):
            total_U = np.kron(total_U,np.eye(2))
        # build the swap gate for swapping the target qubits with the first n qubits 
        swap = arbitrary_swap(circuit_size,targets,list(range(num_sub_qubits)))
        # U_final = swap * U * swap
        # reorders the operator to be in the proper order
        return swap@total_U@swap
    # gate is controlled
    else:
        # if no specific bitstring is given, assume all 1's
        if control_bitstring == '': control_bitstring = '1'*len(controls)
        if(int(np.log2(np.shape(U)[0])) != num_sub_qubits): return "error matrix size does not match number of target bits"
        final_U = np.zeros((2**circuit_size,2**circuit_size))
        # We build the controlled operator up from what we want it to do for each basis state
        for bitint in range(2**circuit_size):
            bitstring=int_to_bitstring(bitint,circuit_size)
            #compute the bitstring after swapping the the target qubits with the first n qubits
            swapped_bitstring = swap_bitstring(int_to_bitstring(bitint,circuit_size),targets,list(range(len(targets))))
            # if all control bits are their respective control values add a term to the full operator
            if all_bits_on(bitstring,controls,control_bitstring):
                final_U = final_U + np.kron(U,projection_for_bitstring(swapped_bitstring[len(targets):]))
            else:
                final_U = final_U + np.kron(np.eye(2**len(targets)),projection_for_bitstring(swapped_bitstring[len(targets):]))
        swap = arbitrary_swap(circuit_size,targets,list(range(len(targets))))
        final_U = swap@final_U@swap
        normalization = np.linalg.norm(final_U, axis=-1)[:, np.newaxis]
        return final_U/normalization


def checkOperator(U,form='bitstring'):
    """prints the result states after a given operator acts on the computational basis states

    Args:
        U (numpy.ndarray): matrix to check (in the computational basis)
    """
    num_qubits=int(np.log2(np.shape(U)[0]))
    for i in range(2**num_qubits):
        bitstring=int_to_bitstring(i,num_qubits)
        transformed_bitstring = state_as_string(U@bitstrings_to_vector(bitstring))
        if form == 'bitstring':
            print(bitstring + " -> " + transformed_bitstring)
        else:
            print(str(int(bitstring,2)) + " -> " + str(int(transformed_bitstring,2)))
    
def shannon_ent(probs):
    """returns the shannon entropy of a given probability vector

    Args:
        probs (list or np.ndarray): 1-D array of floats describing probabilities

    Returns:
        float: the shannon entropy of the probability vector
    """
    sum = 0
    for p_i in probs:
        if p_i > 1e-10:
            sum = sum + p_i*np.log(p_i)
    return -1*sum

def partial_trace(p,trace_out):
    """finds the reduced density matrix after tracing out the given indices

    Args:
        p (numpy.ndarray (2-d)): full density matrix
        trace_out (list): indices to trace out
    """
    d_b = len(trace_out)
    d_a = int(np.log2(np.shape(p)[0])) - d_b
    sum = 0
    for j in range(2**d_b):
        vec = np.zeros(2**d_b)
        vec[j]=1.
        #Necessary because 1-d numpy vectors are equivalent to their transpose, i.e. .conj() will not turn a 1-d row vector into a column vector w/o shape
        vec.shape=(2**d_b,1)
        #for this matrix multiplication to be valid we need the LHS to be the identity x a column vector, and RHS to be the identity x a row vector
        sum = sum + np.kron(np.eye(2**d_a),vec.conj().T)@p@np.kron(np.eye(2**d_a),vec)
    return sum

def bipartitioning(qubs):
    """
    Generate a list of all bipartitions of a list

    Parameters:
    qubs (list): list to find all bipartitions of
    
    Returns: 
    bipartitions (list): list of all bipartitions where each element has 2 lists, the list of elements in the first piece, and the remining list of its complement
    """

    #Number of elements in the overall system
    N=len(qubs)
    bipartitions = list()

    #Loop over number of elements in the first subsystem from 0 to N/2
    for size_A in range(int(np.ceil(N/2))+1):
        #Generates a tuple of all combinations that can be made of size_A qubits out of the system
        combs = combinations(qubs,size_A)
        #Loop over every possible combination
        for c in combs:
            combination = list(c)
            combination.sort()
            #Determine the complement of the combination that completes the system
            complement = qubs[:]
            for bit in combination:
                complement.remove(bit)
            complement.sort()
            #Each bipartition has 2 subsystems, the list of pieces in the first subsystem, and its complement
            new_bipartition = [combination,complement]
            new_bipartition.sort()
            #Do not store duplicate bipartitions
            if new_bipartition not in bipartitions:
                bipartitions.append(new_bipartition)
    bipartitions.sort()
    return bipartitions

def tripartitioning(qubs):
    """
    Generate a list of all tripartitions of a list

    Parameters:
    qubs (list): list to find all tripartitions of
    
    Returns: 
    tripartitions (list): list of all bipartitions where each element has 3 lists, the list of elements in the first piece, and the remining list of its complements
    """
    tripartitions = list()
    #Bipartition a single piece of each bipartition to form a tripartition
    for bipartition in bipartitioning(qubs):
        for piece in range(2):
            for subpartition in bipartitioning(bipartition[piece]):
                new_tripartition = [subpartition[0],subpartition[1],bipartition[1-piece]]
                new_tripartition.sort()
                if new_tripartition not in tripartitions:
                    tripartitions.append(new_tripartition)
    tripartitions.sort()
    return tripartitions


# This is the single object we will use to simulate the quantum computer and track the entropic quantities we are interested in
class QCircuit:


    def __init__(self,circuit_size,initial_state=None,initial_circuit=None):
        """constructor for QCircuit class 

        Args:
            circuit_size (int): number of qubits in the circuit
            initial_state (np.ndarray, optional): the initial state of the qubits in the circuit. Defaults to '000...'.
            initial_circuit(QCircuit, optional): allows this circuit to be a continuation of another. 
        """
        if initial_circuit == None:
            self.circuit_size = circuit_size
            if initial_state is None: 
                self.statevector = bitstrings_to_vector('0'*circuit_size)
            else:self.statevector=initial_state/np.linalg.norm(initial_state)
            self.density_matrix = np.outer(self.statevector,self.statevector)
            self.unitary = np.eye(2**circuit_size)
            
            # a list of all qubits in the circuit
            self.all_qubits = list(range(circuit_size))
            # a list of all possible subsystems to be analyzed in the circuit
            subsystems = list(chain.from_iterable(bipartitioning(list(range(self.circuit_size)))))
            subsystems.sort(key=len)
            subsystems.pop(0)
            subsystems.sort()
            # These are all the quantities of interest we will be looking at at specific checkpoints in the. They will all be populated with each call of subsystem_analysis()
            self.avg_sas = []
            self.avg_ssas = []
            self.avg_mmis = []
            self.norms = []
            self.entropies = []
            self.single_qubit_entropy_series =[]
            for i in range(circuit_size):
                self.single_qubit_entropy_series.append([])
        else: self.__dict__.update(initial_circuit.__dict__)
        
        
    def subsysem_analysis(self):
        """calculates subadditivity, strong subadditivity, and monogamy of mutual information inequality saturations as well as the complete entropy vector and the entropies for individual qubits and populates the associated lists  
        """
        norm = 0
        entropy=[]
        sum_sa = 0
        num_sa_checks = 0
        sum_ssa = 0
        num_ssa_checks = 0
        sum_mmi = 0
        
        for subsystem in self.subsystems:
            p_12 = partial_trace(self.density_matrix,list(np.delete(self.all_qubits)))
            #to look at ancilla in grovers
            #if len(subsystem) == N-1:
            #    if 5 not in subsystem:
            #        print(p_12)
            
            probs = np.linalg.eigvals(np.array(p_12.data)).tolist()
            e = shannon_ent([p_i.real for p_i in probs])
            entropy.append(e)

            if len(subsystem) == 1:
                self.single_qubit_entropy_series
            norm = norm + e**2

            for subsystem_bipartition in bipartitioning(subsystem):
                p_1 = quantum_info.partial_trace(self.density_matrix,subsystem_bipartition[0])
                p_2 = quantum_info.partial_trace(self.density_matrix,subsystem_bipartition[1])
                probs_1 = np.linalg.eigvals(np.array(p_1.data)).tolist()
                probs_2 = np.linalg.eigvals(np.array(p_2.data)).tolist()
                e_1 = shannon_ent([p_i.real for p_i in probs_1])
                e_2 = shannon_ent([p_i.real for p_i in probs_2])
                sa_saturation = e_1+e_2-e
                num_sa_checks = num_sa_checks + 1
                sum_sa = sum_sa + sa_saturation
                if sa_saturation < -1e-10:
                    print("subadditivity failed for ",end="")
                    print(subsystem_bipartition)
                    print("Saturation: "+str(sa_saturation))
                
                # if len(subsystem_bipartition[0]) == 1 or len(subsystem_bipartition[1]) == 1:
                #     print("Bipartition: ",end="")
                #     print(subsystem_bipartition)
                #     print("SA Saturation: "+str(sa_saturation))
            for subsystem_tripartition in tripartitioning(subsystem):
                p_1 = quantum_info.partial_trace(self.density_matrix,subsystem_tripartition[0])
                probs_1 = np.linalg.eigvals(np.array(p_1.data)).tolist()
                e_1 = shannon_ent([p_i.real for p_i in probs_1])
                p_2 = quantum_info.partial_trace(self.density_matrix,subsystem_tripartition[1])
                probs_2 = np.linalg.eigvals(np.array(p_2.data)).tolist()
                e_2 = shannon_ent([p_i.real for p_i in probs_2])
                p_3 = quantum_info.partial_trace(self.density_matrix,subsystem_tripartition[2])
                probs_3 = np.linalg.eigvals(np.array(p_3.data)).tolist()
                e_3 = shannon_ent([p_i.real for p_i in probs_3])
                p_12 = quantum_info.partial_trace(self.density_matrix,subsystem_tripartition[0]+subsystem_tripartition[1])
                probs_12 = np.linalg.eigvals(np.array(p_12.data)).tolist()
                e_12 = shannon_ent([p_i.real for p_i in probs_12])
                p_13 = quantum_info.partial_trace(self.density_matrix,subsystem_tripartition[0]+subsystem_tripartition[2])
                probs_13 = np.linalg.eigvals(np.array(p_13.data)).tolist()
                e_13 = shannon_ent([p_i.real for p_i in probs_13])
                p_23  = quantum_info.partial_trace(self.density_matrix,subsystem_tripartition[1]+subsystem_tripartition[2])
                probs_23 = np.linalg.eigvals(np.array(p_23.data)).tolist()
                e_23 = shannon_ent([p_i.real for p_i in probs_23])
                ssa_saturation = e_12+e_23-e_2-e
                sum_ssa = sum_ssa + ssa_saturation
                num_ssa_checks = num_ssa_checks + 1
                if ssa_saturation < -1e-10:
                    print("strong subadditivity failed for ",end="")
                    print(subsystem_tripartition)
                    print("Saturation: "+str(ssa_saturation))
                mmi_saturation = e_12+e_13+e_23-e_1-e_2-e_3-e
                sum_mmi = sum_mmi + mmi_saturation
        self.entropies.append(entropy)
        self.norms.append(np.sqrt(norm)/2**.5)
        self.avg_sas.append(sum_sa/num_sa_checks)
        self.avg_ssas.append(sum_ssa/num_ssa_checks)
        self.avg_mmis.append(sum_mmi/num_ssa_checks)
        self.avg_ings.append(sum_ing/num_ing_checks)
        for i in range(len(self.subsystems)):
            if len(self.subsystems[i]) == 1:
                self.single_qubit_entropy_series[self.subsystems[i][0]].append(entropy[i])
    
    
    def apply_to_circuit(self,U,track_entropies=True):
        """applies the given unitary to the qubits. unitary must be the correct size for the number of qubits in the circuit. 

        Args:
            U (np.ndarray): _description_
            track_entropies (bool, optional): _description_. Defaults to True.
        """
        self.statevector = U@self.statevector
        self.density_matrix = U@self.density_matrix@U.conj().T
        self.unitary = self.unitary@U
        if track_entropies == True: self.subsysem_analysis()
        
    def plot_saturations(self,sa=True,ssa=True,mmi=True,ing=True,norms=True):
        if sa: plt.plot(self.avg_sas, label="Average SA Saturation")
        if ssa: plt.plot(self.avg_ssas, label="Average SSA Saturation")
        if mmi: plt.plot(self.avg_mmis, label="Average MMI Saturation")
        if ing: plt.plot(self.avg_ings, label="Average ingleton Saturation")
        plt.plot(self.norms, label="Entropy Norm")
        plt.legend()
        plt.show()
    
    def plot_single_qubit_entropy(self):
        for i in range(self.N):
            plt.plot(self.single_qubit_entropy_series[i], label="q"+str(i))
        plt.legend()
        plt.show()
        
    def measure(self, bits):
        """ determines probabilities for each outcome of measuring given qubits. does not actually change the system

        Args:
            bits (list): indices of qubits in the circuit to measure
        """
        state_dict = bitstrings_dict_from_vector(self.statevector)
        measured_dict = {}
        for bitstring in state_dict:
            new_bitstring = ""
            for bit in range(self.circuit_size):
                if bit in bits:
                    new_bitstring = new_bitstring + bitstring[bit]
            if new_bitstring in measured_dict:
                measured_dict[new_bitstring] = measured_dict[new_bitstring] + np.abs(state_dict[bitstring])**2
            else: measured_dict[new_bitstring] = np.abs(state_dict[bitstring])**2
        print(measured_dict)
    
    # The rest are just standard gates to make it easier when using the QCiruit class to run an algorithm
    def x(self,bit):
        self.apply_to_circuit(arbitrary_U(x,self.circuit_size,[bit]))
    
    def y(self,bit):
        self.apply_to_circuit(arbitrary_U(y,self.circuit_size,[bit]))
        
    def z(self,bit):
        self.apply_to_circuit(arbitrary_U(z,self.circuit_size,[bit]))
    
    def h(self,bit):
        self.apply_to_circuit(arbitrary_U(h,self.circuit_size,[bit]))
        
    def s(self,bit):
        self.apply_to_circuit(arbitrary_U(s,self.circuit_size,[bit]))
        
    def p(self,bit,phi):
        self.apply_to_circuit(arbitrary_U(np.array([[1.,0.],[0.,np.exp(phi*1.j)]]),self.circuit_size,[bit]))    
    
    def reverse_bits(self):
        # two qubits - just swap them
        if self.circuit_size == 2:
            self.apply_to_circuit(arbitrary_swap(self.circuit_size,[0],[1]))
        # three qubits - just swap the first and third
        if self.circuit_size == 3:
            self.apply_to_circuit(arbitrary_swap(self.circuit_size,[0],[2]))
        # >three - swap pairwise
        if self.circuit_size > 3:
            self.apply_to_circuit(arbitrary_swap(self.circuit_size,list(range(self.circuit_size))[:int(self.circuit_size/2)],list(range(self.circuit_size))[int(np.ceil(self.circuit_size/2)):]))