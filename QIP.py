import numpy as np
from itertools import combinations, chain, permutations
import matplotlib.pyplot as plt
from scipy.linalg import expm
import pickle

# call this map to go from character '0' or '1' to the vector representation
qubit_map = {'0':np.array([1,0]),'1':np.array([0,1])}    

# define the matrices for standard quantum gates here
x = np.array([[0.,1.],[1.,0.]])
y = np.array([[0.,-1.j],[1.j,0.]])
z = np.array([[1.,0.],[0.,-1.]])
h = 1/2**.5*np.array([[1.,1.],[1.,-1.]])
s = np.array([[1.,0.],[0.,1.j]])
t = np.array([[1., 0.],[0., np.exp(np.pi*1.j/4)]])


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
    return lgm.T

def reorder_U(circuit_size,bits):
    """_summary_

    Args:
        circuit_size (int): number of bits in circuit
        bits (list): bits to move to the end

    Returns:
        ndarray: unitary permutation matrix which reorders the given bits
    """
    lgm = np.zeros((2**circuit_size,2**circuit_size))
    for bitint in range(2**circuit_size):
        bitstring = int_to_bitstring(bitint,circuit_size)
        transformed_bitstring = [bitstring[i] for i in bits] + [bitstring[i] for i in range(len(bitstring)) if i not in bits]
        row = np.array(1.)
        for bit in transformed_bitstring:
            row = np.kron(row,qubit_map[bit])
        lgm[bitint]=row
    return lgm.T

def arbitrary_U(U,circuit_size,targets,controls=[],control_bitstring=''):
    """builds the unitary matrix applying on the full circuit given a un itary acting on a smaller number of qubits. handles controlled gates. must specify target bits and total circuit size 

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
        swap = reorder_U(circuit_size,targets)
        # U_final = swap * U * swap*
        # reorders the operator to be in the proper order
        final_U = swap.T@total_U@swap
        normalization = np.linalg.norm(final_U, axis=-1)[:, np.newaxis]
        return final_U/normalization
    # gate is controlled
    else:
        # if no specific bitstring is given, assume all 1's
        if control_bitstring == '': control_bitstring = '1'*len(controls)
        if(int(np.log2(np.shape(U)[0])) != num_sub_qubits): return "error matrix size does not match number of target bits"
        final_U = np.zeros((2**circuit_size,2**circuit_size))
        # We build the controlled operator up from what we want it to do for each basis state
        for bitint in range(2**circuit_size):
            bitstring=int_to_bitstring(bitint,circuit_size)
            #compute the bitstring after moving the target qubits to the first n qubits
            swapped_bitstring = [bitstring[i] for i in targets] + [bitstring[i] for i in range(len(bitstring)) if i not in targets]
            # if all control bits are their respective control values add a term to the full operator
            if all_bits_on(bitstring,controls,control_bitstring):
                final_U = final_U + np.kron(U,projection_for_bitstring(swapped_bitstring[len(targets):]))
            else:
                final_U = final_U + np.kron(np.eye(2**len(targets)),projection_for_bitstring(swapped_bitstring[len(targets):]))
        swap = reorder_U(circuit_size,targets)
        final_U = swap.T@final_U@swap
        normalization = np.linalg.norm(final_U, axis=-1)[:, np.newaxis]
        return final_U/normalization

def generate_random_unitary_matrix(n):
    """
    Generate a 2^n x 2^n random unitary matrix.
    """
    size = 2**n
    # Generate a random complex matrix
    random_matrix = np.random.randn(size, size) + 1j * np.random.randn(size, size)
    # Perform QR decomposition to obtain a unitary matrix
    q, _ = np.linalg.qr(random_matrix)
    return q


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
            sum = sum + p_i*np.log2(p_i)
    return -1*sum

def partial_trace(p,trace_out):
    """ finds the reduced density matrix given indices of bits to trace over

    Args:
        p (ndarray): density matrix to reduce
        trace_out (list): indices of bits to trace over

    Returns:
        ndarray: reduced density matrix after tracing out the given bits
    """
    d_b = len(trace_out)
    d_a = int(np.log2(np.shape(p)[0])) - d_b
    p_A = np.zeros((2**d_a,2**d_a),dtype=complex)
    for i in range(2**d_a):
        for k in range(2**d_a):
            for j in range(2**d_b):
                # build the bitstring of the indices, summing over only j
                bra_bitstring = ""
                ket_bitstring = ""
                ik_c = 0 # keep track of which binary digit of i,k 
                j_c = 0 # keep track of which binary digit of j, summed over
                for l in range(d_b+d_a):
                    if l in trace_out:
                        bra_bitstring = bra_bitstring + int_to_bitstring(j,d_b)[j_c]
                        ket_bitstring = ket_bitstring + int_to_bitstring(j,d_b)[j_c]
                        j_c = j_c + 1
                    else:
                        bra_bitstring = bra_bitstring + int_to_bitstring(i,d_a)[ik_c]
                        ket_bitstring = ket_bitstring + int_to_bitstring(k,d_a)[ik_c]
                        ik_c = ik_c + 1
                p_A[i][k] = p_A[i][k]+ p[int(bra_bitstring,2)][int(ket_bitstring,2)]
    return p_A

def n_partitioning(qubs, n, allow_empty_systems=True):
    """
    Generate a list of all n-partitions of a list.

    Parameters:
    lst (list): List to find all n-partitions of
    n (int): Number of subsets in each partition

    Returns:
    partitions (list): List of all partitions where each partition is a list of n subsets
    """
    N=len(qubs)
    partitions = []
    if n==1: return list(range(N))
    
    if n==2: 
        if allow_empty_systems: rng = range(int(np.floor(N/2))+1)
        else: rng = range(1, int(np.floor(N/2))+1)
        #Loop over number of elements in the first subsystem from 0 to N/2
        for size_A in rng:
            #Generates a tuple of all combinations that can be made of size_A qubits out of the system
            combs = combinations(qubs,size_A)
            #Loop over every possible combination
            for c in combs:
                combination = list(c)
                #Determine the complement of the combination that completes the system
                complement = qubs[:]
                for bit in combination:
                    complement.remove(bit)
                if size_A == int(N/2):
                    if [complement, combination] in partitions: break
                #Each bipartition has 2 subsystems, the list of pieces in the first subsystem, and its complement
                partitions.append([combination,complement])
        return partitions
    
    else:
        for partition in n_partitioning(qubs,n-1,allow_empty_systems):
            for piece in range(n-1):
                for subpartition in n_partitioning(partition[piece],2,allow_empty_systems):
                    new_partition=partition[:piece]+[sub for sub in subpartition] + partition[piece+1:]
                    new_partition.sort()
                    if new_partition not in partitions:
                        partitions.append(new_partition)
    return partitions

def mutual(x,y,entropies_map):
    return entropies_map[tuple(x)] + entropies_map[tuple(y)] - entropies_map[tuple(sorted(x+y))] 

def mutual_conditional(x,y,z,entropies_map):
    return entropies_map[tuple(sorted(x+z))] + entropies_map[tuple(sorted(y+z))] - entropies_map[tuple(sorted(x+y+z))] - entropies_map[tuple(z)]

def load_from_file(filename="analysis_results.pkl"):
    """ loads a saved object from file

    Args:
        filename (str, optional): pkl file which stores the object data. Defaults to "analysis_results.pkl".

    Returns:
        _type_: object with the saved data
    """
    with open(filename, "rb") as file:
        return pickle.load(file)

# This is the single object we will use to simulate the quantum computer and track the entropic quantities we are interested in
class QCircuit:


    def __init__(self,circuit_size,initial_state=None,initial_circuit=None,allow_empty_systems=True,pure_state=True):
        """constructor for QCircuit class 

        Args:
            circuit_size (int): number of qubits in the circuit
            initial_state (np.ndarray, optional): the initial state of the qubits in the circuit. Defaults to '000...'.
            initial_circuit(QCircuit, optional): allows this circuit to be a continuation of another. 
        """
        self.allow_empty_systems = allow_empty_systems
        self.pure_state = pure_state
        
        if initial_state is None: self.statevector = bitstrings_to_vector('0'*circuit_size)
        else: self.statevector = initial_state/np.linalg.norm(initial_state)
        self.density_matrix = np.outer(self.statevector,self.statevector.conj())
        
        if initial_circuit == None:
            self.circuit_size = circuit_size
            
            
            self.unitary = np.eye(2**circuit_size)
            
            # a list of all qubits in the circuit
            self.all_qubits = list(range(circuit_size))
            # a list of all possible subsystems to be analyzed in the circuit
            self.subsystems = list(chain.from_iterable(n_partitioning(list(range(self.circuit_size)),2,self.allow_empty_systems)))
            self.subsystems.sort(key=len)
            # These are all the quantities of interest we will be looking at at specific checkpoints in the. They will all be populated with each call of subsystem_analysis()
            self.sa = {'avg_saturation':[],'percent_fail':[],'avg_fail_saturation':[]}
            self.ssa = {'avg_saturation':[],'percent_fail':[],'avg_fail_saturation':[]}
            self.mmi = {'avg_saturation':[],'percent_fail':[],'avg_fail_saturation':[]}
            self.ing = {'avg_saturation':[],'percent_fail':[],'avg_fail_saturation':[]}
            self.norms = []
            self.entropies = []
            self.single_qubit_entropy_series =[]
            for i in range(circuit_size):
                self.single_qubit_entropy_series.append([])
        else: self.__dict__.update(initial_circuit.__dict__)
        
        
    def subsystem_analysis(self):
        
        entropies_map = dict()
        entropy_vector = []
        sa_i={'sat':0,'fail_sat':0,'num_checks':0,'num_failures':0}
        ssa_i={'sat':0,'fail_sat':0,'num_checks':0,'num_failures':0}
        mmi_i={'sat':0,'fail_sat':0,'num_checks':0,'num_failures':0}
        ing_i={'sat':0,'fail_sat':0,'num_checks':0,'num_failures':0}
        
        
        # calculate the entropy of every subsystem ahead of time
        for subsystem in n_partitioning(self.all_qubits,2,self.allow_empty_systems):
            reduced_dm = partial_trace(self.density_matrix,subsystem[1])
            probs = np.linalg.eigvals(np.array(reduced_dm.data)).tolist()
            entropy = shannon_ent([p_i.real for p_i in probs])
            entropies_map[tuple(subsystem[0])] = entropy
            entropy_vector.append(entropy)
            if self.pure_state:
                # S_A always = S_B if S_AB = 0 (S_AB is pure) so we dont waste time calculating it again
                entropies_map[tuple(subsystem[1])] = entropy
            else:
                reduced_dm = partial_trace(self.density_matrix,subsystem[0])
                probs = np.linalg.eigvals(np.array(reduced_dm.data)).tolist()
                entropy = shannon_ent([p_i.real for p_i in probs])
                entropies_map[tuple(subsystem[1])] = entropy
                entropy_vector.append(entropy)
        
        
        
        # now for all our entropy inequality checks (see https://arxiv.org/pdf/1505.07839)
        # we need to look at partitions of each subsystem
        for subsystem in self.subsystems:
            # remember the entropy of this subsystem
            this_subsystem_entropy = entropies_map[tuple(subsystem)]
            
            # for all biparitions check subadditivity
            for subsystem_bipartition in n_partitioning(subsystem,2,self.allow_empty_systems):
                sa_saturation = entropies_map[tuple(subsystem_bipartition[0])] + entropies_map[tuple(subsystem_bipartition[1])] - this_subsystem_entropy
                sa_i['sat'] = sa_i['sat'] + sa_saturation
                sa_i['num_checks'] = sa_i['num_checks'] + 1
                if sa_saturation < -1e-10:
                    sa_i['fail_sat'] = sa_i['fail_sat'] + sa_saturation
                    sa_i['num_failures'] = sa_i['num_failures'] + 1
                    
            # for all tripartitions check strong subadditivity and monogamy of mutual information    
            for subsystem_tripartition in n_partitioning(subsystem,3,self.allow_empty_systems):
                
                # for ssa, need to check all permutations
                for permutation in permutations(subsystem_tripartition):
                    s_ab = entropies_map[tuple(sorted(permutation[0]+permutation[1]))]
                    s_bc = entropies_map[tuple(sorted(permutation[1]+permutation[2]))]
                    s_b  = entropies_map[tuple(permutation[1])]
                    # eq. (2.1) in https://arxiv.org/pdf/1505.07839
                    ssa_saturation = s_ab+s_bc-s_b-this_subsystem_entropy
                    ssa_i['sat'] = ssa_i['sat'] + ssa_saturation
                    ssa_i['num_checks'] = ssa_i['num_checks'] + 1
                    

                    if ssa_saturation < -1e-10:
                        ssa_i['fail_sat'] = ssa_i['fail_sat'] + ssa_saturation
                        ssa_i['num_failures'] = ssa_i['num_failures'] + 1
                        
                        
                # for mmi no permutations are needed
                s_a = entropies_map[tuple(subsystem_tripartition[0])]
                s_b  = entropies_map[tuple(permutation[1])] 
                s_c = entropies_map[tuple(subsystem_tripartition[2])]        
                s_ab = entropies_map[tuple(sorted(permutation[0]+permutation[1]))]
                s_bc = entropies_map[tuple(sorted(permutation[1]+permutation[2]))]
                s_ac = entropies_map[tuple(sorted(subsystem_tripartition[0]+subsystem_tripartition[2]))]
                # eq. (2.2) in https://arxiv.org/pdf/1505.07839
                mmi_saturation = s_ab+s_ac+s_bc-s_a-s_b-s_c-this_subsystem_entropy
                mmi_i['sat'] = mmi_i['sat'] + mmi_saturation
                mmi_i['num_checks'] = mmi_i['num_checks'] + 1
                
                if mmi_saturation < -1e-10:
                        mmi_i['fail_sat'] = mmi_i['fail_sat'] + mmi_saturation
                        mmi_i['num_failures'] = mmi_i['num_failures'] + 1
                        
            # for all quadpartitions? check ingleton's inequality
            for subsystem_quadpartition in n_partitioning(subsystem,4,self.allow_empty_systems):
                
                # for ingletons, need to check all permutations
                for permutation in permutations(subsystem_quadpartition):
                    # e_1  = entropies_map[tuple(permutation[0])]
                    # e_2  = entropies_map[tuple(permutation[1])]
                    # e_123 = entropies_map[tuple(sorted(permutation[0]+permutation[1]+permutation[2]))]
                    # e_124 = entropies_map[tuple(sorted(permutation[0]+permutation[1]+permutation[3]))]
                    # e_34 = entropies_map[tuple(sorted(permutation[2]+permutation[3]))]
                    # e_12 = entropies_map[tuple(sorted(permutation[0]+permutation[1]))]
                    # e_13 = entropies_map[tuple(sorted(permutation[0]+permutation[2]))]
                    # e_14 = entropies_map[tuple(sorted(permutation[0]+permutation[3]))]
                    # e_23 = entropies_map[tuple(sorted(permutation[1]+permutation[2]))]
                    # e_24 = entropies_map[tuple(sorted(permutation[1]+permutation[3]))]
                    # ing_saturation = e_12+e_13+e_14+e_23+e_24-(e_1+e_2+e_123+e_124+e_34)
                    ing_saturation = mutual_conditional(subsystem_quadpartition[0],subsystem_quadpartition[1],subsystem_quadpartition[2],entropies_map)+mutual_conditional(subsystem_quadpartition[0],subsystem_quadpartition[1],subsystem_quadpartition[3],entropies_map)+mutual(subsystem_quadpartition[2],subsystem_quadpartition[3],entropies_map)-mutual(subsystem_quadpartition[0],subsystem_quadpartition[1],entropies_map)
                    ing_i['sat'] = ing_i['sat'] + ing_saturation
                    ing_i['num_checks'] = ing_i['num_checks'] + 1

                    if ing_saturation < -1e-10:
                        ing_i['fail_sat'] = ing_i['fail_sat'] + ing_saturation
                        ing_i['num_failures'] = ing_i['num_failures'] + 1
                            
        self.entropies.append(entropies_map)
        self.norms.append(np.linalg.norm(entropy_vector))
        ineq_time_series = [self.sa,self.ssa,self.mmi,self.ing]
        instant_ineq = [sa_i,ssa_i,mmi_i,ing_i]
        for inequality in range(4):
            if instant_ineq[inequality]['num_checks'] != 0:
                ineq_time_series[inequality]['avg_saturation'].append(instant_ineq[inequality]['sat']/instant_ineq[inequality]['num_checks'])
                ineq_time_series[inequality]['percent_fail'].append(instant_ineq[inequality]['num_failures']/instant_ineq[inequality]['num_checks'])
            # else: 
            #     ineq_time_series[inequality]['avg_saturation'].append(0.)
            #     ineq_time_series[inequality]['percent_fail'].append(0.)   
            if instant_ineq[inequality]['num_failures'] != 0:
                ineq_time_series[inequality]['avg_fail_saturation'].append(-1*instant_ineq[inequality]['fail_sat']/instant_ineq[inequality]['num_failures'])
            # else: 
            #     ineq_time_series[inequality]['avg_fail_saturation'].append(0.)
        for bit in range(self.circuit_size):
            self.single_qubit_entropy_series[bit].append(entropies_map[(bit,)])
      
            
    def apply_to_circuit(self,U,track_entropies=True):
        """applies the given unitary to the qubits. unitary must be the correct size for the number of qubits in the circuit. 

        Args:
            U (np.ndarray): _description_
            track_entropies (bool, optional): _description_. Defaults to True.
        """
        self.statevector = U@self.statevector
        self.density_matrix = U@self.density_matrix@U.conj().T
        self.unitary = U@self.unitary
        if track_entropies == True: self.subsystem_analysis()
        
    def plot_saturations(self,savefiles=False,folder=""):
        """plots all inequality check data which was saved in the sa, ssa, mmi, ing dictionaries

        Args:
            savefiles (bool, optional): save plots to file. Defaults to False.
            folder (str, optional): folder name to save the plots into. Defaults to "".
        """
        ineq_time_series = [self.sa,self.ssa,self.mmi,self.ing]
        inequality_names = ["Subadditivity","Strong Subadditivity","Monogamy of Mutual Information","Ingleton's Inequality"]
        for inequality in range(4):
            fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15,5))
            fig.suptitle(inequality_names[inequality])
            ax1.plot(ineq_time_series[inequality]["avg_saturation"])
            ax1.set_title("Average Saturation (bits)")
            ax1.set_xlabel("Number of gates")
            ax2.plot(ineq_time_series[inequality]["percent_fail"])
            ax2.set_title("Percent Failure")
            ax2.set_xlabel("Number of gates")
            ax3.plot(ineq_time_series[inequality]["avg_fail_saturation"])
            ax3.set_title("Average Fail Saturation (bits)")
            ax3.set_xlabel("Number of gates")
            
            if savefiles:
                ax1.figure.savefig(folder+f"{inequality_names[inequality].replace(' ', '_')}.jpg", dpi=300)

    def save_to_file(self, filename="analysis_results.pkl"):
        """saves an object to file

        Args:
            filename (str, optional): name of file to write data to. I think? this might need a .pkl extension. Defaults to "analysis_results.pkl".
        """
        with open(filename, "wb") as file:
            pickle.dump(self, file)
    
    
    def plot_single_qubit_entropy(self):
        """plots only the entropies of the single qubit systems"""
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
    
    def h_on_set(self,bits):
        op = 1.
        for q in bits:
            op = np.kron(op,h)
        self.apply_to_circuit(arbitrary_U(op,self.circuit_size,bits))
    
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