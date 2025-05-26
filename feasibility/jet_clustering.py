import fastjet as fj
import awkward as ak
import numpy as np
import vector

vector.register_awkward()


def get_delta_R(left_four_vector, right_four_vector):
    '''
    Calculates difference in angular radius. 
    Note: both input arrays should be the same shape. 
    For calculating quantities like lesub, the jet array must be reshaped to have the same shape as the constituents. 
    
    Args:
        left_four_vector (awkward array): awkward array containing events consisting of four-vectors (ie. events of particles or jets).
        right_four_vector (awkward array): awkward array containing events consisting of four-vectors (ie. events of particles or jets).
    
    Returns:
        awkward array of delta R values.
    '''
    
    delta_eta = left_four_vector.eta - right_four_vector.eta
    delta_phi = left_four_vector.phi - right_four_vector.phi
    
    return np.sqrt((delta_phi)**2 + (delta_eta)**2)


def get_girth(jets, constituents):
    '''
    Calculate jet girth. 
    
    Args:
        jets (akward array): Events of jets. These jets should be registered with the package "vector" to perform coord. transformations. 
        constituents (awkward array): Events of jet constituents. These should also be registered. 
        
    Returns:
        girth (awkward array): array of jet girths with the same size as jets. 
    '''
    pairs = ak.cartesian((jets, constituents), axis=2) # create pairs of jets and their constituents
    jet_axes, _ = ak.unzip(pairs) # jet_axes is now an array of jets, with the same shape as the constituent array
    
    delta_R = get_delta_R(jet_axes, constituents)
    girth = ak.sum(constituents.pt * delta_R, axis=2) / jets.pt
    
    return girth


def get_lesub(constituents):
    '''
    Calculate jet lesub (leading - subleading). 
    
    Args:
        constituents (awkward array): Events of jet constituents. These should be registered with "vector". 
        
    Returns:
        lesub (awkward array): array of jet girths with the same size as jets. IMPORTANT NOTE: if a jet has only one constituent, the lesub will be assigned value None. 
        This is so I can treat lesub as a substructure observable that characterizes any given jet. 
    '''
    
    descending_pt = ak.argsort(constituents.pt, axis=2, ascending=False) # gives indicices of constituents sorted by pt for each jet

    masked_con = ak.mask(constituents[descending_pt], ak.num(constituents, axis=2) > 1) # not all jets have more than one constituent. we assign those jets 
    lesub = masked_con[:, :, :1].pt - masked_con[:, :, 1:2].pt # leading - subleading
    
    return lesub


def cluster_jets(particles, radius=0.4, min_pt=10, cluster_algo=fj.antikt_algorithm, do_substructure=True):
    '''
    particles (awkward array): Events of particles. These should be registered with "vector" as four-vectors to make calculations easier later.  
    radius: radius parameter for jet clustering. 
    min_pt: minimum pt cut for jets.
    cluster_algo: clustering algorithm to use. 
    '''
    
    jet_def = fj.JetDefinition(cluster_algo, radius)
    cluster = fj.ClusterSequence(particles, jet_def)

    jets = cluster.inclusive_jets(min_pt=min_pt)
    constituents = cluster.constituents(min_pt=min_pt)
    
    if do_substructure:
        
        jets['girth'] = get_girth(jets, constituents)
        jets['lesub'] = get_lesub(constituents)
    
    
    return jets, constituents