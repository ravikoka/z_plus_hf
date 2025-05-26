import awkward as ak
import numpy as np

def get_invariant_mass(left_particles, right_particles):
    
    mother_four_vector = ak.zip(
        {'px': (left_particles + right_particles).px,
        'py': (left_particles + right_particles).py,
        'pz': (left_particles + right_particles).pz,
        'E': (left_particles + right_particles).E}, 
        with_name='Momentum4D'
        )
    
    return mother_four_vector.m

def invariant_mass(pairs, left_mass, right_mass):
    '''
    Calculates the invariant mass between pairs of particles. I assume these are pairs of particles with different identities. This means I won't have pairs of the same particle. If for some reason, you wanted to calculate the invariant mass using the same particle, this code needs to be modified (really at the level of the cartesian product).
    
    pairs (awkward array): an array of events filled with pairs of particles. This can be gotten via a Cartesian product between two arrays of particles. 
    left_mass (float): the mass of the left particles in each pair. Note: I am assuming that the left element of every pair is the same type of particle (and the same assumption goes for the right element).
    right_mass (float): the mass of the right particles in each pair. 
    
    can just sum and then use .m if registered with awkward.
    '''
    left, right = ak.unzip(pairs)
    left_energy = np.sqrt(left.px**2 + left.py**2 + left.pz**2 + left_mass**2)
    right_energy = np.sqrt(right.px**2 + right.py**2 + right.pz**2 + right_mass**2)
    return np.sqrt((left_energy + right_energy)**2 -
                   (left.px + right.px)**2 -
                   (left.py + right.py)**2 -
                   (left.pz + right.pz)**2)

def opposite(pairs):
    left, right = ak.unzip(pairs)
    return pairs[left.pdg == -right.pdg]

def get_dimuon_invariant_mass(particles):
    '''
    Given an array of events of particles, calculate the dimuon invariant mass.
    
    particles (awkward array): events filled with particles.  
    
    To do:
    Only look at events with one muon and one antimuon?
    
    '''
    muon_mass = 0.1056583755
    muon_filter = (particles.pdg == 13) & (particles.is_final)
    muons = particles[muon_filter]

    anti_muon_filter = (particles.pdg == -13) & (particles.is_final)
    anti_muons = particles[anti_muon_filter]
    
    pairs = ak.cartesian([muons, anti_muons])
    
    return ak.flatten(invariant_mass(pairs, muon_mass, muon_mass))