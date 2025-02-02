import pythia8 
import os
import time

import awkward as ak
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

from multiprocessing import Pool


def process_event(pythia_event):
    px = [] # consider prealloc np array
    py = []
    pz = []
    E = []
    pdg = []
    generator_status = []
    mother_lists = []
    daughter_lists = []
    is_final_state = []
    
    for i, particle in enumerate(pythia_event):
        if i == 0:
            continue 
        
        px.append(particle.px())
        py.append(particle.py())
        pz.append(particle.pz())
        E.append(particle.e())
        pdg.append(particle.id())
        generator_status.append(particle.status())
        mother_lists.append(particle.motherList())
        daughter_lists.append(particle.daughterList())
        is_final_state.append(particle.isFinal())
        
    particle_data = {
        "px": np.array(px),
        "py": np.array(py),
        "pz": np.array(pz),
        "E":  np.array(E),
        "pdg": np.array(pdg),
        "gen_status": np.array(generator_status),
        "mother_list": mother_lists, # check if motherlist is indexed with zero as first particle or row 0 in event record (event level info)
        "daughter_list": daughter_lists,
        "is_final": np.array(is_final_state),    
    }
        
    return ak.Array([particle_data])


def generate_events(run_num, rng, parent_data_dir, num_batches, batch_size):
    
    pythia = pythia8.Pythia()

    pythia.readString("Beams:eCM = 5020")
    pythia.readString("HardQCD:all = on")
    pythia.readString("PhaseSpace:pTHatMin = 100.") # check this
    pythia.readString("PhaseSpace:pTHatMin = 500.")
    
    pythia.readString("WeakBosonAndParton:all = on")
    pythia.readString("WeakBosonExchange:all = on")
    pythia.readString("WeakDoubleBoson:all = on")
    pythia.readString("WeakSingleBoson:all = on")
    
    seed = rng.integers(1, 9e8+1, dtype=int) # the min and max seed values in pythia are 1 and 9 mil., respectively
    pythia.readString("Random:setSeed = on")
    pythia.readString(f"Random:Seed = {seed}")
    
    pythia.init()
    
    for batch_num in range(num_batches):

        events = ak.Array([])
        sigma_gens = []
        sigma_errs = []
        for event_num in range(batch_size):
            
            if not pythia.next():
                continue
            
            particle_data = process_event(pythia.event)
            sigma_gens.append(pythia.infoPython().sigmaGen())
            sigma_errs.append(pythia.infoPython().sigmaErr())
            
            events = ak.concatenate([events, particle_data])
    
        print(f'{run_num} here')
        print(f'batch num: {batch_num}, num events processed: {(batch_num + 1) * batch_size}')
        
        #print(f'{pythia.infoPython().sigmaGen()}')
        #print(f'{pythia.infoPython().sigmaErr()}')
        
        out_file_dir = parent_data_dir + f'run{run_num}/'
        os.makedirs(os.path.dirname(out_file_dir), exist_ok=True)
        with open(out_file_dir + f'pp_Z_production_13600_{batch_num}.pkl', 'wb') as out_file:
            pkl.dump([events, sigma_gens, sigma_errs], out_file)
            
            
def main():
    t0 = time.time()
    RAW_DATA_DIR = '/Users/ravikoka/repos/z_plus_hf/feasibility/data/multi/'

    NUM_EVENTS = 125000 #500 #250000 #125000 #int(1e3) 
    BATCH_SIZE = 500
    NUM_BATCHES = NUM_EVENTS // BATCH_SIZE
    
    NUM_PROC = 4 # number of cores to use, data for each process/run is stored in a separate directory  
    
    seed = 157787644118040162645699200457491913582 # generated with secrets.randbits(128), avoids bias in human chosen seeds
    parent_rng = np.random.default_rng(seed) # uses PCG64 as default pseudorandom number generator
    child_rngs = parent_rng.spawn(NUM_PROC) # creates independent child rngs. We could also use SeedSequence, but we'd get the same result.

    args = [(proc_num, child_rngs[proc_num], RAW_DATA_DIR, NUM_BATCHES, BATCH_SIZE) for proc_num in range(NUM_PROC)]
    print(args)
    
    with Pool(NUM_PROC) as pool:
        proc = pool.starmap(generate_events, args)
    
    t1 = time.time()
    
    print(f'total time: {t1-t0}')
    print(f'num events generated: {NUM_PROC * NUM_BATCHES * BATCH_SIZE}') # num events run over is given by floor division. this shouldn't really be an issue usually but let's print the number generated as a safeguard
    print(f'events/s: {NUM_EVENTS * NUM_PROC / (t1-t0)}')
    

if __name__ == '__main__':
    main()
    
    
    
    

    