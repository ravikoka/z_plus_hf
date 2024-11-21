import pythia8 

import awkward as ak
import numpy as np
import pickle as pkl
import hist 
import matplotlib.pyplot as plt
import uproot

import time


def process_event(batch_num, event_num, pythia_event):
    px = [] 
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


if __name__ == '__main__':
    
    RAW_DATA_DIR = '/Users/ravikoka/repos/z_plus_hf/feasibility/data/raw/'

    NUM_EVENTS = int(1e5) 
    BATCH_SIZE = 500
    NUM_BATCHES = NUM_EVENTS // BATCH_SIZE

    pythia = pythia8.Pythia()

    pythia.readString("Beams:eCM = 13600")
    pythia.readString("HardQCD:all = on")
    pythia.readString("PhaseSpace:pTHatMin = 20.") # check this

    pythia.readString("WeakBosonAndParton:all = on")
    pythia.readString("WeakBosonExchange:all = on")
    pythia.readString("WeakDoubleBoson:all = on")
    pythia.readString("WeakSingleBoson:all = on")
    pythia.readString("SoftQCD:all = off")

    pythia.init()


    for batch_num in range(NUM_BATCHES):

        events = ak.Array([])
        for event_num in range(BATCH_SIZE):
            
            if not pythia.next():
                continue
            
            particle_data = process_event(batch_num, event_num, pythia.event)
        
            events = ak.concatenate([events, particle_data])
            
        
        print(f'batch num: {batch_num}, num events processed: {(batch_num + 1) * BATCH_SIZE}')
        
        with open(f'{RAW_DATA_DIR}pp_Z_production_13600_{batch_num}.pkl', 'wb') as out_file:
            pkl.dump(events, out_file)