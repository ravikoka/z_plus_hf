import numpy as np
import multiprocessing

# inspired by https://blog.scientific-python.org/numpy/numpy-rng/
# and https://numpy.org/doc/stable/reference/random/parallel.html

def generate_random_ints(process_num, child_rng):
    
    print(f'Process: {process_num}, Rand. Integers: {[int(child_rng.integers(1, 20)) for _ in range(5)]}')
    
    return 0

if __name__ == '__main__':

    parent_rng = np.random.default_rng(2024)

    child_rngs = parent_rng.spawn(4)

    args = [(i, child_rng) for (i, child_rng) in enumerate(child_rngs)]

    with multiprocessing.Pool(4) as pool:
        proc = pool.starmap(generate_random_ints, args)