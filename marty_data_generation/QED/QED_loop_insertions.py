import os
from subprocess import call
from itertools import combinations_with_replacement, product
from tqdm import tqdm
from icecream import ic

particles_list = [
        "electron",
        "anti_electron",
        "photon"
]

#This function actually calls the compiled MARTY code to run the amplitude calculations
def calc_amplitude(particles, ampl_file="out/ampl.txt",
        fdexpr_file = "out/fdexpr.txt",
        amflow_file = "out/amflow_output.wl",
        sqampl_file="out/ampl_sq.txt",
        insertions_file="out/insertions.txt",
        log_file=False
        ):

    options = "--particles=" + particles + " -e" + " -a " + ampl_file + " -f " + fdexpr_file + " -m" + amflow_file + " -s " + sqampl_file + " -i " + insertions_file
    if log_file:
        options = options + " > " + log_file

    # _ = call("./QED_IO.x " + options, shell=True)
    _ = call("sudo ./QED_AllParticles_IO.x " + options, shell=True)
    

def particles_format(particles_list):
    return ",".join(particles_list)


def get_possible_n_to_m(particles_list, n, m):
    in_list = ["in_"+p for p in particles_list]
    out_list = ["out_"+p for p in particles_list]

    # Creates all possible combos of n in and m out particles
    # returns a list of strings for each process with the in and out particles:
    # e.g., 'in_electron,in_electron,out_electron,out_antielectron'
    possible_two_in = combinations_with_replacement(in_list, n)
    possible_two_out = combinations_with_replacement(out_list, m)
    possible_two_to_two = list(product(possible_two_in, possible_two_out))
    possible_two_to_two = [sum(p, ()) for p in possible_two_to_two]
    possible_two_to_two = [particles_format(p) for p in possible_two_to_two]

    return possible_two_to_two

def get_limited_n_to_m(particles_list, n, m):
    # For now, heavily limit the number of calculated diagrams/processes, just to see if it works.
    in_list = ["in_"+p for p in particles_list]
    out_list = ["out_"+p for p in particles_list]

    two_in = combinations_with_replacement(in_list, 2)
    two_out = combinations_with_replacement(out_list, 2)
    two_to_two = list(product(two_in, two_out))
    two_to_two = [sum(p, ()) for p in two_to_two]
    two_to_two = [particles_format(p) for p in two_to_two]
    return two_to_two


def delete_file(file):
    try:
        os.remove(file)
        print("Out file", file, "existed before. Deleted")
    except:
        pass


if __name__== "__main__":

    #possible_two_to_two = get_possible_n_to_m(particles_list, 2, 2)
    #possible_three_to_three = get_possible_n_to_m(particles_list, 3, 3)

    limited_two_to_two = get_limited_n_to_m(particles_list, 2, 2)

    # These get the length of the possible n to m processes.
    #ic(len(possible_two_to_two))
    #ic(len(possible_three_to_three))
    
    ic(len(limited_two_to_two))

    ampl_file = "out/ampl.txt"
    fdexpr_file = "out/fdexpr.txt"
    amflow_file = "out/amflow_output.wl"
    sqampl_file = "out/ampl_sq.txt"
    insertions_file = "out/insertions.txt"
    log_file = "out/log.txt"
    diagrams_file = "out/diagrams.txt"

    delete_file(ampl_file)
    delete_file(fdexpr_file)
    delete_file(amflow_file)
    delete_file(sqampl_file)
    delete_file(insertions_file)
    delete_file(log_file)
    delete_file(diagrams_file)

    print("Calculating 2-2 amplitudes and squares")
    # for p in tqdm(possible_two_to_two):
    for p in tqdm(limited_two_to_two):
        calc_amplitude(p,
                ampl_file=ampl_file,
                fdexpr_file=fdexpr_file,
                amflow_file=amflow_file,
                sqampl_file=sqampl_file,
                insertions_file=insertions_file,
                log_file=log_file
                )

    # print("Calculating 3-3 amplitudes and squares")
    # for p in tqdm(possible_three_to_three):
    #     calc_amplitude(p,
    #             ampl_file=ampl_file,
    #             sqampl_file=sqampl_file,
    #             insertions_file=insertions_file,
    #             log_file=log_file
    #             )
