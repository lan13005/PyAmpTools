from pyamptools.utility.simulation import setup_generator

if __name__ == "__main__":
    # Parses commandline arguments to condition the generator
    generator = setup_generator("gen_vec_ps")
    generator.generate()
