from pyamptools.utility.generator import setup_generator

if __name__ == "__main__":
    # Parses commandline arguments to condition the generator
    generator = setup_generator("gen_amp")
    generator.generate()
