from mofid.run_mofid import cif2mofid

def generatePrecursor(directory_to_cif):
    """
    Using CIF, generate the metal node and SMILES representation of the linker for the MOF Precursors.
    Documentation of MOFid tool can be found here: https://github.com/snurr-group/mofid
    """
    mofid_dict = cif2mofid(directory_to_cif)
    
    precursor = mofid_dict['mofid'].split(' MOFid-v1')[0]
    return precursor