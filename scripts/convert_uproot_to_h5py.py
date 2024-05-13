import numpy as np
import awkward as ak
import vector
import uproot
import h5py
import matplotlib.pyplot as plt

def convert_uproot_to_h5py(files, output_file_name, example):

    example_data = h5py.File(example, "r")["delphes"]

    print("Converting MET")

    # Convert MET
    MET = uproot.concatenate(files, ["met_pt", "met_phi"], library = "np")
    MET = np.squeeze(np.stack((MET["met_pt"].astype(np.float32), MET["met_phi"].astype(np.float32)), -1).view([('MET', np.float32), ('phi', np.float32)]))
    met_nans = [np.sum(np.isnan(MET["MET"])), np.sum(np.isnan(MET["phi"]))]

    examp_MET = example_data["MET"]

    print("Converting leptons")

    # Convert Leptons
    # Assume type 0 is electron, type 1 is muon
    leptons = uproot.concatenate(files, ["l_pt", "l_eta", "l_phi", "l_mass", "l_pdgid", "lbar_pt", "lbar_eta", "lbar_phi", "lbar_mass", "lbar_pdgid"], library = "np")
    lepton_fourvectors = np.stack((np.stack((leptons["l_pt"],      leptons["l_eta"],      leptons["l_phi"],      leptons["l_mass"]),      axis = -1),
                                      np.stack((leptons["lbar_pt"], leptons["lbar_eta"], leptons["lbar_phi"], leptons["lbar_mass"]), axis = -1))
                                   ).view([("pt", np.float32), ("eta", np.float32), ("phi", np.float32), ("mass", np.float32)]).view(vector.MomentumNumpy4D)[..., 0]

    # 11 is e-, 13 is mu-, -11 is e+, -13 is mu+
    lepton_pid = np.stack((leptons["l_pdgid"], leptons["lbar_pdgid"]), axis = -1)
    lepton_type = np.logical_or(lepton_pid == 13, lepton_pid == -13)
    lepton_charge = -1 * np.sign(lepton_pid)
    leptons = np.squeeze(np.stack((lepton_fourvectors.pt.T, lepton_fourvectors.eta.T, lepton_fourvectors.phi.T, lepton_fourvectors.E.T, lepton_charge, lepton_type), -1).view([('pt', np.float32), ('eta', np.float32), ('phi', np.float32), ('energy', np.float32), ('charge', np.float32), ('type', np.float32)]))
    lepton_nans = [np.sum(np.isnan(leptons["pt"])), np.sum(np.isnan(leptons["eta"])), np.sum(np.isnan(leptons["phi"])), np.sum(np.isnan(leptons["energy"])), np.sum(np.isnan(leptons["charge"])), np.sum(np.isnan(leptons["type"]))]

    examp_leptons = example_data["leptons"]
    print("Converting Neutrinos")
    
    # Convert Neutrinos
    neutrinos = uproot.concatenate(files, ["gen_nu_pt", "gen_nu_eta", "gen_nu_phi", "gen_nubar_pt", "gen_nubar_eta", "gen_nubar_phi"], library = "np")
    neutrinos = np.squeeze(np.stack((np.stack((np.ones_like(neutrinos["gen_nu_pt"]) * 12, neutrinos["gen_nu_pt"],      neutrinos["gen_nu_eta"],      neutrinos["gen_nu_phi"],      np.zeros_like(neutrinos["gen_nu_pt"])),      axis = -1),
                                      np.stack((np.ones_like(neutrinos["gen_nu_pt"]) * -12, neutrinos["gen_nubar_pt"], neutrinos["gen_nubar_eta"], neutrinos["gen_nubar_phi"], np.zeros_like(neutrinos["gen_nubar_pt"])), axis = -1))
                                   ).view([("PDGID", np.float32), ("pt", np.float32), ("eta", np.float32), ("phi", np.float32), ("mass", np.float32)])).T
    neutrino_nans = [np.sum(np.isnan(neutrinos["PDGID"])), np.sum(np.isnan(neutrinos["pt"])), np.sum(np.isnan(neutrinos["eta"])), np.sum(np.isnan(neutrinos["phi"])), np.sum(np.isnan(neutrinos["mass"]))]

    examp_neutrinos = example_data["neutrinos"]

    print("Converting Misc")

    # Convert Extra Jet Info
    njets = uproot.concatenate(files, "jet_multiplicity", library = "np")["jet_multiplicity"].astype(np.int16)
    nbjets = uproot.concatenate(files, "bjet_multiplicity", library = "np")["bjet_multiplicity"].astype(np.int16)

    njets_nans = np.sum(np.isnan(njets))
    nbjets_nans = np.sum(np.isnan(nbjets))

    examp_njets = example_data["njets"]
    examp_nbjets = example_data["nbjets"]

    print("Converting Jets")
    

    # Convert Jets
    bjets = uproot.concatenate(files, ["b_pt", "b_eta", "b_phi", "b_mass", "bbar_pt", "bbar_eta", "bbar_phi", "bbar_mass"], library = "np")
    
    b_jet_vec = np.squeeze(np.stack((np.stack((bjets["b_pt"], bjets["b_eta"],      bjets["b_phi"],      bjets["b_mass"]),      axis = -1),
                                      np.stack((bjets["bbar_pt"], bjets["bbar_eta"], bjets["bbar_phi"], bjets["bbar_mass"]), axis = -1))
                                   ).view([("pt", np.float32), ("eta", np.float32), ("phi", np.float32), ("mass", np.float32)])).view(vector.MomentumNumpy4D)
    
    
    jets = np.squeeze(np.stack((np.stack((b_jet_vec[0].pt, b_jet_vec[0].eta,      b_jet_vec[0].phi,      b_jet_vec[0].E, np.ones_like(b_jet_vec[0].E, dtype = np.float32)),      axis = -1),
                                      np.stack((b_jet_vec[1].pt, b_jet_vec[1].eta, b_jet_vec[1].phi, b_jet_vec[1].E, np.ones_like(b_jet_vec[0].E, dtype = np.float32)), axis = -1))
                                   ).view([("pt", np.float32), ("eta", np.float32), ("phi", np.float32), ("energy", np.float32), ('is_tagged', np.float32)])).T
    
    # Create jet fourvectors
    examp_jets = example_data["jets"]

    with h5py.File(output_file_name + ".h5", "w") as f:
        group = f.create_group("delphes")
        group.create_dataset("MET", data = MET)
        group.create_dataset("leptons", data = leptons)
        group.create_dataset("neutrinos", data = neutrinos)
        group.create_dataset("njets", data = njets)
        group.create_dataset("nbjets", data = nbjets)
        group.create_dataset("jets", data = jets)

    return


def convert_uproot_to_h5py_recalculate_neutrinos(files, output_file_name, example):

    example_data = h5py.File(example, "r")["delphes"]

    print("Converting MET")

    # Convert MET
    MET = uproot.concatenate(files, ["met_pt", "met_phi"], library = "np")
    MET = np.squeeze(np.stack((MET["met_pt"].astype(np.float32), MET["met_phi"].astype(np.float32)), -1).view([('MET', np.float32), ('phi', np.float32)]))
    met_nans = [np.sum(np.isnan(MET["MET"])), np.sum(np.isnan(MET["phi"]))]

    examp_MET = example_data["MET"]

    print("Converting leptons")

    # Convert Leptons
    # Assume type 0 is electron, type 1 is muon
    leptons = uproot.concatenate(files, ["l_pt", "l_eta", "l_phi", "l_mass", "l_pdgid", "lbar_pt", "lbar_eta", "lbar_phi", "lbar_mass", "lbar_pdgid"], library = "np")
    lepton_fourvectors = np.stack((np.stack((leptons["l_pt"],      leptons["l_eta"],      leptons["l_phi"],      leptons["l_mass"]),      axis = -1),
                                      np.stack((leptons["lbar_pt"], leptons["lbar_eta"], leptons["lbar_phi"], leptons["lbar_mass"]), axis = -1))
                                   ).view([("pt", np.float32), ("eta", np.float32), ("phi", np.float32), ("mass", np.float32)]).view(vector.MomentumNumpy4D)[..., 0]

    # 11 is e-, 13 is mu-, -11 is e+, -13 is mu+
    lepton_pid = np.stack((leptons["l_pdgid"], leptons["lbar_pdgid"]), axis = -1)
    lepton_type = np.logical_or(lepton_pid == 13, lepton_pid == -13)
    lepton_charge = -1 * np.sign(lepton_pid)
    leptons = np.squeeze(np.stack((lepton_fourvectors.pt.T, lepton_fourvectors.eta.T, lepton_fourvectors.phi.T, lepton_fourvectors.E.T, lepton_charge, lepton_type), -1).view([('pt', np.float32), ('eta', np.float32), ('phi', np.float32), ('energy', np.float32), ('charge', np.float32), ('type', np.float32)]))
    lepton_nans = [np.sum(np.isnan(leptons["pt"])), np.sum(np.isnan(leptons["eta"])), np.sum(np.isnan(leptons["phi"])), np.sum(np.isnan(leptons["energy"])), np.sum(np.isnan(leptons["charge"])), np.sum(np.isnan(leptons["type"]))]

    examp_leptons = example_data["leptons"]

    print("Converting Jets")
    

    # Convert Jets
    bjets = uproot.concatenate(files, ["b_pt", "b_eta", "b_phi", "b_mass", "bbar_pt", "bbar_eta", "bbar_phi", "bbar_mass"], library = "np")
    
    b_jet_vec = np.squeeze(np.stack((np.stack((bjets["b_pt"], bjets["b_eta"],      bjets["b_phi"],      bjets["b_mass"]),      axis = -1),
                                      np.stack((bjets["bbar_pt"], bjets["bbar_eta"], bjets["bbar_phi"], bjets["bbar_mass"]), axis = -1))
                                   ).view([("pt", np.float32), ("eta", np.float32), ("phi", np.float32), ("mass", np.float32)])).view(vector.MomentumNumpy4D)
    
    
    jets = np.squeeze(np.stack((np.stack((b_jet_vec[0].pt, b_jet_vec[0].eta,      b_jet_vec[0].phi,      b_jet_vec[0].E, np.ones_like(b_jet_vec[0].E, dtype = np.float32)),      axis = -1),
                                      np.stack((b_jet_vec[1].pt, b_jet_vec[1].eta, b_jet_vec[1].phi, b_jet_vec[1].E, np.ones_like(b_jet_vec[0].E, dtype = np.float32)), axis = -1))
                                   ).view([("pt", np.float32), ("eta", np.float32), ("phi", np.float32), ("energy", np.float32), ('is_tagged', np.float32)])).T
    
    # Create jet fourvectors
    examp_jets = example_data["jets"]
    
    print("Converting Neutrinos")
    
    # Convert Neutrinos
    gen_tops = uproot.concatenate(files, ["gen_top_pt", "gen_top_eta", "gen_top_phi", "gen_top_mass", "gen_tbar_pt", "gen_tbar_eta", "gen_tbar_phi", "gen_tbar_mass"], library = "np")
    
    gen_tops = np.squeeze(np.stack((np.stack((gen_tops["gen_top_pt"], gen_tops["gen_top_eta"],      gen_tops["gen_top_phi"],      gen_tops["gen_top_mass"]),      axis = -1),
                                      np.stack((gen_tops["gen_tbar_pt"], gen_tops["gen_tbar_eta"], gen_tops["gen_tbar_phi"], gen_tops["gen_tbar_mass"]), axis = -1))
                                   ).view([("pt", np.float32), ("eta", np.float32), ("phi", np.float32), ("mass", np.float32)])).view(vector.MomentumNumpy4D)
    
    recovered_neutrinos = gen_tops - b_jet_vec.add(lepton_fourvectors[[1, 0]])
                                                   
    neutrinos = np.squeeze(np.stack((np.stack((np.ones_like(recovered_neutrinos[0].pt) * 12, recovered_neutrinos[0].pt, recovered_neutrinos[0].eta, recovered_neutrinos[0].phi,      np.zeros_like(recovered_neutrinos[0].pt)), axis = -1),
                                      np.stack((np.ones_like(recovered_neutrinos[0].pt) * -12, recovered_neutrinos[1].pt, recovered_neutrinos[1].eta, recovered_neutrinos[1].phi, np.zeros_like(recovered_neutrinos[1].pt)), axis = -1))
                                   ).view([("PDGID", np.float32), ("pt", np.float32), ("eta", np.float32), ("phi", np.float32), ("mass", np.float32)])).T
    neutrino_nans = [np.sum(np.isnan(neutrinos["PDGID"])), np.sum(np.isnan(neutrinos["pt"])), np.sum(np.isnan(neutrinos["eta"])), np.sum(np.isnan(neutrinos["phi"])), np.sum(np.isnan(neutrinos["mass"]))]
    
    examp_neutrinos = example_data["neutrinos"]

    print("Converting Misc")

    # Convert Extra Jet Info
    njets = uproot.concatenate(files, "jet_multiplicity", library = "np")["jet_multiplicity"].astype(np.int16)
    nbjets = uproot.concatenate(files, "bjet_multiplicity", library = "np")["bjet_multiplicity"].astype(np.int16)

    njets_nans = np.sum(np.isnan(njets))
    nbjets_nans = np.sum(np.isnan(nbjets))

    examp_njets = example_data["njets"]
    examp_nbjets = example_data["nbjets"]

    with h5py.File(output_file_name + ".h5", "w") as f:
        group = f.create_group("delphes")
        group.create_dataset("MET", data = MET)
        group.create_dataset("leptons", data = leptons)
        group.create_dataset("neutrinos", data = neutrinos)
        group.create_dataset("njets", data = njets)
        group.create_dataset("nbjets", data = nbjets)
        group.create_dataset("jets", data = jets)

    return
    
if __name__ == '__main__':
    example = "./nu2flow_data/test.h5"
    
    for s in range(1):
        start_index = 10 * s
        stop_index = 10 * (s + 1)
    
        files = [f"./eos-purdue/store/user/jthieman/2018/spinCorrInput_2018_January2023/Nominal/ee/ee_ttbarsignalviatau_fromDilepton_2018UL_{i}.root:Step7" for i in range(start_index, stop_index)]
    
        convert_uproot_to_h5py(files, f"./work/users/thastrei/full_sim_{start_index}_{stop_index - 1}", example)
