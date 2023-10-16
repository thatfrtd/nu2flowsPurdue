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
    MET = uproot.concatenate(files, ["MET", "MET_phi"], library = "np")
    MET = np.squeeze(np.stack((MET["MET"].astype(np.float32), MET["MET_phi"].astype(np.float32)), -1).view([('MET', np.float32), ('phi', np.float32)]))
    met_nans = [np.sum(np.isnan(MET["MET"])), np.sum(np.isnan(MET["phi"]))]

    examp_MET = example_data["MET"]

    #plt.hist(MET["MET"], density = True, bins = 30, alpha = 0.5)
    #plt.hist(examp_MET["MET"], density = True, bins = 30, alpha = 0.5)
    #plt.show()
    #
    #plt.close()
    #plt.hist(MET["phi"], density = True, bins = 30, alpha = 0.5)
    #plt.hist(examp_MET["phi"], density = True, bins = 30, alpha = 0.5)
    #plt.show()

    print("Converting leptons")

    # Convert Leptons
    # Assume type 0 is electron, type 1 is muon
    leptons = uproot.concatenate(files, ["lep_pt", "lep_eta", "lep_phi", "lep_mass", "lep_pdgid", "alep_pt", "alep_eta", "alep_phi", "alep_mass", "alep_pdgid"], library = "np")
    lepton_fourvectors = np.stack((np.stack((leptons["lep_pt"],      leptons["lep_eta"],      leptons["lep_phi"],      leptons["lep_mass"]),      axis = -1),
                                      np.stack((leptons["alep_pt"], leptons["alep_eta"], leptons["alep_phi"], leptons["alep_mass"]), axis = -1))
                                   ).view([("pt", np.float64), ("eta", np.float64), ("phi", np.float64), ("mass", np.float64)]).view(vector.MomentumNumpy4D)[..., 0]

    # 11 is e-, 13 is mu-, -11 is e+, -13 is mu+
    lepton_pid = np.stack((leptons["lep_pdgid"], leptons["alep_pdgid"]), axis = -1)
    lepton_type = np.logical_or(lepton_pid == 13, lepton_pid == -13)
    lepton_charge = -1 * np.sign(lepton_pid)
    leptons = np.squeeze(np.stack((lepton_fourvectors.pt.T, lepton_fourvectors.eta.T, lepton_fourvectors.phi.T, lepton_fourvectors.E.T, lepton_charge, lepton_type), -1).view([('pt', np.float64), ('eta', np.float64), ('phi', np.float64), ('energy', np.float64), ('charge', np.float64), ('type', np.float64)]))
    lepton_nans = [np.sum(np.isnan(leptons["pt"])), np.sum(np.isnan(leptons["eta"])), np.sum(np.isnan(leptons["phi"])), np.sum(np.isnan(leptons["energy"])), np.sum(np.isnan(leptons["charge"])), np.sum(np.isnan(leptons["type"]))]

    examp_leptons = example_data["leptons"]

    #plt.close()
    #plt.hist(np.ravel(leptons["pt"]), density = True, bins = 30, alpha = 0.5)
    #plt.hist(np.ravel(examp_leptons["pt"]), density = True, bins = 30, alpha = 0.5)
    #plt.show
    #
    #plt.close()
    #plt.hist(np.ravel(leptons["eta"]), density = True, bins = 30, alpha = 0.5)
    #plt.hist(np.ravel(examp_leptons["eta"]), density = True, bins = 30, alpha = 0.5)
    #plt.show()
    #
    #plt.close()
    #plt.hist(np.ravel(leptons["phi"]), density = True, bins = 30, alpha = 0.5)
    #plt.hist(np.ravel(examp_leptons["phi"]), density = True, bins = 30, alpha = 0.5)
    #plt.show()
    #
    #plt.close()
    #plt.hist(np.ravel(leptons["energy"]), density = True, bins = 30, alpha = 0.5)
    #plt.hist(np.ravel(examp_leptons["energy"]), density = True, bins = 30, alpha = 0.5)
    #plt.show()
    #
    #plt.close()
    #plt.hist(np.ravel(leptons["charge"]), density = True, bins = 30, alpha = 0.5)
    #plt.hist(np.ravel(examp_leptons["charge"]), density = True, bins = 30, alpha = 0.5)
    #plt.show()
    #
    #plt.close()
    #plt.hist(np.ravel(leptons["type"]), density = True, bins = 30, alpha = 0.5)
    #plt.hist(np.ravel(examp_leptons["type"]), density = True, bins = 30, alpha = 0.5)
    #plt.show()

    print("Converting Neutrinos")

    # Convert Neutrinos
    neutrinos = uproot.concatenate(files, ["gen_neu_pt", "gen_neu_eta", "gen_neu_phi", "gen_neu_pdgid", "gen_aneu_pt", "gen_aneu_eta", "gen_aneu_phi", "gen_aneu_pdgid"], library = "np")
    neutrinos = np.squeeze(np.stack((np.stack((neutrinos["gen_neu_pdgid"], neutrinos["gen_neu_pt"],      neutrinos["gen_neu_eta"],      neutrinos["gen_neu_phi"],      np.zeros_like(neutrinos["gen_neu_pt"])),      axis = -1),
                                      np.stack((neutrinos["gen_aneu_pdgid"], neutrinos["gen_aneu_pt"], neutrinos["gen_aneu_eta"], neutrinos["gen_aneu_phi"], np.zeros_like(neutrinos["gen_aneu_pt"])), axis = -1))
                                   ).view([("PDGID", np.float64), ("pt", np.float64), ("eta", np.float64), ("phi", np.float64), ("mass", np.float64)])).T
    neutrino_nans = [np.sum(np.isnan(neutrinos["PDGID"])), np.sum(np.isnan(neutrinos["pt"])), np.sum(np.isnan(neutrinos["eta"])), np.sum(np.isnan(neutrinos["phi"])), np.sum(np.isnan(neutrinos["mass"]))]

    examp_neutrinos = example_data["neutrinos"]

    #plt.close()
    #plt.hist(np.ravel(neutrinos["PDGID"]), density = True, bins = 30, alpha = 0.5)
    #plt.hist(np.ravel(examp_neutrinos["PDGID"]), density = True, bins = 30, alpha = 0.5)
    #plt.show()
    #
    #plt.close()
    #plt.hist(np.ravel(neutrinos["pt"]), density = True, bins = 30, alpha = 0.5)
    #plt.hist(np.ravel(examp_neutrinos["pt"]), density = True, bins = 30, alpha = 0.5)
    #plt.show()
    #
    #plt.close()
    #plt.hist(np.ravel(neutrinos["eta"]), density = True, bins = 30, alpha = 0.5)
    #plt.hist(np.ravel(examp_neutrinos["eta"]), density = True, bins = 30, alpha = 0.5)
    #plt.show()
    #
    #plt.close()
    #plt.hist(np.ravel(neutrinos["phi"]), density = True, bins = 30, alpha = 0.5)
    #plt.hist(np.ravel(examp_neutrinos["phi"]), density = True, bins = 30, alpha = 0.5)
    #plt.show()
    #
    #plt.close()
    #plt.hist(np.ravel(neutrinos["mass"]), density = True, bins = 30, alpha = 0.5)
    #plt.hist(np.ravel(examp_neutrinos["mass"]), density = True, bins = 30, alpha = 0.5)
    #plt.show()

    print("Converting Misc")

    # Convert Extra Jet Info
    njets = uproot.concatenate(files, "jet_size", library = "np")["jet_size"]
    nbjets = ak.sum(uproot.concatenate(files, "jet_btag", library = "ak")["jet_btag"], axis = -1).to_numpy()

    njets_nans = np.sum(np.isnan(njets))
    nbjets_nans = np.sum(np.isnan(nbjets))

    examp_njets = example_data["njets"]
    examp_nbjets = example_data["nbjets"]

    #plt.close()
    #plt.hist(njets, density = True, bins = 30, alpha = 0.5)
    #plt.hist(examp_njets, density = True, bins = 30, alpha = 0.5)
    #plt.show()
    #
    #plt.close()
    #plt.hist(nbjets, density = True, bins = 30, alpha = 0.5)
    #plt.hist(examp_nbjets, density = True, bins = 30, alpha = 0.5)
    #plt.show()

    print("Converting Jets")
    

    # Convert Jets
    jets = uproot.concatenate(files, ["jet_pt", "jet_eta", "jet_phi", "jet_mass", "jet_btag"], library = "ak")

    # Make vector recognize awkward
    vector.register_awkward()

    # Create jet fourvectors
    jet_fourvectors = ak.zip(({"pt" : jets["jet_pt"], "eta" : jets["jet_eta"], "phi" : jets["jet_phi"], "mass" : jets["jet_mass"]}), with_name = "Momentum4D")

    jets = ak.zip(({"pt" : jet_fourvectors.pt, "eta" : jet_fourvectors.eta, "phi" : jet_fourvectors.phi, "energy" : jet_fourvectors.energy, "btag" : jets["jet_btag"]}))

    '''jet_pt_check = jets["jet_pt"] >= 30
    jet_eta_check = abs(jets["jet_eta"]) <= 2.4
    jet_filter = np.logical_and(jet_pt_check, jet_eta_check)'''
    jets_sort = ak.argsort(jets["pt"], ascending = False)
    jets = jets[jets_sort]
    jets = ak.pad_none(jets, 13, axis = 1, clip = True)
    jets_pt = ak.fill_none(jets.pt, 0).to_numpy()
    jets_eta = ak.fill_none(jets.eta, 0).to_numpy()
    jets_phi = ak.fill_none(jets.phi, 0).to_numpy()
    jets_energy = ak.fill_none(jets.energy, 0).to_numpy()
    jets_btag = ak.fill_none(jets.btag, 0).to_numpy()

    jets = np.squeeze(np.stack((jets_pt, jets_eta, jets_phi, jets_energy, jets_btag), -1).view([('pt', np.float64), ('eta', np.float64), ('phi', np.float64), ('energy', np.float64), ('is_tagged', np.float64)]))

    jet_nans = [np.sum(np.isnan(jets["pt"])), np.sum(np.isnan(jets["eta"])), np.sum(np.isnan(jets["phi"])), np.sum(np.isnan(jets["energy"])), np.sum(np.isnan(jets["is_tagged"]))]

    examp_jets = example_data["jets"]

    #plt.close()
    #plt.hist(np.ravel(jets["pt"]), density = True, bins = 30, alpha = 0.5)
    #plt.hist(np.ravel(examp_jets["pt"]), density = True, bins = 30, alpha = 0.5)
    #plt.show()
    #
    #plt.close()
    #plt.hist(np.ravel(jets["eta"]), density = True, bins = 30, alpha = 0.5)
    #plt.hist(np.ravel(examp_jets["eta"]), density = True, bins = 30, alpha = 0.5)
    #plt.show()
    #
    #plt.close()
    #plt.hist(np.ravel(jets["phi"]), density = True, bins = 30, alpha = 0.5)
    #plt.hist(np.ravel(examp_jets["phi"]), density = True, bins = 30, alpha = 0.5)
    #plt.show()
    #
    #plt.close()
    #plt.hist(np.ravel(jets["energy"]), density = True, bins = 30, alpha = 0.5)
    #plt.hist(np.ravel(examp_jets["energy"]), density = True, bins = 30, alpha = 0.5)
    #plt.show()
    #
    #plt.close()
    #plt.hist(np.ravel(jets["is_tagged"]), density = True, bins = 30, alpha = 0.5)
    #plt.hist(np.ravel(examp_jets["is_tagged"]).astype(np.float32), density = True, bins = 30, alpha = 0.5)
    #plt.show()

    with h5py.File(output_file_name + ".h5", "w") as f:
        group = f.create_group("delphes")
        group.create_dataset("MET", data = MET)
        group.create_dataset("leptons", data = leptons)
        group.create_dataset("neutrinos", data = neutrinos)
        group.create_dataset("njets", data = njets)
        group.create_dataset("nbjets", data = nbjets)
        group.create_dataset("jets", data = jets)

    return

def replace_part_of_data(example_file, part, files):

    with h5py.File(example_file, "r+") as output:
        output_data = output["delphes"]

        event_number = output_data["MET"]["phi"].size

        if "MET" in part:
            MET = uproot.concatenate(files, ["MET", "MET_phi"], library = "np")
            MET = np.squeeze(np.stack((MET["MET"], MET["MET_phi"]), -1).view([('MET', np.float64), ('phi', np.float64)]))
            k = output_data["MET"]
            output_data["MET"]["MET"] = MET["MET"][0:event_number]
            output_data["MET"]["phi"] = MET["phi"][0:event_number]
        if "leptons" in part:
            leptons = uproot.concatenate(files, ["lep_pt", "lep_eta", "lep_phi", "lep_mass", "lep_pdgid", "alep_pt", "alep_eta", "alep_phi", "alep_mass", "alep_pdgid"], library = "np")
            lepton_fourvectors = np.stack((np.stack((leptons["lep_pt"],      leptons["lep_eta"],      leptons["lep_phi"],      leptons["lep_mass"]),      axis = -1),
                                              np.stack((leptons["alep_pt"], leptons["alep_eta"], leptons["alep_phi"], leptons["alep_mass"]), axis = -1))
                                           ).view([("pt", np.float64), ("eta", np.float64), ("phi", np.float64), ("mass", np.float64)]).view(vector.MomentumNumpy4D)[..., 0]

            # 11 is e-, 13 is mu-, -11 is e+, -13 is mu+
            lepton_pid = np.stack((leptons["lep_pdgid"], leptons["alep_pdgid"]), axis = -1)
            lepton_type = np.logical_or(lepton_pid == 13, lepton_pid == -13)
            lepton_charge = -1 * np.sign(lepton_pid)
            leptons = np.squeeze(np.stack((lepton_fourvectors.pt.T, lepton_fourvectors.eta.T, lepton_fourvectors.phi.T, lepton_fourvectors.E.T, lepton_charge, lepton_type), -1).view([('pt', np.float64), ('eta', np.float64), ('phi', np.float64), ('energy', np.float64), ('charge', np.float64), ('type', np.float64)]))

            output_data["leptons"]["pt"] = leptons["pt"][0:event_number]
            output_data["leptons"]["eta"] = leptons["eta"][0:event_number]
            output_data["leptons"]["phi"] = leptons["phi"][0:event_number]
            output_data["leptons"]["energy"] = leptons["energy"][0:event_number]
            output_data["leptons"]["charge"] = leptons["charge"][0:event_number]
            output_data["leptons"]["type"] = leptons["type"][0:event_number]
        if "neutrinos" in part:
            neutrinos = uproot.concatenate(files, ["gen_neu_pt", "gen_neu_eta", "gen_neu_phi", "gen_neu_pdgid", "gen_aneu_pt", "gen_aneu_eta", "gen_aneu_phi", "gen_aneu_pdgid"], library = "np")
            neutrinos = np.squeeze(np.stack((np.stack((neutrinos["gen_neu_pdgid"], neutrinos["gen_neu_pt"],      neutrinos["gen_neu_eta"],      neutrinos["gen_neu_phi"],      np.zeros_like(neutrinos["gen_neu_pt"])),      axis = -1),
                                          np.stack((neutrinos["gen_aneu_pdgid"], neutrinos["gen_aneu_pt"], neutrinos["gen_aneu_eta"], neutrinos["gen_aneu_phi"], np.zeros_like(neutrinos["gen_aneu_pt"])), axis = -1))
                                       ).view([("PDGID", np.float64), ("pt", np.float64), ("eta", np.float64), ("phi", np.float64), ("mass", np.float64)])).T

            output_data["neutrinos"]["PDGID"] = neutrinos["PDGID"][0:event_number]
            output_data["neutrinos"]["pt"] = neutrinos["pt"][0:event_number]
            output_data["neutrinos"]["eta"] = neutrinos["eta"][0:event_number]
            output_data["neutrinos"]["phi"] = neutrinos["phi"][0:event_number]
            output_data["neutrinos"]["mass"] = neutrinos["mass"][0:event_number]
        if "misc" in part:
            njets = uproot.concatenate(files, "jet_size", library = "np")["jet_size"]
            nbjets = ak.sum(uproot.concatenate(files, "jet_btag", library = "ak")["jet_btag"], axis = -1).to_numpy()

            output_data["njets"] = njets[0:event_number]
            output_data["nbjets"] = nbjets[0:event_number]

        if "jets" in part:
            num_jets = output_data["jets"]["pt"].shape[1]

            jets = uproot.concatenate(files, ["jet_pt", "jet_eta", "jet_phi", "jet_mass", "jet_btag"], library = "ak")

            '''jet_pt_check = jets["jet_pt"] >= 30
            jet_eta_check = abs(jets["jet_eta"]) <= 2.4
            jet_filter = np.logical_and(jet_pt_check, jet_eta_check)'''
            jets_sort = ak.argsort(jets["jet_pt"], ascending = False)
            jets = jets[jets_sort]
            jets = ak.pad_none(jets, num_jets, axis = 1, clip = True)
            jets = ak.fill_none(jets, 0)
            jets = np.squeeze(np.stack((jets["jet_pt"].to_numpy(), jets["jet_eta"].to_numpy(), jets["jet_phi"].to_numpy(), jets["jet_mass"].to_numpy(), jets["jet_btag"].to_numpy()), -1).view([('pt', np.float64), ('eta', np.float64), ('phi', np.float64), ('energy', np.float64), ('is_tagged', np.float64)]))

            min_e = np.min(jets["energy"][0:event_number])
            max_e = np.max(jets["energy"][0:event_number])

            output_data["jets"]["pt"] = jets["pt"][0:event_number]
            output_data["jets"]["eta"] = jets["eta"][0:event_number]
            output_data["jets"]["phi"] = jets["phi"][0:event_number]
            output_data["jets"]["energy"] = jets["energy"][0:event_number]
            output_data["jets"]["is_tagged"] = jets["is_tagged"][0:event_number]

    return

    
if __name__ == '__main__':
    example = "/Users/thatf/OneDrive/Documents/Simulations/nu2flows/nu2flows_data/pythia_test_delph.h5"

    for s in range(10):
        start_index = 100 * s
        stop_index = 100 * (s + 1)
    
        files = [f"/Users/thatf/OneDrive/Documents/Simulations/CMS Research/Full_Delphes/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_minitrees_{i}.root:Step7" for i in range(start_index, stop_index)]

        #replace_part_of_data(example, ["jets"], files)

        convert_uproot_to_h5py(files, f"/Users/thatf/OneDrive/Documents/Simulations/CMS Research/Full_Delphes/full_delphes_{start_index}_{stop_index - 1}", example)
