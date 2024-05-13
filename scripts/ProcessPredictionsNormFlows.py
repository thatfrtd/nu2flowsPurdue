import uproot
import numpy as np
import awkward as ak
import vector
from numpy.random import default_rng
import matplotlib.pyplot as plt
import h5py

def extract_and_process_predictions(root_file, prediction_file, output_file_name):

    leptons = uproot.concatenate(root_file, ["l_pt", "l_eta", "l_phi", "l_mass", "l_pdgid", "lbar_pt", "lbar_eta", "lbar_phi", "lbar_mass", "lbar_pdgid"], library = "np")
    leptons = np.stack((np.stack((leptons["l_pt"],      leptons["l_eta"],      leptons["l_phi"],      leptons["l_mass"]),      axis = -1),
                                  np.stack((leptons["lbar_pt"], leptons["lbar_eta"], leptons["lbar_phi"], leptons["lbar_mass"]), axis = -1))
                               ).view([("pt", np.float32), ("eta", np.float32), ("phi", np.float32), ("mass", np.float32)]).view(vector.MomentumNumpy4D)[..., 0]
  
    '''
    gen_leptons = uproot.concatenate(root_file, ["gen_lep_pt", "gen_lep_eta", "gen_lep_phi", "gen_lep_mass", "gen_alep_pt", "gen_alep_eta", "gen_alep_phi", "gen_alep_mass"], library = "np")
    gen_leptons = np.stack((np.stack((gen_leptons["gen_lep_pt"], gen_leptons["gen_lep_eta"], gen_leptons["gen_lep_phi"], gen_leptons["gen_lep_mass"]), axis = -1),
                                    np.stack((gen_leptons["gen_alep_pt"], gen_leptons["gen_alep_eta"], gen_leptons["gen_alep_phi"], gen_leptons["gen_alep_mass"]), axis = -1))
                                 ).view([("pt", np.float64), ("eta", np.float64), ("phi", np.float64), ("mass", np.float64)]).view(vector.MomentumNumpy4D)[..., 0]
    '''
    '''
    b_jets = uproot.concatenate(root_file, ["b_pt", "b_eta", "b_phi", "b_mass", "ab_pt", "ab_eta", "ab_phi", "ab_mass"], library = "np")
    b_jets = np.stack((np.stack((b_jets["b_pt"], b_jets["b_eta"], b_jets["b_phi"], b_jets["b_mass"]), axis = -1),
                                    np.stack((b_jets["ab_pt"], b_jets["ab_eta"], b_jets["ab_phi"], b_jets["ab_mass"]), axis = -1))
                                 ).view([("pt", np.float32), ("eta", np.float32), ("phi", np.float32), ("mass", np.float32)]).view(vector.MomentumNumpy4D)[..., 0]
    '''
    b_pt     = uproot.concatenate(root_file, ["b_pt"], library = "np")["b_pt"]
    b_eta    = uproot.concatenate(root_file, ["b_eta"], library = "np")["b_eta"]
    b_phi    = uproot.concatenate(root_file, ["b_phi"], library = "np")["b_phi"]
    b_mass = uproot.concatenate(root_file, ["b_mass"], library = "np")["b_mass"]
    
    bbar_pt     = uproot.concatenate(root_file, ["bbar_pt"], library = "np")["bbar_pt"]
    bbar_eta    = uproot.concatenate(root_file, ["bbar_eta"], library = "np")["bbar_eta"]
    bbar_phi    = uproot.concatenate(root_file, ["bbar_phi"], library = "np")["bbar_phi"]
    bbar_mass = uproot.concatenate(root_file, ["bbar_mass"], library = "np")["bbar_mass"]

    b_jets = np.stack((np.stack((b_pt,      b_eta,      b_phi,      b_mass), axis = -1),
                            np.stack((bbar_pt, bbar_eta, bbar_phi, bbar_mass), axis = -1))
                         ).view([("pt", np.float32), ("eta", np.float32), ("phi", np.float32), ("mass", np.float32)]).view(vector.MomentumNumpy4D).reshape(2,-1)
    '''
    gen_b_jets = uproot.concatenate(root_file, ["gen_b_pt", "gen_b_eta", "gen_b_phi", "gen_b_mass", "gen_ab_pt", "gen_ab_eta", "gen_ab_phi", "gen_ab_mass"], library = "np")
    gen_b_jets = np.stack((np.stack((gen_b_jets["gen_b_pt"], gen_b_jets["gen_b_eta"], gen_b_jets["gen_b_phi"], gen_b_jets["gen_b_mass"]), axis = -1),
                                    np.stack((gen_b_jets["gen_ab_pt"], gen_b_jets["gen_ab_eta"], gen_b_jets["gen_ab_phi"], gen_b_jets["gen_ab_mass"]), axis = -1))
                                 ).view([("pt", np.float64), ("eta", np.float64), ("phi", np.float64), ("mass", np.float64)]).view(vector.MomentumNumpy4D)[..., 0]
    '''

    # Extract Predictions
    with h5py.File(prediction_file, "r") as f:
        data = f["gen_nu"][:, :1]

        pred_neutrino_components = np.stack((data[:, 0, 0], data[:, 0, 1]), axis = 0)

        pred_neutrinos = np.stack((pred_neutrino_components[..., 0], pred_neutrino_components[..., 1], pred_neutrino_components[..., 2], np.zeros_like(pred_neutrino_components[..., 0])), -1).view([("px", np.float32), ("py", np.float32), ("pz", np.float32), ("M", np.float32)]).view(vector.MomentumNumpy4D)[..., 0]
    '''
    gen_neutrinos = uproot.concatenate(root_file, ["gen_neu_pt", "gen_neu_eta", "gen_neu_phi", "gen_aneu_pt", "gen_aneu_eta", "gen_aneu_phi"], library = "np")
    gen_neutrinos = np.stack((np.stack((gen_neutrinos["gen_neu_pt"], gen_neutrinos["gen_neu_eta"], gen_neutrinos["gen_neu_phi"], np.zeros_like(gen_neutrinos["gen_neu_pt"])), axis = -1),
                                    np.stack((gen_neutrinos["gen_aneu_pt"], gen_neutrinos["gen_aneu_eta"], gen_neutrinos["gen_aneu_phi"], np.zeros_like(gen_neutrinos["gen_neu_pt"])), axis = -1))
                                 ).view([("pt", np.float64), ("eta", np.float64), ("phi", np.float64), ("mass", np.float64)]).view(vector.MomentumNumpy4D)[..., 0]

    gen_tops = uproot.concatenate(root_file, ["gen_top_pt", "gen_top_eta", "gen_top_phi", "gen_top_mass", "gen_atop_pt", "gen_atop_eta", "gen_atop_phi", "gen_atop_mass"], library = "np")
    gen_tops = np.stack((np.stack((gen_tops["gen_top_pt"], gen_tops["gen_top_eta"], gen_tops["gen_top_phi"], gen_tops["gen_top_mass"]), axis = -1),
                                    np.stack((gen_tops["gen_atop_pt"], gen_tops["gen_atop_eta"], gen_tops["gen_atop_phi"], gen_tops["gen_top_mass"]), axis = -1))
                                 ).view([("pt", np.float64), ("eta", np.float64), ("phi", np.float64), ("mass", np.float64)]).view(vector.MomentumNumpy4D)[..., 0]
    '''

    # Reconstruct Tops
    pred_w_bosons = leptons[[1, 0]].add(pred_neutrinos)
    pred_tops = b_jets.add(pred_w_bosons)
    ''''
    gen_w_bosons = gen_leptons[[1, 0]].add(gen_neutrinos)
    gen_tops_2 = gen_b_jets.add(gen_w_bosons)
    '''
    pred_ttbar_invariant_mass = pred_tops[0].add(pred_tops[1]).M
    '''gen_ttbar_invariant_mass = gen_tops[0].add(gen_tops[1]).M
    gen_ttbar_invariant_mass_2 = gen_tops_2[0].add(gen_tops_2[1]).M

    plt.hist(pred_ttbar_invariant_mass, 100, [200, 1200], alpha = 0.5, density = True)
    plt.hist(gen_ttbar_invariant_mass, 100, [200, 1200], alpha = 0.5, density = True)
    plt.hist(gen_ttbar_invariant_mass_2, 100, [200, 1200], alpha = 0.5, density = True)
    plt.show()

    plt.hist(pred_tops.M.flat, 60, [75, 275], alpha = 0.5, density = True)
    plt.hist(gen_tops.M.flat, 60, [175, 275], alpha = 0.5, density = True)
    plt.hist(gen_tops_2.M.flat, 60, [175, 275], alpha = 0.5, density = True)
    plt.show()'''

    gen_leptons = uproot.concatenate(root_file, ["gen_l_pt", "gen_l_eta", "gen_l_phi", "gen_l_mass", "gen_lbar_pt", "gen_lbar_eta", "gen_lbar_phi", "gen_lbar_mass"], library = "np")

    gen_leptons_pt = gen_leptons["gen_l_pt"]
    gen_leptons_eta = gen_leptons["gen_l_eta"]
    gen_leptons_phi = gen_leptons["gen_l_phi"]
    gen_leptons_mass = gen_leptons["gen_l_mass"]
    
    gen_anti_leptons_pt = gen_leptons["gen_lbar_pt"]
    gen_anti_leptons_eta = gen_leptons["gen_lbar_eta"]
    gen_anti_leptons_phi = gen_leptons["gen_lbar_phi"]
    gen_anti_leptons_mass = gen_leptons["gen_lbar_mass"]

    gen_b_jets = uproot.concatenate(root_file, ["gen_b_pt", "gen_b_eta", "gen_b_phi", "gen_b_mass", "gen_bbar_pt", "gen_bbar_eta", "gen_bbar_phi", "gen_bbar_mass"], library = "np")

    gen_b_jets_pt = gen_b_jets["gen_b_pt"]
    gen_b_jets_eta = gen_b_jets["gen_b_eta"]
    gen_b_jets_phi = gen_b_jets["gen_b_phi"]
    gen_b_jets_mass = gen_b_jets["gen_b_mass"]

    gen_anti_b_jets_pt = gen_b_jets["gen_bbar_pt"]
    gen_anti_b_jets_eta = gen_b_jets["gen_bbar_eta"]
    gen_anti_b_jets_phi = gen_b_jets["gen_bbar_phi"]
    gen_anti_b_jets_mass = gen_b_jets["gen_bbar_mass"]
        
    gen_neutrinos = uproot.concatenate(root_file, ["gen_nu_pt", "gen_nu_eta", "gen_nu_phi", "gen_nubar_pt", "gen_nubar_eta", "gen_nubar_phi"], library = "np")

    gen_neutrinos_pt = gen_neutrinos["gen_nu_pt"]
    gen_neutrinos_eta = gen_neutrinos["gen_nu_eta"]
    gen_neutrinos_phi = gen_neutrinos["gen_nu_phi"]
    #gen_neutrinos_mass = gen_neutrinos["gen_nu_mass"]

    gen_anti_neutrinos_pt = gen_neutrinos["gen_nubar_pt"]
    gen_anti_neutrinos_eta = gen_neutrinos["gen_nubar_eta"]
    gen_anti_neutrinos_phi = gen_neutrinos["gen_nubar_phi"]
    #gen_anti_neutrinos_mass = gen_neutrinos["gen_aneu_mass"]

    gen_tops = uproot.concatenate(root_file, ["gen_top_pt", "gen_top_eta", "gen_top_phi", "gen_top_mass", "gen_tbar_pt", "gen_tbar_eta", "gen_tbar_phi", "gen_tbar_mass"], library = "np")

    gen_tops_pt = gen_tops["gen_top_pt"]
    gen_tops_eta = gen_tops["gen_top_eta"]
    gen_tops_phi = gen_tops["gen_top_phi"]
    gen_tops_mass = gen_tops["gen_top_mass"]

    gen_anti_tops_pt = gen_tops["gen_tbar_pt"]
    gen_anti_tops_eta = gen_tops["gen_tbar_eta"]
    gen_anti_tops_phi = gen_tops["gen_tbar_phi"]
    gen_anti_tops_mass = gen_tops["gen_tbar_mass"]


    # Save Outputs
    reconstructed_events_file = uproot.recreate(output_file_name)
    reconstructed_events_file["NormalizingFlow"] = {"b_jets_pt": b_jets[0].pt,
                                                    "b_jets_eta": b_jets[0].eta,
                                                    "b_jets_phi": b_jets[0].phi,
                                                    "b_jets_mass": b_jets[0].mass,
                                                    "anti_b_jets_pt": b_jets[1].pt,
                                                    "anti_b_jets_eta": b_jets[1].eta,
                                                    "anti_b_jets_phi": b_jets[1].phi,
                                                    "anti_b_jets_mass": b_jets[1].mass,
                                                    "leptons_pt": leptons[0].pt,
                                                    "leptons_eta": leptons[0].eta,
                                                    "leptons_phi": leptons[0].phi,
                                                    "leptons_mass": leptons[0].mass,
                                                    "anti_leptons_pt": leptons[1].pt,
                                                    "anti_leptons_eta": leptons[1].eta,
                                                    "anti_leptons_phi": leptons[1].phi,
                                                    "anti_leptons_mass": leptons[1].mass,
                                                    "neutrinos_pt": pred_neutrinos[0].pt,
                                                    "neutrinos_eta": pred_neutrinos[0].eta,
                                                    "neutrinos_phi": pred_neutrinos[0].phi,
                                                    "anti_neutrinos_pt": pred_neutrinos[1].pt,
                                                    "anti_neutrinos_eta": pred_neutrinos[1].eta,
                                                    "anti_neutrinos_phi": pred_neutrinos[1].phi,
                                                    "tops_pt": pred_tops[0].pt,
                                                    "tops_eta": pred_tops[0].eta,
                                                    "tops_phi": pred_tops[0].phi,
                                                    "tops_mass": pred_tops[0].mass,
                                                    "anti_tops_pt": pred_tops[1].pt,
                                                    "anti_tops_eta": pred_tops[1].eta,
                                                    "anti_tops_phi": pred_tops[1].phi,
                                                    "anti_tops_mass": pred_tops[1].mass,
                                                    "ttbar_mass": pred_ttbar_invariant_mass,
                                                    "gen_b_jets_pt": gen_b_jets_pt,
                                                    "gen_b_jets_eta": gen_b_jets_eta,
                                                    "gen_b_jets_phi": gen_b_jets_phi,
                                                    "gen_b_jets_mass": gen_b_jets_mass,
                                                    "gen_anti_b_jets_pt": gen_anti_b_jets_pt,
                                                    "gen_anti_b_jets_eta": gen_anti_b_jets_eta,
                                                    "gen_anti_b_jets_phi": gen_anti_b_jets_phi,
                                                    "gen_anti_b_jets_mass": gen_anti_b_jets_mass,
                                                    "gen_leptons_pt": gen_leptons_pt,
                                                    "gen_leptons_eta": gen_leptons_eta,
                                                    "gen_leptons_phi": gen_leptons_phi,
                                                    "gen_leptons_mass": gen_leptons_mass,
                                                    "gen_anti_leptons_pt": gen_anti_leptons_pt,
                                                    "gen_anti_leptons_eta": gen_anti_leptons_eta,
                                                    "gen_anti_leptons_phi": gen_anti_leptons_phi,
                                                    "gen_anti_leptons_mass": gen_anti_leptons_mass,
                                                    "gen_neutrinos_pt": gen_neutrinos_pt,
                                                    "gen_neutrinos_eta": gen_neutrinos_eta,
                                                    "gen_neutrinos_phi": gen_neutrinos_phi,
                                                    "gen_anti_neutrinos_pt": gen_anti_neutrinos_pt,
                                                    "gen_anti_neutrinos_eta": gen_anti_neutrinos_eta,
                                                    "gen_anti_neutrinos_phi": gen_anti_neutrinos_phi,
                                                    "gen_tops_pt": gen_tops_pt,
                                                    "gen_tops_eta": gen_tops_eta,
                                                    "gen_tops_phi": gen_tops_phi,
                                                    "gen_tops_mass": gen_tops_mass,
                                                    "gen_anti_tops_pt": gen_anti_tops_pt,
                                                    "gen_anti_tops_eta": gen_anti_tops_eta,
                                                    "gen_anti_tops_phi": gen_anti_tops_phi,
                                                    "gen_anti_tops_mass": gen_anti_tops_mass,
                                                }

def nu2flows_extract_and_process_predictions_h5py(h5py_file, prediction_file, output_file_name):

    # Extract Particles
    with h5py.File(h5py_file, "r") as f:

        lepton_data = f["leptons"]
        b_loc = f.jets_indices == 0
        bjet = np.zeros((len(f.jets_indices), 1, 4))
        bjet[np.any(b_loc, axis=-1)] = f.jets[b_loc].mom[:, None]


        gen_data = f["truth_particles"]

        gen_lepton_components = np.stack((gen_data[:, 1:4, 8], gen_data[:, 1:4, 3]), axis = 0)
        gen_leptons = np.stack((gen_lepton_components[..., 0], gen_lepton_components[..., 1], gen_lepton_components[..., 2], np.zeros_like(gen_lepton_components[..., 0])), -1).view([("pt", np.float32), ("eta", np.float32), ("phi", np.float32), ("M", np.float32)]).view(vector.MomentumNumpy4D)[..., 0]

        gen_b_jet_components = np.stack((gen_data[:, 1:4, 1], gen_data[:, 1:4, 6]), axis = 0)
        gen_b_jets = np.stack((gen_b_jet_components[..., 0], gen_b_jet_components[..., 1], gen_b_jet_components[..., 2], np.zeros_like(gen_b_jet_components[..., 0])), -1).view([("pt", np.float32), ("eta", np.float32), ("phi", np.float32), ("M", np.float32)]).view(vector.MomentumNumpy4D)[..., 0]

        gen_neutrino_components = np.stack((gen_data[:, 1:4, 4], gen_data[:, 1:4, 9]), axis = 0)
        gen_neutrinos = np.stack((gen_neutrino_components[..., 0], gen_neutrino_components[..., 1], gen_neutrino_components[..., 2], np.zeros_like(gen_neutrino_components[..., 0])), -1).view([("pt", np.float32), ("eta", np.float32), ("phi", np.float32), ("M", np.float32)]).view(vector.MomentumNumpy4D)[..., 0]

        gen_top_components = np.stack((gen_data[:, 1:4, 0], gen_data[:, 1:4, 6]), axis = 0)
        gen_tops = np.stack((gen_top_components[..., 0], gen_top_components[..., 1], gen_top_components[..., 2], np.zeros_like(gen_top_components[..., 0])), -1).view([("pt", np.float32), ("eta", np.float32), ("phi", np.float32), ("M", np.float32)]).view(vector.MomentumNumpy4D)[..., 0]

    # Extract Predictions
    with h5py.File(prediction_file, "r") as f:
        data = f["gen_nu"][:, :1]

        pred_neutrino_components = np.stack((data[:, 0, 0], data[:, 0, 1]), axis = 0)

        pred_neutrinos = np.stack((pred_neutrino_components[..., 0], pred_neutrino_components[..., 1], pred_neutrino_components[..., 2], np.zeros_like(pred_neutrino_components[..., 0])), -1).view([("px", np.float32), ("py", np.float32), ("pz", np.float32), ("M", np.float32)]).view(vector.MomentumNumpy4D)[..., 0]
    '''
    gen_neutrinos = uproot.concatenate(root_file, ["gen_neu_pt", "gen_neu_eta", "gen_neu_phi", "gen_aneu_pt", "gen_aneu_eta", "gen_aneu_phi"], library = "np")
    gen_neutrinos = np.stack((np.stack((gen_neutrinos["gen_neu_pt"], gen_neutrinos["gen_neu_eta"], gen_neutrinos["gen_neu_phi"], np.zeros_like(gen_neutrinos["gen_neu_pt"])), axis = -1),
                                    np.stack((gen_neutrinos["gen_aneu_pt"], gen_neutrinos["gen_aneu_eta"], gen_neutrinos["gen_aneu_phi"], np.zeros_like(gen_neutrinos["gen_neu_pt"])), axis = -1))
                                 ).view([("pt", np.float64), ("eta", np.float64), ("phi", np.float64), ("mass", np.float64)]).view(vector.MomentumNumpy4D)[..., 0]

    gen_tops = uproot.concatenate(root_file, ["gen_top_pt", "gen_top_eta", "gen_top_phi", "gen_top_mass", "gen_atop_pt", "gen_atop_eta", "gen_atop_phi", "gen_atop_mass"], library = "np")
    gen_tops = np.stack((np.stack((gen_tops["gen_top_pt"], gen_tops["gen_top_eta"], gen_tops["gen_top_phi"], gen_tops["gen_top_mass"]), axis = -1),
                                    np.stack((gen_tops["gen_atop_pt"], gen_tops["gen_atop_eta"], gen_tops["gen_atop_phi"], gen_tops["gen_top_mass"]), axis = -1))
                                 ).view([("pt", np.float64), ("eta", np.float64), ("phi", np.float64), ("mass", np.float64)]).view(vector.MomentumNumpy4D)[..., 0]
    '''

    # Reconstruct Tops
    pred_w_bosons = leptons[[1, 0]].add(pred_neutrinos)
    pred_tops = b_jets.add(pred_w_bosons)
    ''''
    gen_w_bosons = gen_leptons[[1, 0]].add(gen_neutrinos)
    gen_tops_2 = gen_b_jets.add(gen_w_bosons)
    '''
    pred_ttbar_invariant_mass = pred_tops[0].add(pred_tops[1]).M
    '''gen_ttbar_invariant_mass = gen_tops[0].add(gen_tops[1]).M
    gen_ttbar_invariant_mass_2 = gen_tops_2[0].add(gen_tops_2[1]).M

    plt.hist(pred_ttbar_invariant_mass, 100, [200, 1200], alpha = 0.5, density = True)
    plt.hist(gen_ttbar_invariant_mass, 100, [200, 1200], alpha = 0.5, density = True)
    plt.hist(gen_ttbar_invariant_mass_2, 100, [200, 1200], alpha = 0.5, density = True)
    plt.show()

    plt.hist(pred_tops.M.flat, 60, [75, 275], alpha = 0.5, density = True)
    plt.hist(gen_tops.M.flat, 60, [175, 275], alpha = 0.5, density = True)
    plt.hist(gen_tops_2.M.flat, 60, [175, 275], alpha = 0.5, density = True)
    plt.show()'''

    # Save Outputs
    reconstructed_events_file = uproot.recreate(output_file_name)
    reconstructed_events_file["NormalizingFlow"] = {"b_jets_pt": b_jets[0].pt,
                                                    "b_jets_eta": b_jets[0].eta,
                                                    "b_jets_phi": b_jets[0].phi,
                                                    "b_jets_mass": b_jets[0].mass,
                                                    "anti_b_jets_pt": b_jets[1].pt,
                                                    "anti_b_jets_eta": b_jets[1].eta,
                                                    "anti_b_jets_phi": b_jets[1].phi,
                                                    "anti_b_jets_mass": b_jets[1].mass,
                                                    "leptons_pt": leptons[0].pt,
                                                    "leptons_eta": leptons[0].eta,
                                                    "leptons_phi": leptons[0].phi,
                                                    "leptons_mass": leptons[0].mass,
                                                    "anti_leptons_pt": leptons[1].pt,
                                                    "anti_leptons_eta": leptons[1].eta,
                                                    "anti_leptons_phi": leptons[1].phi,
                                                    "anti_leptons_mass": leptons[1].mass,
                                                    "neutrinos_pt": pred_neutrinos[0].pt,
                                                    "neutrinos_eta": pred_neutrinos[0].eta,
                                                    "neutrinos_phi": pred_neutrinos[0].phi,
                                                    "anti_neutrinos_pt": pred_neutrinos[1].pt,
                                                    "anti_neutrinos_eta": pred_neutrinos[1].eta,
                                                    "anti_neutrinos_phi": pred_neutrinos[1].phi,
                                                    "tops_pt": pred_tops[0].pt,
                                                    "tops_eta": pred_tops[0].eta,
                                                    "tops_phi": pred_tops[0].phi,
                                                    "tops_mass": pred_tops[0].mass,
                                                    "anti_tops_pt": pred_tops[1].pt,
                                                    "anti_tops_eta": pred_tops[1].eta,
                                                    "anti_tops_phi": pred_tops[1].phi,
                                                    "anti_tops_mass": pred_tops[1].mass,
                                                    "ttbar_mass": pred_ttbar_invariant_mass,
                                                    "gen_b_jets_pt": gen_b_jets[0].pt,
                                                    "gen_b_jets_eta": gen_b_jets[0].eta,
                                                    "gen_b_jets_phi": gen_b_jets[0].phi,
                                                    "gen_b_jets_mass": gen_b_jets[0].mass,
                                                    "gen_anti_b_jets_pt": gen_b_jets[1].pt,
                                                    "gen_anti_b_jets_eta": gen_b_jets[1].eta,
                                                    "gen_anti_b_jets_phi": gen_b_jets[1].phi,
                                                    "gen_anti_b_jets_mass": gen_b_jets[1].mass,
                                                    "gen_leptons_pt": gen_leptons[0].pt,
                                                    "gen_leptons_eta": gen_leptons[0].eta,
                                                    "gen_leptons_phi": gen_leptons[0].phi,
                                                    "gen_leptons_mass": gen_leptons[0].mass,
                                                    "gen_anti_leptons_pt": gen_leptons[1].pt,
                                                    "gen_anti_leptons_eta": gen_leptons[1].eta,
                                                    "gen_anti_leptons_phi": gen_leptons[1].phi,
                                                    "gen_anti_leptons_mass": gen_leptons[1].mass,
                                                    "gen_neutrinos_pt": gen_neutrinos.pt[0],
                                                    "gen_neutrinos_eta": gen_neutrinos.eta[0],
                                                    "gen_neutrinos_phi": gen_neutrinos.phi[0],
                                                    "gen_anti_neutrinos_pt": gen_neutrinos[1].pt,
                                                    "gen_anti_neutrinos_eta": gen_neutrinos[1].eta,
                                                    "gen_anti_neutrinos_phi": gen_neutrinos[1].phi,
                                                    "gen_tops_pt": gen_tops.pt[0],
                                                    "gen_tops_eta": gen_tops.eta[0],
                                                    "gen_tops_phi": gen_tops.phi[0],
                                                    "gen_tops_mass": gen_tops.mass[0],
                                                    "gen_anti_tops_pt": gen_tops[1].pt,
                                                    "gen_anti_tops_eta": gen_tops[1].eta,
                                                    "gen_anti_tops_phi": gen_tops[1].phi,
                                                    "gen_anti_tops_mass": gen_tops[1].mass,
                                                }

#distributions_file = "./KinReco_input.root"
#pred_file_folder_paper = r"C:\Users\thatf\OneDrive\Documents\Simulations\nu2flowsPurdue\nu2flows_models\example_model\outputs" 
#pred_file_folder_purdue = r"D:\nu2flows_data\full_delphes_model_0_299\outputs"
#pred_file_folder = r"C:\Users\thatf\OneDrive\Documents\Simulations\nu2flowsPurdue\models\example_model\outputs"
#output_folder = "/Users/thatf/OneDrive/Documents/Simulations/CMS Research/Full_Delphes_Predictions"
#
#
##extract_and_process_predictions_h5py("/Users/thatf/OneDrive/Documents/Simulations/nu2flows/nu2flows_data/test.h5", "/Users/thatf/OneDrive/Documents/Simulations/nu2flows/nu2flows_models/example_model/outputs/test.h5", "testing")
#
#
#for s in range(7, 8):
#    start_index = 100 * s
#    stop_index = 100 * (s + 1)
#    
#    files_7 = [f"/Users/thatf/OneDrive/Documents/Simulations/CMS Research/Full_Delphes/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_minitrees_{i}.root:Step7" for i in range(start_index, stop_index)]
#    #files_8 = [f"/Users/thatf/OneDrive/Documents/Simulations/CMS Research/Full_Delphes/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_minitrees_{i}.root:Step8" for i in range(start_index, stop_index)]
#
#    extract_and_process_predictions(files_7, f"{pred_file_folder}/full_delphes_{start_index}_{stop_index - 1}.h5", distributions_file, f"{output_folder}/full_delphes_step7_transfer_{start_index}_{stop_index - 1}_predictions.root")