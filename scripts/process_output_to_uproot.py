
import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

from pathlib import Path
import h5py
import numpy as np
from dotmap import DotMap

from utils.plotting import plot_multi_hists_2
from src.datamodules.physics import Mom4Vec
from src.utils import read_dilepton_file
import uproot
import awkward as ak
import vector
from src.utils import read_dilepton_file
from src.datamodules.physics import Mom4Vec
import matplotlib.pyplot as plt

# Calculate the weight of the smear
def get_weight_mlb_awkward(b_jets, leptons, mbl_probability_function, mbl_probability_bin_edges) :

    lepton_b_system_invariant_mass = leptons.add(b_jets).M

    counts = ak.num(lepton_b_system_invariant_mass)

    bins = np.digitize(ak.flatten(lepton_b_system_invariant_mass).to_numpy(), mbl_probability_bin_edges) - 1
    in_range_bins = np.where(bins == 100, 0, bins)

    mlb_probabilities = np.where(bins < mbl_probability_function.size, mbl_probability_function[in_range_bins], 0)
    
    return ak.unflatten(mlb_probabilities, counts) / 100000000

def get_best_leptons_b_jets_lep_alep_with_filter(jet_btag, jet_pt, jet_eta, jet_phi, jet_mass, leptons, distributions_root_file):

    # Distributions for smearing and weighting
    distributionFile = uproot.open(distributions_root_file)

    d_mblProbability = distributionFile["KinReco_mbl_true_step0"].to_numpy()

    # Get the number of events
    event_number = len(jet_pt)

    # Make vector recognize awkward
    vector.register_awkward()

    # Create jet fourvectors
    single_jets = ak.zip(({"pt" : jet_pt, "eta" : jet_eta, "phi" : jet_phi, "mass" : jet_mass}), with_name = "Momentum4D")

    # Get all permutations of jets
    jet_combos = ak.combinations(single_jets, 2, axis = 1)
    b_jets = ak.concatenate((jet_combos["0"][np.newaxis], jet_combos["1"][np.newaxis]))

    # Get all permutations of btags
    btag_combos = ak.combinations(jet_btag, 2, axis = 1)
    
    # Filter out events based on pt, eta, and btag
    jet_pt_check = np.logical_not(ak.any(b_jets.pt < 30, 0))
    jet_eta_check = np.logical_not(ak.any(abs(b_jets.eta) > 2.4, 0))
    jet_btag_check = np.logical_not(np.logical_and(btag_combos["0"] == 0, btag_combos["1"] == 0))
    b_jet_exists_filter = np.logical_and(np.logical_and(jet_pt_check, jet_eta_check), jet_btag_check)

    # Apply pt, eta, and btag filters to jets so they can be said to be b jets
    b_jets = b_jets[:, :event_number][:, b_jet_exists_filter[:event_number]]
    btag_combos = btag_combos[:len(btag_combos)][b_jet_exists_filter[:len(btag_combos)]]

    # Create a mask based on deltaR between leptons and b jets for all pairs of leptons, anti leptons, and jets
    deltaR_filter = np.logical_not(ak.any(ak.any(leptons[[[0, 1], [1, 0]]].deltaR(b_jets[np.newaxis, :, :]) < 0.4, 0), 0))
    b_jets[:, :len(deltaR_filter)][:, deltaR_filter[:len(deltaR_filter)]] # DON'T MATCH LEPTON SIZE NOW!!!!!!!!!!!!!!!!!!!!
    btag_combos = btag_combos[:len(btag_combos)][deltaR_filter[:len(btag_combos)]]

    # Calculate the mass of lepton-bjet system weights
    mlb_weight_1 = get_weight_mlb_awkward(b_jets[0], leptons[1], d_mblProbability[0], d_mblProbability[1]) * get_weight_mlb_awkward(b_jets[1], leptons[0], d_mblProbability[0], d_mblProbability[1])
    mlb_weight_2 = get_weight_mlb_awkward(b_jets[1], leptons[1], d_mblProbability[0], d_mblProbability[1]) * get_weight_mlb_awkward(b_jets[0], leptons[0], d_mblProbability[0], d_mblProbability[1])
    
    # Process results based on if exists higher mlb double jet else highest mlb single jet
    # The jet with the highest mlb weight is the b jet, second highest is the anti b jet
    is_double_jet = np.logical_and(btag_combos["0"] != 0, btag_combos["1"] != 0)

    # if any 2 b jet, take best 2 b jet else take top b jet pair
    picked_jets_1 = ak.where(ak.any(is_double_jet, axis = -1), ak.firsts(ak.argsort(mlb_weight_1.mask[is_double_jet], ascending = False)), ak.firsts(ak.argsort(mlb_weight_1, ascending = False)))[..., np.newaxis]
    picked_jets_2 = ak.where(ak.any(is_double_jet, axis = -1), ak.firsts(ak.argsort(mlb_weight_2.mask[is_double_jet], ascending = False)), ak.firsts(ak.argsort(mlb_weight_2, ascending = False)))[..., np.newaxis]

    # See if jet combo 1 or jet combo 2 is better
    jet_combo_1_better = (mlb_weight_1[picked_jets_1] >= mlb_weight_2[picked_jets_2])[:, 0].to_numpy()

    # Pick the best b jets and anti b jets and put them into one array
    best_b_jets = np.stack((ak.drop_none(ak.where(jet_combo_1_better, b_jets[0][:event_number][picked_jets_1[:event_number]], b_jets[1][:event_number][picked_jets_2[:event_number]])).to_numpy(),
                            ak.drop_none(ak.where(jet_combo_1_better, b_jets[1][:event_number][picked_jets_2[:event_number]], b_jets[0][:event_number][picked_jets_1[:event_number]])).to_numpy()), axis = 0).view(vector.MomentumNumpy4D)[..., 0]
 
    valid_events = ak.all(ak.num(b_jets, 2) != 0, 0).to_numpy()

    return best_b_jets, valid_events

def extract_and_process_predictions_h5py(h5py_file, prediction_file, output_file_name, distribution_root_file):

    file_data = read_dilepton_file(Path(h5py_file))

    leptons = file_data.leptons
    leptons = np.stack((leptons.pt, leptons.eta, leptons.phi, leptons.mass), -1).view([("pt", np.float32), ("eta", np.float32), ("phi", np.float32), ("M", np.float32)]).view(vector.MomentumNumpy4D)[..., 0]
    leptons = leptons[..., 0].T

    jets = file_data.jets

    b_tags = np.logical_or(file_data.jets_indices == 0, file_data.jets_indices == 1)
    b_tags_indices = np.zeros_like(b_tags, dtype = np.float32)
    b_tags_indices[:] = np.arange(b_tags.shape[1])
    
    b_tags = ak.drop_none(np.where(b_tags_indices < file_data.njets[..., np.newaxis], b_tags, -1).astype(np.float32))

    b_tags = b_tags[b_tags != -1]

    jets_pt = ak.Array(np.squeeze(jets.pt))[b_tags != -1]
    jets_eta = ak.Array(np.squeeze(jets.eta))[b_tags != -1]
    jets_phi = ak.Array(np.squeeze(jets.phi))[b_tags != -1]
    jets_mass = ak.Array(np.squeeze(jets.mass))[b_tags != -1]

    b_jets, valid_events = get_best_leptons_b_jets_lep_alep_with_filter(b_tags, jets_pt, jets_eta, jets_phi, jets_mass, leptons, distribution_root_file)

    leptons = leptons[:, valid_events]

    # Extract Particles
    gen_data = file_data["truth_particles"]

    gen_lepton_components = np.stack((gen_data[:, 8, 1:5], gen_data[:, 3, 1:5]), axis = 0)[:, valid_events]
    gen_leptons = np.stack((gen_lepton_components[..., 0], gen_lepton_components[..., 1], gen_lepton_components[..., 2], gen_lepton_components[..., 3]), -1).view([("pt", np.float64), ("eta", np.float64), ("phi", np.float64), ("M", np.float64)]).view(vector.MomentumNumpy4D)[..., 0]

    gen_b_jet_components = np.stack((gen_data[:, 1, 1:5], gen_data[:, 6, 1:5]), axis = 0)[:, valid_events]
    gen_b_jets = np.stack((gen_b_jet_components[..., 0], gen_b_jet_components[..., 1], gen_b_jet_components[..., 2], gen_b_jet_components[..., 3]), -1).view([("pt", np.float64), ("eta", np.float64), ("phi", np.float64), ("M", np.float64)]).view(vector.MomentumNumpy4D)[..., 0]

    gen_neutrino_components = np.stack((gen_data[:, 4, 1:5], gen_data[:, 9, 1:5]), axis = 0)[:, valid_events]
    gen_neutrinos = np.stack((gen_neutrino_components[..., 0], gen_neutrino_components[..., 1], gen_neutrino_components[..., 2], gen_neutrino_components[..., 3]), -1).view([("pt", np.float64), ("eta", np.float64), ("phi", np.float64), ("M", np.float64)]).view(vector.MomentumNumpy4D)[..., 0]

    gen_top_components = np.stack((gen_data[:, 0, 1:5], gen_data[:, 5, 1:5]), axis = 0)[:, valid_events]
    gen_tops = np.stack((gen_top_components[..., 0], gen_top_components[..., 1], gen_top_components[..., 2], gen_top_components[..., 3]), -1).view([("pt", np.float64), ("eta", np.float64), ("phi", np.float64), ("M", np.float64)]).view(vector.MomentumNumpy4D)[..., 0]

    # Extract Predictions
    with h5py.File(prediction_file, "r") as f:
        data = f["gen_nu"][:, :1]

        pred_neutrino_components = np.stack((data[:, 0, 0], data[:, 0, 1]), axis = 0)[:, valid_events]

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
    
    gennu_w_bosons = leptons[[1, 0]].add(gen_neutrinos)
    gen_tops_2 = b_jets.add(gennu_w_bosons)
   
    pred_ttbar_invariant_mass = pred_tops[0].add(pred_tops[1]).M
    gen_ttbar_invariant_mass = gen_tops[0].add(gen_tops[1]).M
    gennu_ttbar_invariant_mass = gen_tops_2[0].add(gen_tops_2[1]).M

    plt.hist(pred_ttbar_invariant_mass, 100, [200, 1200], alpha = 0.5, density = True)
    plt.hist(gen_ttbar_invariant_mass, 100, [200, 1200], alpha = 0.5, density = True)
    plt.hist(gennu_ttbar_invariant_mass, 100, [200, 1200], alpha = 0.5, density = True)
    plt.show()

    plt.hist(pred_tops.M.flat, 60, [75, 275], alpha = 0.5, density = True)
    plt.hist(gen_tops.M.flat, 60, [75, 275], alpha = 0.5, density = True)
    plt.hist(gen_tops_2.M.flat, 60, [75, 275], alpha = 0.5, density = True)
    plt.show()

    plt.hist(pred_w_bosons.M.flat, 60, [50, 110], alpha = 0.5, density = True)
    plt.hist(gen_leptons[[1, 0]].add(gen_neutrinos).M.flat, 60, [50, 110], alpha = 0.5, density = True)
    plt.hist(gennu_w_bosons.M.flat, 60, [50, 110], alpha = 0.5, density = True)
    plt.yscale("log")
    plt.show()

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

distribution_root_file = r"C:\Users\thatf\OneDrive\Documents\Simulations\CMS Research/KinReco_input.root"

extract_and_process_predictions_h5py("/Users/thatf/OneDrive/Documents/Simulations/nu2flows/nu2flows_data/test.h5", "/Users/thatf/OneDrive/Documents/Simulations/nu2flows/nu2flows_models/example_model/outputs/test.h5", "testing", distribution_root_file)
