# Experiment code for "Response Time Improves Choice Prediction and Function Estimation for Gaussian Process Models of Perception and Preferences"

**Important note**: the modeling code is currently under review for inclusion in [AEPsych](https://aepsych.org), specifically here: https://github.com/facebookresearch/aepsych/pull/361. If you would like to use the models in your work, you should only need the code in that PR. 

This repo only contains code and data needed to reproduce most of the experiments in the paper (except the recommender system experiments, for which the data is proprietary). A prerequisite is installing AEPsych from the linked PR. 

Note that we are in the process of restructuring this repo to accomodate the refactoring that was needed for inclusion in AEPsych, so things might still move around a bit. If you are reading this and needing support before everything is fully cleaned up or while something is missing, please contact the authors. 

* `data/` contains the new gait dataset as well as the psychophysics dataset with response times added. 
* `run_csf_experiments.py` is needed to make the data for the CSF results (figure 6, left)
* `run_gait_preference_experiments.py` is needed to make the data for the robot gait results (figure 7, right). 
* `stacked_model.py` implements the Stacked RT-choice GP model. 

We are still cleaning up the plotting code and ancillary figures, they should be available in this repo within the coming days / weeks. 

This code is licensed under the license mentioned in LICENSE.md file at the root of this repo. 