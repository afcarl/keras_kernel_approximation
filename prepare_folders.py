import os

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'

if not os.path.exists("output"):
    os.makedirs("output")


for folder_name in ["models_baseline", "models_mlp", "models_rff", "models_rnnrff", "pred"]:
    if not os.path.exists(os.path.join("output", folder_name)):
        os.makedirs(os.path.join("output", folder_name))
