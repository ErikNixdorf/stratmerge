import stratmerge
from pathlib import Path
import numpy as np

#first test
config = stratmerge.StratMerge.read_config(
    Path('data','examples','ex1','ex1_stratmerge.yml')
)
model = stratmerge.StratMerge(conf=config)
model.merge_stratigraphic_layers()
model = model.calculate_hydrogeoproperty_distributions()
model.get_layer_stats()
model.save()

haha
#%%second one

config = stratmerge.StratMerge.read_config(
    Path('data','examples','ex1','ex1_stratmerge.yml')
)
model2 = stratmerge.StratMerge(conf=config)
av_model = model2.generate_vertical_averages(base_layer = model2.config['generate_planar_model']['base_layer'],
                                                base_layer_thickness = model2.config['generate_planar_model']['base_layer_thickness'])

av_model.get_layer_stats()

#%% third one
config = stratmerge.StratMerge.read_config(
    Path('data','examples','ex1','ex1_stratmerge.yml')
)
model = stratmerge.StratMerge(conf=config)
model.calculate_hydrogeoproperty_distributions()
extruded_mesh = model.extrude_layers()
extruded_mesh.array_names

unique_values,counts = np.unique(extruded_mesh['MaterialIDs'],return_counts=True)