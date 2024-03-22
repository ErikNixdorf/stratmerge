from pathlib import Path
import pytest
from stratmerge import stratmerge


@pytest.fixture(scope="module")
def merge_layers():
    config = stratmerge.StratMerge.read_config(
        Path(Path(__file__).parents[2],'data','examples','ex1','ex1_stratmerge.yml')
    )
    model = stratmerge.StratMerge(conf=config)
    model.merge_stratigraphic_layers()
    model = model.calculate_hydrogeoproperty_distributions()
    return model


@pytest.fixture(scope="module")
def vertical_averaging():
    config = stratmerge.StratMerge.read_config(
        Path(Path(__file__).parents[2],'data','examples','ex1','ex1_stratmerge.yml')
    )
    model = stratmerge.StratMerge(conf=config)
    av_model = model.generate_vertical_averages(base_layer = model.config['generate_planar_model']['base_layer'],
                                                    base_layer_thickness = model.config['generate_planar_model']['base_layer_thickness'])

    return av_model

@pytest.fixture(scope="module")
def extrude_layers():
    config = stratmerge.StratMerge.read_config(
        Path(Path(__file__).parents[2],'data','examples','ex1','ex1_stratmerge.yml')
    )
    model = stratmerge.StratMerge.Model(conf=config)
    model.calculate_hydrogeoproperty_distributions()
    extruded_mesh = model.extrude_layers()
    
    return extruded_mesh

