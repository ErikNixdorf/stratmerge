"""
A Python Module to simplify stratigraphic geological layers by merging and to use 
equivalent values for their hydrogeological properties 
"""

from pathlib import Path
from util_functions import read_ascii_grid, write_ascii_grid,asciigrid_to_datarray
from util_functions import harmonic_weight, arithmetic_weight, has_duplicates, are_values_equal
import numpy as np
import yaml
import numpy as np
from copy import copy, deepcopy
import xarray as xr
from typing import Callable
import pandas as pd
from itertools import combinations
import secrets
from typing import Union
import sys
import os
import pyvista as pv
#%% some related functions
def call_extrusion_tool(mesh_path,
                        layerlist_path,
                        output_path,
                        tool_path,
                        min_acceptable_thickness=0):
    """
    Calls the OGS TOOL to mesh


    Parameters
    ----------
    mesh_path : TYPE
        DESCRIPTION.
    layerlistpath : TYPE
        DESCRIPTION.
    outputpath : TYPE
        DESCRIPTION.
    min_acceptable_thickness : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    None.

    """
    Path.mkdir(output_path.parent,exist_ok=True)
    command_str=f'{tool_path} -i {mesh_path} -r {layerlist_path} -o {output_path} -t {min_acceptable_thickness}'
    print('Extrude 2D Mesh to 3D...',end='')
    os.system(command_str)
    print('...done')


def dataarray_to_ascii(dataarray: xr.DataArray, 
                       metadata: dict,
                       no_data_value: Union[int, float] = -9999, 
                       output_path: str = None,
                       flipud: bool = True) -> None:
    """
    Convert xarray DataArray to ASCII grid format.

    Parameters
    ----------
    dataarray : xr.DataArray
        The DataArray to be converted to ASCII.
    no_data_value : Union[int, float]
        The value to be used as the NODATA value.
    output_path : str
        The path where the ASCII grid will be saved.

    Returns
    -------
    None.
    """
    # Convert DataArray to numpy array
    numpy_array = dataarray.to_numpy()
    if flipud:
        numpy_array = np.flipud(numpy_array)

    # Extract metadata from DataArray attributes
    metadata['nodata_value'] = no_data_value
    
    # Replace NaN values with the specified no_data_value
    numpy_array[np.isnan(numpy_array)] = no_data_value
    
    # Write the ASCII grid using the provided function
    write_ascii_grid(numpy_array, metadata, output_path)
    

def merge_dataset_layers(dataset: xr.Dataset, 
                             layers_to_merge: list, 
                             merged_layer: str, 
                             merge_type: str,
                             flag_bedding_plane_orientation : dict,
                             ) -> xr.Dataset:
    """
     Merge layers in a dataset according to specified rules.
    
     Parameters
     ----------
     dataset : xr.Dataset
         The dataset containing the layers to be merged.
     layers_to_merge : list
         List of layer names to be merged.
     merged_layer : str
         Name of the new merged layer.
     merge_type : str, optional
         The type of merge operation to perform ('max' or 'min'), by default 'max'.
    flag_bedding_plane_orientation; dict : 
        keywords: 'activate'
        keywords: Threshold : relative deviation
     Returns
     -------
     xr.Dataset
         The modified dataset after merging the specified layers.
    
     Raises
     ------
     ValueError
         If the merge type is not implemented.
    """
    # define the merge functions
    merge_functions = {
    'max': lambda ds: ds.max(dim='layer'),
    'min': lambda ds: ds.min(dim='layer')
    }

    if merge_type not in merge_functions:
        raise ValueError(f'Merge Type of type {merge_type} not implemented')
    
    merge_function = merge_functions[merge_type]

    #start algorithm
    ds_subset = dataset[layers_to_merge].to_array(dim='layer')
    da_subset = merge_function(ds_subset)
    da_subset.name = merged_layer
    da_subset.attrs = dataset[layers_to_merge[0]].attrs
    
    if flag_bedding_plane_orientation['activate']:
        def compute_slope(array,cell_size=100,calculate_aspect = False):
            """
            
            

            Parameters
            ----------
            array : TYPE
                DESCRIPTION.
            varargs : TYPE
                DESCRIPTION.
            calculate_aspect : TYPE, optional
                DESCRIPTION. The default is True.

            Returns
            -------
            None.

            """
            # check dimenstion
            if len(array.dims)!=2:
                raise ValueError('Array Dimension does not equal 2, only 2D array are supported')
            
            gradient = np.gradient(array,cell_size)
            # slope_degrees = ATAN (rise_run) * 57.29578
            #  rise_run = √ ([dz/dx]2 + [dz/dy]2]
            # for details, https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-analyst/how-slope-works.htm
            # 
            slope = np.arctan(np.sqrt(gradient[0]**2 + gradient[1]**2)) * np.pi/180
            
            if calculate_aspect:
            #https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-analyst/how-aspect-works.htm
             aspect = 57.29578 * np.arctan2( ([dz/dy], -[dz/dx])
            
        ds_subset.groupby('layer').apply(lambda x: compute_slope(x.squeeze(dim='layer',drop=True),cell_size))
        print('not ok')
          
    # drop the layers which have been mergedfrom top
    dataset = dataset.drop_vars(layers_to_merge)
    
    #add the merged layer
    dataset[merged_layer]=da_subset
    
    return dataset

class StratMerge:
    def __init__(self,conf:dict):
        """
        loads config and reads tata


        Parameters
        ----------
        config_path : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.config = conf
        self.paths: dict = {"root": Path(__file__).parents[1]}

        #generate a specific identifier:
        self.identifier= secrets.token_hex(nbytes=4)
        
        self._build_wd()

        
        #%% read the files
        self.stratlayers=dict()
        stratlayer_dir = self.paths['input_dir'] /'layers'

        
        #get the names of all layers
        self.layer_names = list(self.config['stratigraphic_layers'].keys())
        stratlayer_filenames= [file.name for file in stratlayer_dir.iterdir() if file.is_file()]
        
        #we check the layer_names by splitting scheme
        file_layer_names = set([file.split('_')[0] for file in stratlayer_filenames])
        
        #if there is a difference we raise a Error
        missing_files = file_layer_names.difference(self.layer_names)
        
        if len(missing_files) > 0:
            raise ValueError(f'Ascii files missing for layers {missing_files}')
            
        # extract the order of layers
        layer_order={}
        for layer in self.layer_names:
            layer_order[layer] = int(self.config['stratigraphic_layers'][layer]['layer_id'])
        
        #sort the dictionary
        self.layer_order=pd.Series(layer_order)
            
        
        #%% Extract the different defined hydrogeological properties and their type
        hydrogeoproperty_names = list(
            [   inner_key
                for outer_key in self.config['stratigraphic_layers']
                for inner_key in self.config['stratigraphic_layers'][outer_key]['properties']
            ]
        )
        # check which of the properties is a scalar and which is a tensor
        property_is_scalar= list([
                self.config['stratigraphic_layers'][outer_key]['properties'][inner_key]['is_isotropic']
                for outer_key in self.config['stratigraphic_layers']
                for inner_key in self.config['stratigraphic_layers'][outer_key]['properties']
            ])
        
        #merge it together
        properties_with_type= {key: value for key, value in zip(hydrogeoproperty_names, property_is_scalar)}
        
        if len(properties_with_type) > len(set(hydrogeoproperty_names)):
            raise ValueError('Hydrogeological property has to defined same type scalar/not scalar for each layer the same')
        
        self.hydrogeoproperty_is_scalar = properties_with_type
        self.hydrogeoproperty_names =list(properties_with_type.keys())
        

        self.layer_stats = pd.DataFrame()
        
        #%% We create three different Datasets covering top bottom and so on of each layer 
        # Create dictionaries for tops, bases, and thicks
        tops = {}
        bases = {}
        thicks = {}
        
        #we loop trough our data
        for layer_name in self.layer_names:
            
            #generate the dataarrays
            da_top = asciigrid_to_datarray(stratlayer_dir / f'{layer_name}_top.asc',name= layer_name,nodata_is_nan=True)
            da_base = asciigrid_to_datarray(stratlayer_dir / f'{layer_name}_base.asc',name= layer_name,nodata_is_nan=True)
            #compute the thickness
            da_thick = da_top-da_base
            da_thick.attrs = da_base.attrs
            
            tops[layer_name] = da_top
            bases[layer_name] = da_base
            thicks[layer_name] = da_thick
            
        self.ds_tops= xr.Dataset(tops)
        self.ds_bases = xr.Dataset(bases)
        self.ds_thicks = xr.Dataset(thicks)
        #get general attributes
        self.da_attrs=da_base.attrs
            
    @staticmethod
    def read_config(config_file_path : Path
             ) -> dict:
        """Creates a dictionary out of a YAML file."""
        with open(config_file_path) as c:
            conf = yaml.safe_load(c)
        return conf
    
    
    def _build_wd(self):
        """Builds the working directory. Reads paths from configuration files and creates output directories."""

        self.paths["input_dir"] = Path(self.paths["root"],
                                       self.config['data_io']['data_dir'])

        self.paths["output_dir"] = Path(self.paths["root"],
                                        self.config['data_io']['output_dir'],
                                        self.config['info']['model_name'],self.identifier)
        
        if any(value for value in self.config['data_io']['save_subsets'].values()):
            self.paths["output_dir"].mkdir(parents=True, exist_ok=True)
                
    def weight_property(self, property_name: str, calc_function: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> xr.Dataset:
        """
        A function that weights all properties by the thickness of each layer.

        Parameters
        ----------
        property_name : str
            The name of the property to be weighted.

        calc_function : callable
            A function that takes the property values array and the layer thickness array as arguments
            and returns the calculated weight array.

        Returns
        -------
        xr.Dataset
            A dataset containing weighted property values for each layer and statistical measure.
        """

        #%% We generate dataarray for each parameter
        ones_array =np.ones((self.ds_tops.x.size,self.ds_tops.y.size))

        weight_layers_dict={}
        for layer in self.layer_names:
            
            initstats=True
            for stat in ['min','mean','max']:
                try:
                    property_value = self.config['stratigraphic_layers'][layer]['properties'][property_name][stat]
                except Exception as e:
                    print(e)
                    print('use np.nan instead')
                    property_value= np.nan
                #create an array

                prop_array = float(property_value) * ones_array

                #generate the weight array
                weight_array = calc_function(prop_array,self.ds_thicks[layer].values)
                
                #make a data_array out of it
                coords={'x':np.arange(weight_array.shape[0]),
                        'y':np.arange(weight_array.shape[1]),
                        'stat':[stat],
                        }
                
                weight_dataarray = xr.DataArray(np.expand_dims(weight_array, 2), 
                                     dims=list(coords.keys()), 
                                     coords=coords, 
                                     name=layer
                                     )
                if initstats:
                    stat_weight_da = weight_dataarray
                    initstats= False
                else:
                    stat_weight_da = xr.concat([stat_weight_da,weight_dataarray],dim = 'stat')
            
            #we stack the dataarrays together to a have dataset comprising all layers
            weight_layers_dict[layer]=stat_weight_da
        
        #convert to dataset
        ds_attrs = self.da_attrs
        ds_attrs['name'] = property_name
        weight_layers_ds= xr.Dataset(weight_layers_dict,attrs=ds_attrs)
            

        return weight_layers_ds
    
    def weight_hydrogeoproperties(self) -> None:
        """
        Top-level function to calculate the thickness-weighted information on each hydrogeological property.
        
        This function calculates the thickness-weighted information for each hydrogeological property,
        including arithmetic and harmonic weights where applicable.

        Returns
        -------
        None
            This function modifies the class instance by adding property weight information.
        """
        # we loop trough each property of each layer and generate the dataset
        self.property_weights = {}
        for property_name in self.hydrogeoproperty_names:
            
            #we calculate the arithmetic mean as default
            self.property_weights[property_name]={'arithmetic_weight':
                                                  self.weight_property(property_name,arithmetic_weight)
                                                  }
            if self.hydrogeoproperty_is_scalar[property_name] is False:
                print(f'{property_name} is a tensor, we calculate harmonic weight as well')
                self.property_weights[property_name]['harmonic_weight'] = self.weight_property(property_name,harmonic_weight)

                    
        return None


        
    
    def merge_stratigraphic_layers(self,merge_rules={}):
        """
        Merge stratigraphic layers by combining layer extents and updating weighted properties.
    
        This function takes the stratigraphic merging configuration from the class attribute `config`,
        merges layer extents according to the specified rules, and updates associated weighted properties.
    
        Returns
        -------
        None.
    
        Raises
        ------
        ValueError
            If a stratigraphic layer is part of more than one merged layer.
    
        """

        #load the merge setting
        if not merge_rules:
            print('read merge rules from yaml user input')
            
            merge_rules = self.config['merge_stratigraphiclayers']['merge_rules']
            
        all_layers_to_merge = [layer for layers in merge_rules.values() for layer in layers]
        
        if has_duplicates(all_layers_to_merge):
            raise ValueError('Each stratigraphic layer can only be part of one merged layer, check data input')
    
        # check whether the weighted hydrogeoproperties have been calculated already
        if not hasattr(self, 'property_weights'):
            print('Calculating property weights first...')
            self.weight_hydrogeoproperties()
        
        for merged_layer,layers_to_merge in merge_rules.items():
            print(f'Creating merged layer {merged_layer}...', end='')
            
            # Merge top  and base layers
            bedding_dip_conf = self.config['merge_stratigraphiclayers']['flag_bedding_plane_orientation']
            self.ds_tops = merge_dataset_layers(self.ds_tops,
                                           layers_to_merge,
                                           merged_layer,
                                           merge_type='max',
                                           flag_bedding_plane_orientation = bedding_dip_conf)
            
            self.ds_bases = merge_dataset_layers(self.ds_bases,
                                           layers_to_merge,
                                           merged_layer,
                                           merge_type='min',
                                           flag_bedding_plane_orientation = bedding_dip_conf)
            
            #we update the weighted properties as well by just sum then up
            for prop in self.property_weights:
                for weight_type, weight_data in self.property_weights[prop].items():
                    da_sublayers_prop = weight_data[layers_to_merge].to_array(dim='layer')
                    #sum it up
                    da_sublayer_prop = da_sublayers_prop.sum(dim='layer')
                    #we fix a bug in xarray that nan+nan is zero https://github.com/pydata/xarray/issues/5693
                    da_sublayer_prop= xr.where(da_sublayer_prop == 0, np.nan, da_sublayer_prop)
                    da_sublayer_prop.name = merged_layer
                    #drop the weights of layers which need to be merged
                    self.property_weights[prop][weight_type] = weight_data.drop_vars(layers_to_merge)
                    self.property_weights[prop][weight_type][merged_layer]=da_sublayer_prop
            
            # update the layer names
            self.layer_names=list(set(self.layer_names).difference(layers_to_merge))
            self.layer_names.append(merged_layer)
            #update the layer order
            merged_layer_id=self.layer_order[layers_to_merge].min()
            self.layer_order.drop(layers_to_merge,inplace=True)
            #get the minimum difference larger than the merge_order_id
            self.layer_order[self.layer_order-merged_layer_id>0]= self.layer_order - len(layers_to_merge) + 1
            #add the merged layer
            self.layer_order=pd.concat([self.layer_order,pd.Series({merged_layer:merged_layer_id})])
            print('...done')
        #sort the layer order
        self.layer_order.sort_values(inplace=True)
            
        #finally we update the thicknesses
        thicks_attrs = self.ds_thicks.attrs
        self.ds_thicks = self.ds_tops - self.ds_bases
        self.ds_thicks.attrs = thicks_attrs

                    
    def calculate_hydrogeoproperty_distributions(self):
        """
        Calculate hydrogeoproperty distributions using property weights and layer thicknesses.
    
        This function calculates hydrogeoproperty distributions based on previously calculated
        property weights and layer thicknesses.
    
        Returns
        -------
        None.
    
        """
        # Check if property_weights have been calculated, and calculate them if not
        if not hasattr(self, 'property_weights'):
            print('Calculating property weights first...')
            self.weight_hydrogeoproperties()
            
        # Initialize a dictionary to store hydrogeoproperty layers
        self.hydrogeoproperty_layers = {}
        
        # Iterate through properties and weight types
        for property_name, weight_type_dict in self.property_weights.items():
            for weight_type, weight_data in weight_type_dict.items():
                if weight_type == 'arithmetic_weight':
                    # Calculate property distribution using arithmetic weights
                    ds_property = weight_data / self.ds_thicks
                    ds_property.attrs['name'] = f'{property_name}_x'                    
                elif weight_type == 'harmonic_weight':
                    # Calculate property distribution using harmonic weights
                    ds_property = self.ds_thicks / weight_data
                    ds_property.attrs['name'] = f'{property_name}_z'
                    
                # Store calculated property distribution
                self.hydrogeoproperty_layers[ds_property.attrs['name']] = ds_property
        
        return deepcopy(self)
                
    def generate_vertical_averages(self,
                                   base_layer='layer',
                                   base_layer_thickness=5):
        """
        Generate a single layer of average hydrogeological properties as well as thicknesses.
    
        Returns
        -------
        v_average_instance : instance of the class
            A new instance of the class with averaged properties.
    
        """
        # we generate a new instance of the class
        v_average_instance = deepcopy(self)
        
        # we adapt the base layer thickness
        
        print('Correct the thickness of the base layer')
        #we correct the thickness of the base_layer
        if base_layer not in v_average_instance.layer_names or base_layer is not None :
            print(f'{base_layer} not in the stratigraphic layer system or not provided, we calculate with defaulr')
        else:
            print(f'Correct the thickness of the {base_layer}')
            #adapt the base_layer_thickness
            v_average_instance.ds_thicks[base_layer] = base_layer_thickness * v_average_instance.ds_thicks[base_layer] / v_average_instance.ds_thicks[base_layer]
            #we have to adapt the bottom
            v_average_instance.ds_bases[base_layer] = v_average_instance.ds_tops[base_layer] - base_layer_thickness
        
        
        print('A 2D vertically averaged planar model is generated') 
        # we define new merge rules based on all layers
        merge_rule={'single_layer':v_average_instance.layer_names}
        v_average_instance.merge_stratigraphic_layers(merge_rule)
        v_average_instance.calculate_hydrogeoproperty_distributions()
        
        #we can remove the ones in vertical direction as we have horizontal flow only
        props = list(v_average_instance.hydrogeoproperty_layers.keys())
        for property_name in props:
            if property_name.endswith('_z'):
                del v_average_instance.hydrogeoproperty_layers[property_name]
        
        return v_average_instance

    def generate_stratigraphic_ensemble(self,
                                        basic_layer='pt',
                                        vertical_averaging = True,
                                        all_layers_only = True,
                                        stats_to_use = ['mean']):
        """
        Generate stratigraphic ensembles by creating various layer combinations.

        Parameters
        ----------
        basic_layer : str, optional
            The basic layer to include in combinations, default is 'pt'.
        vertical_averaging : bool, optional
            Whether to generate vertical averages, default is True.

        Returns
        -------
        None.
        """
        
        #%% very first we have to check whether layers have to be merged because
        # it influences the number of iterations
        # Perform merging and weight calculation
        if self.config['merge_stratigraphiclayers']['activate']:
            self.merge_stratigraphic_layers()
        if not hasattr(self, 'property_weights'):
            print('Calculating property weights first...')
            self.weight_hydrogeoproperties()
        
        #%% Generate the possible layer combinations
        #%% get all combinations of the layers
        layer_combinations = []
        # Generate combinations of 1, 2, 3, and 4 entries
        if all_layers_only:
            layers_min = len(self.layer_names)
        else:
            layers_min = 1
        for r in range(layers_min, len(self.layer_names) + 1):
            for combo in combinations(self.layer_names, r):
                layer_combinations.append(combo)
        #remove all combinations which do not include the homogeneous layer
        layer_combinations=[layer_combi for layer_combi in layer_combinations if basic_layer in layer_combi]
        # before we loop we make a copy of the original_istance
        full_model = deepcopy(self)
        
        #prepare the the setup csv
        ensemble_stats=pd.DataFrame()
        #%% we loop trough the layer_combinations, generate new instances of the class and replace some data there
        for layer_combination in layer_combinations:
            #generate a binary dictionary which layer is in the combination
            layer_exists_dic = {entry: 1 if entry in list(layer_combination) else 0 for entry in self.layer_names}
            #generate a new instance of the class
            combiner=deepcopy(full_model)

            # now we reduce all data by removing layers
            combiner.ds_bases= combiner.ds_bases[list(layer_combination)]
            combiner.ds_tops= combiner.ds_tops[list(layer_combination)]
            combiner.ds_thicks= combiner.ds_thicks[list(layer_combination)]
            
            #also fix the weights
            combination_prop_weights = copy(combiner.property_weights)
            for prop in combination_prop_weights:
                for weight_type, weight_data in combination_prop_weights[prop].items():

                    weight_data = weight_data[list(layer_combination)]

                    #overwrite
                    combiner.property_weights[prop][weight_type] = weight_data
            
            # calculate the hydrogeological properties
            combiner.calculate_hydrogeoproperty_distributions()
            
            #overwrite the meta_data
            #layer_names
            combiner.layer_names=list(layer_combination)
            #layer_order
            combiner.layer_order[list(layer_combination)]
            #reset to zero
            combiner.layer_order[:]=range(len(combiner.layer_order))
            
            # if requested we generate the vertical average
            if vertical_averaging:
                combiner=combiner.generate_vertical_averages()
            
            # we save each ensemble writing out all statistics as independent layers
            combiner.output_dir=combiner.output_dir / combiner.identifier / 'stratigraphic_ensembles'
            for stat in stats_to_use:
                combiner.identifier = secrets.token_hex(nbytes=4)
                
                
                #save layers
                run_stats = combiner.save(save_ascii=True,
                                              save_nc=False, 
                                              write_statistics =True,
                                              select_specific_statistic_only= stat,
                                              )

                #remove the stat information from columns
                col_new = ['_'.join(col.split('_')[:-1]) if len(col.split('_')) > 1 else col for col in run_stats.columns]
                run_stats.columns=col_new
                run_stats['stat_mode'] = stat
                #we add the layer information to the average stats
                if vertical_averaging:
                    run_stats.index=[combiner.identifier]
                    layer_binary = pd.DataFrame({combiner.identifier:layer_exists_dic}).T
                    run_stats=pd.concat([run_stats,layer_binary],axis=1)
                    run_stats['layer']= combiner.layer_names
                else:
                    run_stats = run_stats.reset_index().rename(columns={'variable':'layer'})
                    run_stats.index = len(run_stats)*[combiner.identifier]
                #get ensemble stats
                ensemble_stats=pd.concat([ensemble_stats,run_stats])
        
        #write ensemble stats
        ensemble_stats.to_csv(combiner.output_dir / 'ensemble_stats.csv')
        return ensemble_stats
                
    def get_layer_stats(self):
        """
        Calculate statistics for each layer of hydrogeological properties.
        
        This method computes statistics such as thickness, minimum, and mean for different hydrogeological properties 
        across different layers.
        
        Returns:
            layer_stats (pandas.DataFrame): DataFrame containing statistics for each layer.
        """
        
        
        #lets get the layer_thickness
        self.layer_stats['thickness']=self.ds_thicks.mean().to_array()
        self.layer_stats.index = list(self.ds_thicks.data_vars)
        # now we calculate all the min and and mean properties
        for property_name,prop_data in self.hydrogeoproperty_layers.items():

            stats_to_calculate = prop_data.coords['stat'].values
            # calculate the spatial mean
            for stat in stats_to_calculate:
                data_subset = prop_data.sel(stat=stat)
                average_property = data_subset.mean().to_array().reset_coords('stat',drop=True)
                average_property.name = f'{property_name}_{stat}'
                average_property = average_property.to_dataframe()
                
                self.layer_stats=pd.concat([self.layer_stats,average_property],axis=1)
        return self.layer_stats

    def save(self,
             save_ascii=True,
             save_nc=True, 
             identifier = None
             ):
        """
        Save data to ASCII and NetCDF files, and calculate statistics.

        Args:
            save_ascii (bool, optional): Flag to save ASCII files for layers. Default is True.
            save_nc (bool, optional): Flag to save NetCDF files. Default is True.
            write_statistics (bool, optional): Flag to write statistics to CSV file. Default is True.
            select_specific_statistic_only (str, optional): Calculate statistics for a specific statistic only. Default is None.

        Returns:
            pd.DataFrame: DataFrame containing calculated average statistics.
        """
        if identifier is None:
            output_dir = self.paths['output_dir']
        else:
            output_dir = self.paths['output_dir'].parent/Path(identifier)
        

        # Save layer boundary data in ASCII format
        if save_ascii:
            ascii_dir = Path(output_dir)/Path('ascii_files')
            ascii_dir.mkdir(parents=True,exist_ok=True)
            for layer in self.layer_names:
                for data_type in ['top', 'base', 'thick']:
                    file_name = f'{layer}_{data_type}.asc'
                    data = self.__getattribute__(f'ds_{data_type}s')[layer].copy()
                    dataarray_to_ascii(data, 
                                       self.da_attrs, 
                                       no_data_value = self.config['data_io']['nodata_value'], 
                                       output_path = ascii_dir / file_name
                                       )
         
        # Save layer boundary data in NetCDF format
        if save_nc:
            nc_dir = Path(output_dir)/Path('nc_files')
            nc_dir.mkdir(parents=True,exist_ok=True)
            for data_type in ['tops', 'bases', 'thicks']:
                data = self.__getattribute__('ds_' + data_type)
                data.to_netcdf(nc_dir / f'layers_{data_type[:-1]}.nc')
        
        # now we write out the properties
        for property_name,prop_data in self.hydrogeoproperty_layers.items():
            if save_nc:
                prop_data.to_netcdf(nc_dir / f'{property_name}.nc')
                
            # calculate the spatial mean
            for stat in prop_data.coords['stat'].values:
                data_subset = prop_data.sel(stat=stat)
            
                #more complicated are the ascii files
                if save_ascii:
                    filename_suffix=f'{stat}'
                    
                    for layer in self.layer_names:
                        ascii_filename= f'{layer}_{property_name}_{filename_suffix}.asc'
                        dataarray_to_ascii(data_subset[layer].copy(),
                                           self.da_attrs,
                                           no_data_value = self.config['data_io']['nodata_value'],
                                           output_path = ascii_dir /  Path(ascii_filename)
                                           )
                    
        #write out the statistics
        self.layer_stats.to_csv(output_dir / 'stats_layer_averages.csv')
    
    
    def extrude_layers(self):
        """
        Extrude 2D layers to create a 3D mesh.
    
        This function performs the following steps:
        1. Checks if ASCII layers are saved; if not, it saves them.
        2. Loads the Digital Elevation Model (DEM) and ensures compatibility with ASCII layers.
        3. Writes out the DEM.
        4. Generates a dictionary of layers to add, including DEM and base layer.
        5. Rearranges the order of layers for extrusion.
        6. Writes the layer list to a text file.
        7. Runs the extrusion tool.
        8. Optionally removes the soil layer from the extruded mesh.
    
        Returns:
            None
        """
        # Extract configuration options
        extrusion_opts = self.config['build_3d_mesh']
        #add paths to executable
        if not Path(extrusion_opts['path_to_extruder']).is_absolute():
            path_to_extrusion_module = self.paths['input_dir'] / extrusion_opts['path_to_extruder']
        else:
            path_to_extrusion_module = (extrusion_opts['path_to_extruder'])
        #2d mesh
        if not Path(extrusion_opts['path_to_planar_mesh']).is_absolute():
            path_to_mesh = self.paths['input_dir'] / extrusion_opts['path_to_planar_mesh']
        else:
            path_to_mesh = (extrusion_opts['path_to_planar_mesh'])
        #path to dem
        if not Path(extrusion_opts['path_to_dem']).is_absolute():
            path_to_dem = self.paths['input_dir'] / extrusion_opts['path_to_dem']
        else:
            path_to_dem = (extrusion_opts['path_to_dem'])
            
        
        minimum_thickness = extrusion_opts['minimum_layer_depth']
        remove_soil_layer = extrusion_opts['remove_soil_layer']
        #%% first we check whether the ascii directory is not empty
        layer_dir = Path(self.paths['output_dir'] /'ascii_files')
        
        if 'ascii_files' not in self.paths['output_dir'].iterdir():
            print('Ascii layers are not saved yet,repeat saving')
            self.save(save_ascii=True,identifier=deepcopy(self.identifier))
        
        #%% we load the dem and save it to the sam folder
        dem = asciigrid_to_datarray(path_to_dem,name= 'dem',
                                    nodata_is_nan=True)
        
        # if the attributes are not identical we cant match it
        if not are_values_equal(dem.attrs,self.da_attrs):
            raise ValueError('The Extention or Cell Size of the DEM is not equal to ascii layers, adapt preprocessing')
        
        # we write out the dem
        dataarray_to_ascii(dem, 
                           dem.attrs,
                           no_data_value = self.config['data_io']['nodata_value'],
                           output_path = layer_dir / 'dem.asc',
                           )
        
        
        #%% generate the dictionary of the dem and the top of first level
        layers_to_add = {'dem':{'order':0,
                           'layer_path' : layer_dir / 'dem.asc'}
                    }
        base_layer = self.layer_order.idxmax()
        layers_to_add.update( {f'{base_layer}_base' : {'order':self.layer_order.max()+1,
                                                  'layer_path' : layer_dir / f'{base_layer}_base.asc'}
                            }
                          )
        
        
        #%% next we rearrange the order
        extrusion_order=self.layer_order.copy(deep=True)
        extrusion_order.name = 'order'
        extrusion_order = extrusion_order + 1
        # generate the names of the ascii layers
        layer_path_series= extrusion_order.index.to_series().apply(lambda x: f'{layer_dir / (x+"_top.asc")}')
        layer_path_series.name = 'layer_path'
        extrusion_order = pd.concat([extrusion_order,layer_path_series],axis = 1)
        #add the top layers
        extrusion_order = pd.concat([extrusion_order,
                                     pd.DataFrame(layers_to_add).T])
        #we sort
        extrusion_order = extrusion_order.sort_values(by='order')
        
        #%% write layer list
        f = open(layer_dir / 'layer_list.txt', 'w')
        for layer in extrusion_order['layer_path']:
            f.write(str(layer) + "\n")
        f.close()
        
        
        #%% run the extrusion:
        output_mesh_name = path_to_mesh.parts[-1].split('.')[0]+'_extruded.vtu'
        extruded_dir = layer_dir.parent /'extruded'
        Path.mkdir(extruded_dir,exist_ok = True)
        output_path = extruded_dir / output_mesh_name
        
        call_extrusion_tool(path_to_mesh,
                            layer_dir / 'layer_list.txt',
                            output_path,
                            Path(path_to_extrusion_module),
                            min_acceptable_thickness=minimum_thickness)
        
        mesh = pv.read(output_path)
        #%% remove the soil layer
        if remove_soil_layer:
            mesh = mesh.threshold(value=(0,mesh['MaterialIDs'].max()-1),scalars='MaterialIDs')
            mesh.save(output_path)
        
        return mesh
        
        
    
def main(config_path=None):
    """
    Default method to run the code and perform various tasks based on configuration.

    Returns:
    new_instance: An updated instance of the class.
    geomodel_stats: Statistics of the generated geological model.
    """
    if config_path:
        configuration = StratMerge.read_config(config_path)
    else:
        print('No config_file given, take default')
        configuration = StratMerge.read_config(Path(Path(__file__).parents[1], "data/examples/stratmerge.yml"))

    
    model = StratMerge(conf=configuration)
    config = model.config
    generate_planar_model = config['generate_planar_model']['activate']
    build_3d_mesh = config['build_3d_mesh']['activate']
    merge_stratigraphic_layers = config['merge_stratigraphiclayers']['activate']
    save_model_statistics = config['data_io']['save_subsets']['save_model_statistics']
    save_nc_layers = config['data_io']['save_subsets']['save_layer_nc']
    save_ascii_layers = build_3d_mesh or config['data_io']['save_subsets']['save_layer_ascii']
            
    if generate_planar_model and build_3d_mesh:
        raise ValueError('Only one of generate_planar_model or build_3d_mesh can be activated, not both')

    if generate_planar_model:        
        new_instance = model.generate_vertical_averages(base_layer = model.config['generate_planar_model']['base_layer'],
                                                        base_layer_thickness = model.config['generate_planar_model']['base_layer_thickness'])
    else:
        if merge_stratigraphic_layers:
            model.merge_stratigraphic_layers()
        new_instance = model.calculate_hydrogeoproperty_distributions()
    #compute statistics
    geomodel_stats = new_instance.save(
            save_ascii=save_ascii_layers,
            save_nc=save_nc_layers,
            write_statistics=save_model_statistics
        )
    
    if build_3d_mesh:
        print('Extruding the mesh')
        model.extrude_layers()

    return new_instance, geomodel_stats
        
  

if __name__ == "__main__":
    if len(sys.argv) > 1:
        cfg_file = sys.argv[1]  # Get the first command-line argument
        print(f'Reading {cfg_file}')
        main(path_to_config_file=cfg_file)
    else:
        main()  
            
            
            
        
        
    