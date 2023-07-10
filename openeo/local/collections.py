import datetime
import logging
import xarray as xr
import numpy as np
import rioxarray
from pathlib import Path
from pyproj import Transformer
from typing import Dict, List, Tuple, Union
import h5py
import fsspec
import requests
from bs4 import BeautifulSoup
import os 

_log = logging.getLogger(__name__)

def _get_dimension(dims: dict, candidates: List[str]):
    for name in candidates:
        if name in dims:
            return name
    error = f'Dimension matching one of the candidates {candidates} not found! The available ones are {dims}. Please rename the dimension accordingly and try again. This local collection will be skipped.'
    raise Exception(error)
        
def _get_netcdf_zarr_metadata(files):
  data_list = []
  for file_path in files:
    print(file_path)
    if file_path.endswith('.zarr'):
      data_list.append(xr.open_dataset(file_path,chunks={},engine='zarr'))
    elif file_path.endswith('.grib2'):
      data_list.append(xr.open_dataset(file_path,chunks={},engine='rasterio'))
    else:
      if file_path.startswith('http'):
        file_path_bytes = file_path + '#mode=bytes'
      data_list.append(xr.open_dataset(file_path_bytes,chunks={}))
  data = xr.concat(data_list,dim='time')
  try:
      t_dim = _get_dimension(data.dims, ['t', 'time', 'temporal', 'DATE'])
  except:
      t_dim = None
  try:
      x_dim = _get_dimension(data.dims, ['x', 'X', 'lon', 'longitude'])
      y_dim = _get_dimension(data.dims, ['y', 'Y', 'lat', 'latitude'])
  except Exception as e:
      _log.warning(e)
      raise Exception(f'Error creating metadata for {file_path}') from e
  split_info = file_path.split('/')
  metadata = {}
  metadata['stac_version'] = '1.0.0-rc.2'
  metadata['type'] = 'Collection'
  metadata['id'] = split_info[-2]
  data_attrs_lowercase = [x.lower() for x in data.attrs]
  data_attrs_original  = [x for x in data.attrs]
  data_attrs = dict(zip(data_attrs_lowercase,data_attrs_original))
  
  if 'title' in data_attrs_lowercase:
    metadata['title'] = data.attrs[data_attrs['title']]
  else:
      metadata['title'] = split_info[-1]
  if 'description' in data_attrs_lowercase:
      metadata['description'] = data.attrs[data_attrs['description']]
  else:
    try:
      headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
      url_readme = '/'.join(file_path.split('/')[:-1]) + '/' +  'read-me.txt'
      info = requests.get(url_readme,headers).content.decode()
      metadata['description'] = info
    except:
      metadata['description'] = ''
  if 'license' in data_attrs_lowercase:
      metadata['license'] = data.attrs[data_attrs['license']]
  else:
      metadata['license'] = ''
  providers = [{'name':split_info[-2],
                'roles':['producer'],
                'url':'https://www.cptec.inpe.br/'}]
  if 'providers' in data_attrs_lowercase:
      providers[0]['name'] = data.attrs[data_attrs['providers']]
      metadata['providers'] = providers
  elif 'institution' in data_attrs_lowercase:
      providers[0]['name'] = data.attrs[data_attrs['institution']]
      metadata['providers'] = providers
  else:
      metadata['providers'] = providers
  if 'links' in data_attrs_lowercase:
      metadata['links'] = data.attrs[data_attrs['links']]
  else:
      metadata['links'] = ''
  x_min = data[x_dim].min().item(0)
  x_max = data[x_dim].max().item(0)
  y_min = data[y_dim].min().item(0)
  y_max = data[y_dim].max().item(0)

  crs_present = False
  bands = list(data.data_vars)
  if 'crs' in bands:
      bands.remove('crs')
      crs_present = True
  extent = {}
  if crs_present:
      if 'crs_wkt' in data.crs.attrs:
          transformer = Transformer.from_crs(data.crs.attrs['crs_wkt'],'epsg:4326')
          lat_min,lon_min = transformer.transform(x_min,y_min)
          lat_max,lon_max = transformer.transform(x_max,y_max)               
          extent['spatial'] = {'bbox': [[lon_min, lat_min, lon_max, lat_max]]}

  if t_dim is not None:
      t_min = str(data[t_dim].min().values)
      t_max = str(data[t_dim].max().values)
      extent['temporal'] = {'interval': [[t_min,t_max]]}

  metadata['extent'] = extent

  t_dimension = {}
  if t_dim is not None:
      t_dimension = {t_dim: {'type': 'temporal', 'extent':[t_min,t_max]}}

  x_dimension = {x_dim: {'type': 'spatial','axis':'x','extent':[x_min,x_max]}}
  y_dimension = {y_dim: {'type': 'spatial','axis':'y','extent':[y_min,y_max]}}
  if crs_present:
      if 'crs_wkt' in data.crs.attrs:
          x_dimension[x_dim]['reference_system'] = data.crs.attrs['crs_wkt']
          y_dimension[y_dim]['reference_system'] = data.crs.attrs['crs_wkt']

  b_dimension = {}
  if len(bands)>0:
      b_dimension = {'bands': {'type': 'bands', 'values':bands}}

  metadata['cube:dimensions'] = {**t_dimension,**x_dimension,**y_dimension,**b_dimension}

  return data,metadata

def _get_geotiff_metadata(file_path):
    data = rioxarray.open_rasterio(file_path.as_posix(),chunks={},band_as_variable=True)
    file_path = file_path.as_posix()
    try:
        t_dim = _get_dimension(data.dims, ['t', 'time', 'temporal', 'DATE'])
    except:
        t_dim = None
    try:
        x_dim = _get_dimension(data.dims, ['x', 'X', 'lon', 'longitude'])
        y_dim = _get_dimension(data.dims, ['y', 'Y', 'lat', 'latitude'])
    except Exception as e:
        _log.warning(e)
        raise Exception(f'Error creating metadata for {file_path}') from e
        
    metadata = {}
    metadata['stac_version'] = '1.0.0-rc.2'
    metadata['type'] = 'Collection'
    metadata['id'] = file_path
    data_attrs_lowercase = [x.lower() for x in data.attrs]
    data_attrs_original  = [x for x in data.attrs]
    data_attrs = dict(zip(data_attrs_lowercase,data_attrs_original))
    if 'title' in data_attrs_lowercase:
        metadata['title'] = data.attrs[data_attrs['title']]
    else:
        metadata['title'] = file_path
    if 'description' in data_attrs_lowercase:
        metadata['description'] = data.attrs[data_attrs['description']]
    else:
        metadata['description'] = ''
    if 'license' in data_attrs_lowercase:
        metadata['license'] = data.attrs[data_attrs['license']]
    else:
        metadata['license'] = ''
    providers = [{'name':'',
                 'roles':['producer'],
                 'url':''}]
    if 'providers' in data_attrs_lowercase:
        providers[0]['name'] = data.attrs[data_attrs['providers']]
        metadata['providers'] = providers
    elif 'institution' in data_attrs_lowercase:
        providers[0]['name'] = data.attrs[data_attrs['institution']]
        metadata['providers'] = providers
    else:
        metadata['providers'] = providers
    if 'links' in data_attrs_lowercase:
        metadata['links'] = data.attrs[data_attrs['links']]
    else:
        metadata['links'] = ''
    x_min = data[x_dim].min().item(0)
    x_max = data[x_dim].max().item(0)
    y_min = data[y_dim].min().item(0)
    y_max = data[y_dim].max().item(0)

    crs_present = False
    coords = list(data.coords)
    if 'spatial_ref' in coords:
        # bands.remove('crs')
        crs_present = True
    bands = []
    for d in data.data_vars:
        data_attrs_lowercase = [x.lower() for x in data[d].attrs]
        data_attrs_original  = [x for x in data[d].attrs]
        data_attrs = dict(zip(data_attrs_lowercase,data_attrs_original))
        if 'description' in data_attrs_lowercase:
            bands.append(data[d].attrs[data_attrs['description']])
        else:
            bands.append(d)
    extent = {}
    if crs_present:
        if 'crs_wkt' in data.spatial_ref.attrs:
            transformer = Transformer.from_crs(data.spatial_ref.attrs['crs_wkt'], 'epsg:4326')
            lat_min,lon_min = transformer.transform(x_min,y_min)
            lat_max,lon_max = transformer.transform(x_max,y_max)
            extent['spatial'] = {'bbox': [[lon_min, lat_min, lon_max, lat_max]]}

    if t_dim is not None:
        t_min = str(data[t_dim].min().values)
        t_max = str(data[t_dim].max().values)
        extent['temporal'] = {'interval': [[t_min,t_max]]}

    metadata['extent'] = extent

    t_dimension = {}
    if t_dim is not None:
        t_dimension = {t_dim: {'type': 'temporal', 'extent':[t_min,t_max]}}

    x_dimension = {x_dim: {'type': 'spatial','axis':'x','extent':[x_min,x_max]}}
    y_dimension = {y_dim: {'type': 'spatial','axis':'y','extent':[y_min,y_max]}}
    if crs_present:
        if 'crs_wkt' in data.spatial_ref.attrs:
            x_dimension[x_dim]['reference_system'] = data.spatial_ref.attrs['crs_wkt']
            y_dimension[y_dim]['reference_system'] = data.spatial_ref.attrs['crs_wkt']

    b_dimension = {}
    if len(bands)>0:
        b_dimension = {'bands': {'type': 'bands', 'values':bands}}

    metadata['cube:dimensions'] = {**t_dimension,**x_dimension,**y_dimension,**b_dimension}

    return metadata

def _get_local_collections(local_collections_path):

  def find_files_with_extensions(folder_path,walk, extensions):
    found_files = []
    
    if walk:
      for root, dirs, files in os.walk(folder_path):
          for file in files:
              if any(file.endswith(ext) for ext in extensions):
                  found_files.append(os.path.join(root, file))
    else:
      for file in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file)) and any(file.endswith(ext) for ext in extensions):
          found_files.append(os.path.join(folder_path, file))
    
    return found_files


  def get_INMET_MERGE_datasets(url):
    try:
      headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
      # Send a GET request to the webpage
      response = requests.get(url, headers=headers)
      # Check if the response was successful (status code 200)
      response.raise_for_status()
    except requests.exceptions.HTTPError as err:
      print(f"HTTP error occurred: {err}")
    except requests.exceptions.RequestException as err:
      print(f"An error occurred: {err}")

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, "html.parser")

    # Find all <a> tags with href attribute
    links = soup.find_all("a", href=True)

    # Extract the last part of the link and remove the extension
    data_dict = {}
    ursl = []
    for link in links:
      href = link["href"]
      path = Path(href)
      name = path.stem
      if href.endswith(".nc"):
        data_dict[name] = href
        ursl.append(url + '/' + href)
      elif href.endswith(".grib2"):
        data_dict[name] = href
        ursl.append(url + '/' + href)
    return ursl

  if isinstance(local_collections_path,str):
    local_collections_path = [local_collections_path]
  local_collections_list = []
  for flds in local_collections_path:
    if flds.startswith('http'):
      local_collections_netcdf_zarr = get_INMET_MERGE_datasets(flds)
    #for local_file in local_collections_netcdf_zarr:
    else:
      local_collections_netcdf_zarr = find_files_with_extensions(flds,False,['.nc','.zarr', '.grib2'])
    try:
        data,metadata = _get_netcdf_zarr_metadata(local_collections_netcdf_zarr)
        local_collections_list.append(metadata)
    except Exception as e:
        _log.error(e)
        continue
    '''
    local_collections_geotiffs = [p for p in Path(flds).rglob('*') if p.suffix in  ['.tif','.tiff']]
    for local_file in local_collections_geotiffs: 
        try:
            metadata = _get_geotiff_metadata(local_file)
            local_collections_list.append(metadata)
        except Exception as e:
            _log.error(e)
            continue
    '''
  local_collections_dict = {'collections':local_collections_list}
    
  return data,local_collections_dict