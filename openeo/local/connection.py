import datetime
import logging
from pathlib import Path
import xarray as xr
from typing import Dict, List, Tuple, Union, Callable, Optional, Any, Iterator

import requests
from bs4 import BeautifulSoup

from openeo.metadata import CollectionMetadata, SpatialDimension, TemporalDimension, BandDimension, Band
from openeo.internal.graph_building import PGNode, as_flat_graph
from openeo.rest.datacube import DataCube
from openeo.internal.jupyter import VisualDict, VisualList
from openeo.local.collections import _get_local_collections, _get_netcdf_zarr_metadata, _get_geotiff_metadata
from openeo.local.processing import PROCESS_REGISTRY
from openeo_pg_parser_networkx.graph import OpenEOProcessGraph

class LocalConnection():
    """
    Connection to no backend, for local processing.
    """

    def __init__(self,local_collections_path: Union[str,List]):
        """
        Constructor of LocalConnection.

        :param local_collections_path: String or list of strings, path to the folder(s) with
        the local collections in netCDF, geoTIFF or ZARR.
        """
        self.local_collections_path = local_collections_path
        
    def list_collections(self) -> List[dict]:
        """
        List basic metadata of all collections provided in the local collections folder.

        .. caution::
        :return: list of dictionaries with basic collection metadata.
        """
        data, metadata = _get_local_collections(self.local_collections_path)
        self.data = data
        self.metadata = metadata["collections"]
        return VisualList("collections", data=self.metadata)

    def describe_collection(self, collection_id: str) -> dict:
      """
      Get full collection metadata for given collection id.
      
      .. seealso::
      
          :py:meth:`~openeo.rest.connection.Connection.list_collection_ids`
          to list all collection ids provided by the back-end.

      :param collection_id: collection id
      :return: collection metadata.
      """
      if self.metadata:
        if collection_id == self.metadata['id']:
          return VisualDict("collection", data=self.metadata)
        else:
          print('Invalid collection ID')
          return VisualDict("collection", data='')
      else:
        print('No collections loaded')
        return VisualDict("collection", data='')

    
    def collection_metadata(self, collection_id: str) -> CollectionMetadata:
        # TODO: duplication with `Connection.describe_collection`: deprecate one or the other?
        return CollectionMetadata(self.describe_collection(collection_id))

    
    def load_collection(
            self,
            collection_id: str,
            spatial_extent: Optional[Dict[str, float]] = None,
            temporal_extent: Optional[List[Union[str, datetime.datetime, datetime.date]]] = None,
            bands: Optional[List[str]] = None,
            properties: Optional[Dict[str, Union[str, PGNode, Callable]]] = None,
            fetch_metadata=True,
    ) -> DataCube:
        """
        Load a DataCube by collection id.

        :param collection_id: image collection identifier
        :param spatial_extent: limit data to specified bounding box or polygons
        :param temporal_extent: limit data to specified temporal interval
        :param bands: only add the specified bands
        :param properties: limit data by metadata property predicates
        :return: a datacube containing the requested data
        """
        return DataCube.load_collection(
            collection_id=collection_id, connection=self,
            spatial_extent=spatial_extent, temporal_extent=temporal_extent, bands=bands, properties=properties,
            fetch_metadata=fetch_metadata,
        )

    
    def datacube_from_process(self, process_id: str, namespace: str = None, **kwargs) -> DataCube:
        """
        Load a data cube from a (custom) process.
        :param process_id: The process id.
        :param namespace: optional: process namespace
        :param kwargs: The arguments of the custom process
        :return: A :py:class:`DataCube`, without valid metadata, as the client is not aware of this custom process.
        """
        graph = PGNode(process_id, namespace=namespace, arguments=kwargs)
        return DataCube(graph=graph, connection=self)

    def load_stac(
        self,
        url: str,
        spatial_extent: Optional[Dict[str, float]] = None,
        temporal_extent: Optional[List[Union[str, datetime.datetime, datetime.date]]] = None,
        bands: Optional[List[str]] = None,
        properties: Optional[dict] = None,
    ) -> DataCube:
        """
        Loads data from a static STAC catalog or a STAC API Collection and returns the data as a processable :py:class:`DataCube`.
        A batch job result can be loaded by providing a reference to it.
        If supported by the underlying metadata and file format, the data that is added to the data cube can be
        restricted with the parameters ``spatial_extent``, ``temporal_extent`` and ``bands``.
        If no data is available for the given extents, a ``NoDataAvailable`` error is thrown.
        Remarks:
        * The bands (and all dimensions that specify nominal dimension labels) are expected to be ordered as
          specified in the metadata if the ``bands`` parameter is set to ``null``.
        * If no additional parameter is specified this would imply that the whole data set is expected to be loaded.
          Due to the large size of many data sets, this is not recommended and may be optimized by back-ends to only
          load the data that is actually required after evaluating subsequent processes such as filters.
          This means that the values should be processed only after the data has been limited to the required extent
          and as a consequence also to a manageable size.
        :param url: The URL to a static STAC catalog (STAC Item, STAC Collection, or STAC Catalog)
            or a specific STAC API Collection that allows to filter items and to download assets.
            This includes batch job results, which itself are compliant to STAC.
            For external URLs, authentication details such as API keys or tokens may need to be included in the URL.
            Batch job results can be specified in two ways:
            - For Batch job results at the same back-end, a URL pointing to the corresponding batch job results
              endpoint should be provided. The URL usually ends with ``/jobs/{id}/results`` and ``{id}``
              is the corresponding batch job ID.
            - For external results, a signed URL must be provided. Not all back-ends support signed URLs,
              which are provided as a link with the link relation `canonical` in the batch job result metadata.
        :param spatial_extent:
            Limits the data to load to the specified bounding box or polygons.
            For raster data, the process loads the pixel into the data cube if the point at the pixel center intersects
            with the bounding box or any of the polygons (as defined in the Simple Features standard by the OGC).
            For vector data, the process loads the geometry into the data cube if the geometry is fully within the
            bounding box or any of the polygons (as defined in the Simple Features standard by the OGC).
            Empty geometries may only be in the data cube if no spatial extent has been provided.
            The GeoJSON can be one of the following feature types:
            * A ``Polygon`` or ``MultiPolygon`` geometry,
            * a ``Feature`` with a ``Polygon`` or ``MultiPolygon`` geometry, or
            * a ``FeatureCollection`` containing at least one ``Feature`` with ``Polygon`` or ``MultiPolygon`` geometries.
            Set this parameter to ``None`` to set no limit for the spatial extent.
            Be careful with this when loading large datasets. It is recommended to use this parameter instead of
            using ``filter_bbox()`` or ``filter_spatial()`` directly after loading unbounded data.
        :param temporal_extent:
            Limits the data to load to the specified left-closed temporal interval.
            Applies to all temporal dimensions.
            The interval has to be specified as an array with exactly two elements:
            1.  The first element is the start of the temporal interval.
                The specified instance in time is **included** in the interval.
            2.  The second element is the end of the temporal interval.
                The specified instance in time is **excluded** from the interval.
            The second element must always be greater/later than the first element.
            Otherwise, a `TemporalExtentEmpty` exception is thrown.
            Also supports open intervals by setting one of the boundaries to ``None``, but never both.
            Set this parameter to ``None`` to set no limit for the temporal extent.
            Be careful with this when loading large datasets. It is recommended to use this parameter instead of
            using ``filter_temporal()`` directly after loading unbounded data.
        :param bands:
            Only adds the specified bands into the data cube so that bands that don't match the list
            of band names are not available. Applies to all dimensions of type `bands`.
            Either the unique band name (metadata field ``name`` in bands) or one of the common band names
            (metadata field ``common_name`` in bands) can be specified.
            If the unique band name and the common name conflict, the unique band name has a higher priority.
            The order of the specified array defines the order of the bands in the data cube.
            If multiple bands match a common name, all matched bands are included in the original order.
            It is recommended to use this parameter instead of using ``filter_bands()`` directly after loading unbounded data.
        :param properties:
            Limits the data by metadata properties to include only data in the data cube which
            all given conditions return ``True`` for (AND operation).
            Specify key-value-pairs with the key being the name of the metadata property,
            which can be retrieved with the openEO Data Discovery for Collections.
            The value must be a condition (user-defined process) to be evaluated against a STAC API.
            This parameter is not supported for static STAC.
        .. versionadded:: 0.17.0
        """
        # TODO: detect actual metadata from URL
        metadata = CollectionMetadata(
            {},
            dimensions=[
                SpatialDimension(name="x", extent=[]),
                SpatialDimension(name="y", extent=[]),
                TemporalDimension(name="time", extent=[]),
                BandDimension(name="band", bands=[Band("unknown")]),
            ],
        )
        arguments = {"url": url}
        # TODO: more normalization/validation of extent/band parameters and `properties`
        if spatial_extent:
            arguments["spatial_extent"] = spatial_extent
        if temporal_extent:
            arguments["temporal_extent"] = DataCube._get_temporal_extent(temporal_extent)
        if bands:
            arguments["bands"] = bands
        if properties:
            arguments["properties"] = properties
        cube = self.datacube_from_process(process_id="load_stac", **arguments)
        cube.metadata = metadata
        return cube
    
    def execute(self, process_graph: Union[dict, str, Path]) -> xr.DataArray:
        """
        Execute locally the process graph and return the result as an xarray.DataArray.

        :param process_graph: (flat) dict representing a process graph, or process graph as raw JSON string,
        :return: a datacube containing the requested data
        """
        process_graph = as_flat_graph(process_graph)
        return OpenEOProcessGraph(process_graph).to_callable(PROCESS_REGISTRY)()